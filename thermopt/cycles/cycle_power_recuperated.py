import copy
import numpy as np
import jaxprop as cpx

from .. import utilities

from ..components import (
    compression_process,
    expansion_process,
    heat_exchanger,
)

COLORS_MATLAB = utilities.COLORS_MATLAB


def evaluate_cycle(
    variables,
    parameters,
    constraints,
    objective_function,
    recuperated=True,
):
    # Create copies to not change the originals
    variables = copy.deepcopy(variables)
    parameters = copy.deepcopy(parameters)

    # Initialize fluid objects
    working_fluid = cpx.Fluid(
        **parameters.pop("working_fluid"), identifier="working_fluid"
    )
    heating_fluid = cpx.Fluid(
        **parameters.pop("heating_fluid"), identifier="heating_fluid"
    )
    cooling_fluid = cpx.Fluid(
        **parameters.pop("cooling_fluid"), identifier="cooling_fluid"
    )

    # Extract special points
    special_points = parameters.pop("special_points")

    # Extract heat source/sink parameters and give short names
    T_source_out_min = parameters["heat_source"].pop("minimum_temperature")
    p_source_out = parameters["heat_source"].pop("exit_pressure")
    p_sink_out = parameters["heat_sink"].pop("exit_pressure")

    # Compute coldest state at the heat source exit
    source_out_min = heating_fluid.get_state(
        cpx.PT_INPUTS, p_source_out, T_source_out_min
    )

    # Ambient reference state for the heat source (for η_utilization / η_ambient)
    # Dead state (T₀, p₀) per DiPippo utilization efficiency convention
    T_ambient = special_points["ambient_temperature"]
    p_ambient = special_points["ambient_pressure"]
    source_out_ambient = heating_fluid.get_state(cpx.PT_INPUTS, p_ambient, T_ambient)
    T0 = T_ambient
    p0 = p_ambient

    # Extract pressure drops and give short names
    dp_heater_h = parameters["heater"].pop("pressure_drop_hot_side")
    dp_heater_c = parameters["heater"].pop("pressure_drop_cold_side")
    dp_cooler_h = parameters["cooler"].pop("pressure_drop_hot_side")
    dp_cooler_c = parameters["cooler"].pop("pressure_drop_cold_side")
    if recuperated:
        dp_recup_h = parameters["recuperator"].pop("pressure_drop_hot_side")
        dp_recup_c = parameters["recuperator"].pop("pressure_drop_cold_side")
    else:
        dp_recup_c, dp_recup_h = 0.0, 0.0

    # Extract design variables from dictionary (make sure all are used)
    expander_inlet_p = variables.pop("expander_inlet_pressure")
    expander_inlet_h = variables.pop("expander_inlet_enthalpy")
    compressor_inlet_p = variables.pop("compressor_inlet_pressure")
    compressor_inlet_h = variables.pop("compressor_inlet_enthalpy")
    heat_source_temperature_out = variables.pop("heat_source_exit_temperature")
    heat_sink_temperature_out = variables.pop("heat_sink_exit_temperature")
    if recuperated:
        recuperator_cold_outlet_enthalpy = variables.pop("recuperator_cold_outlet_enthalpy")
    else:
        recuperator_cold_outlet_enthalpy = 0.0

    # Mass flow rates are design variables
    m_total = variables.pop("mass_flow_rate_cycle")
    m_sink = variables.pop("mass_flow_rate_sink")
    m_source = variables.pop("well_mass_flow_rate")

    # Evaluate  compressor
    dp = (1.0 - dp_heater_c) * (1.0 - dp_recup_c)
    compressor_outlet_p = expander_inlet_p / dp
    compressor_eff = parameters["compressor"].pop("efficiency")
    compressor_eff_type = parameters["compressor"].pop("efficiency_type")
    compressor = compression_process(
        working_fluid,
        compressor_inlet_h,
        compressor_inlet_p,
        compressor_outlet_p,
        compressor_eff,
        compressor_eff_type,
        mass_flow=m_total,
        T0=T0,
        p0=p0,
    )

    # Evaluate expander
    dp = (1.0 - dp_cooler_h) * (1.0 - dp_recup_h)
    expander_outlet_p = compressor_inlet_p / dp
    expander_efficiency = parameters["expander"].pop("efficiency", None)
    expander_efficiency_type = parameters["expander"].pop("efficiency_type")

    # Build data_in for correlation-based efficiency types
    expander_data_in = {}
    n_turbines_parallel = parameters["expander"].pop("n_turbines_parallel", 1)
    if expander_efficiency_type == "astolfi-stacking":
        expander_data_in["n_stages"] = parameters["expander"].pop("n_stages")
        if "RPM" in parameters["expander"]:
            expander_data_in["RPM"] = parameters["expander"].pop("RPM")

        # Intermediate stage pressures may be declared in the YAML in one
        # of two equivalent parameterizations, chosen per-stage:
        #
        #   (a) expander_intermediate_pressure_i — absolute pressure [Pa]
        #
        #   (b) expander_intermediate_ratio_i    — nested log expansion
        #       ratio in (0, 1). The absolute pressure is reconstructed as
        #           p_i = p_out * (p_{i-1} / p_out) ** r_i,
        #       with p_0 = p_in. Because 0 < r_i < 1 we have
        #       p_out < p_i < p_{i-1} by construction, so the stage
        #       ordering p_in > p_1 > ... > p_{n-1} > p_out cannot be
        #       violated — not by a gradient probe, not by a line-search
        #       step, not by anything. Using ratios eliminates the class
        #       of NaN crashes that arise when SLSQP steps into a region
        #       where Dh_is_stg is non-positive.
        intermediate_pressures = []
        intermediate_ratios = []
        for i in range(1, 10):
            p_key = f"expander_intermediate_pressure_{i}"
            r_key = f"expander_intermediate_ratio_{i}"
            if p_key in variables and r_key in variables:
                raise ValueError(
                    f"Both '{p_key}' and '{r_key}' are declared as design "
                    f"variables. Use exactly one parameterization per stage."
                )
            if p_key in variables:
                intermediate_pressures.append(variables.pop(p_key))
            elif r_key in variables:
                intermediate_ratios.append(variables.pop(r_key))

        # Reconstruct absolute pressures from ratios (nested log form).
        # This happens on every evaluation, so the optimizer sees ratios
        # but the cycle model sees pressures — expansion_process and
        # everything downstream are untouched.
        if intermediate_ratios:
            p_prev = expander_inlet_p
            for r in intermediate_ratios:
                p_i = expander_outlet_p * (p_prev / expander_outlet_p) ** r
                intermediate_pressures.append(p_i)
                p_prev = p_i

        if intermediate_pressures:
            expander_data_in["intermediate_pressures"] = intermediate_pressures

    turbine_shafts = [
        expansion_process(
            working_fluid,
            expander_inlet_h,
            expander_inlet_p,
            expander_outlet_p,
            expander_efficiency,
            expander_efficiency_type,
            mass_flow=m_total / n_turbines_parallel,
            data_in=expander_data_in,
            T0=T0,
            p0=p0,
        )
        for _ in range(n_turbines_parallel)
    ]
    expander = turbine_shafts[0]
    expander["mass_flow"] = m_total
    expander["n_turbines_parallel"] = n_turbines_parallel
    expander["power"] = sum(s["power"] for s in turbine_shafts)
    expander["energy_analysis"]["power"] = expander["power"]
    expander["exergy_analysis"]["E_fuel"] = sum(
        s["exergy_analysis"]["E_fuel"] for s in turbine_shafts
    )
    expander["exergy_analysis"]["E_product"] = sum(
        s["exergy_analysis"]["E_product"] for s in turbine_shafts
    )
    expander["exergy_analysis"]["E_D"] = sum(
        s["exergy_analysis"]["E_D"] for s in turbine_shafts
    )

    # Evaluate recuperator
    p_in_cold = compressor["state_out"].p
    h_in_cold = compressor["state_out"].h
    p_out_cold = p_in_cold * (1.0 - dp_recup_c)
    h_out_cold_actual = recuperator_cold_outlet_enthalpy
    p_in_hot = expander["state_out"].p
    h_in_hot = expander["state_out"].h
    p_out_hot = p_in_hot * (1.0 - dp_recup_h)
    h_out_hot = h_in_hot - (h_out_cold_actual - h_in_cold)
    if recuperated:
        num_elements = parameters["recuperator"].pop("num_elements")
    else:
        num_elements = 2
    recuperator = heat_exchanger(
        working_fluid,
        h_in_hot,
        h_out_hot,
        p_in_hot,
        p_out_hot,
        working_fluid,
        h_in_cold,
        h_out_cold_actual,
        p_in_cold,
        p_out_cold,
        mass_flow_hot=m_total,
        mass_flow_cold=m_total,
        counter_current=True,
        num_steps=num_elements,
        T0=T0,
        p0=p0,
    )

    # Evaluate heater
    h_in_cold = recuperator["cold_side"]["state_out"].h
    p_in_cold = recuperator["cold_side"]["state_out"].p
    h_out_cold = expander["state_in"].h
    p_out_cold = expander["state_in"].p
    p_in_hot = parameters["heat_source"].pop("inlet_pressure")
    h_in_hot = parameters["heat_source"].pop("inlet_enthalpy")
    T_out_hot = heat_source_temperature_out
    p_out_hot = p_in_hot * (1 - dp_heater_h)
    h_out_hot = heating_fluid.get_state(cpx.PT_INPUTS, p_out_hot, T_out_hot).h
    num_elements = parameters["heater"].pop("num_elements")
    heater = heat_exchanger(
        heating_fluid,
        h_in_hot,
        h_out_hot,
        p_in_hot,
        p_out_hot,
        working_fluid,
        h_in_cold,
        h_out_cold,
        p_in_cold,
        p_out_cold,
        mass_flow_hot=m_source,
        mass_flow_cold=m_total,
        counter_current=True,
        num_steps=num_elements,
        T0=T0,
        p0=p0,
    )

    # Evaluate heat source pump
    h_in = heater["hot_side"]["state_out"].h
    p_in = heater["hot_side"]["state_out"].p
    p_out = p_source_out
    efficiency = parameters["heat_source_pump"].pop("efficiency")
    efficiency_type = parameters["heat_source_pump"].pop("efficiency_type")
    heat_source_pump = compression_process(
        heating_fluid,
        h_in,
        p_in,
        p_out,
        efficiency,
        efficiency_type,
        mass_flow=m_source,
        T0=T0,
        p0=p0,
    )

    # Evaluate heat sink pump
    T_in = parameters["heat_sink"].pop("inlet_temperature")
    p_in = parameters["heat_sink"].pop("inlet_pressure")
    h_in = cooling_fluid.get_state(cpx.PT_INPUTS, p_in, T_in).h
    p_out = p_sink_out / (1 - dp_cooler_c)
    efficiency = parameters["heat_sink_pump"].pop("efficiency")
    efficiency_type = parameters["heat_sink_pump"].pop("efficiency_type")
    heat_sink_pump = compression_process(
        cooling_fluid,
        h_in,
        p_in,
        p_out,
        efficiency,
        efficiency_type,
        mass_flow=m_sink,
        T0=T0,
        p0=p0,
    )

    # Evaluate cooler
    p_in_cold = heat_sink_pump["state_out"].p
    h_in_cold = heat_sink_pump["state_out"].h
    p_out_cold = p_in_cold * (1 - dp_cooler_c)
    T_out_cold = heat_sink_temperature_out
    h_out_cold = cooling_fluid.get_state(cpx.PT_INPUTS, p_out_cold, T_out_cold).h
    h_in_hot = recuperator["hot_side"]["state_out"].h
    p_in_hot = recuperator["hot_side"]["state_out"].p
    h_out_hot = compressor["state_in"].h
    p_out_hot = compressor["state_in"].p
    num_elements = parameters["cooler"].pop("num_elements")
    cooler = heat_exchanger(
        working_fluid,
        h_in_hot,
        h_out_hot,
        p_in_hot,
        p_out_hot,
        cooling_fluid,
        h_in_cold,
        h_out_cold,
        p_in_cold,
        p_out_cold,
        mass_flow_hot=m_total,
        mass_flow_cold=m_sink,
        counter_current=True,
        num_steps=num_elements,
        T0=T0,
        p0=p0,
    )

    components = {
        "expander": expander,
        "compressor": compressor,
        "recuperator": recuperator,
        "heater": heater,
        "cooler": cooler,
        "heat_source_pump": heat_source_pump,
        "heat_sink_pump": heat_sink_pump,
    }

    # First-law analysis
    W_out = expander["energy_analysis"]["power"]
    W_comp = compressor["energy_analysis"]["power"]
    W_aux = (
        heat_source_pump["energy_analysis"]["power"]
        + heat_sink_pump["energy_analysis"]["power"]
    )
    W_in = W_comp + W_aux
    Q_in = heater["energy_analysis"]["Q_hot"]
    Q_out = cooler["energy_analysis"]["Q_hot"]
    W_net = W_out - W_comp  # net cycle power (turbine minus WF pump)

    Q_in_max = m_source * (heater["hot_side"]["state_in"].h - source_out_min.h)
    Q_in_max_ambient = m_source * (
        heater["hot_side"]["state_in"].h - source_out_ambient.h
    )
    cycle_efficiency = (W_out - W_in) / Q_in
    system_efficiency = (W_out - W_in) / Q_in_max
    system_efficiency_ambient = (W_out - W_in) / Q_in_max_ambient
    backwork_ratio = W_comp / W_out
    energy_balance = (Q_in + W_comp) - (W_out + Q_out)

    energy_analysis = {
        "heater_heat_flow": Q_in,
        "heater_heat_flow_max": Q_in_max,
        "heater_heat_flow_max_ambient": Q_in_max_ambient,
        "recuperator_heat_flow": recuperator["energy_analysis"]["Q_hot"],
        "cooler_heat_flow": Q_out,
        "net_cycle_power": W_net,
        "net_system_power": W_out - W_in,
        "mass_flow_heating_fluid": m_source,
        "mass_flow_working_fluid": m_total,
        "mass_flow_cooling_fluid": m_sink,
        "cycle_efficiency": cycle_efficiency,
        "system_efficiency": system_efficiency,
        "system_efficiency_ambient": system_efficiency_ambient,
        "backwork_ratio": backwork_ratio,
        "energy_balance": energy_balance,
    }

    # Cycle-level exergy analysis

    _aux_pump_names = {"heat_source_pump", "heat_sink_pump"}

    E_fuel_system = heater["exergy_analysis"]["E_fuel"]

    E_product_system = energy_analysis["net_system_power"]

    E_loss_cooler = cooler["exergy_analysis"]["E_product"]

    E_D_total = sum(comp["exergy_analysis"]["E_D"] for comp in components.values())
    E_D_internal = sum(
        comp["exergy_analysis"]["E_D"]
        for name, comp in components.items()
        if name not in _aux_pump_names
    )

    eta_exergy = E_product_system / E_fuel_system if E_fuel_system != 0 else 0.0

    balance_residual = E_fuel_system - (W_net + E_D_internal + E_loss_cooler)

    exergy_analysis = {
        "T0": T0,
        "p0": p0,
        "E_fuel_system": E_fuel_system,
        "E_product_system": E_product_system,
        "E_loss_cooler": E_loss_cooler,
        "E_D_total": E_D_total,
        "E_D_internal": E_D_internal,
        "eta_exergy": eta_exergy,
        "W_net_cycle": W_net,
        "balance_residual": balance_residual,
    }

    # Evaluate objective function and constraints
    variables_out = {
        "mass_flow_rate_cycle": m_total,
        "mass_flow_rate_sink": m_sink,
    }
    if expander_efficiency_type == "astolfi-stacking":
        # Always emit absolute pressures so downstream constraints / reports
        # that reference expander_intermediate_pressure_i work identically
        # regardless of which input parameterization the YAML chose.
        for i, p in enumerate(intermediate_pressures):
            variables_out[f"expander_intermediate_pressure_{i+1}"] = p
        # If the YAML used the ratio form, also emit the ratios so
        # constraint expressions that reference $variables.expander_intermediate_ratio_i
        # can resolve.
        for i, r in enumerate(intermediate_ratios):
            variables_out[f"expander_intermediate_ratio_{i+1}"] = r
    output = {
        "components": components,
        "energy_analysis": energy_analysis,
        "exergy_analysis": exergy_analysis,
        "variables": variables_out,
    }
    f = utilities.evaluate_objective_function(output, objective_function)
    c_eq, c_ineq, constraint_report = utilities.evaluate_constraints(
        output, constraints
    )

    # Automatically add heat exchanger energy balance residuals as equality constraints
    c_eq_hx = np.array(
        [
            component["energy_balance_residual"]
            for name, component in components.items()
            if component.get("type") == "heat_exchanger"
            and name != "recuperator"  # Recuperator is handeled by the effectiveness
        ]
    )
    if len(c_eq_hx) > 0:
        c_eq = np.concatenate([c_eq, c_eq_hx])

    # Add energy balance residuals to constraint_report so they appear in
    # the optimization report. The residual is (Q_hot - Q_cold) / Q_hot,
    # so the equality target is 0.0 and no normalization is needed.
    tol = 1e-4
    for name, component in components.items():
        if component.get("type") == "heat_exchanger" and name != "recuperator":
            residual = float(component["energy_balance_residual"])
            constraint_report.append(
                {
                    "name": f"$components.{name}.energy_balance_residual",
                    "value": residual,
                    "type": "=",
                    "target": 0.0,
                    "mismatch": residual,
                    "normalized_mismatch": residual,
                    "satisfied": abs(residual) < tol,
                    "normalize": None,
                }
            )

    # Set colors for plotting
    orange = COLORS_MATLAB[1]
    blue = COLORS_MATLAB[0]
    red = COLORS_MATLAB[6]
    heater["hot_side"]["plot_params"] = {"color": red, "linestyle": "-"}
    heater["cold_side"]["plot_params"] = {"color": orange, "linestyle": "-"}
    recuperator["hot_side"]["plot_params"] = {"color": orange, "linestyle": "-"}
    recuperator["cold_side"]["plot_params"] = {"color": orange, "linestyle": "-"}
    cooler["hot_side"]["plot_params"] = {"color": orange, "linestyle": "-"}
    cooler["cold_side"]["plot_params"] = {"color": blue, "linestyle": "-"}
    expander["plot_params"] = {"color": orange, "linestyle": "-"}
    compressor["plot_params"] = {"color": orange, "linestyle": "-"}

    # Check if any fixed parameter or design variable was not used
    utilities.check_for_unused_keys(parameters, "parameters", raise_error=True)
    utilities.check_for_unused_keys(variables, "variables", raise_error=True)

    # Cycle performance summary
    output = {
        **output,
        "working_fluid": working_fluid,
        "heating_fluid": heating_fluid,
        "cooling_fluid": cooling_fluid,
        "components": components,
        "objective_function": f,
        "equality_constraints": c_eq,
        "inequality_constraints": c_ineq,
        "constraints_report": constraint_report,
    }

    return output
