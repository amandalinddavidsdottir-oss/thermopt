import copy
import numpy as np
import jaxprop as cpx

from .. import utilities

from ..components import (
    compression_process,
    expansion_process,
    heat_exchanger,
    compute_component_energy_flows,
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
        recuperator_effectiveness = variables.pop("recuperator_effectiveness")
    else:
        recuperator_effectiveness = 0.0

    # Mass flow rates are always design variables
    m_total = variables.pop("mass_flow_rate_cycle")
    m_sink = variables.pop("mass_flow_rate_sink")
    if "well_mass_flow_rate" in parameters:
        m_source = parameters.pop("well_mass_flow_rate")
    else:
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
    )

    # Evaluate expander
    dp = (1.0 - dp_cooler_h) * (1.0 - dp_recup_h)
    expander_outlet_p = compressor_inlet_p / dp
    expander_efficiency = parameters["expander"].pop("efficiency", None)
    expander_efficiency_type = parameters["expander"].pop("efficiency_type")

    # Build data_in for correlation-based efficiency types
    expander_data_in = {}
    if expander_efficiency_type == "astolfi-stacking":
        expander_data_in["n_stages"] = parameters["expander"].pop("n_stages")
        if "RPM" in parameters["expander"]:
            expander_data_in["RPM"] = parameters["expander"].pop("RPM")
        intermediate_pressures = []
        for i in range(1, 10):
            key = f"expander_intermediate_pressure_{i}"
            if key in variables:
                intermediate_pressures.append(variables.pop(key))
        if intermediate_pressures:
            expander_data_in["intermediate_pressures"] = intermediate_pressures

    expander = expansion_process(
        working_fluid,
        expander_inlet_h,
        expander_inlet_p,
        expander_outlet_p,
        expander_efficiency,
        expander_efficiency_type,
        mass_flow=m_total,
        data_in=expander_data_in,
    )

    # Evaluate recuperator
    eps = recuperator_effectiveness
    T_in_hot = expander["state_out"].T
    p_in_cold = compressor["state_out"].p
    h_in_cold = compressor["state_out"].h
    p_out_cold = p_in_cold * (1.0 - dp_recup_c)
    h_out_cold_ideal = working_fluid.get_state(cpx.PT_INPUTS, p_out_cold, T_in_hot).h
    h_out_cold_ideal = working_fluid.get_state(cpx.PT_INPUTS, p_out_cold, T_in_hot).h
    h_out_cold_actual = h_in_cold + eps * (h_out_cold_ideal - h_in_cold)
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
        enforce_energy_balance=False,
        counter_current=True,
        num_steps=num_elements,
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
    )

    # Assign mass flows to turbomachinery
    expander["mass_flow"] = m_total
    compressor["mass_flow"] = m_total
    heat_source_pump["mass_flow"] = m_source
    heat_sink_pump["mass_flow"] = m_sink

    W_net = (expander["specific_work"] - compressor["specific_work"]) * m_total

    # Summary of components
    components = {
        "expander": expander,
        "compressor": compressor,
        "recuperator": recuperator,
        "heater": heater,
        "cooler": cooler,
        "heat_source_pump": heat_source_pump,
        "heat_sink_pump": heat_sink_pump,
    }
    compute_component_energy_flows(components)

    # First-law analysis
    Q_in = heater["heat_flow"]
    Q_out = cooler["heat_flow"]
    W_out = expander["power"]
    W_comp = compressor["power"]
    W_aux = heat_source_pump["power"] + heat_sink_pump["power"]
    W_in = W_comp + W_aux
    Q_in_max = m_source * (heater["hot_side"]["state_in"].h - source_out_min.h)
    Q_in_max_ambient = m_source * (heater["hot_side"]["state_in"].h - source_out_ambient.h)
    cycle_efficiency = (W_out - W_in) / Q_in
    system_efficiency = (W_out - W_in) / Q_in_max
    system_efficiency_ambient = (W_out - W_in) / Q_in_max_ambient
    backwork_ratio = W_comp / W_out
    energy_balance = (Q_in + W_comp + W_aux) - (W_out + Q_out)

    # Define dictionary with 1st Law analysis
    energy_analysis = {
        "heater_heat_flow": Q_in,
        "heater_heat_flow_max": Q_in_max,
        "heater_heat_flow_max_ambient": Q_in_max_ambient,
        "recuperator_heat_flow": recuperator["heat_flow"],
        "cooler_heat_flow": Q_out,
        "expander_power": W_out,
        "compressor_power": W_comp,
        "heat_source_pump_power": heat_source_pump["power"],
        "heat_sink_pump_power": heat_sink_pump["power"],
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

    # Evaluate objective function and constraints
    variables_out = {
        "mass_flow_rate_cycle": m_total,
        "mass_flow_rate_sink": m_sink,
    }
    if expander_efficiency_type == "astolfi-stacking":
        for i, p in enumerate(intermediate_pressures):
            variables_out[f"expander_intermediate_pressure_{i+1}"] = p
    output = {
        "components": components,
        "energy_analysis": energy_analysis,
        "variables": variables_out,
    }
    f = utilities.evaluate_objective_function(output, objective_function)
    c_eq, c_ineq, constraint_report = utilities.evaluate_constraints(
        output, constraints
    )

    # Automatically add heat exchanger energy balance residuals as equality constraints
    c_eq_hx = np.array([
        component["energy_balance_residual"]
        for component in components.values()
        if component.get("type") == "heat_exchanger"
        and component.get("energy_balance_residual") is not None
    ])
    if len(c_eq_hx) > 0:
        c_eq = np.concatenate([c_eq, c_eq_hx])

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
