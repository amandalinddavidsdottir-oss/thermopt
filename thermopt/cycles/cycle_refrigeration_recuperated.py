import copy
import jaxprop as cpx

from .. import utilities

from ..components import compression_process, expansion_process, heat_exchanger, compute_component_energy_flows

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

    # Extract pressure drops and give short names
    p_source_out = parameters["heat_source"].pop("exit_pressure")
    p_sink_out   = parameters["heat_sink"].pop("exit_pressure")
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
        recuperator_inlet_enthalpy_hot = variables.pop("recuperator_inlet_enthalpy_hot")
    else:
        recuperator_inlet_enthalpy_hot = expander_inlet_h

    # Evaluate compressor
    dp = (1.0 - dp_cooler_h) * (1.0 - dp_recup_h)
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
    dp = (1.0 - dp_heater_c) * (1.0 - dp_recup_c)
    expander_outlet_p = compressor_inlet_p / dp
    expander_efficiency = parameters["expander"].pop("efficiency")
    expander_efficiency_type = parameters["expander"].pop("efficiency_type")
    expander = expansion_process(
        working_fluid,
        expander_inlet_h,
        expander_inlet_p,
        expander_outlet_p,
        expander_efficiency,
        expander_efficiency_type,
    )

    # Evaluate recuperator
    p_out_hot = expander["state_in"].p
    h_out_hot = expander["state_in"].h
    h_in_hot = recuperator_inlet_enthalpy_hot
    p_in_hot = p_out_hot / (1.0 - dp_recup_h)
    p_out_cold = compressor["state_in"].p
    h_out_cold = compressor["state_in"].h
    p_in_cold = p_out_cold / (1.0 - dp_recup_c)
    h_in_cold = h_out_cold - (h_in_hot - h_out_hot)
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
        h_out_cold,
        p_in_cold,
        p_out_cold,
        counter_current=True,
        num_steps=num_elements,
    )

    # Evaluate heater
    h_in_cold = expander["state_out"].h
    p_in_cold = expander["state_out"].p
    h_out_cold = recuperator["cold_side"]["state_in"].h
    p_out_cold = recuperator["cold_side"]["state_in"].p
    T_in_hot = parameters["heat_source"].pop("inlet_temperature")
    p_in_hot = parameters["heat_source"].pop("inlet_pressure")
    h_in_hot = heating_fluid.get_state(cpx.PT_INPUTS, p_in_hot, T_in_hot).h
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
        counter_current=True,
        num_steps=num_elements,
    )

    # Evaluate heat source pump
    # TODO place on other side?
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
    # TODO place on other side?
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
    h_in_hot = compressor["state_out"].h
    p_in_hot = compressor["state_out"].p
    h_out_hot = recuperator["hot_side"]["state_in"].h
    p_out_hot = recuperator["hot_side"]["state_in"].p
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
        counter_current=True,
        num_steps=num_elements,
    )

    # Compute mass flow rates
    Q_duty = parameters.pop("heat_duty")
    m_sink = Q_duty / cooler["q_cold_side"]
    m_total = m_sink * cooler["mass_flow_ratio"]
    m_source = m_total * heater["mass_flow_ratio"]

    # Add the mass flow to the components
    heater["hot_side"]["mass_flow"] = m_source
    heater["cold_side"]["mass_flow"] = m_total
    recuperator["hot_side"]["mass_flow"] = m_total
    recuperator["cold_side"]["mass_flow"] = m_total
    cooler["hot_side"]["mass_flow"] = m_total
    cooler["cold_side"]["mass_flow"] = m_sink
    expander["mass_flow"] = m_total
    compressor["mass_flow"] = m_total
    heat_source_pump["mass_flow"] = m_source
    heat_sink_pump["mass_flow"] = m_sink

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
    COP_heat_pump = Q_out / (W_in - W_out)
    COP_refrigeration = Q_in / (W_in - W_out)
    backwork_ratio = W_out / W_comp
    energy_balance = (Q_in + W_comp) - (W_out + Q_out)  # Ignore pumps

    # Define dictionary with 1st Law analysis
    energy_analysis = {
        "mass_flow_heating_fluid": m_source,
        "mass_flow_working_fluid": m_total,
        "mass_flow_cooling_fluid": m_sink,
        "heater_heat_flow": Q_in,
        "recuperator_heat_flow": recuperator["heat_flow"],
        "cooler_heat_flow": Q_out,
        "expander_power": W_out,
        "compressor_power": W_comp,
        "heat_source_pump_power": heat_source_pump["power"],
        "heat_sink_pump_power": heat_sink_pump["power"],
        "COP_heat_pump": COP_heat_pump,
        "COP_refrigeration": COP_refrigeration,
        "backwork_ratio": backwork_ratio,
        "energy_balance": energy_balance,
    }

    # Evaluate objective function and constraints
    # output = {"components": components}
    output = {"components": components, "energy_analysis": energy_analysis}
    f = utilities.evaluate_objective_function(output, objective_function)
    c_eq, c_ineq, constraint_report = utilities.evaluate_constraints(output, constraints)

    # Set colors and linestyles for plotting
    orange = COLORS_MATLAB[1]
    blue = COLORS_MATLAB[0]
    red = COLORS_MATLAB[6]
    heater["hot_side"]["plot_params"] = {"color": blue, "linestyle": "-"}
    heater["cold_side"]["plot_params"] = {"color": orange, "linestyle": "-"}
    recuperator["hot_side"]["plot_params"] = {"color": orange, "linestyle": "-"}
    recuperator["cold_side"]["plot_params"] = {"color": orange, "linestyle": "-"}
    cooler["hot_side"]["plot_params"] = {"color": orange, "linestyle": "-"}
    cooler["cold_side"]["plot_params"] = {"color": red, "linestyle": "-"}
    expander["plot_params"] = {"color": orange, "linestyle": "-"}
    compressor["plot_params"] = {"color": orange, "linestyle": "-"}

    # Check if any fixed parameter or design variable was not used
    utilities.check_for_unused_keys(parameters, "parameters", raise_error=True)
    utilities.check_for_unused_keys(variables, "variables", raise_error=True)

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
