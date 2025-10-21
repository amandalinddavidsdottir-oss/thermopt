import copy
import jaxprop as cpx

from .. import utilities

from ..components import compression_process, expansion_process, heat_exchanger

COLORS_MATLAB = utilities.COLORS_MATLAB


def evaluate_cycle(
    variables,
    parameters,
    constraints,
    objective_function,
):
    
    # --------------------------------------------------------------------- #
    # ---------------------- Charge cycle definition ---------------------- #
    # --------------------------------------------------------------------- #

    # Create copies to not change the originals
    variables = copy.deepcopy(variables)
    parameters = copy.deepcopy(parameters)

    # Initialize fluid objects
    working_fluid = cpx.Fluid(
        **parameters.pop("working_fluid"), identifier="working_fluid"
    )
    heating_fluid = cpx.Fluid(
        **parameters.pop("hot_storage_fluid"), identifier="hot_storage_fluid"
    )
    cooling_fluid = cpx.Fluid(
        **parameters.pop("cold_storage_fluid"), identifier="cold_storage_fluid"
    )

    # Extract special points
    special_points = parameters.pop("special_points")

    # Extract pressure drops and give short names
    cold_storage_pressure = parameters["cold_storage"].pop("pressure")
    hot_storage_pressure = parameters["hot_storage"].pop("pressure")
    dp_heater_h = parameters["heater_charge"].pop("pressure_drop_hot_side")  # Unused now that there are no pumps
    dp_heater_c = parameters["heater_charge"].pop("pressure_drop_cold_side")
    dp_cooler_h = parameters["cooler_charge"].pop("pressure_drop_hot_side")
    dp_cooler_c = parameters["cooler_charge"].pop("pressure_drop_cold_side")  # Unused now that there are no pumps
    dp_recup_h = parameters["recuperator_charge"].pop("pressure_drop_hot_side")
    dp_recup_c = parameters["recuperator_charge"].pop("pressure_drop_cold_side")
   
    # Extract design variables from dictionary (make sure all are used)
    cold_storage_upper_temperature = variables.pop("cold_storage_upper_temperature")
    cold_storage_lower_temperature = variables.pop("cold_storage_lower_temperature")
    hot_storage_upper_temperature = variables.pop("hot_storage_upper_temperature")
    hot_storage_lower_temperature = variables.pop("hot_storage_lower_temperature")
    expander_inlet_p = variables.pop("expander_inlet_pressure_charge")
    expander_inlet_h = variables.pop("expander_inlet_enthalpy_charge")
    compressor_inlet_p = variables.pop("compressor_inlet_pressure_charge")
    compressor_inlet_h = variables.pop("compressor_inlet_enthalpy_charge")
    recuperator_inlet_enthalpy_hot_charge = variables.pop("recuperator_inlet_enthalpy_hot_charge")
    
    # Extract variables for non-dimensional turbomachinery models
    mass_flow_rate_charge = variables.pop("mass_flow_rate_charge")
    mass_flow_rate_discharge = variables.pop("mass_flow_rate_discharge")

    expander_charge_data = parameters["expander_charge"].pop("data_in")
    expander_charge_data["specific_speed"] = variables.pop("expander_specific_speed_charge")

    expander_discharge_data = parameters["expander_discharge"].pop("data_in")
    expander_discharge_data["specific_speed"] = variables.pop("expander_specific_speed_discharge")

    compressor_charge_data = parameters["compressor_charge"].pop("data_in")
    compressor_charge_data["backsweep"] = variables.pop("compressor_backsweep_charge")
    compressor_charge_data["flow_coefficient"] = variables.pop("compressor_flow_coefficient_charge")

    compressor_discharge_data = parameters["compressor_discharge"].pop("data_in")
    compressor_discharge_data["backsweep"] = variables.pop("compressor_backsweep_discharge")
    compressor_discharge_data["flow_coefficient"] = variables.pop("compressor_flow_coefficient_discharge")

    # Evaluate compressor
    dp = (1.0 - dp_cooler_h) * (1.0 - dp_recup_h)
    compressor_outlet_p = expander_inlet_p / dp
    compressor_eff = parameters["compressor_charge"].pop("efficiency")
    compressor_eff_type = parameters["compressor_charge"].pop("efficiency_type")
    compressor_charge = compression_process(
        working_fluid,
        compressor_inlet_h,
        compressor_inlet_p,
        compressor_outlet_p,
        compressor_eff,
        compressor_eff_type,
        mass_flow_rate_charge,
        compressor_charge_data,
    )

    # Evaluate expander
    dp = (1.0 - dp_heater_c) * (1.0 - dp_recup_c)
    expander_outlet_p = compressor_inlet_p / dp
    expander_efficiency = parameters["expander_charge"].pop("efficiency")
    expander_efficiency_type = parameters["expander_charge"].pop("efficiency_type")
    expander_charge = expansion_process(
        working_fluid,
        expander_inlet_h,
        expander_inlet_p,
        expander_outlet_p,
        expander_efficiency,
        expander_efficiency_type,
        mass_flow_rate_charge,
        expander_charge_data,
    )

    # Evaluate recuperator
    p_out_hot = expander_charge["state_in"].p
    h_out_hot = expander_charge["state_in"].h
    h_in_hot = recuperator_inlet_enthalpy_hot_charge
    p_in_hot = p_out_hot / (1.0 - dp_recup_h)
    p_out_cold = compressor_charge["state_in"].p
    h_out_cold = compressor_charge["state_in"].h
    p_in_cold = p_out_cold / (1.0 - dp_recup_c)
    h_in_cold = h_out_cold - (h_in_hot - h_out_hot)
    num_elements = parameters["recuperator_charge"].pop("num_elements")
    recuperator_charge = heat_exchanger(
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
    h_in_cold = expander_charge["state_out"].h
    p_in_cold = expander_charge["state_out"].p
    h_out_cold = recuperator_charge["cold_side"]["state_in"].h
    p_out_cold = recuperator_charge["cold_side"]["state_in"].p
    T_in_hot = cold_storage_upper_temperature
    p_in_hot = cold_storage_pressure
    h_in_hot = heating_fluid.get_state(cpx.PT_INPUTS, p_in_hot, T_in_hot).h
    T_out_hot = cold_storage_lower_temperature
    p_out_hot = cold_storage_pressure
    h_out_hot = heating_fluid.get_state(cpx.PT_INPUTS, p_out_hot, T_out_hot).h
    num_elements = parameters["heater_charge"].pop("num_elements")
    heater_charge = heat_exchanger(
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

    # Evaluate cooler
    T_in_cold = hot_storage_lower_temperature
    p_in_cold = hot_storage_pressure
    h_in_cold = cooling_fluid.get_state(cpx.PT_INPUTS, p_in_cold, T_in_cold).h
    T_out_cold = hot_storage_upper_temperature
    p_out_cold = hot_storage_pressure
    h_out_cold = cooling_fluid.get_state(cpx.PT_INPUTS, p_out_cold, T_out_cold).h
    h_in_hot = compressor_charge["state_out"].h
    p_in_hot = compressor_charge["state_out"].p
    h_out_hot = recuperator_charge["hot_side"]["state_in"].h
    p_out_hot = recuperator_charge["hot_side"]["state_in"].p
    num_elements = parameters["cooler_charge"].pop("num_elements")
    cooler_charge = heat_exchanger(
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
    W_charge = parameters.pop("charging_power")
    m_total_charge = W_charge / (compressor_charge["specific_work"] - expander_charge["specific_work"]) 
    m_sink_charge = m_total_charge*(cooler_charge["q_hot_side"])/(cooler_charge["q_cold_side"])
    m_source_charge = m_total_charge*(heater_charge["q_cold_side"])/(heater_charge["q_hot_side"])
    m_error_charge = (m_total_charge - mass_flow_rate_charge) / m_total_charge
    
    # Add the mass flow to the components
    heater_charge["hot_side"]["mass_flow"] = m_source_charge
    heater_charge["cold_side"]["mass_flow"] = m_total_charge
    recuperator_charge["hot_side"]["mass_flow"] = m_total_charge
    recuperator_charge["cold_side"]["mass_flow"] = m_total_charge
    cooler_charge["hot_side"]["mass_flow"] = m_total_charge
    cooler_charge["cold_side"]["mass_flow"] = m_sink_charge
    expander_charge["mass_flow"] = m_total_charge
    compressor_charge["mass_flow"] = m_total_charge


    # --------------------------------------------------------------------- #
    # -------------------- Discharge cycle definition --------------------- #
    # --------------------------------------------------------------------- #

    # Extract pressure drops and give short names
    dp_heater_h = parameters["heater_discharge"].pop("pressure_drop_hot_side")  # Unused now that there are no pumps
    dp_heater_c = parameters["heater_discharge"].pop("pressure_drop_cold_side")
    dp_cooler_h = parameters["cooler_discharge"].pop("pressure_drop_hot_side")
    dp_cooler_c = parameters["cooler_discharge"].pop("pressure_drop_cold_side")  # Unused now that there are no pumps
    dp_recup_h = parameters["recuperator_discharge"].pop("pressure_drop_hot_side")
    dp_recup_c = parameters["recuperator_discharge"].pop("pressure_drop_cold_side")
 
    # Extract design variables from dictionary (make sure all are used)
    cold_storage_upper_temperature_discharge = variables.pop("cold_storage_upper_temperature_discharge")
    expander_inlet_p = variables.pop("expander_inlet_pressure_discharge")
    expander_inlet_h = variables.pop("expander_inlet_enthalpy_discharge")
    compressor_inlet_p = variables.pop("compressor_inlet_pressure_discharge")
    compressor_inlet_h = variables.pop("compressor_inlet_enthalpy_discharge")
    recuperator_outlet_enthalpy_hot_discharge = variables.pop("recuperator_outlet_enthalpy_hot_discharge")

    # Evaluate  compressor
    dp = (1.0 - dp_heater_c) * (1.0 - dp_recup_c)
    compressor_outlet_p = expander_inlet_p / dp
    compressor_eff = parameters["compressor_discharge"].pop("efficiency")
    compressor_eff_type = parameters["compressor_discharge"].pop("efficiency_type")
    compressor_discharge = compression_process(
        working_fluid,
        compressor_inlet_h,
        compressor_inlet_p,
        compressor_outlet_p,
        compressor_eff,
        compressor_eff_type,
        mass_flow_rate_discharge,
        compressor_discharge_data,
    )

    # Evaluate expander
    dp = (1.0 - dp_cooler_h) * (1.0 - dp_recup_h)
    expander_outlet_p = compressor_inlet_p / dp
    expander_efficiency = parameters["expander_discharge"].pop("efficiency")
    expander_efficiency_type = parameters["expander_discharge"].pop("efficiency_type")
    expander_discharge = expansion_process(
        working_fluid,
        expander_inlet_h,
        expander_inlet_p,
        expander_outlet_p,
        expander_efficiency,
        expander_efficiency_type,
        mass_flow_rate_discharge,
        expander_discharge_data,
    )

    # Evaluate recuperator
    p_in_hot = expander_discharge["state_out"].p
    h_in_hot = expander_discharge["state_out"].h
    p_out_hot = p_in_hot * (1.0 - dp_recup_h)
    h_out_hot = recuperator_outlet_enthalpy_hot_discharge
    p_in_cold = compressor_discharge["state_out"].p
    h_in_cold = compressor_discharge["state_out"].h
    p_out_cold = p_in_cold * (1.0 - dp_recup_c)
    h_out_cold = h_in_cold + (h_in_hot - h_out_hot)
    num_elements = parameters["recuperator_discharge"].pop("num_elements")
    recuperator_discharge = heat_exchanger(
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
    h_in_cold = recuperator_discharge["cold_side"]["state_out"].h
    p_in_cold = recuperator_discharge["cold_side"]["state_out"].p
    h_out_cold = expander_discharge["state_in"].h
    p_out_cold = expander_discharge["state_in"].p
    T_in_hot = hot_storage_upper_temperature
    p_in_hot = hot_storage_pressure
    h_in_hot = heating_fluid.get_state(cpx.PT_INPUTS, p_in_hot, T_in_hot).h
    T_out_hot = hot_storage_lower_temperature
    p_out_hot = hot_storage_pressure
    h_out_hot = heating_fluid.get_state(cpx.PT_INPUTS, p_out_hot, T_out_hot).h
    num_elements = parameters["heater_discharge"].pop("num_elements")
    heater_discharge = heat_exchanger(
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

    # Evaluate cooler
    T_in_cold = cold_storage_lower_temperature
    p_in_cold = cold_storage_pressure
    h_in_cold = cooling_fluid.get_state(cpx.PT_INPUTS, p_in_cold, T_in_cold).h
    T_out_cold = cold_storage_upper_temperature_discharge
    p_out_cold = cold_storage_pressure
    h_out_cold = cooling_fluid.get_state(cpx.PT_INPUTS, p_out_cold, T_out_cold).h
    h_in_hot = recuperator_discharge["hot_side"]["state_out"].h
    p_in_hot = recuperator_discharge["hot_side"]["state_out"].p
    h_out_hot = compressor_discharge["state_in"].h
    p_out_hot = compressor_discharge["state_in"].p
    num_elements = parameters["cooler_discharge"].pop("num_elements")
    cooler_discharge = heat_exchanger(
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
    m_source_discharge = m_sink_charge
    m_total_discharge = m_source_discharge*heater_discharge["q_hot_side"]/heater_discharge["q_cold_side"] #an addition
    m_sink_discharge = m_total_discharge*(cooler_discharge["q_hot_side"])/(cooler_discharge["q_cold_side"]) #an addition
    m_error_discharge = (m_total_discharge - mass_flow_rate_discharge) / m_total_discharge
    m_mismatch = (m_sink_discharge - m_source_charge) / m_sink_discharge

    # Add the mass flow to the components
    heater_discharge["hot_side"]["mass_flow"] = m_source_discharge
    heater_discharge["cold_side"]["mass_flow"] = m_total_discharge
    recuperator_discharge["hot_side"]["mass_flow"] = m_total_discharge
    recuperator_discharge["cold_side"]["mass_flow"] = m_total_discharge
    cooler_discharge["hot_side"]["mass_flow"] = m_total_discharge
    cooler_discharge["cold_side"]["mass_flow"] = m_sink_discharge
    expander_discharge["mass_flow"] = m_total_discharge
    compressor_discharge["mass_flow"] = m_total_discharge


    # --------------------------------------------------------------------- #
    # ------------ Energy analysis and postprocessing --------------------- #
    # --------------------------------------------------------------------- #

    # Summary of components
    components = {
        "expander_charge": expander_charge,
        "compressor_charge": compressor_charge,
        "recuperator_charge": recuperator_charge,
        "heater_charge": heater_charge,
        "cooler_charge": cooler_charge,
        "expander_discharge": expander_discharge,
        "compressor_discharge": compressor_discharge,
        "recuperator_discharge": recuperator_discharge,
        "heater_discharge": heater_discharge,
        "cooler_discharge": cooler_discharge,
    }

    # Compute energy balances
    for name, component in components.items():
        if component["type"] == "heat_exchanger":
            hot = component["hot_side"]
            cold = component["cold_side"]
            Q1 = hot["mass_flow"] * (hot["state_in"].h - hot["state_out"].h)
            Q2 = cold["mass_flow"] * (cold["state_out"].h - cold["state_in"].h)
            component["heat_flow_rate"] = Q1
            component["heat_flow_rate_"] = Q2
            component["heat_balance"] = Q1 - Q2
            component["power"] = 0.0

        if component["type"] in ["compressor", "expander"]:
            component["power"] = component["mass_flow"] * component["specific_work"]
            component["heat_flow_rate"] = 0.00

    # First law analysis of energy flows
    Q_in_charge = heater_charge["heat_flow_rate"]
    Q_out_charge = cooler_charge["heat_flow_rate"]
    W_out_charge = expander_charge["power"]
    W_in_charge = compressor_charge["power"]
    W_net_charge = W_out_charge - W_in_charge
    COP_heat_pump = Q_out_charge / (W_in_charge - W_out_charge)
    COP_refrigeration = Q_in_charge / (W_in_charge - W_out_charge)
    backwork_ratio_charge = W_out_charge / W_in_charge
    energy_balance_charge = (Q_in_charge + W_in_charge) - (W_out_charge + Q_out_charge)
    Q_in_discharge = heater_discharge["heat_flow_rate"]
    Q_out_discharge = cooler_discharge["heat_flow_rate"]
    W_out_discharge = expander_discharge["power"]
    W_in_discharge = compressor_discharge["power"]
    W_net_discharge = W_out_discharge - W_in_discharge
    cycle_efficiency_discharge = (W_out_discharge - W_in_discharge) / Q_in_discharge
    backwork_ratio_discharge = W_in_discharge / W_out_discharge
    energy_balance_discharge = (Q_in_discharge + W_in_discharge) - (W_out_discharge + Q_out_discharge)  # Add additional rejected heat in sink
    roundtrip_efficiency = COP_heat_pump*cycle_efficiency_discharge
    energy_analysis = {
        "mass_flow_heating_fluid_charge": m_source_charge,
        "mass_flow_working_fluid_charge": m_total_charge,
        "mass_flow_cooling_fluid_charge": m_sink_charge,
        "heater_heat_flow_charge": Q_in_charge,
        "recuperator_heat_flow_charge": recuperator_charge["heat_flow_rate"],
        "cooler_heat_flow_charge": Q_out_charge,
        "expander_power_charge": W_out_charge,
        "compressor_power_charge": W_in_charge,
        "net_cycle_power_charge": W_net_charge,
        "COP_heat_pump": COP_heat_pump,
        "COP_refrigeration": COP_refrigeration,
        "backwork_ratio_charge": backwork_ratio_charge,
        "energy_balance_charge": energy_balance_charge,
        "mass_flow_heating_fluid_discharge": m_source_discharge,
        "mass_flow_working_fluid_discharge": m_total_discharge,
        "mass_flow_cooling_fluid_discharge": m_sink_discharge,
        "heater_heat_flow_discharge": Q_in_discharge,
        "recuperator_heat_flow_discharge": recuperator_discharge["heat_flow_rate"],
        "cooler_heat_flow_discharge": Q_out_discharge,
        "expander_power_discharge": W_out_discharge,
        "compressor_power_discharge": W_in_discharge,
        "net_cycle_power_discharge": W_net_discharge,
        "cycle_efficiency_discharge": cycle_efficiency_discharge,
        "backwork_ratio_discharge": backwork_ratio_discharge,
        "energy_balance_discharge": energy_balance_discharge,
        "roundtrip_efficiency": roundtrip_efficiency,
        "mass_flow_mismatch": m_mismatch,
        "cold_storage_lower_temperature": cold_storage_lower_temperature,
        "cold_storage_upper_temperature": cold_storage_upper_temperature,
        "cold_storage_upper_temperature_discharge": cold_storage_upper_temperature_discharge,
        "cold_storage_pressure": cold_storage_pressure,
        "hot_storage_lower_temperature": hot_storage_lower_temperature,
        "hot_storage_upper_temperature": hot_storage_upper_temperature,
        "hot_storage_pressure": hot_storage_pressure,
        "m_error_charge": m_error_charge,
        "m_error_discharge": m_error_discharge,
    } 

    # Evaluate objective function and constraints
    output = {"components": components, "energy_analysis": energy_analysis}
    f = utilities.evaluate_objective_function(output, objective_function)
    c_eq, c_ineq, constraint_report = utilities.evaluate_constraints(output, constraints)

    # Set colors and linestyles for plotting
    orange = COLORS_MATLAB[1]
    blue = COLORS_MATLAB[0]
    red = COLORS_MATLAB[6]
    heater_charge["hot_side"]["plot_params"] = {"color": blue, "linestyle": "-"}
    heater_charge["cold_side"]["plot_params"] = {"color": orange, "linestyle": "-"}
    recuperator_charge["hot_side"]["plot_params"] = {"color": orange, "linestyle": "-"}
    recuperator_charge["cold_side"]["plot_params"] = {"color": orange, "linestyle": "-"}
    cooler_charge["hot_side"]["plot_params"] = {"color": orange, "linestyle": "-"}
    cooler_charge["cold_side"]["plot_params"] = {"color": red, "linestyle": "-"}
    expander_charge["plot_params"] = {"color": orange, "linestyle": "-"}
    compressor_charge["plot_params"] = {"color": orange, "linestyle": "-"}
    heater_discharge["hot_side"]["plot_params"] = {"color": red, "linestyle": "--", "marker": "s"}
    heater_discharge["cold_side"]["plot_params"] = {"color": orange, "linestyle": "--", "marker": "s"}
    recuperator_discharge["hot_side"]["plot_params"] = {"color": orange, "linestyle": "--", "marker": "s"}
    recuperator_discharge["cold_side"]["plot_params"] = {"color": orange, "linestyle": "--", "marker": "s"}
    cooler_discharge["hot_side"]["plot_params"] = {"color": orange, "linestyle": "--", "marker": "s"}
    cooler_discharge["cold_side"]["plot_params"] = {"color": blue, "linestyle": "--", "marker": "s"}
    expander_discharge["plot_params"] = {"color": orange, "linestyle": "--", "marker": "s"}
    compressor_discharge["plot_params"] = {"color": orange, "linestyle": "--", "marker": "s"}

    # Check if any fixed parameter or design variable was not used
    utilities.check_for_unused_keys(parameters, "parameters", raise_error=True)
    utilities.check_for_unused_keys(variables, "variables", raise_error=True)

    # print("Efficiency summary")
    # print(components["expander_charge"]["data_out"]["isentropic_efficiency"])
    # print(components["expander_discharge"]["data_out"]["isentropic_efficiency"])
    # print(components["compressor_charge"]["data_out"]["isentropic_efficiency"])
    # print(components["compressor_discharge"]["data_out"]["isentropic_efficiency"])
    # print()




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


