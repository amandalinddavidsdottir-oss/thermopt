import copy
import jaxprop as cpx
from .. import utilities
from ..components import (
    compression_process, expansion_process,
    heat_exchanger, compute_component_energy_flows,
)
COLORS_MATLAB = utilities.COLORS_MATLAB


def evaluate_cycle(variables, parameters, constraints, objective_function):
    variables = copy.deepcopy(variables)
    parameters = copy.deepcopy(parameters)

    # Initialize fluids
    working_fluid = cpx.Fluid(**parameters.pop("working_fluid"),
                               identifier="working_fluid")
    heating_fluid = cpx.Fluid(**parameters.pop("heating_fluid"),
                               identifier="heating_fluid")
    cooling_fluid = cpx.Fluid(**parameters.pop("cooling_fluid"),
                               identifier="cooling_fluid")
    special_points = parameters.pop("special_points")

    # Extract heat source/sink parameters and give short names
    T_source_out_min = parameters["heat_source"].pop("minimum_temperature")
    p_source_out = parameters["heat_source"].pop("exit_pressure")
    p_sink_out = parameters["heat_sink"].pop("exit_pressure")
    
    # Compute coldest state at the heat source exit
    source_out_min = heating_fluid.get_state(
        cpx.PT_INPUTS, p_source_out, T_source_out_min)

    # Extract pressure drops and give short names
    dp_hp_evap_h = parameters["hp_evaporator"].pop("pressure_drop_hot_side")
    dp_hp_evap_c = parameters["hp_evaporator"].pop("pressure_drop_cold_side")
    dp_lp_evap_h = parameters["lp_evaporator"].pop("pressure_drop_hot_side")
    dp_lp_evap_c = parameters["lp_evaporator"].pop("pressure_drop_cold_side")
    dp_preheater_h = parameters["preheater"].pop("pressure_drop_hot_side")
    dp_preheater_c = parameters["preheater"].pop("pressure_drop_cold_side")
    dp_cooler_h = parameters["cooler"].pop("pressure_drop_hot_side")
    dp_cooler_c = parameters["cooler"].pop("pressure_drop_cold_side")

    # Extract design variables from dictionary (make sure all are used)
    hp_expander_inlet_p = variables.pop("hp_expander_inlet_pressure")
    hp_expander_inlet_h = variables.pop("hp_expander_inlet_enthalpy")
    lp_expander_inlet_p = variables.pop("lp_expander_inlet_pressure")
    lp_evap_outlet_h = variables.pop("lp_evaporator_outlet_enthalpy")
    compressor_inlet_p = variables.pop("compressor_inlet_pressure")
    compressor_inlet_h = variables.pop("compressor_inlet_enthalpy")
    mass_split_fraction = variables.pop("mass_split_fraction")
    heat_source_mid_temp = variables.pop("heat_source_mid_temperature")
    heat_sink_exit_temp = variables.pop("heat_sink_exit_temperature")
    preheater_outlet_h = variables.pop("preheater_outlet_enthalpy")
    x = mass_split_fraction

    # === 1. LP Pump (all fluid: state 1 → 2) ===
    lp_pump_outlet_p = lp_expander_inlet_p / (
        (1.0 - dp_preheater_c) * (1.0 - dp_lp_evap_c)) ## LP pump outlet pressure: back-calculated from the LP expander inlet pressure (design variable) accounting for the cold-side pressure drops through the preheater and LP evaporator.
    lp_pump_eff = parameters["lp_pump"].pop("efficiency")
    lp_pump_eff_type = parameters["lp_pump"].pop("efficiency_type")
    lp_pump = compression_process(
        working_fluid, compressor_inlet_h, compressor_inlet_p,
        lp_pump_outlet_p, lp_pump_eff, lp_pump_eff_type)

    # === 2. State 3 = preheater outlet (design variable) ===
    p_3 = lp_pump_outlet_p * (1.0 - dp_preheater_c)
    h_3 = preheater_outlet_h

    # === 3. HP Pump (HP branch: state 3 → 3') ===
    hp_pump_outlet_p = hp_expander_inlet_p / (1.0 - dp_hp_evap_c)
    hp_pump_eff = parameters["hp_pump"].pop("efficiency")
    hp_pump_eff_type = parameters["hp_pump"].pop("efficiency_type")
    hp_pump = compression_process(
        working_fluid, h_3, p_3, hp_pump_outlet_p,
        hp_pump_eff, hp_pump_eff_type)

    # === 4. HP Expander (state 6 → 7, P_HP → P_LP) ===
    hp_exp_outlet_p = lp_expander_inlet_p
    hp_exp_eff = parameters["hp_expander"].pop("efficiency")
    hp_exp_eff_type = parameters["hp_expander"].pop("efficiency_type")
    hp_expander = expansion_process(
        working_fluid, hp_expander_inlet_h, hp_expander_inlet_p,
        hp_exp_outlet_p, hp_exp_eff, hp_exp_eff_type)

    # === 5. Mixing (state 7 + 4 → 8) ===
    h_7 = hp_expander["state_out"].h
    h_4 = lp_evap_outlet_h
    h_8 = x * h_7 + (1.0 - x) * h_4 # Energy balance for adiabatic mixing

    # === 6. LP Expander (state 8 → 9, P_LP → P_cond) ===
    lp_exp_outlet_p = compressor_inlet_p / (1.0 - dp_cooler_h)
    lp_exp_eff = parameters["lp_expander"].pop("efficiency")
    lp_exp_eff_type = parameters["lp_expander"].pop("efficiency_type")
    lp_expander = expansion_process(
        working_fluid, h_8, lp_expander_inlet_p,
        lp_exp_outlet_p, lp_exp_eff, lp_exp_eff_type)

    # === 7. HP Evaporator ===
    h_in_cold_hp = hp_pump["state_out"].h
    p_in_cold_hp = hp_pump["state_out"].p
    h_out_cold_hp = hp_expander["state_in"].h
    p_out_cold_hp = hp_expander["state_in"].p
    T_in_hot_brine = parameters["heat_source"].pop("inlet_temperature")
    p_in_hot_brine = parameters["heat_source"].pop("inlet_pressure")
    h_in_hot_brine = heating_fluid.get_state(
        cpx.PT_INPUTS, p_in_hot_brine, T_in_hot_brine).h
    T_HS1 = heat_source_mid_temp
    p_out_hot_hp = p_in_hot_brine * (1.0 - dp_hp_evap_h)
    h_out_hot_hp = heating_fluid.get_state(
        cpx.PT_INPUTS, p_out_hot_hp, T_HS1).h
    num_el_hp = parameters["hp_evaporator"].pop("num_elements")
    hp_evaporator = heat_exchanger(
        heating_fluid, h_in_hot_brine, h_out_hot_hp,
        p_in_hot_brine, p_out_hot_hp,
        working_fluid, h_in_cold_hp, h_out_cold_hp,
        p_in_cold_hp, p_out_cold_hp,
        counter_current=True, num_steps=num_el_hp)

    # === 8. LP Evaporator (T_HS,2 from mass_flow_ratio chain) ===
    h_in_cold_lp = h_3
    p_in_cold_lp = p_3
    h_out_cold_lp = lp_evap_outlet_h
    p_out_cold_lp = p_3 * (1.0 - dp_lp_evap_c)

    ratio_hp = hp_evaporator["mass_flow_ratio"]
    dh_lp_cold = h_out_cold_lp - h_in_cold_lp
    p_in_hot_lp = p_out_hot_hp
    h_in_hot_lp = h_out_hot_hp
    dh_brine_lp = ((1.0 - x) / (x * ratio_hp)) * dh_lp_cold
    h_out_hot_lp = h_in_hot_lp - dh_brine_lp
    p_out_hot_lp = p_in_hot_lp * (1.0 - dp_lp_evap_h)
    num_el_lp = parameters["lp_evaporator"].pop("num_elements")
    lp_evaporator = heat_exchanger(
        heating_fluid, h_in_hot_lp, h_out_hot_lp,
        p_in_hot_lp, p_out_hot_lp,
        working_fluid, h_in_cold_lp, h_out_cold_lp,
        p_in_cold_lp, p_out_cold_lp,
        counter_current=True, num_steps=num_el_lp)

    # === 9. Preheater (T_HS,out COMPUTED from energy balance) ===
    h_in_cold_pre = lp_pump["state_out"].h
    p_in_cold_pre = lp_pump["state_out"].p
    h_out_cold_pre = h_3
    p_out_cold_pre = p_3

    # Compute brine outlet enthalpy from energy balance:
    # m_brine × (h_brine,HS2 - h_brine,out) = m_total × (h_3 - h_2)
    # h_brine,out = h_brine,HS2 - (m_total/m_brine) × (h_3 - h_2)
    # m_total/m_brine = 1/(x × ratio_hp)
    dh_pre_cold = h_out_cold_pre - h_in_cold_pre  # h_3 - h_2
    dh_brine_pre = (1.0 / (x * ratio_hp)) * dh_pre_cold
    p_in_hot_pre = p_out_hot_lp
    h_in_hot_pre = h_out_hot_lp
    h_out_hot_pre = h_in_hot_pre - dh_brine_pre    # ← COMPUTED, not from design variable
    p_out_hot_pre = p_in_hot_pre * (1.0 - dp_preheater_h)
    num_el_pre = parameters["preheater"].pop("num_elements")
    preheater = heat_exchanger(
        heating_fluid, h_in_hot_pre, h_out_hot_pre,
        p_in_hot_pre, p_out_hot_pre,
        working_fluid, h_in_cold_pre, h_out_cold_pre,
        p_in_cold_pre, p_out_cold_pre,
        counter_current=True, num_steps=num_el_pre)

    # Brine exit temperature (for reinjection constraint)
    brine_exit_temperature = preheater["hot_side"]["state_out"].T

    # === 10. Heat source pump ===
    eff_hs = parameters["heat_source_pump"].pop("efficiency")
    eff_type_hs = parameters["heat_source_pump"].pop("efficiency_type")
    heat_source_pump = compression_process(
        heating_fluid,
        preheater["hot_side"]["state_out"].h,
        preheater["hot_side"]["state_out"].p,
        p_source_out, eff_hs, eff_type_hs)

    # === 11. Heat sink pump ===
    T_in_sink = parameters["heat_sink"].pop("inlet_temperature")
    p_in_sink = parameters["heat_sink"].pop("inlet_pressure")
    h_in_sink = cooling_fluid.get_state(
        cpx.PT_INPUTS, p_in_sink, T_in_sink).h
    p_out_sink = p_sink_out / (1.0 - dp_cooler_c)
    eff_sink = parameters["heat_sink_pump"].pop("efficiency")
    eff_type_sink = parameters["heat_sink_pump"].pop("efficiency_type")
    heat_sink_pump = compression_process(
        cooling_fluid, h_in_sink, p_in_sink, p_out_sink,
        eff_sink, eff_type_sink)

    # === 12. Condenser ===
    p_in_cold_cond = heat_sink_pump["state_out"].p
    h_in_cold_cond = heat_sink_pump["state_out"].h
    p_out_cold_cond = p_in_cold_cond * (1.0 - dp_cooler_c)
    h_out_cold_cond = cooling_fluid.get_state(
        cpx.PT_INPUTS, p_out_cold_cond, heat_sink_exit_temp).h
    h_in_hot_cond = lp_expander["state_out"].h
    p_in_hot_cond = lp_expander["state_out"].p
    h_out_hot_cond = compressor_inlet_h
    p_out_hot_cond = compressor_inlet_p
    num_el_cond = parameters["cooler"].pop("num_elements")
    cooler = heat_exchanger(
        working_fluid, h_in_hot_cond, h_out_hot_cond,
        p_in_hot_cond, p_out_hot_cond,
        cooling_fluid, h_in_cold_cond, h_out_cold_cond,
        p_in_cold_cond, p_out_cold_cond,
        counter_current=True, num_steps=num_el_cond)

    # === 13. Mass flow rates ===
    W_net = parameters.pop("net_power")
    hp_turb_w = x * hp_expander["specific_work"] #The HP components process only fraction x of the total flow, so their specific work (per kg of their own flow) is multiplied by x to get work per kg of total flow. The LP expander and LP pump handle the full flow, so no scaling needed.
    lp_turb_w = lp_expander["specific_work"]
    lp_pump_w = lp_pump["specific_work"]
    hp_pump_w = x * hp_pump["specific_work"]
    net_spec_w = (hp_turb_w + lp_turb_w) - (lp_pump_w + hp_pump_w)
    m_total = W_net / net_spec_w
    m_HP = x * m_total
    m_LP = (1.0 - x) * m_total
    m_brine = m_HP * hp_evaporator["mass_flow_ratio"]
    m_sink = m_total / cooler["mass_flow_ratio"]

    # === 14. Assign mass flows ===
    hp_evaporator["hot_side"]["mass_flow"] = m_brine
    hp_evaporator["cold_side"]["mass_flow"] = m_HP
    lp_evaporator["hot_side"]["mass_flow"] = m_brine
    lp_evaporator["cold_side"]["mass_flow"] = m_LP
    preheater["hot_side"]["mass_flow"] = m_brine
    preheater["cold_side"]["mass_flow"] = m_total
    cooler["hot_side"]["mass_flow"] = m_total
    cooler["cold_side"]["mass_flow"] = m_sink
    hp_expander["mass_flow"] = m_HP
    lp_expander["mass_flow"] = m_total
    lp_pump["mass_flow"] = m_total
    hp_pump["mass_flow"] = m_HP
    heat_source_pump["mass_flow"] = m_brine
    heat_sink_pump["mass_flow"] = m_sink

    # === 15. Components and energy flows ===
    components = {
        "hp_expander": hp_expander, "lp_expander": lp_expander,
        "lp_pump": lp_pump, "hp_pump": hp_pump,
        "hp_evaporator": hp_evaporator,
        "lp_evaporator": lp_evaporator,
        "preheater": preheater, "cooler": cooler,
        "heat_source_pump": heat_source_pump,
        "heat_sink_pump": heat_sink_pump,
    }
    compute_component_energy_flows(components)

    # === 15b. Mixer pseudo-components (for T-s diagram plotting) ===
    # The mixing point (state 8) is not a thermodynamic "component" in ThermOpt,
    # so the plotter doesn't draw the connections 7→8 and 4→8 automatically.
    # Adding these as pseudo-components lets ThermOpt's built-in plotter draw them.
    # No new thermodynamics — just connecting states that already exist:
    #   state 7 = hp_expander["state_out"]
    #   state 4 = lp_evaporator["cold_side"]["state_out"]
    #   state 8 = lp_expander["state_in"]  (computed from h_8 = x*h_7 + (1-x)*h_4)
    components["mixer_from_hp"] = {
        "type": "mixer",
        "states": hp_expander["state_out"] + lp_expander["state_in"],
        "state_in": hp_expander["state_out"],
        "state_out": lp_expander["state_in"],
        "mass_flow": m_HP,
        "specific_work": 0.0,
        "power": 0.0,
    }
    components["mixer_from_lp"] = {
        "type": "mixer",
        "states": lp_evaporator["cold_side"]["state_out"] + lp_expander["state_in"],
        "state_in": lp_evaporator["cold_side"]["state_out"],
        "state_out": lp_expander["state_in"],
        "mass_flow": m_LP,
        "specific_work": 0.0,
        "power": 0.0,
    }

    # === 16. First-law analysis ===
    Q_hp = hp_evaporator["heat_flow"]
    Q_lp = lp_evaporator["heat_flow"]
    Q_pre = preheater["heat_flow"]
    Q_in = Q_hp + Q_lp + Q_pre
    Q_out = cooler["heat_flow"]
    W_out = hp_expander["power"] + lp_expander["power"]
    W_comp = lp_pump["power"] + hp_pump["power"]
    W_aux = heat_source_pump["power"] + heat_sink_pump["power"]
    W_in = W_comp + W_aux
    Q_in_max = m_brine * (
        heating_fluid.get_state(
            cpx.PT_INPUTS, p_in_hot_brine, T_in_hot_brine).h
        - source_out_min.h)
    cycle_efficiency = (W_out - W_in) / Q_in
    system_efficiency = (W_out - W_in) / Q_in_max
    backwork_ratio = W_comp / W_out
    energy_balance = (Q_in + W_comp) - (W_out + Q_out)

    energy_analysis = {
        "hp_evaporator_heat_flow": Q_hp,
        "lp_evaporator_heat_flow": Q_lp,
        "preheater_heat_flow": Q_pre,
        "total_heat_input": Q_in,
        "heater_heat_flow_max": Q_in_max,
        "cooler_heat_flow": Q_out,
        "hp_expander_power": hp_expander["power"],
        "lp_expander_power": lp_expander["power"],
        "total_expander_power": W_out,
        "lp_pump_power": lp_pump["power"],
        "hp_pump_power": hp_pump["power"],
        "heat_source_pump_power": heat_source_pump["power"],
        "heat_sink_pump_power": heat_sink_pump["power"],
        "net_cycle_power": W_net,
        "net_system_power": W_out - W_in,
        "mass_flow_heating_fluid": m_brine,
        "mass_flow_working_fluid": m_total,
        "mass_flow_hp": m_HP, "mass_flow_lp": m_LP,
        "mass_flow_cooling_fluid": m_sink,
        "split_fraction": x,
        "cycle_efficiency": cycle_efficiency,
        "system_efficiency": system_efficiency,
        "backwork_ratio": backwork_ratio,
        "energy_balance": energy_balance,
        "brine_exit_temperature": brine_exit_temperature,
    }

    # === 17. Objective function and constraints ===
    output = {"components": components, "energy_analysis": energy_analysis}
    f = utilities.evaluate_objective_function(output, objective_function)
    c_eq, c_ineq, constraint_report = utilities.evaluate_constraints(
        output, constraints)

    # === 18. Plot colors ===
    orange = COLORS_MATLAB[1]; blue = COLORS_MATLAB[0]
    red = COLORS_MATLAB[6]; green = COLORS_MATLAB[4]
    purple = COLORS_MATLAB[3]
    hp_evaporator["hot_side"]["plot_params"] = {"color": red, "linestyle": "-"}
    hp_evaporator["cold_side"]["plot_params"] = {"color": red, "linestyle": "--"}
    lp_evaporator["hot_side"]["plot_params"] = {"color": red, "linestyle": "-"}
    lp_evaporator["cold_side"]["plot_params"] = {"color": orange, "linestyle": "--"}
    preheater["hot_side"]["plot_params"] = {"color": red, "linestyle": "-"}
    preheater["cold_side"]["plot_params"] = {"color": green, "linestyle": "--"}
    cooler["hot_side"]["plot_params"] = {"color": orange, "linestyle": "-"}
    cooler["cold_side"]["plot_params"] = {"color": blue, "linestyle": "-"}
    hp_expander["plot_params"] = {"color": red, "linestyle": "-"}
    lp_expander["plot_params"] = {"color": orange, "linestyle": "-"}
    lp_pump["plot_params"] = {"color": green, "linestyle": "-"}
    hp_pump["plot_params"] = {"color": red, "linestyle": "-"}
    components["mixer_from_hp"]["plot_params"] = {"color": purple, "linestyle": "--"}
    components["mixer_from_lp"]["plot_params"] = {"color": purple, "linestyle": "--"}

    # === 19. Check unused keys ===
    utilities.check_for_unused_keys(parameters, "parameters", raise_error=True)
    utilities.check_for_unused_keys(variables, "variables", raise_error=True)

    # === 20. Return ===
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