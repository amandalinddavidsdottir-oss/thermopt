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
    variables, parameters, constraints, objective_function, recuperated=True
):
    """
    Dual-pressure ORC cycle with a single shared brine source.

    All brine from the wells is throttled to the same pressure and passes
    sequentially through the HP evaporator, LP evaporator, and preheater
    before being reinjected by a single heat-source pump.

    The well mass flow rate (``well_mass_flow_rate``) must be provided as a
    fixed parameter or design variable. The working fluid cycle mass flow
    (``mass_flow_rate_cycle``) and cooling fluid mass flow (``mass_flow_rate_sink``)
    are design variables. Heat exchanger energy balances are automatically enforced
    as equality constraints inside the heat exchanger model.
    """
    variables = copy.deepcopy(variables)
    parameters = copy.deepcopy(parameters)

    # Initialize fluids
    working_fluid = cpx.Fluid(
        **parameters.pop("working_fluid"), identifier="working_fluid"
    )
    heating_fluid = cpx.Fluid(
        **parameters.pop("heating_fluid"), identifier="heating_fluid"
    )
    cooling_fluid = cpx.Fluid(
        **parameters.pop("cooling_fluid"), identifier="cooling_fluid"
    )
    special_points = parameters.pop("special_points")

    # Extract heat source / sink parameters
    T_source_out_min = parameters["heat_source"].pop("minimum_temperature")
    p_source_out = parameters["heat_source"].pop("exit_pressure")
    p_sink_out = parameters["heat_sink"].pop("exit_pressure")

    # Coldest allowed state at the heat-source exit (for Q_in_max)
    source_out_min = heating_fluid.get_state(
        cpx.PT_INPUTS, p_source_out, T_source_out_min
    )

    # Ambient reference state for the heat source (for η_utilization / η_ambient)
    # Dead state (T₀, p₀) per DiPippo utilization efficiency convention
    T_ambient = special_points["ambient_temperature"]
    p_ambient = special_points["ambient_pressure"]
    source_out_ambient = heating_fluid.get_state(cpx.PT_INPUTS, p_ambient, T_ambient)

    # Extract pressure drops
    dp_hp_evap_h = parameters["hp_evaporator"].pop("pressure_drop_hot_side")
    dp_hp_evap_c = parameters["hp_evaporator"].pop("pressure_drop_cold_side")
    dp_lp_evap_h = parameters["lp_evaporator"].pop("pressure_drop_hot_side")
    dp_lp_evap_c = parameters["lp_evaporator"].pop("pressure_drop_cold_side")
    dp_preheater_h = parameters["preheater"].pop("pressure_drop_hot_side")
    dp_preheater_c = parameters["preheater"].pop("pressure_drop_cold_side")
    dp_cooler_h = parameters["cooler"].pop("pressure_drop_hot_side")
    dp_cooler_c = parameters["cooler"].pop("pressure_drop_cold_side")
    if recuperated:
        dp_recup_h = parameters["recuperator"].pop("pressure_drop_hot_side")
        dp_recup_c = parameters["recuperator"].pop("pressure_drop_cold_side")
    else:
        dp_recup_h, dp_recup_c = 0.0, 0.0

    # Extract design variables
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
    if recuperated:
        recuperator_effectiveness = variables.pop("recuperator_effectiveness")
    else:
        recuperator_effectiveness = 0.0
    x = mass_split_fraction

    # Mass flow rates as design variables — all known upfront, no ratio algebra needed
    if "well_mass_flow_rate" in parameters:
        well_mass_flow_rate = parameters.pop("well_mass_flow_rate")
    else:
        well_mass_flow_rate = variables.pop("well_mass_flow_rate")
    m_brine = well_mass_flow_rate
    m_total = variables.pop("mass_flow_rate_cycle")
    m_sink = variables.pop("mass_flow_rate_sink")
    m_HP = x * m_total
    m_LP = (1.0 - x) * m_total

    # ------------------------------------------------------------------
    # Evaluate LP pump
    # ------------------------------------------------------------------
    lp_pump_outlet_p = lp_expander_inlet_p / (
        (1.0 - dp_recup_c) * (1.0 - dp_preheater_c) * (1.0 - dp_lp_evap_c)
    )
    lp_pump_eff = parameters["lp_pump"].pop("efficiency")
    lp_pump_eff_type = parameters["lp_pump"].pop("efficiency_type")
    lp_pump = compression_process(
        working_fluid,
        compressor_inlet_h,
        compressor_inlet_p,
        lp_pump_outlet_p,
        lp_pump_eff,
        lp_pump_eff_type,
    )

    # ------------------------------------------------------------------
    # Evaluate HP pump
    # ------------------------------------------------------------------
    p_3 = lp_pump_outlet_p * (1.0 - dp_recup_c) * (1.0 - dp_preheater_c)
    h_3 = preheater_outlet_h

    hp_pump_outlet_p = hp_expander_inlet_p / (1.0 - dp_hp_evap_c)
    hp_pump_eff = parameters["hp_pump"].pop("efficiency")
    hp_pump_eff_type = parameters["hp_pump"].pop("efficiency_type")
    hp_pump = compression_process(
        working_fluid, h_3, p_3, hp_pump_outlet_p, hp_pump_eff, hp_pump_eff_type
    )

    # ------------------------------------------------------------------
    # Evaluate HP expander
    # ------------------------------------------------------------------
    hp_exp_outlet_p = lp_expander_inlet_p
    hp_exp_eff = parameters["hp_expander"].pop("efficiency", None)
    hp_exp_eff_type = parameters["hp_expander"].pop("efficiency_type")

    hp_exp_data_in = {}
    if hp_exp_eff_type == "astolfi-stacking":
        hp_exp_data_in["n_stages"] = parameters["hp_expander"].pop("n_stages")
        if "RPM" in parameters["hp_expander"]:
            hp_exp_data_in["RPM"] = parameters["hp_expander"].pop("RPM")

    hp_expander = expansion_process(
        working_fluid,
        hp_expander_inlet_h,
        hp_expander_inlet_p,
        hp_exp_outlet_p,
        hp_exp_eff,
        hp_exp_eff_type,
        mass_flow=m_HP,
        data_in=hp_exp_data_in,
    )

    # ------------------------------------------------------------------
    # Mix HP expander outlet with LP evaporator outlet
    # ------------------------------------------------------------------
    h_7 = hp_expander["state_out"].h
    h_4 = lp_evap_outlet_h
    h_8 = x * h_7 + (1.0 - x) * h_4  # Adiabatic mixing energy balance

    # ------------------------------------------------------------------
    # Evaluate LP expander
    # ------------------------------------------------------------------
    lp_exp_outlet_p = compressor_inlet_p / ((1.0 - dp_cooler_h) * (1.0 - dp_recup_h))
    lp_exp_eff = parameters["lp_expander"].pop("efficiency", None)
    lp_exp_eff_type = parameters["lp_expander"].pop("efficiency_type")

    lp_exp_data_in = {}
    if lp_exp_eff_type == "astolfi-stacking":
        lp_exp_data_in["n_stages"] = parameters["lp_expander"].pop("n_stages")
        if "RPM" in parameters["lp_expander"]:
            lp_exp_data_in["RPM"] = parameters["lp_expander"].pop("RPM")

    lp_expander = expansion_process(
        working_fluid,
        h_8,
        lp_expander_inlet_p,
        lp_exp_outlet_p,
        lp_exp_eff,
        lp_exp_eff_type,
        mass_flow=m_total,
        data_in=lp_exp_data_in,
    )

    # ------------------------------------------------------------------
    # Evaluate recuperator
    # ------------------------------------------------------------------
    eps = recuperator_effectiveness
    h_in_hot_recup = lp_expander["state_out"].h
    p_in_hot_recup = lp_expander["state_out"].p
    T_in_hot_recup = lp_expander["state_out"].T
    p_out_hot_recup = p_in_hot_recup * (1.0 - dp_recup_h)

    h_in_cold_recup = lp_pump["state_out"].h
    p_in_cold_recup = lp_pump["state_out"].p
    p_out_cold_recup = p_in_cold_recup * (1.0 - dp_recup_c)

    h_out_cold_ideal = working_fluid.get_state(
        cpx.PT_INPUTS, p_out_cold_recup, T_in_hot_recup
    ).h
    h_out_cold_recup = h_in_cold_recup + eps * (h_out_cold_ideal - h_in_cold_recup)
    h_out_hot_recup = h_in_hot_recup - (h_out_cold_recup - h_in_cold_recup)

    if recuperated:
        num_el_recup = parameters["recuperator"].pop("num_elements")
    else:
        num_el_recup = 2
    recuperator = heat_exchanger(
        working_fluid,
        h_in_hot_recup,
        h_out_hot_recup,
        p_in_hot_recup,
        p_out_hot_recup,
        working_fluid,
        h_in_cold_recup,
        h_out_cold_recup,
        p_in_cold_recup,
        p_out_cold_recup,
        counter_current=True,
        num_steps=num_el_recup,
        mass_flow_hot=m_total,
        mass_flow_cold=m_total,
        enforce_energy_balance=False,
    )

    # ------------------------------------------------------------------
    # Evaluate HP evaporator
    # Brine enters fresh from the well; exits to the LP evaporator inlet.
    # ------------------------------------------------------------------
    h_in_cold_hp = hp_pump["state_out"].h
    p_in_cold_hp = hp_pump["state_out"].p
    h_out_cold_hp = hp_expander["state_in"].h
    p_out_cold_hp = hp_expander["state_in"].p

    p_in_hot_brine = parameters["heat_source"].pop("inlet_pressure")
    h_in_hot_brine = parameters["heat_source"].pop("inlet_enthalpy")

    p_out_hot_hp = p_in_hot_brine * (1.0 - dp_hp_evap_h)
    h_out_hot_hp = heating_fluid.get_state(
        cpx.PT_INPUTS, p_out_hot_hp, heat_source_mid_temp
    ).h
    num_el_hp = parameters["hp_evaporator"].pop("num_elements")
    hp_evaporator = heat_exchanger(
        heating_fluid,
        h_in_hot_brine,
        h_out_hot_hp,
        p_in_hot_brine,
        p_out_hot_hp,
        working_fluid,
        h_in_cold_hp,
        h_out_cold_hp,
        p_in_cold_hp,
        p_out_cold_hp,
        counter_current=True,
        num_steps=num_el_hp,
        mass_flow_hot=m_brine,
        mass_flow_cold=m_HP,
    )

    # ------------------------------------------------------------------
    # Evaluate LP evaporator
    # Brine inlet is the HP evaporator hot-side outlet.
    # h_out_hot is computed directly from the energy balance using known mass flows.
    # ------------------------------------------------------------------
    h_in_cold_lp = h_3
    p_in_cold_lp = p_3
    h_out_cold_lp = lp_evap_outlet_h
    p_out_cold_lp = p_3 * (1.0 - dp_lp_evap_c)

    p_in_hot_lp = p_out_hot_hp
    h_in_hot_lp = h_out_hot_hp
    h_out_hot_lp = h_in_hot_lp - (m_LP / m_brine) * (h_out_cold_lp - h_in_cold_lp)
    p_out_hot_lp = p_in_hot_lp * (1.0 - dp_lp_evap_h)

    num_el_lp = parameters["lp_evaporator"].pop("num_elements")
    lp_evaporator = heat_exchanger(
        heating_fluid,
        h_in_hot_lp,
        h_out_hot_lp,
        p_in_hot_lp,
        p_out_hot_lp,
        working_fluid,
        h_in_cold_lp,
        h_out_cold_lp,
        p_in_cold_lp,
        p_out_cold_lp,
        counter_current=True,
        num_steps=num_el_lp,
        mass_flow_hot=m_brine,
        mass_flow_cold=m_LP,
    )

    # ------------------------------------------------------------------
    # Evaluate preheater
    # Brine inlet is the LP evaporator hot-side outlet.
    # ------------------------------------------------------------------
    h_in_cold_pre = recuperator["cold_side"]["state_out"].h
    p_in_cold_pre = recuperator["cold_side"]["state_out"].p
    h_out_cold_pre = h_3
    p_out_cold_pre = p_3

    p_in_hot_pre = p_out_hot_lp
    h_in_hot_pre = h_out_hot_lp
    h_out_hot_pre = h_in_hot_pre - (m_total / m_brine) * (h_out_cold_pre - h_in_cold_pre)
    p_out_hot_pre = p_in_hot_pre * (1.0 - dp_preheater_h)

    num_el_pre = parameters["preheater"].pop("num_elements")
    preheater = heat_exchanger(
        heating_fluid,
        h_in_hot_pre,
        h_out_hot_pre,
        p_in_hot_pre,
        p_out_hot_pre,
        working_fluid,
        h_in_cold_pre,
        h_out_cold_pre,
        p_in_cold_pre,
        p_out_cold_pre,
        mass_flow_hot=m_brine,
        mass_flow_cold=m_total,
        counter_current=True,
        num_steps=num_el_pre,
    )

    brine_exit_temperature = preheater["hot_side"]["state_out"].T

    # ------------------------------------------------------------------
    # Evaluate heat-source pump (single shared brine loop)
    # ------------------------------------------------------------------
    eff_hs = parameters["heat_source_pump"].pop("efficiency")
    eff_type_hs = parameters["heat_source_pump"].pop("efficiency_type")
    heat_source_pump = compression_process(
        heating_fluid,
        preheater["hot_side"]["state_out"].h,
        preheater["hot_side"]["state_out"].p,
        p_source_out,
        eff_hs,
        eff_type_hs,
    )

    # ------------------------------------------------------------------
    # Evaluate heat-sink pump
    # ------------------------------------------------------------------
    T_in_sink = parameters["heat_sink"].pop("inlet_temperature")
    p_in_sink = parameters["heat_sink"].pop("inlet_pressure")
    h_in_sink = cooling_fluid.get_state(cpx.PT_INPUTS, p_in_sink, T_in_sink).h
    p_out_sink = p_sink_out / (1.0 - dp_cooler_c)
    eff_sink = parameters["heat_sink_pump"].pop("efficiency")
    eff_type_sink = parameters["heat_sink_pump"].pop("efficiency_type")
    heat_sink_pump = compression_process(
        cooling_fluid, h_in_sink, p_in_sink, p_out_sink, eff_sink, eff_type_sink
    )

    # ------------------------------------------------------------------
    # Evaluate cooler
    # ------------------------------------------------------------------
    p_in_cold_cond = heat_sink_pump["state_out"].p
    h_in_cold_cond = heat_sink_pump["state_out"].h
    p_out_cold_cond = p_in_cold_cond * (1.0 - dp_cooler_c)
    h_out_cold_cond = cooling_fluid.get_state(
        cpx.PT_INPUTS, p_out_cold_cond, heat_sink_exit_temp
    ).h
    h_in_hot_cond = recuperator["hot_side"]["state_out"].h
    p_in_hot_cond = recuperator["hot_side"]["state_out"].p
    h_out_hot_cond = compressor_inlet_h
    p_out_hot_cond = compressor_inlet_p
    num_el_cond = parameters["cooler"].pop("num_elements")
    cooler = heat_exchanger(
        working_fluid,
        h_in_hot_cond,
        h_out_hot_cond,
        p_in_hot_cond,
        p_out_hot_cond,
        cooling_fluid,
        h_in_cold_cond,
        h_out_cold_cond,
        p_in_cold_cond,
        p_out_cold_cond,
        mass_flow_hot=m_total,
        mass_flow_cold=m_sink,
        counter_current=True,
        num_steps=num_el_cond,
    )

    # Compute net power from known mass flows
    hp_turb_w = x * hp_expander["specific_work"]
    lp_turb_w = lp_expander["specific_work"]
    lp_pump_w = lp_pump["specific_work"]
    hp_pump_w = x * hp_pump["specific_work"]
    net_spec_w = (hp_turb_w + lp_turb_w) - (lp_pump_w + hp_pump_w)
    W_net = net_spec_w * m_total

    # Assign mass flows to turbomachinery
    hp_expander["mass_flow"] = m_HP
    lp_expander["mass_flow"] = m_total
    lp_pump["mass_flow"] = m_total
    hp_pump["mass_flow"] = m_HP
    heat_source_pump["mass_flow"] = m_brine
    heat_sink_pump["mass_flow"] = m_sink

    # ------------------------------------------------------------------
    # Assemble components dict
    # ------------------------------------------------------------------
    components = {
        "hp_expander": hp_expander,
        "lp_expander": lp_expander,
        "lp_pump": lp_pump,
        "hp_pump": hp_pump,
        "hp_evaporator": hp_evaporator,
        "lp_evaporator": lp_evaporator,
        "preheater": preheater,
        "recuperator": recuperator,
        "cooler": cooler,
        "heat_source_pump": heat_source_pump,
        "heat_sink_pump": heat_sink_pump,
    }
    compute_component_energy_flows(components)

    # Mixer pseudo-components for T-s diagram plotting
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

    # ------------------------------------------------------------------
    # First-law analysis
    # ------------------------------------------------------------------
    Q_hp = hp_evaporator["heat_flow"]
    Q_lp = lp_evaporator["heat_flow"]
    Q_pre = preheater["heat_flow"]
    Q_recup = recuperator["heat_flow"]
    Q_in = Q_hp + Q_lp + Q_pre
    Q_out = cooler["heat_flow"]
    W_out = hp_expander["power"] + lp_expander["power"]
    W_comp = lp_pump["power"] + hp_pump["power"]
    W_aux = heat_source_pump["power"] + heat_sink_pump["power"]
    W_in = W_comp + W_aux
    Q_in_max = m_brine * (h_in_hot_brine - source_out_min.h)
    Q_in_max_ambient = m_brine * (h_in_hot_brine - source_out_ambient.h)
    cycle_efficiency = (W_out - W_in) / Q_in
    system_efficiency = (W_out - W_in) / Q_in_max
    system_efficiency_ambient = (W_out - W_in) / Q_in_max_ambient
    backwork_ratio = W_comp / W_out
    energy_balance = (Q_in + W_comp + W_aux) - (W_out + Q_out)

    energy_analysis = {
        "hp_evaporator_heat_flow": Q_hp,
        "lp_evaporator_heat_flow": Q_lp,
        "preheater_heat_flow": Q_pre,
        "recuperator_heat_flow": Q_recup,
        "total_heat_input": Q_in,
        "heater_heat_flow_max": Q_in_max,
        "heater_heat_flow_max_ambient": Q_in_max_ambient,
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
        "mass_flow_hp": m_HP,
        "mass_flow_lp": m_LP,
        "mass_flow_cooling_fluid": m_sink,
        "split_fraction": x,
        "cycle_efficiency": cycle_efficiency,
        "system_efficiency": system_efficiency,
        "system_efficiency_ambient": system_efficiency_ambient,
        "backwork_ratio": backwork_ratio,
        "energy_balance": energy_balance,
        "brine_exit_temperature": brine_exit_temperature,
    }

    # Evaluate objective function and constraints
    output = {
        "components": components,
        "energy_analysis": energy_analysis,
        "variables": {
            "mass_flow_rate_cycle": m_total,
            "mass_flow_rate_sink": m_sink,
        },
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

    # ------------------------------------------------------------------
    # Set colors for plotting
    # ------------------------------------------------------------------
    orange = COLORS_MATLAB[1]
    blue = COLORS_MATLAB[0]
    red = COLORS_MATLAB[6]
    green = COLORS_MATLAB[4]
    purple = COLORS_MATLAB[3]
    hp_evaporator["hot_side"]["plot_params"] = {"color": red, "linestyle": "-"}
    hp_evaporator["cold_side"]["plot_params"] = {"color": red, "linestyle": "--"}
    lp_evaporator["hot_side"]["plot_params"] = {"color": red, "linestyle": "-"}
    lp_evaporator["cold_side"]["plot_params"] = {"color": orange, "linestyle": "--"}
    preheater["hot_side"]["plot_params"] = {"color": red, "linestyle": "-"}
    preheater["cold_side"]["plot_params"] = {"color": green, "linestyle": "--"}
    recuperator["hot_side"]["plot_params"] = {"color": orange, "linestyle": "-"}
    recuperator["cold_side"]["plot_params"] = {"color": orange, "linestyle": "--"}
    cooler["hot_side"]["plot_params"] = {"color": orange, "linestyle": "-"}
    cooler["cold_side"]["plot_params"] = {"color": blue, "linestyle": "-"}
    hp_expander["plot_params"] = {"color": red, "linestyle": "-"}
    lp_expander["plot_params"] = {"color": orange, "linestyle": "-"}
    lp_pump["plot_params"] = {"color": green, "linestyle": "-"}
    hp_pump["plot_params"] = {"color": red, "linestyle": "-"}
    components["mixer_from_hp"]["plot_params"] = {"color": purple, "linestyle": "--"}
    components["mixer_from_lp"]["plot_params"] = {"color": purple, "linestyle": "--"}

    # Check for unused keys
    utilities.check_for_unused_keys(parameters, "parameters", raise_error=True)
    utilities.check_for_unused_keys(variables, "variables", raise_error=True)

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
