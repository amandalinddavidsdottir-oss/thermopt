import copy
import numpy as np
import jaxprop as cpx
from .. import utilities
from ..components import (
    compression_process,
    expansion_process,
    heat_exchanger,
    mixing_process,
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
    T0 = T_ambient
    p0 = p_ambient

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
    compressor_inlet_p = variables.pop("compressor_inlet_pressure")
    compressor_inlet_h = variables.pop("compressor_inlet_enthalpy")
    lp_evap_outlet_h = variables.pop("lp_evaporator_outlet_enthalpy")

    # LP (intermediate) expander inlet pressure, defined as a fraction between
    # condensation pressure (fraction=0) and evaporation pressure (fraction=1):
    #
    #   p_lp = fraction * p_cond + (1 - fraction) * p_evap
    #
    # This way, any initial guess used in multistart will always give a pressure
    # that is between p_cond and p_evap, so LP < HP is always satisfied from
    # the start without needing an explicit ordering constraint.
    lp_fraction = variables.pop("lp_expander_inlet_pressure_fraction")
    lp_expander_inlet_p = (
        lp_fraction * compressor_inlet_p + (1.0 - lp_fraction) * hp_expander_inlet_p
    )
    mass_split_fraction = variables.pop("mass_split_fraction")
    heat_source_hp_evap_exit_fraction = variables.pop(
        "heat_source_hp_evap_exit_temperature_fraction"
    )
    heat_source_lp_evap_exit_fraction = variables.pop(
        "heat_source_lp_evap_exit_temperature_fraction"
    )
    heat_source_preheater_exit_fraction = variables.pop(
        "heat_source_preheater_exit_temperature_fraction"
    )
    heat_sink_exit_temp = variables.pop("heat_sink_exit_temperature")
    preheater_outlet_h = variables.pop("preheater_outlet_enthalpy")
    if recuperated:
        recuperator_cold_outlet_enthalpy = variables.pop("recuperator_cold_outlet_enthalpy")
    else:
        recuperator_cold_outlet_enthalpy = None
    x = mass_split_fraction

    # Mass flow rates as design variables — all known upfront, no ratio algebra needed
    m_brine = variables.pop("well_mass_flow_rate")
    m_total = variables.pop("mass_flow_rate_cycle")
    m_sink = variables.pop("mass_flow_rate_sink")
    m_HP = x * m_total
    m_LP = (1.0 - x) * m_total

    # Evaluate LP pump
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
        mass_flow=m_total,
        T0=T0,
        p0=p0,
    )

    # Evaluate HP pump
    p_3 = lp_pump_outlet_p * (1.0 - dp_recup_c) * (1.0 - dp_preheater_c)
    h_3 = preheater_outlet_h

    hp_pump_outlet_p = hp_expander_inlet_p / (1.0 - dp_hp_evap_c)
    hp_pump_eff = parameters["hp_pump"].pop("efficiency")
    hp_pump_eff_type = parameters["hp_pump"].pop("efficiency_type")
    hp_pump = compression_process(
        working_fluid,
        h_3,
        p_3,
        hp_pump_outlet_p,
        hp_pump_eff,
        hp_pump_eff_type,
        mass_flow=m_HP,
        T0=T0,
        p0=p0,
    )

    # Evaluate HP expander
    hp_exp_outlet_p = lp_expander_inlet_p
    hp_exp_eff = parameters["hp_expander"].pop("efficiency", None)
    hp_exp_eff_type = parameters["hp_expander"].pop("efficiency_type")

    hp_exp_data_in = {}
    n_turbines_parallel_hp = parameters["hp_expander"].pop("n_turbines_parallel", 1)
    if hp_exp_eff_type == "astolfi-stacking":
        hp_exp_data_in["n_stages"] = parameters["hp_expander"].pop("n_stages")
        if "RPM" in parameters["hp_expander"]:
            hp_exp_data_in["RPM"] = parameters["hp_expander"].pop("RPM")
        hp_intermediate_pressures = []
        hp_intermediate_ratios = []
        for i in range(1, 10):
            p_key = f"hp_expander_intermediate_pressure_{i}"
            r_key = f"hp_expander_intermediate_ratio_{i}"
            if p_key in variables and r_key in variables:
                raise ValueError(
                    f"Both '{p_key}' and '{r_key}' are declared as design "
                    f"variables. Use exactly one parameterization per stage."
                )
            if p_key in variables:
                hp_intermediate_pressures.append(variables.pop(p_key))
            elif r_key in variables:
                hp_intermediate_ratios.append(variables.pop(r_key))
        if hp_intermediate_ratios:
            p_prev = hp_expander_inlet_p
            for r in hp_intermediate_ratios:
                p_i = hp_exp_outlet_p * (p_prev / hp_exp_outlet_p) ** r
                hp_intermediate_pressures.append(p_i)
                p_prev = p_i
        if hp_intermediate_pressures:
            hp_exp_data_in["intermediate_pressures"] = hp_intermediate_pressures

    turbine_shafts = [
        expansion_process(
            working_fluid,
            hp_expander_inlet_h,
            hp_expander_inlet_p,
            hp_exp_outlet_p,
            hp_exp_eff,
            hp_exp_eff_type,
            mass_flow=m_HP / n_turbines_parallel_hp,
            data_in=hp_exp_data_in,
            T0=T0,
            p0=p0,
        )
        for _ in range(n_turbines_parallel_hp)
    ]
    hp_expander = turbine_shafts[0]
    hp_expander["mass_flow"] = m_HP
    hp_expander["n_turbines_parallel"] = n_turbines_parallel_hp
    hp_expander["power"] = sum(s["power"] for s in turbine_shafts)
    hp_expander["energy_analysis"]["power"] = hp_expander["power"]
    hp_expander["exergy_analysis"]["E_fuel"] = sum(
        s["exergy_analysis"]["E_fuel"] for s in turbine_shafts
    )
    hp_expander["exergy_analysis"]["E_product"] = sum(
        s["exergy_analysis"]["E_product"] for s in turbine_shafts
    )
    hp_expander["exergy_analysis"]["E_D"] = sum(
        s["exergy_analysis"]["E_D"] for s in turbine_shafts
    )

    # Mix HP expander outlet with LP evaporator outlet using dedicated mixing component
    mixer = mixing_process(
        working_fluid,
        h_in_1=hp_expander["state_out"].h,
        p_in_1=hp_expander["state_out"].p,
        m_1=m_HP,
        h_in_2=lp_evap_outlet_h,
        p_in_2=lp_expander_inlet_p,
        m_2=m_LP,
        p_out=lp_expander_inlet_p,
        T0=T0,
        p0=p0,
    )
    h_8 = mixer["state_out"].h

    # Evaluate LP expander
    lp_exp_outlet_p = compressor_inlet_p / ((1.0 - dp_cooler_h) * (1.0 - dp_recup_h))
    lp_exp_eff = parameters["lp_expander"].pop("efficiency", None)
    lp_exp_eff_type = parameters["lp_expander"].pop("efficiency_type")

    lp_exp_data_in = {}
    n_turbines_parallel_lp = parameters["lp_expander"].pop("n_turbines_parallel", 1)
    if lp_exp_eff_type == "astolfi-stacking":
        lp_exp_data_in["n_stages"] = parameters["lp_expander"].pop("n_stages")
        if "RPM" in parameters["lp_expander"]:
            lp_exp_data_in["RPM"] = parameters["lp_expander"].pop("RPM")
        lp_intermediate_pressures = []
        lp_intermediate_ratios = []
        for i in range(1, 10):
            p_key = f"lp_expander_intermediate_pressure_{i}"
            r_key = f"lp_expander_intermediate_ratio_{i}"
            if p_key in variables and r_key in variables:
                raise ValueError(
                    f"Both '{p_key}' and '{r_key}' are declared as design "
                    f"variables. Use exactly one parameterization per stage."
                )
            if p_key in variables:
                lp_intermediate_pressures.append(variables.pop(p_key))
            elif r_key in variables:
                lp_intermediate_ratios.append(variables.pop(r_key))
        if lp_intermediate_ratios:
            p_prev = lp_expander_inlet_p
            for r in lp_intermediate_ratios:
                p_i = lp_exp_outlet_p * (p_prev / lp_exp_outlet_p) ** r
                lp_intermediate_pressures.append(p_i)
                p_prev = p_i
        if lp_intermediate_pressures:
            lp_exp_data_in["intermediate_pressures"] = lp_intermediate_pressures

    turbine_shafts = [
        expansion_process(
            working_fluid,
            h_8,
            lp_expander_inlet_p,
            lp_exp_outlet_p,
            lp_exp_eff,
            lp_exp_eff_type,
            mass_flow=m_total / n_turbines_parallel_lp,
            data_in=lp_exp_data_in,
            T0=T0,
            p0=p0,
        )
        for _ in range(n_turbines_parallel_lp)
    ]
    lp_expander = turbine_shafts[0]
    lp_expander["mass_flow"] = m_total
    lp_expander["n_turbines_parallel"] = n_turbines_parallel_lp
    lp_expander["power"] = sum(s["power"] for s in turbine_shafts)
    lp_expander["energy_analysis"]["power"] = lp_expander["power"]
    lp_expander["exergy_analysis"]["E_fuel"] = sum(
        s["exergy_analysis"]["E_fuel"] for s in turbine_shafts
    )
    lp_expander["exergy_analysis"]["E_product"] = sum(
        s["exergy_analysis"]["E_product"] for s in turbine_shafts
    )
    lp_expander["exergy_analysis"]["E_D"] = sum(
        s["exergy_analysis"]["E_D"] for s in turbine_shafts
    )

    # Evaluate recuperator
    h_in_hot_recup = lp_expander["state_out"].h
    p_in_hot_recup = lp_expander["state_out"].p
    p_out_hot_recup = p_in_hot_recup * (1.0 - dp_recup_h)

    h_in_cold_recup = lp_pump["state_out"].h
    p_in_cold_recup = lp_pump["state_out"].p
    p_out_cold_recup = p_in_cold_recup * (1.0 - dp_recup_c)

    h_out_cold_recup = recuperator_cold_outlet_enthalpy if recuperated else h_in_cold_recup
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
        T0=T0,
        p0=p0,
    )

    # Evaluate HP evaporator
    h_in_cold_hp = hp_pump["state_out"].h
    p_in_cold_hp = hp_pump["state_out"].p
    h_out_cold_hp = hp_expander["state_in"].h
    p_out_cold_hp = hp_expander["state_in"].p

    p_in_hot_brine = parameters["heat_source"].pop("inlet_pressure")
    h_in_hot_brine = parameters["heat_source"].pop("inlet_enthalpy")

    # Compute brine exit temperatures from fractions — ordering guaranteed by construction:
    #   T_reinjection <= T_preheater_exit <= T_lp_evap_exit <= T_hp_evap_exit <= T_brine_inlet
    T_source_in = heating_fluid.get_state(
        cpx.HmassP_INPUTS, h_in_hot_brine, p_in_hot_brine
    ).T
    heat_source_hp_evap_exit_temp = (
        T_source_out_min
        + heat_source_hp_evap_exit_fraction * (T_source_in - T_source_out_min)
    )
    heat_source_lp_evap_exit_temp = (
        T_source_out_min
        + heat_source_lp_evap_exit_fraction
        * (heat_source_hp_evap_exit_temp - T_source_out_min)
    )
    heat_source_preheater_exit_temp = (
        T_source_out_min
        + heat_source_preheater_exit_fraction
        * (heat_source_lp_evap_exit_temp - T_source_out_min)
    )

    p_out_hot_hp = p_in_hot_brine * (1.0 - dp_hp_evap_h)
    h_out_hot_hp = heating_fluid.get_state(
        cpx.PT_INPUTS, p_out_hot_hp, heat_source_hp_evap_exit_temp
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
        T0=T0,
        p0=p0,
    )

    # Evaluate LP evaporator
    h_in_cold_lp = h_3
    p_in_cold_lp = p_3
    h_out_cold_lp = lp_evap_outlet_h
    p_out_cold_lp = p_3 * (1.0 - dp_lp_evap_c)

    p_in_hot_lp = p_out_hot_hp
    h_in_hot_lp = h_out_hot_hp
    p_out_hot_lp = p_in_hot_lp * (1.0 - dp_lp_evap_h)
    h_out_hot_lp = heating_fluid.get_state(
        cpx.PT_INPUTS, p_out_hot_lp, heat_source_lp_evap_exit_temp
    ).h

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
        T0=T0,
        p0=p0,
    )

    # Evaluate preheater
    h_in_cold_pre = recuperator["cold_side"]["state_out"].h
    p_in_cold_pre = recuperator["cold_side"]["state_out"].p
    h_out_cold_pre = h_3
    p_out_cold_pre = p_3

    p_in_hot_pre = p_out_hot_lp
    h_in_hot_pre = h_out_hot_lp
    p_out_hot_pre = p_in_hot_pre * (1.0 - dp_preheater_h)
    h_out_hot_pre = heating_fluid.get_state(
        cpx.PT_INPUTS, p_out_hot_pre, heat_source_preheater_exit_temp
    ).h

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
        T0=T0,
        p0=p0,
    )

    brine_exit_temperature = preheater["hot_side"]["state_out"].T

    # Evaluate heat-source pump (single shared brine loop)
    eff_hs = parameters["heat_source_pump"].pop("efficiency")
    eff_type_hs = parameters["heat_source_pump"].pop("efficiency_type")
    heat_source_pump = compression_process(
        heating_fluid,
        preheater["hot_side"]["state_out"].h,
        preheater["hot_side"]["state_out"].p,
        p_source_out,
        eff_hs,
        eff_type_hs,
        mass_flow=m_brine,
        T0=T0,
        p0=p0,
    )

    # Evaluate heat-sink pump
    T_in_sink = parameters["heat_sink"].pop("inlet_temperature")
    p_in_sink = parameters["heat_sink"].pop("inlet_pressure")
    h_in_sink = cooling_fluid.get_state(cpx.PT_INPUTS, p_in_sink, T_in_sink).h
    p_out_sink = p_sink_out / (1.0 - dp_cooler_c)
    eff_sink = parameters["heat_sink_pump"].pop("efficiency")
    eff_type_sink = parameters["heat_sink_pump"].pop("efficiency_type")
    heat_sink_pump = compression_process(
        cooling_fluid,
        h_in_sink,
        p_in_sink,
        p_out_sink,
        eff_sink,
        eff_type_sink,
        mass_flow=m_sink,
        T0=T0,
        p0=p0,
    )

    # Evaluate cooler
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
        T0=T0,
        p0=p0,
    )

    # Assemble components dict
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
        "mixer": mixer,
    }

    # First-law analysis
    Q_hp = hp_evaporator["energy_analysis"]["Q_hot"]
    Q_lp = lp_evaporator["energy_analysis"]["Q_hot"]
    Q_pre = preheater["energy_analysis"]["Q_hot"]
    Q_recup = recuperator["energy_analysis"]["Q_hot"]
    Q_in = Q_hp + Q_lp + Q_pre
    Q_out = cooler["energy_analysis"]["Q_hot"]
    W_out = (
        hp_expander["energy_analysis"]["power"]
        + lp_expander["energy_analysis"]["power"]
    )
    W_comp = lp_pump["energy_analysis"]["power"] + hp_pump["energy_analysis"]["power"]
    W_aux = (
        heat_source_pump["energy_analysis"]["power"]
        + heat_sink_pump["energy_analysis"]["power"]
    )
    W_in = W_comp + W_aux
    W_net = W_out - W_comp

    Q_in_max = m_brine * (h_in_hot_brine - source_out_min.h)
    Q_in_max_ambient = m_brine * (h_in_hot_brine - source_out_ambient.h)
    cycle_efficiency = (W_out - W_in) / Q_in
    system_efficiency = (W_out - W_in) / Q_in_max
    system_efficiency_ambient = (W_out - W_in) / Q_in_max_ambient
    backwork_ratio = W_comp / W_out
    energy_balance = (Q_in + W_comp) - (W_out + Q_out)

    energy_analysis = {
        "hp_evaporator_heat_flow": Q_hp,
        "lp_evaporator_heat_flow": Q_lp,
        "preheater_heat_flow": Q_pre,
        "recuperator_heat_flow": Q_recup,
        "total_heat_input": Q_in,
        "heater_heat_flow_max": Q_in_max,
        "heater_heat_flow_max_ambient": Q_in_max_ambient,
        "cooler_heat_flow": Q_out,
        "total_expander_power": W_out,
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

    # Cycle-level exergy analysis (2nd Law)
    _aux_pump_names = {"heat_source_pump", "heat_sink_pump"}

    E_fuel_system = (
        hp_evaporator["exergy_analysis"]["E_fuel"]
        + lp_evaporator["exergy_analysis"]["E_fuel"]
        + preheater["exergy_analysis"]["E_fuel"]
    )

    E_product_system = energy_analysis["net_system_power"]
    E_loss_cooler = cooler["exergy_analysis"]["E_product"]
    E_D_mixer = mixer["exergy_analysis"]["E_D"]

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
        "E_D_mixer": E_D_mixer,
        "eta_exergy": eta_exergy,
        "W_net_cycle": W_net,
        "balance_residual": balance_residual,
    }

    output = {
        "components": components,
        "energy_analysis": energy_analysis,
        "exergy_analysis": exergy_analysis,
        "variables": {
            "mass_flow_rate_cycle": m_total,
            "mass_flow_rate_sink": m_sink,
        },
    }

    f = utilities.evaluate_objective_function(output, objective_function)
    c_eq, c_ineq, constraint_report = utilities.evaluate_constraints(
        output, constraints
    )

    c_eq_hx = np.array(
        [
            component["energy_balance_residual"]
            for name, component in components.items()
            if component.get("type") == "heat_exchanger" and name != "recuperator"
        ]
    )
    if len(c_eq_hx) > 0:
        c_eq = np.concatenate([c_eq, c_eq_hx])

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
    components["mixer"]["stream_1"]["plot_params"] = {
        "color": purple,
        "linestyle": "--",
    }
    components["mixer"]["stream_2"]["plot_params"] = {
        "color": purple,
        "linestyle": "--",
    }

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
