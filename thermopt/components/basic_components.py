import numpy as np

from scipy.integrate import solve_ivp

from .turbomachinery_nondimensional import RadialTurbine, CentrifugalCompressor
from .macchi_astolfi import astolfi_stage_eta, compute_specific_speed

import CoolProp.CoolProp as cp
import jaxprop as props

_astolfi_intermediate_printed = False


def _specific_flow_exergy(h, s, h0, s0, T0):
    """
    Compute specific flow exergy at a thermodynamic state.

    Parameters
    ----------
    h  : float  Specific enthalpy at the state [J/kg]
    s  : float  Specific entropy  at the state [J/(kg·K)]
    h0 : float  Dead-state specific enthalpy    [J/kg]
    s0 : float  Dead-state specific entropy     [J/(kg·K)]
    T0 : float  Dead-state temperature          [K]

    Returns
    -------
    float
        Specific flow exergy  e = (h - h0) - T0*(s - s0)  [J/kg]
    """
    return (h - h0) - T0 * (s - s0)


def heat_exchanger(
    fluid_hot,
    h_in_hot,
    h_out_hot,
    p_in_hot,
    p_out_hot,
    fluid_cold,
    h_in_cold,
    h_out_cold,
    p_in_cold,
    p_out_cold,
    mass_flow_hot,
    mass_flow_cold,
    num_steps=50,
    counter_current=True,
    *,
    T0,
    p0,
):
    """
    Simulate a counter-current or co-current heat exchanger using discretized enthalpy and pressure profiles.

    The heat exchange is discretized along both streams. Thermodynamic states are evaluated at linearly spaced
    enthalpy and pressure points. The resulting states are ordered in the direction of increasing temperature.
    For counter-current flow, the hot-side state array is flipped to enable element-wise temperature difference
    computation.

    Both ``mass_flow_hot`` and ``mass_flow_cold`` must be provided as inputs.
    The energy balance residual ``(Q_hot - Q_cold) / Q_hot`` is computed and
    returned. For all heat exchangers except the recuperator, this residual is
    added to the optimizer equality constraints by the cycle function.

    Parameters
    ----------
    fluid_hot : CoolProp.AbstractState
        Fluid object for the hot side.
    h_in_hot : float
        Inlet specific enthalpy [J/kg] of the hot fluid.
    h_out_hot : float
        Outlet specific enthalpy [J/kg] of the hot fluid.
    p_in_hot : float
        Inlet pressure [Pa] of the hot fluid.
    p_out_hot : float
        Outlet pressure [Pa] of the hot fluid.
    fluid_cold : CoolProp.AbstractState
        Fluid object for the cold side.
    h_in_cold : float
        Inlet specific enthalpy [J/kg] of the cold fluid.
    h_out_cold : float
        Outlet specific enthalpy [J/kg] of the cold fluid.
    p_in_cold : float
        Inlet pressure [Pa] of the cold fluid.
    p_out_cold : float
        Outlet pressure [Pa] of the cold fluid.
    mass_flow_hot : float
        Mass flow rate [kg/s] of the hot fluid.
    mass_flow_cold : float
        Mass flow rate [kg/s] of the cold fluid.
    num_steps : int, optional
        Number of discretization steps (default is 50).
    counter_current : bool, optional
        If True, assumes counter-current flow and flips hot-side state arrays (default is True).
    T0 : float
        Dead-state temperature [K]. Used to compute the exergy analysis.
    p0 : float
        Dead-state pressure [Pa]. Used to compute the exergy analysis.

    Returns
    -------
    dict
        Dictionary containing:
        - 'type': str, component type
        - 'hot_side': dict, state and property arrays for the hot stream
        - 'cold_side': dict, state and property arrays for the cold stream
        - 'q_hot_side': float, specific enthalpy drop [J/kg] on the hot side
        - 'q_cold_side': float, specific enthalpy gain [J/kg] on the cold side
        - 'temperature_hot_side': ndarray, temperatures [K] for the hot side (ordered by increasing T)
        - 'temperature_cold_side': ndarray, temperatures [K] for the cold side
        - 'temperature_difference': ndarray, element-wise temperature difference [K]
        - 'energy_balance_residual': float, normalized energy balance residual [-] (= 0 when satisfied)
        - 'heat_flow': float, total heat duty [W] (hot-side reference: mass_flow_hot * dh_hot)
        - 'energy_analysis': dict with keys 'Q_hot' [W], 'Q_cold' [W], 'heat_balance' [W]
        - 'exergy_analysis': dict with keys 'e_hot_in', 'e_hot_out', 'e_cold_in',
            'e_cold_out' [J/kg], 'E_fuel' [W], 'E_product' [W], 'E_D' [W], 'eta_exergy' [-]
    """

    # Evaluate properties on the hot side
    hot_side = heat_transfer_process(
        fluid=fluid_hot,
        h_1=h_in_hot,
        h_2=h_out_hot,
        p_1=p_in_hot,
        p_2=p_out_hot,
        num_steps=num_steps,
    )

    # Evaluate properties in the cold side
    cold_side = heat_transfer_process(
        fluid=fluid_cold,
        h_1=h_in_cold,
        h_2=h_out_cold,
        p_1=p_in_cold,
        p_2=p_out_cold,
        num_steps=num_steps,
    )

    # Sort values for temperature difference calculation
    if counter_current:
        hot_side["states"] = hot_side["states"].flipped()

    # Compute temperature difference
    dT = hot_side["states"]["T"] - cold_side["states"]["T"]

    dh_hot = hot_side["state_in"].h - hot_side["state_out"].h
    dh_cold = cold_side["state_out"].h - cold_side["state_in"].h

    # Store mass flows
    hot_side["mass_flow"] = mass_flow_hot
    cold_side["mass_flow"] = mass_flow_cold

    # Energy analysis
    Q_hot = mass_flow_hot * dh_hot  # total heat transferred from hot side [W]
    Q_cold = mass_flow_cold * dh_cold  # total heat absorbed by cold side     [W]
    heat_balance = Q_hot - Q_cold  # should be ~0 when energy balance holds

    # Cumulative Q arrays along the HX for T-Q plotting
    Q_hot_array = mass_flow_hot * (hot_side["state_in"]["h"] - hot_side["states"]["h"])
    Q_cold_array = mass_flow_cold * (
        cold_side["states"]["h"] - cold_side["state_in"]["h"]
    )
    hot_side["heat_flow"] = Q_hot_array
    cold_side["heat_flow"] = Q_cold_array

    energy_analysis = {
        "Q_hot": Q_hot,
        "Q_cold": Q_cold,
        "heat_balance": heat_balance,
    }

    energy_balance_residual = (Q_hot - Q_cold) / abs(Q_hot) if Q_hot != 0 else 0.0

    # Exergy analysis
    ds_hot = fluid_hot.get_state(props.PT_INPUTS, p0, T0)
    ds_cold = fluid_cold.get_state(props.PT_INPUTS, p0, T0)
    h0_hot, s0_hot = ds_hot.h, ds_hot.s
    h0_cold, s0_cold = ds_cold.h, ds_cold.s

    # Specific flow exergies — ensure physical inlet/outlet ordering
    hot_in = hot_side["state_in"]
    hot_out = hot_side["state_out"]
    cold_in = cold_side["state_in"]
    cold_out = cold_side["state_out"]


    e_hot_in = _specific_flow_exergy(hot_in.h, hot_in.s, h0_hot, s0_hot, T0)
    e_hot_out = _specific_flow_exergy(hot_out.h, hot_out.s, h0_hot, s0_hot, T0)
    e_cold_in = _specific_flow_exergy(cold_in.h, cold_in.s, h0_cold, s0_cold, T0)
    e_cold_out = _specific_flow_exergy(cold_out.h, cold_out.s, h0_cold, s0_cold, T0)

    E_fuel = mass_flow_hot * (e_hot_in - e_hot_out)  # [W]
    E_product = mass_flow_cold * (e_cold_out - e_cold_in)  # [W]
    E_D = E_fuel - E_product  # [W]  ≥ 0
    eta_exergy = E_product / E_fuel

    exergy_analysis = {
        "e_hot_in": float(e_hot_in),
        "e_hot_out": float(e_hot_out),
        "e_cold_in": float(e_cold_in),
        "e_cold_out": float(e_cold_out),
        "E_fuel": float(E_fuel),
        "E_product": float(E_product),
        "E_D": float(E_D),
        "eta_exergy": eta_exergy,
    }

    # Create result dictionary
    result = {
        "type": "heat_exchanger",
        "hot_side": hot_side,
        "cold_side": cold_side,
        "q_hot_side": dh_hot,
        "q_cold_side": dh_cold,
        "temperature_hot_side": hot_side["states"]["T"],
        "temperature_cold_side": cold_side["states"]["T"],
        "temperature_difference": dT,
        "energy_balance_residual": energy_balance_residual,
        "power": 0.0,
        "energy_analysis": energy_analysis,
        "exergy_analysis": exergy_analysis,
    }

    return result


def heat_transfer_process(fluid, h_1, p_1, h_2, p_2, num_steps=25):
    """
    Compute a discretized heat transfer process by discretizing enthalpy and pressure between inlet and outlet.

    This function generates a sequence of thermodynamic states along a heat transfer path
    by linearly spacing enthalpy and pressure between inlet and outlet values.
    The resulting states are organized in the direction of increasing enthalpy,
    which for sensible heating/cooling corresponds to increasing temperature.

    Parameters
    ----------
    fluid : CoolProp.AbstractState
        CoolProp fluid object used for property evaluation.
    h_1 : float
        Inlet specific enthalpy [J/kg].
    p_1 : float
        Inlet pressure [Pa].
    h_2 : float
        Outlet specific enthalpy [J/kg].
    p_2 : float
        Outlet pressure [Pa].
    num_steps : int, optional
        Number of discretization steps (default is 25).

    Returns
    -------
    dict
        Dictionary containing:
        - 'states': dict of arrays with thermodynamic properties along the process
        - 'fluid_name': str, name of the fluid
        - 'state_in': CoolProp state object at the inlet
        - 'state_out': CoolProp state object at the outlet
        - 'color': str, set to 'black' (optional use for plotting)
    """

    # Generate linearly spaced arrays for pressure and enthalpy
    p_array = np.linspace(p_1, p_2, num_steps)
    h_array = np.linspace(h_1, h_2, num_steps)

    # Compute states for hot or cold side
    states = fluid.get_state(props.HmassP_INPUTS, h_array, p_array)

    # Create result dictionary
    result = {
        "states": states,
        "fluid_name": fluid.name,
        "state_in": states.at_index(0),
        "state_out": states.at_index(-1),
        "color": "black",
    }

    return result


def compression_process(
    fluid,
    h_in,
    p_in,
    p_out,
    efficiency=None,
    efficiency_type="isentropic",
    mass_flow=None,
    data_in={},
    num_steps=10,
    *,
    T0,
    p0,
):
    """
    Calculate properties along a compression process defined by an isentropic or polytropic efficiency.

    Parameters
    ----------
    fluid : Fluid
        The fluid object used to evaluate thermodynamic properties.
    h_in : float
        Enthalpy at the start of the compression process [J/kg].
    p_in : float
        Pressure at the start of the compression process [Pa].
    p_out : float
        Pressure at the end of the compression process [Pa].
    efficiency : float
        The efficiency of the compression process.
    efficiency_type : str, optional
        The type of efficiency to be used ('isentropic', 'polytropic', or
        'non-dimensional'). Default is 'isentropic'.
    mass_flow : float
        Mass flow rate [kg/s].
    data_in : dict, optional
        Additional input data for correlation-based efficiency models.
    num_steps : int, optional
        Number of steps for the polytropic process calculation. Default is 10.
    T0 : float
        Dead-state temperature [K]. Used to compute the exergy analysis.
    p0 : float
        Dead-state pressure [Pa]. Used to compute the exergy analysis.

    Returns
    -------
    dict
        Dictionary containing the compression process results, including:
        - 'states', 'state_in', 'state_out': thermodynamic states
        - 'efficiency', 'efficiency_type': efficiency descriptor
        - 'specific_work', 'isentropic_work': specific work values [J/kg]
        - 'pressure_ratio': outlet-to-inlet pressure ratio
        - 'mass_flow': mass flow rate [kg/s]
        - 'power': shaft power consumed [W]  (= mass_flow * specific_work)
        - 'energy_analysis': dict with 'power' [W], 'specific_work' [J/kg],
          'isentropic_work' [J/kg]
        - 'exergy_analysis': dict with 'e_in', 'e_out' [J/kg], 'E_fuel' [W],
          'E_product' [W], 'E_D' [W], 'eta_exergy' [-]

    Raises
    ------
    ValueError
        If an invalid 'efficiency_type' is provided.
    """
    # Compute inlet state
    state_in = fluid.get_state(props.HmassP_INPUTS, h_in, p_in, supersaturation=True)
    state_out_is = fluid.get_state(
        props.PSmass_INPUTS, p_out, state_in.s, supersaturation=True
    )

    # Evaluate compression process
    if efficiency_type == "isentropic":
        h_out = state_in.h + (state_out_is.h - state_in.h) / efficiency
        state_out = fluid.get_state(
            props.HmassP_INPUTS, h_out, p_out, supersaturation=True
        )
        states = state_in + state_out
        data_out = {"isentropic_efficiency": efficiency}

    elif efficiency_type == "polytropic":
        # Differential equation defining the polytropic compression
        def odefun(p, h):
            state = fluid.get_state(props.HmassP_INPUTS, h, p, supersaturation=True)
            dhdp = 1.0 / (efficiency * state.rho)
            return dhdp, state

        # Solve polytropic compression differential equation
        sol = solve_ivp(
            lambda p, h: odefun(p, h)[0],
            [state_in.p, p_out],
            [state_in.h],
            t_eval=np.linspace(state_in.p, p_out, num_steps),
            method="RK45",
        )

        # Evaluate fluid properties at intermediate states
        states = postprocess_ode(sol.t, sol.y, odefun)
        state_in, state_out = states.at_index(0), states.at_index(-1)
        data_out = {"polytropic_efficiency": efficiency}

    elif efficiency_type == "non-dimensional":
        fluidC = cp.AbstractState("HEOS", fluid.name)
        compr = CentrifugalCompressor()
        compr.set_CoolProp_fluid(fluidC)
        # efficiency: a value is not needed
        # mass_flow: given a value and assigned as an optimization variable
        compr.data_in = data_in  # assigned compressor design parameters
        state_in = fluid.get_state(
            props.HmassP_INPUTS,
            h_in,
            p_in,
            supersaturation=True,
            generalize_quality=True,
        )
        T_in = state_in.T
        compr.CoolProp_solve_outlet_fixed_P(
            mass_flow, p_in, T_in, p_out, iterate_on_enthalpy=True
        )
        T_out_compressor = compr.outlet["T"]
        state_out = fluid.get_state(
            props.PT_INPUTS,
            p_out,
            T_out_compressor,
            supersaturation=True,
            generalize_quality=True,
        )
        states = [state_in, state_out]
        data_out = compr.get_output_data()

    else:
        raise ValueError("Invalid efficiency_type. Use 'isentropic' or 'polytropic'.")

    # Compute work
    isentropic_work = state_out_is.h - state_in.h
    specific_work = state_out.h - state_in.h

    # Energy analysis
    power = mass_flow * specific_work

    energy_analysis = {
        "power": power,
        "specific_work": specific_work,
        "isentropic_work": isentropic_work,
    }

    # Exergy analysis
    ds = fluid.get_state(props.PT_INPUTS, p0, T0)
    h0, s0 = ds.h, ds.s

    e_in = _specific_flow_exergy(state_in.h, state_in.s, h0, s0, T0)
    e_out = _specific_flow_exergy(state_out.h, state_out.s, h0, s0, T0)

    # Fuel = shaft work in, product = exergy gained by fluid
    E_fuel = power
    E_product = mass_flow * (e_out - e_in)
    E_D = E_fuel - E_product
    eta_exergy = E_product / E_fuel

    exergy_analysis = {
        "e_in": float(e_in),
        "e_out": float(e_out),
        "E_fuel": float(E_fuel),
        "E_product": float(E_product),
        "E_D": float(E_D),
        "eta_exergy": eta_exergy,
    }

    # Create result dictionary
    result = {
        "type": "compressor",
        "fluid_name": fluid.name,
        "states": states,
        "state_in": state_in,
        "state_out": state_out,
        "efficiency": efficiency,
        "efficiency_type": efficiency_type,
        "specific_work": specific_work,
        "isentropic_work": isentropic_work,
        "pressure_ratio": state_out.p / state_in.p,
        "mass_flow": mass_flow,
        "power": power,
        "energy_analysis": energy_analysis,
        "exergy_analysis": exergy_analysis,
        "color": "black",
        "data_in": data_in,
        "data_out": data_in | data_out,
    }

    return result


def expansion_process(
    fluid,
    h_in,
    p_in,
    p_out,
    efficiency=None,
    efficiency_type="isentropic",
    mass_flow=None,
    data_in={},
    num_steps=50,
    *,
    T0,
    p0,
):
    """
    Calculate properties along an expansion process defined by an isentropic or polytropic efficiency.

    Parameters
    ----------
    fluid : Fluid
        The fluid object used to evaluate thermodynamic properties.
    h_in : float
        Enthalpy at the start of the expansion process [J/kg].
    p_in : float
        Pressure at the start of the expansion process [Pa].
    p_out : float
        Pressure at the end of the expansion process [Pa].
    efficiency : float
        The efficiency of the expansion process.
    efficiency_type : str, optional
        The type of efficiency ('isentropic', 'polytropic', 'non-dimensional',
        or 'astolfi-stacking'). Default is 'isentropic'.
    mass_flow : float
        Mass flow rate [kg/s].
    data_in : dict, optional
        Additional design parameters for correlation-based efficiency models
        (e.g. RPM, n_stages for 'astolfi-stacking').
    num_steps : int, optional
        Number of steps for the polytropic process calculation. Default is 50.
    T0 : float
        Dead-state temperature [K]. Used to compute the exergy analysis.
    p0 : float
        Dead-state pressure [Pa]. Used to compute the exergy analysis.

    Returns
    -------
    dict
        Dictionary containing the expansion process results, including:
        - 'states', 'state_in', 'state_out': thermodynamic states
        - 'efficiency', 'efficiency_type': efficiency descriptor
        - 'specific_work', 'isentropic_work': specific work values [J/kg]
        - 'pressure_ratio': inlet-to-outlet pressure ratio
        - 'mass_flow': mass flow rate [kg/s]
        - 'power': shaft power produced [W]  (= mass_flow * specific_work)
        - 'energy_analysis': dict with 'power' [W], 'specific_work' [J/kg],
          'isentropic_work' [J/kg]
        - 'exergy_analysis': dict with 'e_in', 'e_out' [J/kg], 'E_fuel' [W],
          'E_product' [W], 'E_D' [W], 'eta_exergy' [-]

    Raises
    ------
    ValueError
        If an invalid 'efficiency_type' is provided.
    """
    # Compute inlet state
    state_in = fluid.get_state(
        props.HmassP_INPUTS, h_in, p_in, supersaturation=True, generalize_quality=True
    )
    state_out_is = fluid.get_state(
        props.PSmass_INPUTS,
        p_out,
        state_in.s,
        supersaturation=True,
        generalize_quality=True,
    )
    if efficiency_type == "isentropic":
        # Compute outlet state according to the definition of isentropic efficiency
        h_out = state_in.h - efficiency * (state_in.h - state_out_is.h)
        state_out = fluid.get_state(
            props.HmassP_INPUTS,
            h_out,
            p_out,
            supersaturation=True,
            generalize_quality=True,
        )
        states = state_in + state_out
        data_out = {"isentropic_efficiency": efficiency}

    elif efficiency_type == "polytropic":
        # Differential equation defining the polytropic expansion
        def odefun(p, h):
            state = fluid.get_state(
                props.HmassP_INPUTS, h, p, supersaturation=True, generalize_quality=True
            )
            dhdp = efficiency / state.rho
            return dhdp, state

        # Solve polytropic expansion differential equation
        sol = solve_ivp(
            lambda p, h: odefun(p, h)[0],
            [state_in.p, p_out],
            [state_in.h],
            t_eval=np.linspace(state_in.p, p_out, num_steps),
            method="RK45",
        )

        # Evaluate fluid properties at intermediate states
        states = postprocess_ode(sol.t, sol.y, odefun)
        state_in, state_out = states.at_index(0), states.at_index(-1)
        data_out = {"polytropic_efficiency": efficiency}

    elif efficiency_type == "non-dimensional":
        fluidT = cp.AbstractState("HEOS", fluid.name)
        turb = RadialTurbine()
        turb.set_CoolProp_fluid(fluidT)
        # efficiency: a value is not needed
        # mass_flow: given a value and assigned as an optimization variable
        turb.data_in = data_in  # assigned turbine design parameters
        state_in = fluid.get_state(
            props.HmassP_INPUTS,
            h_in,
            p_in,
            supersaturation=True,
            generalize_quality=True,
        )
        T_in = state_in.T
        turb.CoolProp_solve_outlet_fixed_P(
            mass_flow, p_in, T_in, p_out, iterate_on_enthalpy=True
        )
        T_out_expander = turb.outlet["T"]
        state_out = fluid.get_state(
            props.PT_INPUTS,
            p_out,
            T_out_expander,
            supersaturation=True,
            generalize_quality=True,
        )
        states = [state_in, state_out]
        data_out = turb.get_output_data()

    elif efficiency_type == "astolfi-stacking":
        # Stage-stacking with Astolfi (2014) Table 6.6 correlation, see section 6.1.5
        # Each stage efficiency is evaluated as f(SP_stage, Vr_stage, Ns_stage)
        n_stages = int(data_in.get("n_stages", 3))
        RPM = data_in.get("RPM", None)
        if RPM is None:
            raise ValueError("Astolfi-stacking efficiency requires RPM in data_in.")

        # Intermediate pressures must be provided as design variables for n_stages > 1
        intermediate_pressures = data_in.get("intermediate_pressures", None)

        if n_stages > 1 and intermediate_pressures is None:
            raise ValueError(
                f"astolfi-stacking with n_stages={n_stages} requires "
                f"{n_stages - 1} intermediate pressure design variable(s) "
                f"(expander_intermediate_pressure_1"
                + (
                    f" ... expander_intermediate_pressure_{n_stages-1}"
                    if n_stages > 2
                    else ""
                )
                + f"), but none were provided in the YAML. "
                f"Add them as design variables or set n_stages=1."
            )

        # Build stage pressure list
        global _astolfi_intermediate_printed
        if not _astolfi_intermediate_printed and n_stages > 1:
            print(
                f"  [Astolfi-stacking] {n_stages} stages, "
                f"{n_stages-1} intermediate pressure variable(s)"
            )
            _astolfi_intermediate_printed = True
        if intermediate_pressures is not None:
            p_stages = [p_in] + list(intermediate_pressures) + [p_out]
        else:
            p_stages = [p_in, p_out]

        # Overall isentropic quantities (for reporting)
        Dh_is_total = state_in.h - state_out_is.h
        V_in_total = mass_flow / state_in.rho
        V_out_is_total = mass_flow / state_out_is.rho
        SP_total = V_out_is_total**0.5 / Dh_is_total**0.25
        Vr_total = V_out_is_total / V_in_total
        Ns_total = compute_specific_speed(RPM, V_out_is_total, Dh_is_total)

        # Stage-by-stage loop
        stage_data = []
        stage_works = []
        current_state = state_in
        any_Vr_over = False
        any_SP_clamped = False
        any_Ns_out_of_range = False

       #print(p_stages)

        for i in range(n_stages):
            p_in_stg = p_stages[i]
            p_out_stg = p_stages[i + 1]

            # Inlet state for this stage
            h_in_stg = current_state.h
            s_in_stg = current_state.s
            rho_in_stg = current_state.rho
            T_in_stg = current_state.T

            # Isentropic outlet of this stage
            state_out_is_stg = fluid.get_state(
                props.PSmass_INPUTS,
                p_out_stg,
                s_in_stg,
                supersaturation=True,
                generalize_quality=True,
            )
            h_out_is_stg = state_out_is_stg.h
            rho_out_is_stg = state_out_is_stg.rho

            # Non-dimensional parameters for this stage
            V_in_stg = mass_flow / rho_in_stg
            V_out_is_stg = mass_flow / rho_out_is_stg
            Dh_is_stg = h_in_stg - h_out_is_stg

            SP_stg = V_out_is_stg**0.5 / Dh_is_stg**0.25
            Vr_stg = V_out_is_stg / V_in_stg
            Ns_stg = compute_specific_speed(RPM, V_out_is_stg, Dh_is_stg)

            # Evaluate Table 6.6 correlation
            eta_stg, SP_true_stg, SP_eval_stg, SP_clamp_stg, Vr_over_stg, Ns_oor_stg = (
                astolfi_stage_eta(SP_stg, Vr_stg, Ns_stg)
            )
            if Vr_over_stg:
                any_Vr_over = True
            if SP_clamp_stg:
                any_SP_clamped = True
            if Ns_oor_stg:
                any_Ns_out_of_range = True

            # Real outlet of this stage
           #print(Dh_is_stg)
           #print(eta_stg)
            h_out_real_stg = h_in_stg - eta_stg * Dh_is_stg
            state_out_real_stg = fluid.get_state(
                props.HmassP_INPUTS,
                h_out_real_stg,
                p_out_stg,
                generalize_quality=True,
            )

            work_stg = h_in_stg - h_out_real_stg
            stage_works.append(work_stg)

            stage_data.append(
                {
                    "stage": i + 1,
                    "p_in": p_in_stg,
                    "p_out": p_out_stg,
                    "T_in": T_in_stg,
                    "T_out_is": state_out_is_stg.T,
                    "T_out_real": state_out_real_stg.T,
                    "h_in": h_in_stg,
                    "h_out_is": h_out_is_stg,
                    "h_out_real": h_out_real_stg,
                    "s_in": s_in_stg,
                    "Dh_is": Dh_is_stg,
                    "SP": SP_stg,  # true physical value
                    "SP_eval": SP_eval_stg,  # clamped value used in correlation
                    "SP_clamped": SP_clamp_stg,
                    "Vr": Vr_stg,
                    "Vr_over_limit": Vr_over_stg,
                    "Ns": Ns_stg,
                    "Ns_out_of_range": Ns_oor_stg,  # flagged if outside [0.045, 0.20]
                    "eta_stage": eta_stg,
                    "work": work_stg,
                }
            )

            current_state = state_out_real_stg

        # Overall efficiency
        total_work = sum(stage_works)
        eta_overall = total_work / Dh_is_total if Dh_is_total > 0 else 0.0

        state_out = current_state
        efficiency = eta_overall
        h_out = state_out.h
        states = state_in + state_out

        SP_total_eval = max(0.02, min(1.0, SP_total))
        data_out = {
            "isentropic_efficiency": eta_overall,
            "n_stages": n_stages,
            "size_parameter": SP_total,
            "size_parameter_eval": SP_total_eval,
            "size_parameter_clamped": any_SP_clamped,
            "volume_ratio": Vr_total,
            "volume_ratio_out_of_range": any_Vr_over,
            "specific_speed": Ns_total,
            "specific_speed_out_of_range": any_Ns_out_of_range,
            "RPM": RPM,
            "stage_data": stage_data,
            "stage_Ns": np.array([s["Ns"] for s in stage_data]),
            "stage_Vr": np.array([s["Vr"] for s in stage_data]),
            "splitting_method": (
                "optimizer_intermediate_pressures"
                if intermediate_pressures is not None
                else "single_stage"
            ),
        }

    else:
        raise ValueError(
            "Invalid efficiency_type. Use 'isentropic', 'polytropic', "
            "'non-dimensional', or 'astolfi-stacking'."
        )

    # Compute work
    isentropic_work = state_in.h - state_out_is.h
    specific_work = state_in.h - state_out.h

    # Energy analysis
    power = mass_flow * specific_work

    energy_analysis = {
        "power": power,
        "specific_work": specific_work,
        "isentropic_work": isentropic_work,
    }

    # Exergy analysis
    ds = fluid.get_state(props.PT_INPUTS, p0, T0)
    h0, s0 = ds.h, ds.s

    e_in = _specific_flow_exergy(state_in.h, state_in.s, h0, s0, T0)
    e_out = _specific_flow_exergy(state_out.h, state_out.s, h0, s0, T0)

    # Fuel = exergy drop of fluid, product = shaft work
    E_fuel = mass_flow * (e_in - e_out)
    E_product = power
    E_D = E_fuel - E_product
    eta_exergy = E_product / E_fuel

    exergy_analysis = {
        "e_in": float(e_in),
        "e_out": float(e_out),
        "E_fuel": float(E_fuel),
        "E_product": float(E_product),
        "E_D": float(E_D),
        "eta_exergy": eta_exergy,
    }

    # Create result dictionary
    result = {
        "type": "expander",
        "fluid_name": fluid.name,
        "states": states,
        "state_in": state_in,
        "state_out": state_out,
        "efficiency": efficiency,
        "efficiency_type": efficiency_type,
        "specific_work": specific_work,
        "isentropic_work": isentropic_work,
        "pressure_ratio": state_in.p / state_out.p,
        "mass_flow": mass_flow,
        "power": power,
        "energy_analysis": energy_analysis,
        "exergy_analysis": exergy_analysis,
        "color": "black",
        "data_in": data_in,
        "data_out": data_in | data_out,
    }

    # For correlation-based types, expose SP, Vr, Ns at top level
    # (e.g. $components.hp_expander.specific_speed in YAML)
    if efficiency_type == "astolfi-stacking":
        result["size_parameter"] = data_out["size_parameter"]
        result["volume_ratio"] = data_out["volume_ratio"]
        result["specific_speed"] = data_out["specific_speed"]

    return result


def isenthalpic_valve(state_in, p_out, fluid, N=50):
    """
    Simulate an isenthalpic throttling process across a valve.

    The enthalpy remains constant across the valve, and pressure is reduced from inlet to outlet.
    The function returns a list of thermodynamic states along the expansion

    This function will yield similar results as an expansion process with a polytropic/isentropic efficiency of 0 %.

    Parameters
    ----------
    state_in : CoolProp.AbstractState
        Inlet thermodynamic state.
    p_out : float
        Outlet pressure [Pa].
    fluid : CoolProp.AbstractState
        Fluid object for property evaluation.
    N : int, optional
        Number of pressure steps between inlet and outlet (default is 50).

    Returns
    -------
    list of CoolProp.AbstractState
        List of thermodynamic states along the isenthalpic path.
    """
    p_array = np.linspace(state_in.p, p_out, N)
    h_array = state_in.h * p_array
    states = fluid.get_state(props.HmassP_INPUTS, h_array, p_array)
    return states


def postprocess_ode(t, y, ode_handle):
    """
    Reconstruct thermodynamic states from ODE solution vectors.

    This function applies a user-defined ODE handle to each time step to extract
    the corresponding thermodynamic state based on the integrated solution.
    It is useful for evaluating additional fluid properties not stored in the raw `y` vector.

    Parameters
    ----------
    t : ndarray
        Time array [s].
    y : ndarray
        Solution array of shape (n_states, n_time_steps).
    ode_handle : callable
        Function that takes (t_i, y_i) and returns a tuple (_, state),
        where `state` is a CoolProp.AbstractState or similar object.

    Returns
    -------
    list
        List of thermodynamic states reconstructed from the ODE trajectory.
    """
    # Collect individual states
    states = []
    for t_i, y_i in zip(t, y.T):
        _, state = ode_handle(t_i, y_i)
        states.append(state)

    # Combine into one batched FluidState
    return props.FluidState.stack(states)


def mixing_process(
    fluid,
    h_in_1,
    p_in_1,
    m_1,
    h_in_2,
    p_in_2,
    m_2,
    p_out,
    *,
    T0,
    p0,
):
    """
    Calculate the adiabatic mixing of two streams of the same fluid.

    The outlet enthalpy is determined by the energy balance:
        h_out = (m_1 * h_in_1 + m_2 * h_in_2) / (m_1 + m_2)

    The outlet pressure is passed explicitly by the caller. In the dual
    pressure cycle, both inlet streams are at the LP expander inlet pressure
    by construction, and this pressure is passed directly as p_out.

    Parameters
    ----------
    fluid : Fluid
        The fluid object used to evaluate thermodynamic properties.
    h_in_1 : float
        Specific enthalpy of stream 1 at the inlet [J/kg].
    p_in_1 : float
        Pressure of stream 1 at the inlet [Pa].
    m_1 : float
        Mass flow rate of stream 1 [kg/s].
    h_in_2 : float
        Specific enthalpy of stream 2 at the inlet [J/kg].
    p_in_2 : float
        Pressure of stream 2 at the inlet [Pa].
    m_2 : float
        Mass flow rate of stream 2 [kg/s].
    p_out : float
        Pressure of the mixed outlet stream [Pa]. Both inlet streams must
        be at this pressure level for adiabatic mixing to be valid.
    T0 : float
        Dead-state temperature [K]. Used to compute the exergy analysis.
    p0 : float
        Dead-state pressure [Pa]. Used to compute the exergy analysis.

    Returns
    -------
    dict
        Dictionary containing:
        - 'type': 'mixer'
        - 'state_in_1': thermodynamic state of stream 1 inlet
        - 'state_in_2': thermodynamic state of stream 2 inlet
        - 'state_out': thermodynamic state of mixed outlet
        - 'mass_flow_1': mass flow rate of stream 1 [kg/s]
        - 'mass_flow_2': mass flow rate of stream 2 [kg/s]
        - 'mass_flow_out': total outlet mass flow rate [kg/s]
        - 'energy_analysis': dict with energy balance information
        - 'exergy_analysis': dict with exergy destruction and efficiency
    """
    # Evaluate inlet states
    state_in_1 = fluid.get_state(props.HmassP_INPUTS, h_in_1, p_in_1)
    state_in_2 = fluid.get_state(props.HmassP_INPUTS, h_in_2, p_in_2)

    # Total mass flow and mixed outlet enthalpy from energy balance
    m_out = m_1 + m_2
    h_out = (m_1 * h_in_1 + m_2 * h_in_2) / m_out

    # Evaluate outlet state at the explicitly provided outlet pressure
    state_out = fluid.get_state(props.HmassP_INPUTS, h_out, p_out)

    # Energy analysis
    energy_balance_residual = (m_1 * h_in_1 + m_2 * h_in_2) - m_out * h_out
    energy_analysis = {
        "mass_flow_1": m_1,
        "mass_flow_2": m_2,
        "mass_flow_out": m_out,
        "h_in_1": h_in_1,
        "h_in_2": h_in_2,
        "h_out": h_out,
        "energy_balance_residual": float(energy_balance_residual),
    }

    # Exergy analysis
    ds = fluid.get_state(props.PT_INPUTS, p0, T0)
    h0, s0 = ds.h, ds.s

    e_in_1 = _specific_flow_exergy(state_in_1.h, state_in_1.s, h0, s0, T0)
    e_in_2 = _specific_flow_exergy(state_in_2.h, state_in_2.s, h0, s0, T0)
    e_out = _specific_flow_exergy(state_out.h, state_out.s, h0, s0, T0)

    E_fuel = m_1 * e_in_1 + m_2 * e_in_2
    E_product = m_out * e_out
    E_D = E_fuel - E_product
    eta_exergy = E_product / E_fuel

    exergy_analysis = {
        "e_in_1": float(e_in_1),
        "e_in_2": float(e_in_2),
        "e_out": float(e_out),
        "E_fuel": float(E_fuel),
        "E_product": float(E_product),
        "E_D": float(E_D),
        "eta_exergy": eta_exergy,
    }

    # Build states arrays for T-s diagram plotting
    # Two streams converging to the mixed outlet — mirrors hot_side/cold_side in heat_exchanger
    states_1 = state_in_1 + state_out  # HP expander outlet -> mixed state
    states_2 = state_in_2 + state_out  # LP evaporator outlet -> mixed state

    stream_1 = {
        "states": states_1,
        "state_in": state_in_1,
        "state_out": state_out,
        "mass_flow": m_1,
    }

    stream_2 = {
        "states": states_2,
        "state_in": state_in_2,
        "state_out": state_out,
        "mass_flow": m_2,
    }

    result = {
        "type": "mixer",
        "stream_1": stream_1,
        "stream_2": stream_2,
        "state_in_1": state_in_1,
        "state_in_2": state_in_2,
        "state_out": state_out,
        "mass_flow_1": m_1,
        "mass_flow_2": m_2,
        "mass_flow_out": m_out,
        "power": 0.0,
        "specific_work": 0.0,
        "energy_analysis": energy_analysis,
        "exergy_analysis": exergy_analysis,
    }

    return result
