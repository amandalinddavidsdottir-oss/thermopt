import warnings
import numpy as np

from scipy.integrate import solve_ivp

# from .. import utilities

from .turbomachinery_nondimensional import RadialTurbine, CentrifugalCompressor
from .macchi_astolfi import astolfi_stage_eta, compute_specific_speed

import CoolProp.CoolProp as cp
import jaxprop as props

# from .. import properties as props


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
    num_steps=50,
    counter_current=True,
):
    """
    Simulate a counter-current or co-current heat exchanger using discretized enthalpy and pressure profiles.

    The heat exchange is discretized along both streams. Thermodynamic states are evaluated at linearly spaced
    enthalpy and pressure points. The resulting states are ordered in the direction of increasing temperature.
    For counter-current flow, the hot-side state array is flipped to enable element-wise temperature difference
    computation.

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
    num_steps : int, optional
        Number of discretization steps (default is 50).
    counter_current : bool, optional
        If True, assumes counter-current flow and flips hot-side state arrays (default is True).

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
        - 'mass_flow_ratio': float, ratio of cold-side to hot-side mass flow required for heat balance
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

    # Compute mass flow ratio
    # Use 1.0 for cases with no heat exchange (i.e., no recuperator)
    dh_hot = hot_side["state_in"].h - hot_side["state_out"].h
    dh_cold = cold_side["state_out"].h - cold_side["state_in"].h
    mass_ratio = 1.00 if (dh_cold == 0 or dh_hot == 0) else dh_cold / dh_hot

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
        "mass_flow_ratio": mass_ratio,
    }

    return result


def heat_transfer_process(fluid, h_1, p_1, h_2, p_2, num_steps=25):
    """
    Compute a discretized heat transfer process by idiscretizing enthalpy and pressure between inlet and outlet.

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
        - 'mass_flow': float, set to NaN (to be populated externally)
        - 'heat_flow': float, set to NaN (to be populated externally)
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
        "mass_flow": np.nan,
        "heat_flow": np.nan,
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
):
    """
    Calculate properties along a compression process defined by a isentropic or polytropic efficiency

    Parameters
    ----------
    fluid : Fluid
        The fluid object used to evaluate thermodynamic properties
    h_in : float
        Enthalpy at the start of the compression process.
    p_in : float
        Pressure at the start of the compression process.
    p_out : float
        Pressure at the end of the compression process.
    efficiency : float
        The efficiency of the compression process.
    efficiency_type : str, optional
        The type of efficiency to be used in the process ('isentropic' or 'polytropic'). Default is 'isentropic'.
    num_steps : int, optional
        The number of steps for the polytropic process calculation. Default is 50.

    Returns
    -------
    tuple
        Tuple containing (state_out, states) where states is a list with all intermediate states

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
        "mass_flow": np.nan,
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
):
    """
    Calculate properties along a compression process defined by a isentropic or polytropic efficiency

    Parameters
    ----------
    fluid : Fluid
        The fluid object used to evaluate thermodynamic properties
    h_in : float
        Enthalpy at the start of the compression process.
    p_in : float
        Pressure at the start of the compression process.
    p_out : float
        Pressure at the end of the compression process.
    efficiency : float
        The efficiency of the compression process.
    efficiency_type : str, optional
        The type of efficiency to be used in the process ('isentropic' or 'polytropic'). Default is 'isentropic'.
    num_steps : int, optional
        The number of steps for the polytropic process calculation. Default is 50.

    Returns
    -------
    tuple
        Tuple containing (state_out, states) where states is a list with all intermediate states

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
        # ── Astolfi PhD (2014) stage-stacking with Table 6.6 correlation ──
        # Models multi-stage axial turbine with common RPM.
        # Each stage efficiency = f(SP_stage, Vr_stage, Ns_stage).
        # See Astolfi PhD thesis, section 6.1.5.
        if mass_flow is None:
            raise ValueError(
                "Astolfi-stacking efficiency requires mass_flow. "
                "Use mass-flow mode (well_mass_flow_rate in YAML)."
            )
        n_stages = int(data_in.get("n_stages", 3))
        RPM = data_in.get("RPM", None)
        if RPM is None:
            raise ValueError("Astolfi-stacking efficiency requires RPM in data_in.")

        # Equal volume ratio per stage (Astolfi section 6.1.5)
        # Vr_stage = Vr_total^(1/n_stages)
        # For each stage, find p_out_stg such that:
        #   rho_in_stg / rho_out_is_stg = Vr_stage
        # This requires a root-finding step per stage since real fluid
        # density is a nonlinear function of pressure.
        from scipy.optimize import brentq

        Vr_total_split = (mass_flow / state_out_is.rho) / (mass_flow / state_in.rho)
        Vr_stage_target = Vr_total_split ** (1.0 / n_stages)

        # Build stage pressure boundaries iteratively
        p_stages = [p_in]
        current_split_state = state_in
        for i in range(n_stages - 1):
            rho_in_stg = current_split_state.rho
            s_in_stg = current_split_state.s
            Vr_target = Vr_stage_target

            def vr_residual(p_guess):
                state_out_guess = fluid.get_state(
                    props.PSmass_INPUTS,
                    p_guess,
                    s_in_stg,
                    supersaturation=True,
                    generalize_quality=True,
                )
                return (rho_in_stg / state_out_guess.rho) - Vr_target

            # Search between current pressure and final outlet pressure
            p_lo = p_stages[-1] * 0.001
            p_hi = p_stages[-1] * 0.9999
            try:
                p_stg_out = brentq(vr_residual, p_lo, p_hi, xtol=1.0, maxiter=100)
            except ValueError:
                # Fallback to equal pressure ratio if root finding fails
                beta_total = p_in / p_out
                beta_stage = beta_total ** (1.0 / n_stages)
                p_stg_out = p_stages[-1] / beta_stage

            p_stages.append(p_stg_out)
            current_split_state = fluid.get_state(
                props.PSmass_INPUTS,
                p_stg_out,
                s_in_stg,
                supersaturation=True,
                generalize_quality=True,
            )

        p_stages.append(p_out)  # final stage always ends at p_out

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
            h_out_real_stg = h_in_stg - eta_stg * Dh_is_stg
            state_out_real_stg = fluid.get_state(
                props.HmassP_INPUTS,
                h_out_real_stg,
                p_out_stg,
                supersaturation=True,
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

            # Advance to next stage
            current_state = state_out_real_stg

        # Overall efficiency
        total_work = sum(stage_works)
        eta_overall = total_work / Dh_is_total if Dh_is_total > 0 else 0.0

        # Final outlet state is the real outlet of the last stage
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
        }

    else:
        raise ValueError(
            "Invalid efficiency_type. Use 'isentropic', 'polytropic', "
            "'non-dimensional', or 'astolfi-stacking'."
        )

    # Compute work
    isentropic_work = state_in.h - state_out_is.h
    specific_work = state_in.h - state_out.h

    # # Compute degree of superheating
    # state_in = props.calculate_superheating(state_in, fluid)
    # state_out = props.calculate_superheating(state_out, fluid)

    # # Compute degree of subcooling
    # state_in = props.calculate_subcooling(state_in, fluid)
    # state_out = props.calculate_subcooling(state_out, fluid)

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
        "mass_flow": np.nan,
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


def compute_component_energy_flows(components):
    """
    Compute power and heat flows for a set of components in a thermodynamic cycle.

    For heat exchangers, the function calculates local and total heat transfer on both hot and cold sides,
    assuming precomputed enthalpy profiles. For turbomachinery (compressors or expanders), it evaluates
    the mechanical power based on mass flow and specific work. The results are stored in-place in the
    `components` dictionary.

    Parameters
    ----------
    components : dict
        Dictionary of components, each with a 'type' field and required properties such as mass flow,
        enthalpy states, and specific work.

    Returns
    -------
    None
        The function modifies the `components` dictionary in place, adding:
        - 'power' [W]
        - 'heat_flow' and 'heat_flow_bis' [W]
        - 'heat_balance' [W] for heat exchangers
    """
    for name, component in components.items():
        if component["type"] == "heat_exchanger":
            hot = component["hot_side"]
            cold = component["cold_side"]
            Q_hot_array = hot["mass_flow"] * (hot["state_in"]["h"] - hot["states"]["h"])
            Q_cold_array = cold["mass_flow"] * (
                cold["states"]["h"] - cold["state_in"]["h"]
            )
            component["hot_side"]["heat_flow"] = Q_hot_array
            component["cold_side"]["heat_flow"] = Q_cold_array
            # component["hot_side"]["states"]["heat_flow"] = Q_hot_array
            # component["cold_side"]["states"]["heat_flow"] = Q_cold_array
            component["heat_flow"] = Q_hot_array[0]
            component["heat_flow_bis"] = Q_cold_array[-1]
            component["heat_balance"] = Q_hot_array[0] - Q_cold_array[-1]
            component["power"] = 0.0

        elif component["type"] in ["compressor", "expander"]:
            component["power"] = component["mass_flow"] * component["specific_work"]
            component["heat_flow_rate"] = 0.0
