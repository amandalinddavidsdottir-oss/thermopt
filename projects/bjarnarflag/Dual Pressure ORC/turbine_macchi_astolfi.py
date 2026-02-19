"""
Macchi & Astolfi Turbine Efficiency Correlation for ORC
=======================================================

Implements the axial-flow turbine efficiency correlation from:

    Macchi, E. & Astolfi, M. (2017), "Axial flow turbines for Organic
    Rankine Cycle applications", Chapter 9 in *Organic Rankine Cycle (ORC)
    Power Systems*, Woodhead Publishing.

The correlation predicts the maximum attainable isentropic efficiency of
an axial-flow turbine at optimized rotational speed, as a function of two
non-dimensional parameters:

    SP  = V_out_is^0.5 / Dh_is^0.25     Size Parameter [m]         Eq. 9.1
    Vr  = V_out_is / V_in                Volume Ratio   [-]         Eq. 9.2

where
    V_out_is  = isentropic volumetric flow rate at turbine exit  [m^3/s]
    V_in      = volumetric flow rate at turbine inlet            [m^3/s]
    Dh_is     = isentropic enthalpy drop across the turbine      [J/kg]

Correlations are provided for 1, 2, and 3 stage turbines (Table 9.1).

The specific speed Ns is also computed for reference (Eq. 9.3):

    Ns  = (RPM/60) * V_out_is^0.5 / Dh_is^0.75

Usage
-----
    # After thermopt optimization:
    from turbine_macchi_astolfi import evaluate_turbine_efficiency

    results = evaluate_turbine_efficiency(
        cycle,
        RPM=3000,                # constrained rotational speed [rev/min]
        stages=[1, 2, 3],       # compare stage counts
    )
    results.print_summary()

Author : Amanda (Master Thesis)
"""

import math
import numpy as np
import CoolProp.CoolProp as cp


# ══════════════════════════════════════════════════════════════════════
#  Regression coefficients from Table 9.1 (Macchi & Astolfi, 2017)
# ══════════════════════════════════════════════════════════════════════
#  Row index n,  F_n term,  A_n for 1/2/3 stages
#  A dash (—) in the book means the coefficient is zero.

_COEFFICIENTS = {
    #  n:  (F_n description,         A_1stage,        A_2stage,        A_3stage)
    #  Signs validated against Table 9.2 (1-stage) and Figs 9.9-9.10
    #  (2/3-stage) from Macchi & Astolfi (2017).
    0:  ("1",                         0.90831500,      0.923406,        0.932274),
    1:  ("ln(SP)",                   -0.05248690,     -0.0221021,      -0.01243),
    2:  ("ln(SP)^2",                 -0.04799080,     -0.0233814,      -0.018),
    3:  ("ln(SP)^3",                 -0.01710380,     -0.00844961,     -0.00716),
    4:  ("ln(SP)^4",                 -0.00244002,     -0.0012978,      -0.00118),
    5:  ("Vr",                        0.0,            -0.00069293,     -0.00044),
    6:  ("ln(Vr)",                    0.04961780,      0.0146911,       0.0),
    7:  ("ln(Vr)^2",                 -0.04894860,     -0.0102795,       0.0),
    8:  ("ln(Vr)^3",                  0.01171650,      0.0,            -0.0016),
    9:  ("ln(Vr)^4",                 -0.00100473,      0.000317241,     0.000298),
    10: ("ln(Vr)*ln(SP)",             0.05645970,      0.0163959,       0.005959),
    11: ("ln(Vr)^2*ln(SP)",          -0.01859440,     -0.00515265,     -0.00163),
    12: ("ln(Vr)*ln(SP)^2",           0.01288860,      0.00358361,      0.001946),
    13: ("ln(Vr)^3*ln(SP)",           0.00178187,      0.000554726,     0.000163),
    14: ("ln(Vr)^3*ln(SP)^2",        -0.00021196,      0.0,             0.0),
    15: ("ln(Vr)^2*ln(SP)^3",         0.00078667,      0.000293607,     0.000211),
}


def _macchi_astolfi_eta(SP, Vr, n_stages):
    """
    Evaluate the Macchi & Astolfi efficiency correlation.

    Parameters
    ----------
    SP : float
        Size parameter [m].  Valid range: 0.02 to 1.0 m.
    Vr : float
        Volume ratio [-].  Valid range: 1.2 to 200.
    n_stages : int
        Number of turbine stages (1, 2, or 3).

    Returns
    -------
    float
        Predicted maximum isentropic total-to-static efficiency [-].
    """
    if n_stages not in (1, 2, 3):
        raise ValueError(f"n_stages must be 1, 2, or 3; got {n_stages}")

    # Column index in the coefficient table:  1-stage=1, 2-stage=2, 3-stage=3
    col = n_stages

    lnSP = math.log(SP)
    lnVr = math.log(Vr)

    # Build the 16 basis functions F_i
    F = [0.0] * 16
    F[0]  = 1.0
    F[1]  = lnSP
    F[2]  = lnSP**2
    F[3]  = lnSP**3
    F[4]  = lnSP**4
    F[5]  = Vr
    F[6]  = lnVr
    F[7]  = lnVr**2
    F[8]  = lnVr**3
    F[9]  = lnVr**4
    F[10] = lnVr * lnSP
    F[11] = lnVr**2 * lnSP
    F[12] = lnVr * lnSP**2
    F[13] = lnVr**3 * lnSP
    F[14] = lnVr**3 * lnSP**2
    F[15] = lnVr**2 * lnSP**3

    eta = sum(_COEFFICIENTS[i][col] * F[i] for i in range(16))

    return eta


def compute_SP(V_out_is, Dh_is):
    """
    Size Parameter (Eq. 9.1).

    Parameters
    ----------
    V_out_is : float
        Isentropic outlet volumetric flow rate [m^3/s].
    Dh_is : float
        Isentropic specific enthalpy drop [J/kg].

    Returns
    -------
    float
        SP [m].
    """
    return V_out_is**0.5 / Dh_is**0.25


def compute_Vr(V_out_is, V_in):
    """
    Volume Ratio (Eq. 9.2).

    Parameters
    ----------
    V_out_is : float
        Isentropic outlet volumetric flow rate [m^3/s].
    V_in : float
        Inlet volumetric flow rate [m^3/s].

    Returns
    -------
    float
        Vr [-].
    """
    return V_out_is / V_in


def compute_Ns(RPM, V_out_is, Dh_is):
    """
    Specific speed (Eq. 9.3).

    Parameters
    ----------
    RPM : float
        Rotational speed [rev/min].
    V_out_is : float
        Isentropic outlet volumetric flow rate [m^3/s].
    Dh_is : float
        Isentropic specific enthalpy drop [J/kg].

    Returns
    -------
    float
        Ns [-]  (in revolutions, not radians).
    """
    return (RPM / 60.0) * V_out_is**0.5 / Dh_is**0.75


def compute_optimal_Ns(SP, Vr):
    """
    Estimate optimal specific speed.

    For most ORC turbines, the optimal Ns is between 0.10 and 0.15
    (Macchi & Astolfi, §9.4.1).  For small SP and high Vr it drops
    to ~0.05.  This function provides a simple estimate.

    Parameters
    ----------
    SP : float
        Size parameter [m].
    Vr : float
        Volume ratio [-].

    Returns
    -------
    float
        Estimated optimal Ns [-].
    """
    # Heuristic based on Fig. 9.4 and the discussion in §9.4.1
    Ns_opt = 0.15
    if SP < 0.05 or Vr > 50:
        Ns_opt = 0.08
    elif SP < 0.1 or Vr > 20:
        Ns_opt = 0.10
    return Ns_opt


def compute_RPM_from_Ns(Ns, V_out_is, Dh_is):
    """
    Back-calculate RPM from a target specific speed.

    Parameters
    ----------
    Ns : float
        Target specific speed [-].
    V_out_is : float
        Isentropic outlet volumetric flow rate [m^3/s].
    Dh_is : float
        Isentropic specific enthalpy drop [J/kg].

    Returns
    -------
    float
        RPM [rev/min].
    """
    return Ns * 60.0 * Dh_is**0.75 / V_out_is**0.5


# ══════════════════════════════════════════════════════════════════════
#  Result container
# ══════════════════════════════════════════════════════════════════════
class TurbineCorrelationResults:
    """
    Container for results of the Macchi & Astolfi evaluation.

    Attributes
    ----------
    fluid_name : str
    m_dot : float           mass flow rate [kg/s]
    p_in, p_out : float     inlet / outlet pressure [Pa]
    T_in : float            inlet temperature [K]
    h_in, h_out_is : float  inlet / isentropic outlet enthalpy [J/kg]
    rho_in, rho_out_is : float   inlet / isentropic outlet density [kg/m3]
    Dh_is : float           isentropic enthalpy drop [J/kg]
    V_in : float            inlet volumetric flow rate [m3/s]
    V_out_is : float        isentropic outlet volumetric flow rate [m3/s]
    SP : float              size parameter [m]
    Vr : float              volume ratio [-]
    stages : list of dict   per-stage-count results
    """

    def __init__(self):
        self.fluid_name = ""
        self.label = "expander"  # display label (e.g. "HP Expander")
        self.m_dot = 0.0
        self.p_in = 0.0
        self.p_out = 0.0
        self.T_in = 0.0
        self.h_in = 0.0
        self.h_out_is = 0.0
        self.rho_in = 0.0
        self.rho_out_is = 0.0
        self.Dh_is = 0.0
        self.V_in = 0.0
        self.V_out_is = 0.0
        self.SP = 0.0
        self.Vr = 0.0
        self.W_is = 0.0          # isentropic power [W]
        self.stages = []         # list of dicts, one per n_stages evaluated
        self.RPM_user = None     # user-specified RPM (or None)

    def print_summary(self):
        """Print formatted summary."""
        print("\n" + "=" * 76)
        print(f"  MACCHI & ASTOLFI TURBINE EFFICIENCY CORRELATION — {self.label}")
        print("=" * 76)
        print(f"  Fluid              : {self.fluid_name}")
        print(f"  Mass flow rate     : {self.m_dot:.3f} kg/s")
        print(f"  Inlet  (p, T)      : {self.p_in/1e5:.2f} bar,  "
              f"{self.T_in - 273.15:.1f} °C")
        print(f"  Outlet pressure    : {self.p_out/1e5:.2f} bar")
        print(f"  Dh_is              : {self.Dh_is/1e3:.2f} kJ/kg")
        print(f"  W_is               : {self.W_is/1e3:.1f} kW")
        print("-" * 76)
        print(f"  V_in               : {self.V_in:.4f} m³/s")
        print(f"  V_out,is           : {self.V_out_is:.4f} m³/s")
        print(f"  SP                 : {self.SP:.4f} m")
        print(f"  Vr                 : {self.Vr:.2f}")

        # Validity warnings
        if self.SP < 0.02 or self.SP > 1.0:
            print(f"  ⚠  SP = {self.SP:.4f} m is OUTSIDE the correlation "
                  f"range [0.02, 1.0] m")
        if self.Vr < 1.2 or self.Vr > 200:
            print(f"  ⚠  Vr = {self.Vr:.2f} is OUTSIDE the correlation "
                  f"range [1.2, 200]")

        print("-" * 76)
        header = (f"  {'Stages':<8s} {'eta_is':>8s} {'W_actual [kW]':>14s}"
                  f" {'Ns_opt':>8s} {'RPM_opt':>10s}")
        if self.RPM_user is not None:
            header += f" {'Ns@{0}RPM'.format(int(self.RPM_user)):>12s}"
        print(header)
        print("-" * 76)

        for s in self.stages:
            line = (f"  {s['n_stages']:<8d} {s['eta_is']:8.4f}"
                    f" {s['W_actual']/1e3:14.2f}"
                    f" {s['Ns_opt']:8.4f}"
                    f" {s['RPM_opt']:10.0f}")
            if self.RPM_user is not None:
                line += f" {s['Ns_user']:12.4f}"
            print(line)

        print("=" * 76)
        best = max(self.stages, key=lambda s: s["eta_is"])
        print(f"  Best option: {best['n_stages']} stage(s) with "
              f"eta_is = {best['eta_is']*100:.2f}%  "
              f"(W = {best['W_actual']/1e3:.2f} kW)")
        print("=" * 76 + "\n")

    def to_dict(self):
        """Return results as a flat dictionary for DataFrame use."""
        rows = []
        for s in self.stages:
            rows.append({
                "fluid": self.fluid_name,
                "m_dot_kg_s": self.m_dot,
                "p_in_bar": self.p_in / 1e5,
                "p_out_bar": self.p_out / 1e5,
                "Dh_is_kJ_kg": self.Dh_is / 1e3,
                "W_is_kW": self.W_is / 1e3,
                "SP_m": self.SP,
                "Vr": self.Vr,
                "n_stages": s["n_stages"],
                "eta_is": s["eta_is"],
                "W_actual_kW": s["W_actual"] / 1e3,
                "Ns_opt": s["Ns_opt"],
                "RPM_opt": s["RPM_opt"],
                "Ns_user": s.get("Ns_user", None),
                "RPM_user": self.RPM_user,
            })
        return rows


# ══════════════════════════════════════════════════════════════════════
#  Main function: evaluate from a thermopt cycle object
# ══════════════════════════════════════════════════════════════════════
def _evaluate_expander_component(exp, wf, RPM, stages, label="expander"):
    """
    Core evaluation logic for a single expander component.

    Parameters
    ----------
    exp : dict
        The expander component dict from cycle_data["components"].
    wf : object
        The working fluid object from cycle_data["working_fluid"].
    RPM : float or None
        Fixed rotational speed [rev/min].
    stages : list of int
        Stage counts to evaluate.
    label : str
        Display label (e.g. "hp_expander", "lp_expander", "expander").

    Returns
    -------
    TurbineCorrelationResults
    """
    # ── Extract thermodynamic states ──────────────────────────────
    state_in = exp["state_in"]
    state_out = exp["state_out"]

    p_in  = state_in.p
    p_out = state_out.p
    h_in  = state_in.h
    T_in  = state_in.T
    s_in  = state_in.s
    rho_in = state_in.rho

    # Mass flow rate
    m_dot = exp["mass_flow"]

    # Get fluid name from the component
    fluid_name = exp.get("fluid_name", "unknown")

    # ── Isentropic outlet state ───────────────────────────────────
    Dh_is = exp["isentropic_work"]                       # [J/kg]
    h_out_is = h_in - Dh_is

    # Get density at the isentropic outlet using the thermopt fluid object
    try:
        import jaxprop
        state_out_is = wf.get_state(jaxprop.PSmass_INPUTS, p_out, s_in,
                                     supersaturation=True,
                                     generalize_quality=True)
        rho_out_is = state_out_is.rho
    except Exception:
        # Fallback: estimate from CoolProp directly with capitalized name
        try:
            name = fluid_name.capitalize() if fluid_name.islower() else fluid_name
            rho_out_is = cp.PropsSI("D", "P", p_out, "S", s_in, name)
        except Exception:
            # Last resort: use actual outlet density as approximation
            rho_out_is = state_out.rho

    # ── Compute non-dimensional parameters ────────────────────────
    V_in = m_dot / rho_in                            # [m³/s]
    V_out_is = m_dot / rho_out_is                    # [m³/s]

    SP = compute_SP(V_out_is, Dh_is)
    Vr = compute_Vr(V_out_is, V_in)

    W_is = m_dot * Dh_is                             # [W]

    # ── Build results ─────────────────────────────────────────────
    res = TurbineCorrelationResults()
    res.fluid_name = fluid_name
    res.label = label
    res.m_dot = m_dot
    res.p_in = p_in
    res.p_out = p_out
    res.T_in = T_in
    res.h_in = h_in
    res.h_out_is = h_out_is
    res.rho_in = rho_in
    res.rho_out_is = rho_out_is
    res.Dh_is = Dh_is
    res.V_in = V_in
    res.V_out_is = V_out_is
    res.SP = SP
    res.Vr = Vr
    res.W_is = W_is
    res.RPM_user = RPM

    for n in stages:
        eta = _macchi_astolfi_eta(SP, Vr, n)
        eta = max(0.0, min(1.0, eta))     # clip to [0, 1]

        Ns_opt = compute_optimal_Ns(SP, Vr)
        RPM_opt = compute_RPM_from_Ns(Ns_opt, V_out_is, Dh_is)

        entry = {
            "n_stages": n,
            "eta_is": eta,
            "W_actual": m_dot * Dh_is * eta,     # actual power [W]
            "Ns_opt": Ns_opt,
            "RPM_opt": RPM_opt,
        }

        if RPM is not None:
            Ns_user = compute_Ns(RPM, V_out_is, Dh_is)
            entry["Ns_user"] = Ns_user

        res.stages.append(entry)

    return res


def evaluate_turbine_efficiency(cycle, RPM=None, stages=None):
    """
    Evaluate the Macchi & Astolfi correlation for the converged expander(s).

    Auto-detects single-pressure (one expander) vs dual-pressure
    (hp_expander + lp_expander) topologies.

    Parameters
    ----------
    cycle : thermopt.ThermodynamicCycleOptimization
        A converged thermopt cycle object.
    RPM : float, optional
        If given, also compute the specific speed at this fixed RPM
        (e.g. 3000 for a 2-pole 50 Hz generator).
    stages : list of int, optional
        Stage counts to evaluate.  Default: [1, 2, 3].

    Returns
    -------
    TurbineCorrelationResults or list of TurbineCorrelationResults
        Single result for single-pressure, list of two for dual-pressure.
    """
    if stages is None:
        stages = [1, 2, 3]

    components = cycle.problem.cycle_data["components"]
    wf = cycle.problem.cycle_data["working_fluid"]

    is_dual = "hp_expander" in components

    if is_dual:
        hp_res = _evaluate_expander_component(
            components["hp_expander"], wf, RPM, stages, label="HP Expander")
        lp_res = _evaluate_expander_component(
            components["lp_expander"], wf, RPM, stages, label="LP Expander")
        return [hp_res, lp_res]
    else:
        return _evaluate_expander_component(
            components["expander"], wf, RPM, stages, label="Expander")


# ══════════════════════════════════════════════════════════════════════
#  Standalone evaluation (no thermopt needed)
# ══════════════════════════════════════════════════════════════════════
def evaluate_turbine_standalone(fluid_name, p_in, T_in, p_out, m_dot,
                                RPM=None, stages=None):
    """
    Evaluate the correlation from raw thermodynamic inputs.

    This is useful when you don't have a thermopt cycle object — e.g.
    for quick what-if studies, or when comparing parallel vs series
    turbine arrangements with different pressure splits.

    Parameters
    ----------
    fluid_name : str
        CoolProp fluid name (e.g. 'butane', 'Isopentane').
    p_in : float
        Turbine inlet pressure [Pa].
    T_in : float
        Turbine inlet temperature [K].
    p_out : float
        Turbine outlet pressure [Pa].
    m_dot : float
        Mass flow rate [kg/s].
    RPM : float, optional
        Fixed rotational speed [rev/min].
    stages : list of int, optional
        Stage counts to evaluate.  Default: [1, 2, 3].

    Returns
    -------
    TurbineCorrelationResults
    """
    if stages is None:
        stages = [1, 2, 3]

    # ── Inlet state ───────────────────────────────────────────────
    h_in = cp.PropsSI("H", "P", p_in, "T", T_in, fluid_name)
    s_in = cp.PropsSI("S", "P", p_in, "T", T_in, fluid_name)
    rho_in = cp.PropsSI("D", "P", p_in, "T", T_in, fluid_name)

    # ── Isentropic outlet ─────────────────────────────────────────
    h_out_is = cp.PropsSI("H", "P", p_out, "S", s_in, fluid_name)
    rho_out_is = cp.PropsSI("D", "P", p_out, "S", s_in, fluid_name)

    # ── Non-dimensional parameters ────────────────────────────────
    Dh_is = h_in - h_out_is
    V_in = m_dot / rho_in
    V_out_is = m_dot / rho_out_is

    SP = compute_SP(V_out_is, Dh_is)
    Vr = compute_Vr(V_out_is, V_in)
    W_is = m_dot * Dh_is

    # ── Build results ─────────────────────────────────────────────
    res = TurbineCorrelationResults()
    res.fluid_name = fluid_name
    res.m_dot = m_dot
    res.p_in = p_in
    res.p_out = p_out
    res.T_in = T_in
    res.h_in = h_in
    res.h_out_is = h_out_is
    res.rho_in = rho_in
    res.rho_out_is = rho_out_is
    res.Dh_is = Dh_is
    res.V_in = V_in
    res.V_out_is = V_out_is
    res.SP = SP
    res.Vr = Vr
    res.W_is = W_is
    res.RPM_user = RPM

    for n in stages:
        eta = _macchi_astolfi_eta(SP, Vr, n)
        eta = max(0.0, min(1.0, eta))

        Ns_opt = compute_optimal_Ns(SP, Vr)
        RPM_opt = compute_RPM_from_Ns(Ns_opt, V_out_is, Dh_is)

        entry = {
            "n_stages": n,
            "eta_is": eta,
            "W_actual": m_dot * Dh_is * eta,
            "Ns_opt": Ns_opt,
            "RPM_opt": RPM_opt,
        }
        if RPM is not None:
            entry["Ns_user"] = compute_Ns(RPM, V_out_is, Dh_is)

        res.stages.append(entry)

    return res


# ══════════════════════════════════════════════════════════════════════
#  Parallel vs. series comparison helper
# ══════════════════════════════════════════════════════════════════════
def compare_parallel_vs_series(fluid_name, p_high, T_in, p_mid, p_low,
                                m_dot_total, RPM=None, stages=None):
    """
    Compare two turbines arranged in parallel vs. in series.

    Use case: dual pressure level ORC where high-pressure and
    low-pressure expansions can either be on separate parallel
    turbines or on a single shaft in series.

    Parallel arrangement:
        Turbine A: (p_high → p_low) at m_dot / 2
        Turbine B: (p_high → p_low) at m_dot / 2

    Series arrangement:
        Turbine A: (p_high → p_mid) at m_dot
        Turbine B: (p_mid  → p_low) at m_dot

    Parameters
    ----------
    fluid_name : str
        CoolProp fluid name.
    p_high, p_mid, p_low : float
        Pressures [Pa].
    T_in : float
        Turbine inlet temperature [K].
    m_dot_total : float
        Total mass flow rate [kg/s].
    RPM : float, optional
        Fixed RPM for both turbines.
    stages : list of int, optional
        Stage counts to evaluate per turbine.

    Returns
    -------
    dict
        {"parallel": [res_A, res_B], "series": [res_A, res_B]}
    """
    # ── Parallel: both turbines expand full pressure ratio, half flow ──
    par_A = evaluate_turbine_standalone(
        fluid_name, p_high, T_in, p_low, m_dot_total / 2, RPM, stages)
    par_B = evaluate_turbine_standalone(
        fluid_name, p_high, T_in, p_low, m_dot_total / 2, RPM, stages)

    # ── Series: first turbine to p_mid, second from p_mid to p_low ────
    ser_A = evaluate_turbine_standalone(
        fluid_name, p_high, T_in, p_mid, m_dot_total, RPM, stages)

    # For the second turbine, we need the exit temperature of the first.
    # Use the best-stage efficiency from ser_A to estimate T_mid.
    best_A = max(ser_A.stages, key=lambda s: s["eta_is"])
    h_out_A = ser_A.h_in - best_A["eta_is"] * ser_A.Dh_is
    T_mid = cp.PropsSI("T", "P", p_mid, "H", h_out_A, fluid_name)

    ser_B = evaluate_turbine_standalone(
        fluid_name, p_mid, T_mid, p_low, m_dot_total, RPM, stages)

    return {
        "parallel": [par_A, par_B],
        "series": [ser_A, ser_B],
    }


def print_comparison(comp, RPM=None):
    """
    Pretty-print the output of compare_parallel_vs_series().

    Parameters
    ----------
    comp : dict
        Output from compare_parallel_vs_series().
    RPM : float, optional
        RPM label for display.
    """
    rpm_str = f" @ {int(RPM)} RPM" if RPM else ""

    print("\n" + "=" * 76)
    print(f"  PARALLEL vs. SERIES TURBINE COMPARISON{rpm_str}")
    print("=" * 76)

    for arrangement, turbines in comp.items():
        print(f"\n  ── {arrangement.upper()} ──")
        total_power = {}
        for idx, t in enumerate(turbines):
            label = "A" if idx == 0 else "B"
            print(f"    Turbine {label}: "
                  f"p = {t.p_in/1e5:.1f} → {t.p_out/1e5:.1f} bar, "
                  f"m = {t.m_dot:.2f} kg/s, "
                  f"SP = {t.SP:.4f} m, Vr = {t.Vr:.1f}")
            for s in t.stages:
                n = s["n_stages"]
                if n not in total_power:
                    total_power[n] = 0.0
                total_power[n] += s["W_actual"]
                print(f"      {n} stage: eta = {s['eta_is']*100:.2f}%, "
                      f"W = {s['W_actual']/1e3:.1f} kW, "
                      f"RPM_opt = {s['RPM_opt']:.0f}")

        print(f"    Combined power (A + B):")
        for n, W in sorted(total_power.items()):
            print(f"      {n} stage: W_total = {W/1e3:.1f} kW")

    print("=" * 76 + "\n")
