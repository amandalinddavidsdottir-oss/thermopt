"""
Macchi & Astolfi (2017) Axial Turbine Efficiency Correlation
=============================================================

Implements the efficiency correlation from:

    Macchi, E. & Astolfi, M. (2017), "Axial flow turbines for Organic
    Rankine Cycle applications", Chapter 9 in *Organic Rankine Cycle (ORC)
    Power Systems*, Woodhead Publishing.

The correlation predicts the maximum attainable isentropic efficiency of
an axial-flow turbine as a function of two non-dimensional parameters:

    SP  = V_out_is^0.5 / Dh_is^0.25     Size Parameter [m]     (Eq. 9.1)
    Vr  = V_out_is / V_in                Volume Ratio   [-]     (Eq. 9.2)

Correlations are provided for 1, 2, and 3 stage turbines (Table 9.1).

The specific speed Ns is also computed for reference (Eq. 9.3):

    Ns  = (RPM/60) * V_out_is^0.5 / Dh_is^0.75

This module is designed to be called by ThermOpt's expansion_process()
during optimization, providing a physics-based efficiency at each
iteration instead of a fixed isentropic assumption.

Author : Amanda (Master Thesis)
"""

import math
import warnings


# ══════════════════════════════════════════════════════════════════════
#  Regression coefficients from Table 9.1 (Macchi & Astolfi, 2017)
# ══════════════════════════════════════════════════════════════════════
#  Row index n,  F_n term,  A_n for 1/2/3 stages

_COEFFICIENTS = {
    #  n:  (F_n description,         A_1stage,        A_2stage,        A_3stage)
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


def macchi_astolfi_eta(SP, Vr, n_stages):
    """
    Evaluate the Macchi & Astolfi efficiency correlation.

    Parameters
    ----------
    SP : float
        Size parameter [m].  Valid range: 0.02 to 1.0 m.
        If SP > 1.0, it is clamped to 1.0 (efficiency plateaus for
        large turbines — see Macchi & Astolfi, 2017, Fig. 9.9b).
        If SP < 0.02, it is clamped to 0.02.
    Vr : float
        Volume ratio [-].  No clamping applied — high Vr genuinely
        reduces efficiency due to increased Mach numbers.
    n_stages : int
        Number of turbine stages (1, 2, or 3).

    Returns
    -------
    eta : float
        Predicted maximum isentropic total-to-static efficiency [-],
        clamped to [0, 1].
    SP_eval : float
        SP value used in the correlation (may be clamped).
    SP_clamped : bool
        True if SP was clamped.
    Vr_out_of_range : bool
        True if Vr is outside [1.2, 200].
    """
    if n_stages not in (1, 2, 3):
        raise ValueError(f"n_stages must be 1, 2, or 3; got {n_stages}")

    # Clamp SP per professor's guidance:
    # efficiency plateaus at large SP (Reynolds number saturation)
    SP_eval = SP
    SP_clamped = False
    if SP > 1.0:
        SP_eval = 1.0
        SP_clamped = True
    elif SP < 0.02:
        SP_eval = 0.02
        SP_clamped = True

    # Check Vr range (no clamping — high Vr genuinely reduces efficiency)
    Vr_out_of_range = (Vr < 1.2 or Vr > 200)

    # Column index: 1-stage=1, 2-stage=2, 3-stage=3
    col = n_stages

    lnSP = math.log(SP_eval)
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

    # Clamp to physical range
    eta = max(0.0, min(1.0, eta))

    return eta, SP_eval, SP_clamped, Vr_out_of_range


def compute_specific_speed(RPM, V_out_is, Dh_is):
    """
    Specific speed Ns (Eq. 9.3).

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
        Ns [-] (in revolutions, not radians).
    """
    return (RPM / 60.0) * V_out_is**0.5 / Dh_is**0.75
