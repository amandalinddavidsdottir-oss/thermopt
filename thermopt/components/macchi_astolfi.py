"""
Astolfi (2014) Single-Stage Axial Turbine Efficiency Correlation
=================================================================

Implements the three-variable efficiency correlation from:

    Astolfi, M. (2014), PhD Thesis, Politecnico di Milano, Table 6.6

    eta_stage = f(SP, Vr, Ns)

where:
    SP  = V_out_is^0.5 / Dh_is^0.25     Size Parameter [m]
    Vr  = V_out_is / V_in                Volume Ratio   [-]
    Ns  = (RPM/60) * V_out_is^0.5 / Dh_is^0.75   Specific Speed [-]

Adjusted R-squared = 0.995.  Valid for Vr <= 5 per stage.

Combined with stage-stacking (Section 6.1.5 of the thesis), this
enables multi-stage turbine modelling where RPM is a meaningful
optimization variable.

Author : Amanda (Master Thesis)
"""

import math
import warnings


# ══════════════════════════════════════════════════════════════════════
#  Regression coefficients from Table 6.6 (Astolfi PhD thesis, 2014)
#  Single-stage correlation: eta = f(SP, Vr, Ns)
#  Adjusted R-squared = 0.995
#  Valid for Vr <= 5 per stage
# ══════════════════════════════════════════════════════════════════════

_COEFFICIENTS_TABLE66 = {
    #  n:  (F_n description,              A_n)
    0:  ("1",                              0.828496),
    1:  ("SP",                            -0.083605),
    2:  ("ln(SP)",                         0.078745),
    3:  ("ln(SP)^2",                       0.030635),
    4:  ("ln(SP)^3",                       0.005738),
    5:  ("Vr",                             0.005011),
    6:  ("ln(Vr)",                        -0.021296),
    7:  ("Ns",                             2.648380),
    8:  ("Ns^2",                         -11.918500),
    9:  ("Ns^3",                          13.241800),
    10: ("Ns^2 * ln(Vr)",                  2.158950),
    11: ("Ns * ln(Vr)^2",                 -0.141356),
    12: ("Ns^3 * ln(Vr)",                 -7.013500),
    13: ("Ns^3 * ln(SP)",                  0.659568),
    14: ("Ns * ln(SP)^3",                 -0.002947),
}


def astolfi_stage_eta(SP, Vr, Ns):
    """
    Evaluate the Astolfi (2014) single-stage efficiency correlation
    from Table 6.6 of the PhD thesis.

    Parameters
    ----------
    SP : float
        Size parameter [m].  Clamped to [0.02, 1.0].
    Vr : float
        Volume ratio [-].  Should be <= 5 for single-stage validity.
    Ns : float
        Specific speed [-] (in revolutions).

    Returns
    -------
    eta : float
        Predicted isentropic total-to-static efficiency.
        Set to 0.0 if the polynomial returns a value outside [0, 1],
        which indicates extrapolation beyond the regression range.
    eta_raw : float
        Raw polynomial value before any clamping (for diagnostics).
    SP_eval : float
        SP value used (may be clamped).
    SP_clamped : bool
        True if SP was clamped.
    Vr_above_limit : bool
        True if Vr > 5 (outside single-stage validity).
    """
    SP_eval = SP
    SP_clamped = False
    if SP > 1.0:
        SP_eval = 1.0
        SP_clamped = True
    elif SP < 0.02:
        SP_eval = 0.02
        SP_clamped = True

    Vr_above_limit = (Vr > 5.0)

    lnSP = math.log(SP_eval)
    lnVr = math.log(Vr)

    # 15 basis functions from Table 6.6
    F = [0.0] * 15
    F[0]  = 1.0
    F[1]  = SP_eval          # SP (not ln(SP)!)
    F[2]  = lnSP
    F[3]  = lnSP**2
    F[4]  = lnSP**3
    F[5]  = Vr
    F[6]  = lnVr
    F[7]  = Ns
    F[8]  = Ns**2
    F[9]  = Ns**3
    F[10] = Ns**2 * lnVr     # cross-term
    F[11] = Ns * lnVr**2     # cross-term
    F[12] = Ns**3 * lnVr     # cross-term
    F[13] = Ns**3 * lnSP     # cross-term
    F[14] = Ns * lnSP**3     # cross-term

    eta_raw = sum(_COEFFICIENTS_TABLE66[i][1] * F[i] for i in range(15))

    # Guard against extrapolation artifacts:
    # - eta_raw < 0: polynomial went negative (Ns too high, bad design) → 0%
    # - eta_raw > 1: polynomial oscillated above 100% (extreme Ns) → 0%
    # Both indicate the operating point is far outside the regression range.
    if eta_raw < 0.0 or eta_raw > 1.0:
        eta = 0.0
    else:
        eta = eta_raw

    return eta, eta_raw, SP_eval, SP_clamped, Vr_above_limit


def compute_specific_speed(RPM, V_out_is, Dh_is):
    """Specific speed Ns (Eq. 9.3)."""
    return (RPM / 60.0) * V_out_is**0.5 / Dh_is**0.75
