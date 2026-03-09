"""
Astolfi (2014) Single-Stage Axial Turbine Efficiency Correlation
=================================================================

Implements the three-variable efficiency correlation from:

    Astolfi, M. (2014), PhD Thesis, Politecnico di Milano, Table 6.6

    eta_stage = f(SP, Vr, Ns)

where:
    SP  = V_out_is^0.5 / Dh_is^0.25          Size Parameter [m],  valid range [0.02, 1.0]
    Vr  = V_out_is / V_in                     Volume Ratio   [-],  valid range [1.2, 5] per stage
    Ns  = (RPM/60) * V_out_is^0.5 / Dh_is^0.75   Specific Speed [-],  valid range [0.045, 0.20]

Adjusted R-squared = 0.995.

    SP  : clamped to [0.02, 1.0] when entering the correlation, true value preserved for reporting
    Vr  : flagged if outside [1.2, 5] — regression was not calibrated outside this range.
          If Vr_stage > 5, more stages are needed. See basic_components.py auto-splitting.
    Ns  : flagged if outside [0.045, 0.20] — correlation was not regressed outside this range,
          evaluation there is extrapolation with no guaranteed accuracy.

    eta : raw polynomial value returned as-is. If Ns is severely out of range the polynomial
          may return unphysical values, but CoolProp will fail on the resulting enthalpy and
          the run will be discarded. The Vr and Ns flags are the correct mechanism for
          identifying invalid results after convergence.

Combined with stage-stacking (Section 6.1.5 of the thesis), this
enables multi-stage turbine modelling where RPM is a meaningful
optimization variable.

Author : Amanda (Master Thesis)
"""

import math


# ══════════════════════════════════════════════════════════════════════
#  Regression coefficients from Table 6.6 (Astolfi PhD thesis, 2014)
#  Single-stage correlation: eta = f(SP, Vr, Ns)
#  Adjusted R-squared = 0.995
#  SP valid range  : [0.02, 1.0]   — clamped when entering correlation
#  Vr valid range  : [1.2, 5]      — flagged if outside; use more stages if > 5
#  Ns valid range  : [0.045, 0.20] — flagged if outside; extrapolation only
# ══════════════════════════════════════════════════════════════════════

_COEFFICIENTS_TABLE66 = {
    #  n:  (F_n description,              A_n)
    #  Verified against Table 6.6, Astolfi PhD thesis (2014)
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

# Vr validity bounds from Astolfi (2014) Table 6.6
VR_LOWER = 1.2   # regression lower bound
VR_UPPER = 5.0   # regression upper bound / design limit for single stage


def astolfi_stage_eta(SP, Vr, Ns):
    """
    Evaluate the Astolfi (2014) single-stage efficiency correlation
    from Table 6.6 of the PhD thesis.

    Parameters
    ----------
    SP : float
        Size parameter [m]. Valid range [0.02, 1.0].
        Clamped to this range only when entering the correlation.
        True value is preserved and returned for reporting.
    Vr : float
        Volume ratio [-]. Valid regression range [1.2, 5] per stage.
        Flagged if outside this range. If Vr > 5, more stages are needed —
        see auto-splitting logic in basic_components.py.
    Ns : float
        Specific speed [-]. Valid regression range [0.045, 0.20].
        Flagged if outside this range — result is extrapolation only.

    Returns
    -------
    eta : float
        Raw polynomial value. May be unphysical if Ns is severely out of range —
        in that case CoolProp will fail on the resulting outlet enthalpy and the
        run is discarded. Use Vr_out_of_range and Ns_out_of_range to assess validity.
    SP_true : float
        True physical SP value (unclamped).
    SP_eval : float
        SP value used inside the correlation (may be clamped).
    SP_clamped : bool
        True if SP was clamped before entering the correlation.
    Vr_out_of_range : bool
        True if Vr is outside [1.2, 5] (outside regression calibration range).
    Ns_out_of_range : bool
        True if Ns is outside [0.045, 0.20] (extrapolation, no accuracy guarantee).
    """
    # SP_true is always the real physical value — used for reporting
    # SP_eval is clamped only for the polynomial evaluation
    SP_eval = SP
    SP_clamped = False
    if SP > 1.0:
        SP_eval = 1.0
        SP_clamped = True
    elif SP < 0.02:
        SP_eval = 0.02
        SP_clamped = True

    # Flag Vr outside the regression calibration range [1.2, 5]
    Vr_out_of_range = (Vr < VR_LOWER or Vr > VR_UPPER)

    Ns_out_of_range = (Ns < 0.045 or Ns > 0.20)

    lnSP = math.log(SP_eval)
    lnVr = math.log(Vr)

    # 15 basis functions from Table 6.6 — verified against thesis
    F = [0.0] * 15
    F[0]  = 1.0
    F[1]  = SP_eval
    F[2]  = lnSP
    F[3]  = lnSP**2
    F[4]  = lnSP**3
    F[5]  = Vr
    F[6]  = lnVr
    F[7]  = Ns
    F[8]  = Ns**2
    F[9]  = Ns**3
    F[10] = Ns**2 * lnVr
    F[11] = Ns * lnVr**2
    F[12] = Ns**3 * lnVr
    F[13] = Ns**3 * lnSP
    F[14] = Ns * lnSP**3

    eta_raw = sum(_COEFFICIENTS_TABLE66[i][1] * F[i] for i in range(15))

    # Return the raw polynomial value without clamping. If Ns is severely out
    # of range the value may be unphysical, but CoolProp will then fail on the
    # resulting outlet enthalpy and the run will be discarded. Clamping would
    # introduce bias into the optimizer by making out-of-range cases appear
    # artificially efficient (eta=1.0) or artificially useless (eta=0.0).
    # The Ns_out_of_range flag is the correct mechanism for post-convergence
    # validity assessment.
    return eta_raw, SP, SP_eval, SP_clamped, Vr_out_of_range, Ns_out_of_range


def compute_specific_speed(RPM, V_out_is, Dh_is):
    """Specific speed Ns = (RPM/60) * V_out_is^0.5 / Dh_is^0.75  (Eq. 9.3)."""
    return (RPM / 60.0) * V_out_is**0.5 / Dh_is**0.75
