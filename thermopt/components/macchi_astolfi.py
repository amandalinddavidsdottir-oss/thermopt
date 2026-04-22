"""
Efficiency correlation for a single-stage axial turbine from Astolfi (2014).

Based on the three-variable regression from Table 6.6 of:
    Astolfi, M. (2014), PhD Thesis, Politecnico di Milano

    eta_stage = f(SP, Vr, Ns)

where:
    SP  = V_out_is^0.5 / Dh_is^0.25              Size parameter [m],   valid range [0.02, 1.0]
    Vr  = V_out_is / V_in                         Volume ratio   [-],   valid range [1.2, 5] per stage
    Ns  = (RPM/60) * V_out_is^0.5 / Dh_is^0.75   Specific speed [-],   valid range [0.045, 0.195]

SP is clamped to [0.02, 1.0] before entering the correlation, but the true value is kept
for reporting. Vr and Ns are flagged if outside their valid ranges.

Used together with the stage-stacking approach from Section 6.1.5 to model multi-stage
turbines where RPM is a free optimization variable.

Author : Amanda (Master Thesis)
"""

import math


# Regression coefficients from Table 6.6 (Astolfi PhD thesis, 2014)
# Single-stage correlation: eta = f(SP, Vr, Ns), adjusted R^2 = 0.995

_COEFFICIENTS_TABLE66 = {
    #  n:  (F_n description,              A_n)
    #  Table 6.6, Astolfi PhD thesis (2014)
    0: ("1", 0.828496),
    1: ("SP", -0.083605),
    2: ("ln(SP)", 0.078745),
    3: ("ln(SP)^2", 0.030635),
    4: ("ln(SP)^3", 0.005738),
    5: ("Vr", 0.005011),
    6: ("ln(Vr)", -0.021296),
    7: ("Ns", 2.648380),
    8: ("Ns^2", -11.918500),
    9: ("Ns^3", 13.241800),
    10: ("Ns^2 * ln(Vr)", 2.158950),
    11: ("Ns * ln(Vr)^2", -0.141356),
    12: ("Ns^3 * ln(Vr)", -7.013500),
    13: ("Ns^3 * ln(SP)", 0.659568),
    14: ("Ns * ln(SP)^3", -0.002947),
}

# Vr validity bounds from Astolfi (2014) Table 6.6
VR_LOWER = 1.2  # regression lower bound
VR_UPPER = 5.0  # regression upper bound / design limit for single stage


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
        Specific speed [-]. Valid regression range [0.045, 0.195].
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
        True if Ns is outside [0.045, 0.195] (extrapolation, no accuracy guarantee).
    """
    # Clamp SP for the polynomial evaluation, but keep the true value for reporting
    SP_eval = SP
    SP_clamped = False
    if SP > 1.0:
        SP_eval = 1.0
        SP_clamped = True
    elif SP < 0.02:
        SP_eval = 0.02
        SP_clamped = True

    # Flag Vr outside the regression calibration range [1.2, 5]
    Vr_out_of_range = Vr < VR_LOWER or Vr > VR_UPPER

    Ns_out_of_range = Ns < 0.045 or Ns > 0.195

    lnSP = math.log(SP_eval)
    lnVr = math.log(Vr)

    # 15 basis functions from Table 6.6
    F = [0.0] * 15
    F[0] = 1.0
    F[1] = SP_eval
    F[2] = lnSP
    F[3] = lnSP**2
    F[4] = lnSP**3
    F[5] = Vr
    F[6] = lnVr
    F[7] = Ns
    F[8] = Ns**2
    F[9] = Ns**3
    F[10] = Ns**2 * lnVr
    F[11] = Ns * lnVr**2
    F[12] = Ns**3 * lnVr
    F[13] = Ns**3 * lnSP
    F[14] = Ns * lnSP**3

    eta_raw = sum(_COEFFICIENTS_TABLE66[i][1] * F[i] for i in range(15))

    # Return raw polynomial value without clamping to avoid biasing the optimizer
    return eta_raw, SP, SP_eval, SP_clamped, Vr_out_of_range, Ns_out_of_range


def compute_specific_speed(RPM, V_out_is, Dh_is):
    """Specific speed Ns = (RPM/60) * V_out_is^0.5 / Dh_is^0.75  (Eq. 9.3)."""
    return (RPM / 60.0) * V_out_is**0.5 / Dh_is**0.75
