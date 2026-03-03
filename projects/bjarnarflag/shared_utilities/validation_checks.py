"""
Post-optimization validation checks for ORC cycles.

Works with ALL cycle configurations:
  - Simple ORC (heater, expander, compressor, cooler)
  - Simple ORC with recuperator (+ recuperator)
  - Dual-pressure ORC (hp/lp evaporator, preheater, hp/lp expander, hp/lp pump)
  - Dual-pressure ORC with recuperator (+ recuperator)

Provides:
  run_validation_checks(cycle, config_file)
    1. Heat flow direction check
    2. Constraint activity summary (limits read from YAML)
    3. Energy balance closure
    4. State point phase verification
    5. Turbine summary & verification (when macchi-astolfi is used)

Usage:
    from validation_checks import run_validation_checks
    issues = run_validation_checks(cycle, config_file="case.yaml")
"""

import math
import numpy as np
import yaml


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def run_validation_checks(cycle, config_file=None, verbose=True):
    """
    Run all validation checks on a converged ThermOpt cycle.

    Parameters
    ----------
    cycle : thermopt.ThermodynamicCycleOptimization
        A cycle object that has been optimized.
    config_file : str or Path, optional
        Path to YAML config file. If provided, constraint limits
        (superheating, subcooling, pinch) are read from it automatically.
    verbose : bool
        If True, print full details. If False, only print warnings.

    Returns
    -------
    int
        Number of warnings or errors found. 0 means all checks passed.
    """
    data = cycle.problem.cycle_data
    components = data["components"]
    energy = data["energy_analysis"]

    # Read constraint limits from YAML (or use defaults)
    limits = _read_limits_from_yaml(config_file)

    issues = 0

    print("\n" + "=" * 76)
    print("  POST-OPTIMIZATION VALIDATION CHECKS")
    print("=" * 76)

    issues += _check_heat_flow_direction(energy, verbose)
    issues += _check_constraint_summary(components, energy, limits, verbose)
    issues += _check_energy_balance(energy, verbose)
    issues += _check_state_phases(components, verbose)

    # ── Final verdict ──
    print()
    if issues == 0:
        print("  ✓ ALL CHECKS PASSED — no issues detected")
    else:
        print(f"  ⚠ {issues} WARNING(S) DETECTED — review output above")
    print("=" * 76 + "\n")

    # ── Turbine summary ──
    _print_turbine_summaries(components)

    return issues


# ══════════════════════════════════════════════════════════════════════════════
#  READ CONSTRAINT LIMITS FROM YAML
# ══════════════════════════════════════════════════════════════════════════════

def _read_limits_from_yaml(config_file):
    """
    Read constraint limits directly from the YAML config file.
    Falls back to defaults if file is not provided or unreadable.
    """
    limits = {
        "pinch": 5.0,
        "subcooling": 1.0,
        "superheating": 5.0,
    }

    if config_file is None:
        return limits

    try:
        with open(config_file, "r") as f:
            cfg = yaml.safe_load(f)
        constraints = cfg.get("problem_formulation", {}).get("constraints", [])
        for c in constraints:
            var = str(c.get("variable", ""))
            val = c.get("value", None)
            if val is None:
                continue
            if "superheating" in var:
                limits["superheating"] = float(val)
            elif "subcooling" in var:
                limits["subcooling"] = float(val)
            elif "temperature_difference" in var:
                # Only update if it's a general pinch constraint
                limits["pinch"] = float(val)
    except (OSError, yaml.YAMLError, TypeError):
        pass  # Use defaults

    return limits


# ══════════════════════════════════════════════════════════════════════════════
#  CHECK 1 — HEAT FLOW DIRECTION
# ══════════════════════════════════════════════════════════════════════════════

def _check_heat_flow_direction(energy, verbose):
    """
    Verify all heat exchangers transfer heat in the correct direction.
    """
    issues = 0

    print()
    print("  ── Check 1: Heat Flow Direction ──")

    hx_keys = {k: k.replace("_heat_flow", "")
               for k in energy
               if k.endswith("_heat_flow") and "_max" not in k
               and k != "total_heat_input"}

    Q_total = abs(energy.get("total_heat_input",
                  energy.get("heater_heat_flow", 1e6)))
    noise_threshold = max(Q_total * 1e-4, 1000)

    for key, name in sorted(hx_keys.items()):
        Q = energy[key]
        Q_kW = Q / 1e3

        if Q < -noise_threshold:
            print(f"    ⚠ {name:20s}: Q = {Q_kW:12.1f} kW  ← NEGATIVE (heat flows wrong way!)")
            issues += 1
        elif abs(Q) < noise_threshold:
            if verbose:
                print(f"    ~ {name:20s}: Q = {Q_kW:12.1f} kW  (≈ 0, effectively inactive)")
        elif verbose:
            print(f"    ✓ {name:20s}: Q = {Q_kW:12.1f} kW")

    return issues


# ══════════════════════════════════════════════════════════════════════════════
#  CHECK 2 — CONSTRAINT ACTIVITY SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

_HX_DISPLAY = {
    "heater": "Evaporator",
    "hp_evaporator": "HP evaporator",
    "lp_evaporator": "LP evaporator",
    "preheater": "Preheater",
    "recuperator": "Recuperator",
    "cooler": "Condenser",
}

def _check_constraint_summary(components, energy, limits, verbose):
    """
    Summarise key constraints. Limits are read from the YAML config.
    """
    issues = 0
    pinch_limit = limits["pinch"]
    subcooling_limit = limits["subcooling"]
    superheating_limit = limits["superheating"]

    print()
    print("  ── Check 2: Constraint Activity Summary ──")
    print(f"    {'Constraint':<36s} {'Value':>10s} {'Limit':>10s} {'Slack':>10s}  {'Status'}")
    print("    " + "─" * 80)

    # --- Pinch points ---
    for hx_name in ["heater", "hp_evaporator", "lp_evaporator",
                     "preheater", "recuperator", "cooler"]:
        if hx_name not in components:
            continue
        comp = components[hx_name]
        if "temperature_difference" not in comp:
            continue

        dT_array = np.array(comp["temperature_difference"])
        min_dT = float(np.min(dT_array))
        slack = min_dT - pinch_limit

        if min_dT < pinch_limit - 0.1:
            status = "⚠ VIOLATED"
            issues += 1
        elif abs(slack) < 0.1:
            status = "BINDING"
        else:
            status = "slack"

        display = _HX_DISPLAY.get(hx_name, hx_name)
        limit_str = f"> {pinch_limit:.1f} K"
        print(f"    {display + ' pinch':<36s} {min_dT:10.2f} {limit_str:>10s} {slack:10.2f}  {status}")

    # --- Subcooling at pump inlet ---
    pump_name = "lp_pump" if "lp_pump" in components else "compressor"
    if pump_name in components:
        pump_in = components[pump_name].get("state_in", None)
        if pump_in is not None and hasattr(pump_in, "subcooling"):
            sc = float(pump_in.subcooling)
            slack = sc - subcooling_limit
            if sc < subcooling_limit - 0.1:
                status = "⚠ VIOLATED"
                issues += 1
            elif abs(slack) < 0.2:
                status = "BINDING"
            else:
                status = "slack"
            label = "LP pump" if pump_name == "lp_pump" else "Pump"
            limit_str = f"> {subcooling_limit:.1f} K"
            print(f"    {label + ' subcooling':<36s} {sc:10.2f} {limit_str:>10s} {slack:10.2f}  {status}")

    # --- Superheating at expander inlet ---
    exp_name = "hp_expander" if "hp_expander" in components else "expander"
    if exp_name in components:
        exp_in = components[exp_name].get("state_in", None)
        if exp_in is not None and hasattr(exp_in, "superheating"):
            sh = float(exp_in.superheating)
            slack = sh - superheating_limit
            if sh < superheating_limit - 0.1:
                status = "⚠ VIOLATED"
                issues += 1
            elif abs(slack) < 0.2:
                status = "BINDING"
            else:
                status = "slack"
            label = "HP expander" if exp_name == "hp_expander" else "Expander"
            limit_str = f"> {superheating_limit:.1f} K"
            print(f"    {label + ' superheating':<36s} {sh:10.2f} {limit_str:>10s} {slack:10.2f}  {status}")

    # --- Brine reinjection temperature (dual-pressure only) ---
    T_brine_out = energy.get("brine_exit_temperature", None)
    if T_brine_out is not None:
        T_out_C = T_brine_out - 273.15
        print(f"    {'Brine exit temperature':<36s} {T_out_C:10.1f} {'°C':>10s} {'':>10s}  (check vs limit)")

    # --- Preheater heat flow Q > 0 (dual-pressure with recuperator) ---
    Q_pre = energy.get("preheater_heat_flow", None)
    if Q_pre is not None and "recuperator" in components:
        Q_kW = Q_pre / 1e3
        Q_total = abs(energy.get("total_heat_input",
                      energy.get("heater_heat_flow", 1e6)))
        noise_kW = max(Q_total * 1e-4, 1000) / 1e3
        if Q_kW < -noise_kW:
            status = "⚠ VIOLATED"
            issues += 1
        elif abs(Q_kW) < noise_kW:
            status = "BINDING (≈ 0)"
        else:
            status = "slack"
        print(f"    {'Preheater heat flow':<36s} {Q_kW:10.1f} {'> 0 kW':>10s} {Q_kW:10.1f}  {status}")

    return issues


# ══════════════════════════════════════════════════════════════════════════════
#  CHECK 3 — ENERGY BALANCE CLOSURE
# ══════════════════════════════════════════════════════════════════════════════

def _check_energy_balance(energy, verbose):
    """Check first-law energy balance: Q_in = W_net + Q_out."""
    issues = 0

    print()
    print("  ── Check 3: Energy Balance (1st Law) ──")

    Q_in = energy.get("total_heat_input", energy.get("heater_heat_flow", 0))
    Q_out = abs(energy.get("cooler_heat_flow", 0))
    W_net = energy.get("net_system_power", energy.get("net_cycle_power", 0))

    residual = Q_in - (W_net + Q_out)
    pct = abs(residual) / abs(Q_in) * 100 if Q_in != 0 else 0

    if verbose:
        print(f"    Q_in  (total heat input)     : {Q_in / 1e3:12.1f} kW")
        print(f"    W_net (net system power)      : {W_net / 1e3:12.1f} kW")
        print(f"    Q_out (cooler rejection)      : {Q_out / 1e3:12.1f} kW")
        print(f"    Residual (Q_in - W - Q_out)   : {residual / 1e3:12.3f} kW  ({pct:.4f}%)")

    if pct > 0.1:
        print(f"    ⚠ Energy balance residual > 0.1% — check for missing components")
        issues += 1
    else:
        print(f"    ✓ Energy balance closed ({pct:.4f}%)")

    return issues


# ══════════════════════════════════════════════════════════════════════════════
#  CHECK 4 — STATE POINT PHASE VERIFICATION
# ══════════════════════════════════════════════════════════════════════════════

def _check_state_phases(components, verbose):
    """Verify key state points are in the expected thermodynamic phase."""
    issues = 0

    print()
    print("  ── Check 4: State Point Phase Verification ──")

    checks = []

    # Pumps (liquid expected at inlet)
    if "compressor" in components:
        checks.append(("Pump inlet",
                        components["compressor"]["state_in"], "liquid"))
    if "lp_pump" in components:
        checks.append(("LP pump inlet",
                        components["lp_pump"]["state_in"], "liquid"))
    if "hp_pump" in components:
        checks.append(("HP pump inlet",
                        components["hp_pump"]["state_in"], "liquid"))

    # Expanders (vapor expected at inlet and outlet)
    if "expander" in components:
        checks.append(("Expander inlet",
                        components["expander"]["state_in"], "vapor"))
        checks.append(("Expander outlet",
                        components["expander"]["state_out"], "vapor"))
    if "hp_expander" in components:
        checks.append(("HP expander inlet",
                        components["hp_expander"]["state_in"], "vapor"))
        checks.append(("HP expander outlet",
                        components["hp_expander"]["state_out"], "vapor"))
    if "lp_expander" in components:
        checks.append(("LP expander inlet",
                        components["lp_expander"]["state_in"], "vapor"))
        checks.append(("LP expander outlet",
                        components["lp_expander"]["state_out"], "vapor"))

    for label, state, expected in checks:
        try:
            phase = state.phase
        except AttributeError:
            try:
                Q = state.Q
                if Q <= 0:
                    phase = "liquid"
                elif Q >= 1:
                    phase = "vapor"
                else:
                    phase = "two-phase"
            except (AttributeError, Exception):
                if verbose:
                    print(f"    ? {label:<28s}: phase info not available")
                continue

        phase_lower = str(phase).lower()
        is_liquid = any(x in phase_lower for x in
                        ["liquid", "subcooled", "compressed"])
        is_vapor = any(x in phase_lower for x in
                       ["gas", "vapor", "superheated", "supercritical_gas"])

        if expected == "liquid" and not is_liquid:
            print(f"    ⚠ {label:<28s}: phase = {phase}  (expected liquid!)")
            issues += 1
        elif expected == "vapor" and not is_vapor:
            print(f"    ⚠ {label:<28s}: phase = {phase}  (expected vapor!)")
            issues += 1
        elif verbose:
            T_C = state.T - 273.15 if hasattr(state, 'T') else float('nan')
            print(f"    ✓ {label:<28s}: {str(phase):<20s} (T = {T_C:.1f} °C)")

    return issues


# ══════════════════════════════════════════════════════════════════════════════
#  CHECK 5 — TURBINE SUMMARY & VERIFICATION
# ══════════════════════════════════════════════════════════════════════════════

# Coefficients from Table 9.1 (Macchi & Astolfi, 2017)
# Kept here so verification is independent of thermopt
_MA_COEFFICIENTS = {
    0:  (0.90831500,  0.923406,    0.932274),
    1:  (-0.05248690, -0.0221021,  -0.01243),
    2:  (-0.04799080, -0.0233814,  -0.018),
    3:  (-0.01710380, -0.00844961, -0.00716),
    4:  (-0.00244002, -0.0012978,  -0.00118),
    5:  (0.0,         -0.00069293, -0.00044),
    6:  (0.04961780,  0.0146911,   0.0),
    7:  (-0.04894860, -0.0102795,  0.0),
    8:  (0.01171650,  0.0,         -0.0016),
    9:  (-0.00100473, 0.000317241, 0.000298),
    10: (0.05645970,  0.0163959,   0.005959),
    11: (-0.01859440, -0.00515265, -0.00163),
    12: (0.01288860,  0.00358361,  0.001946),
    13: (0.00178187,  0.000554726, 0.000163),
    14: (-0.00021196, 0.0,         0.0),
    15: (0.00078667,  0.000293607, 0.000211),
}


def _verify_eta(SP, Vr, n_stages):
    """Recompute η from SP and Vr with same clamping as the optimizer."""
    SP_eval = max(0.02, min(1.0, SP))
    col = n_stages - 1  # 0-indexed: 1-stage=0, 2-stage=1, 3-stage=2

    lnSP = math.log(SP_eval)
    lnVr = math.log(Vr)

    F = [
        1.0, lnSP, lnSP**2, lnSP**3, lnSP**4,
        Vr, lnVr, lnVr**2, lnVr**3, lnVr**4,
        lnVr * lnSP, lnVr**2 * lnSP, lnVr * lnSP**2,
        lnVr**3 * lnSP, lnVr**3 * lnSP**2, lnVr**2 * lnSP**3,
    ]

    eta = sum(_MA_COEFFICIENTS[i][col] * F[i] for i in range(16))
    return max(0.0, min(1.0, eta))


def _print_expander_summary(exp, label="Expander"):
    """Print turbine summary and verification for a single expander."""
    eff_type = exp.get("efficiency_type", "isentropic")
    data_out = exp.get("data_out", {})
    state_in = exp["state_in"]
    state_out = exp["state_out"]
    m_dot = exp.get("mass_flow", float("nan"))

    print("\n" + "=" * 76)
    print(f"  TURBINE SUMMARY — {label}")
    print("=" * 76)
    print(f"  Fluid              : {exp.get('fluid_name', 'unknown')}")
    print(f"  Mass flow rate     : {m_dot:.3f} kg/s")
    print(f"  Inlet  (p, T)      : {state_in.p/1e5:.2f} bar,  "
          f"{state_in.T - 273.15:.1f} °C")
    print(f"  Outlet (p, T)      : {state_out.p/1e5:.2f} bar,  "
          f"{state_out.T - 273.15:.1f} °C")
    print(f"  Pressure ratio     : {exp['pressure_ratio']:.2f}")
    print(f"  Dh_is              : {exp['isentropic_work']/1e3:.2f} kJ/kg")
    print(f"  Specific work      : {exp['specific_work']/1e3:.2f} kJ/kg")

    W_gross = m_dot * exp["specific_work"]
    W_is = m_dot * exp["isentropic_work"]
    print(f"  W_is               : {W_is/1e3:.1f} kW")
    print(f"  W_actual           : {W_gross/1e3:.1f} kW")
    print("-" * 76)

    if eff_type == "macchi-astolfi":
        n_stages = data_out.get("n_stages", "?")
        RPM = data_out.get("RPM", None)
        SP = data_out.get("size_parameter", None)
        SP_eval = data_out.get("size_parameter_eval", None)
        SP_clamped = data_out.get("size_parameter_clamped", False)
        Vr = data_out.get("volume_ratio", None)
        Ns = data_out.get("specific_speed", None)
        eta_opt = exp.get("efficiency", None)

        print(f"  Efficiency mode    : Macchi-Astolfi (computed during optimization)")
        print(f"  Stages             : {n_stages}")
        if RPM is not None:
            print(f"  RPM                : {int(RPM)}")
        print(f"  η_is (optimizer)   : {eta_opt:.4f}  ({eta_opt*100:.2f}%)")
        print("-" * 76)

        if SP is not None:
            print(f"  SP (actual)        : {SP:.4f} m")
            if SP_clamped:
                print(f"  SP (used in corr.) : {SP_eval:.4f} m  "
                      f"(clamped — efficiency plateaus)")
            else:
                print(f"  SP (used in corr.) : {SP:.4f} m")
            if SP < 0.02 or SP > 1.0:
                print(f"  ⚠  SP = {SP:.4f} m is OUTSIDE the correlation "
                      f"range [0.02, 1.0] m")

        if Vr is not None:
            print(f"  Vr                 : {Vr:.2f}")
            if Vr < 1.2 or Vr > 200:
                print(f"  ⚠  Vr = {Vr:.2f} is OUTSIDE the correlation "
                      f"range [1.2, 200]")

        if Ns is not None:
            print(f"  Ns                 : {Ns:.4f}")
            if Ns < 0.05 or Ns > 0.20:
                print(f"  ⚠  Ns = {Ns:.4f} is outside the typical optimal "
                      f"range [0.05, 0.20]")

        # Cross-check
        if SP is not None and Vr is not None and isinstance(n_stages, int):
            eta_verify = _verify_eta(SP, Vr, n_stages)
            diff = abs(eta_verify - eta_opt) * 100
            print("-" * 76)
            print(f"  ── Verification ──")
            print(f"  η_is (recomputed)  : {eta_verify:.4f}  ({eta_verify*100:.2f}%)")
            if diff < 0.01:
                print(f"  ✓ Matches optimizer (Δη = {diff:.4f} pp)")
            else:
                print(f"  ⚠ Differs from optimizer by {diff:.2f} pp")

    else:
        eta = exp.get("efficiency", None)
        print(f"  Efficiency mode    : Fixed isentropic")
        print(f"  η_is               : {eta}")

    print("=" * 76 + "\n")


def _print_turbine_summaries(components):
    """Print turbine summary for all expanders. Auto-detects topology."""
    if "hp_expander" in components:
        _print_expander_summary(components["hp_expander"], label="HP Expander")
        _print_expander_summary(components["lp_expander"], label="LP Expander")
    elif "expander" in components:
        _print_expander_summary(components["expander"], label="Expander")
