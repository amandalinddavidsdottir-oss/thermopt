"""
Post-optimization validation checks for ORC cycles.

Works with ALL cycle configurations:
  - Simple ORC (heater, expander, compressor, cooler)
  - Simple ORC with recuperator (+ recuperator)
  - Dual-pressure ORC (hp/lp evaporator, preheater, hp/lp expander, hp/lp pump)
  - Dual-pressure ORC with recuperator (+ recuperator)

Provides a single function `run_validation_checks(cycle)` that performs:
  1. Heat flow direction check (all heat exchangers)
  2. Constraint activity summary (pinch points, subcooling, superheating, etc.)
  3. Energy balance closure
  4. State point phase verification

Usage:
    from validation_checks import run_validation_checks
    issues = run_validation_checks(cycle)
    # issues = number of warnings/errors found (0 = all clean)
"""

import numpy as np


def run_validation_checks(cycle, verbose=True):
    """
    Run all validation checks on a converged ThermOpt cycle.

    Parameters
    ----------
    cycle : thermopt.ThermodynamicCycleOptimization
        A cycle object that has been optimized.
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
    issues = 0

    print("\n" + "=" * 76)
    print("  POST-OPTIMIZATION VALIDATION CHECKS")
    print("=" * 76)

    issues += _check_heat_flow_direction(energy, verbose)
    issues += _check_constraint_summary(components, energy, verbose)
    issues += _check_energy_balance(energy, verbose)
    issues += _check_state_phases(components, verbose)

    # ── Final verdict ──
    print()
    if issues == 0:
        print("  ✓ ALL CHECKS PASSED — no issues detected")
    else:
        print(f"  ⚠ {issues} WARNING(S) DETECTED — review output above")
    print("=" * 76 + "\n")

    return issues


# ══════════════════════════════════════════════════════════════════════════════
#  CHECK 1 — HEAT FLOW DIRECTION
# ══════════════════════════════════════════════════════════════════════════════

def _check_heat_flow_direction(energy, verbose):
    """
    Verify all heat exchangers transfer heat in the correct direction.
    Auto-detects heat exchangers by scanning for keys ending in '_heat_flow'.
    """
    issues = 0

    print()
    print("  ── Check 1: Heat Flow Direction ──")

    # Find all heat flow keys in energy_analysis, skip "_max" variants
    hx_keys = {k: k.replace("_heat_flow", "")
               for k in energy
               if k.endswith("_heat_flow") and "_max" not in k
               and k != "total_heat_input"}

    # Determine a sensible threshold: 0.01% of total heat input
    Q_total = abs(energy.get("total_heat_input",
                  energy.get("heater_heat_flow", 1e6)))
    noise_threshold = max(Q_total * 1e-4, 1000)  # at least 1 kW

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

# Display names for heat exchangers
_HX_DISPLAY = {
    "heater": "Evaporator",
    "hp_evaporator": "HP evaporator",
    "lp_evaporator": "LP evaporator",
    "preheater": "Preheater",
    "recuperator": "Recuperator",
    "cooler": "Condenser",
}

def _check_constraint_summary(components, energy, verbose):
    """
    Summarise key constraints. Auto-detects which components exist.
    For pinch-point arrays, reports only the minimum ΔT.
    """
    issues = 0

    print()
    print("  ── Check 2: Constraint Activity Summary ──")
    print(f"    {'Constraint':<36s} {'Value':>10s} {'Limit':>10s} {'Slack':>10s}  {'Status'}")
    print("    " + "─" * 80)

    # --- Pinch points (auto-detect heat exchangers with temperature_difference) ---
    pinch_limit = 5.0  # K

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
        print(f"    {display + ' pinch':<36s} {min_dT:10.2f} {'> 5.0 K':>10s} {slack:10.2f}  {status}")

    # --- Subcooling at pump inlet ---
    # Simple ORC: "compressor" | Dual-pressure: "lp_pump"
    pump_name = "lp_pump" if "lp_pump" in components else "compressor"
    if pump_name in components:
        pump_in = components[pump_name].get("state_in", None)
        if pump_in is not None and hasattr(pump_in, "subcooling"):
            sc = float(pump_in.subcooling)
            sc_limit = 1.0
            slack = sc - sc_limit
            if sc < sc_limit - 0.1:
                status = "⚠ VIOLATED"
                issues += 1
            elif abs(slack) < 0.2:
                status = "BINDING"
            else:
                status = "slack"
            label = "LP pump" if pump_name == "lp_pump" else "Pump"
            print(f"    {label + ' subcooling':<36s} {sc:10.2f} {'> 1.0 K':>10s} {slack:10.2f}  {status}")

    # --- Superheating at expander inlet ---
    # Simple ORC: "expander" | Dual-pressure: "hp_expander"
    exp_name = "hp_expander" if "hp_expander" in components else "expander"
    if exp_name in components:
        exp_in = components[exp_name].get("state_in", None)
        if exp_in is not None and hasattr(exp_in, "superheating"):
            sh = float(exp_in.superheating)
            sh_limit = 5.0
            slack = sh - sh_limit
            if sh < sh_limit - 0.1:
                status = "⚠ VIOLATED"
                issues += 1
            elif abs(slack) < 0.2:
                status = "BINDING"
            else:
                status = "slack"
            label = "HP expander" if exp_name == "hp_expander" else "Expander"
            print(f"    {label + ' superheating':<36s} {sh:10.2f} {'> 5.0 K':>10s} {slack:10.2f}  {status}")

    # --- Brine reinjection temperature (dual-pressure only) ---
    T_brine_out = energy.get("brine_exit_temperature", None)
    if T_brine_out is not None:
        T_out_C = T_brine_out - 273.15
        print(f"    {'Brine exit temperature':<36s} {T_out_C:10.1f} {'°C':>10s} {'':>10s}  (check vs limit)")

    # --- Preheater heat flow Q > 0 (dual-pressure with recuperator) ---
    Q_pre = energy.get("preheater_heat_flow", None)
    if Q_pre is not None and "recuperator" in components:
        Q_kW = Q_pre / 1e3
        # Use same noise threshold as Check 1
        Q_total = abs(energy.get("total_heat_input",
                      energy.get("heater_heat_flow", 1e6)))
        noise_kW = max(Q_total * 1e-4, 1000) / 1e3  # in kW
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
    """
    Check first-law energy balance: Q_in = W_net + Q_out.
    Auto-detects which keys are available.
    """
    issues = 0

    print()
    print("  ── Check 3: Energy Balance (1st Law) ──")

    # Q_in: dual-pressure has "total_heat_input", simple has "heater_heat_flow"
    Q_in = energy.get("total_heat_input", energy.get("heater_heat_flow", 0))
    Q_out = abs(energy.get("cooler_heat_flow", 0))

    # W_net: prefer net_system_power (includes aux pumps)
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
    """
    Verify that key state points are in the expected thermodynamic phase.
    Auto-detects which components exist.

    Expected phases:
      - Pump inlets: subcooled liquid
      - Expander inlets: superheated vapor
      - Expander outlets: superheated vapor (for dry fluids like toluene)
    """
    issues = 0

    print()
    print("  ── Check 4: State Point Phase Verification ──")

    # Build checks from whichever components exist
    # Format: (label, state_object, expected_phase)
    checks = []

    # --- Pumps (liquid expected at inlet) ---
    if "compressor" in components:
        checks.append(("Pump inlet",
                        components["compressor"]["state_in"], "liquid"))
    if "lp_pump" in components:
        checks.append(("LP pump inlet",
                        components["lp_pump"]["state_in"], "liquid"))
    if "hp_pump" in components:
        checks.append(("HP pump inlet",
                        components["hp_pump"]["state_in"], "liquid"))

    # --- Expanders (vapor expected at inlet and outlet) ---
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
        # Try to get phase from CoolProp state object
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

        # Normalise CoolProp phase strings
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
