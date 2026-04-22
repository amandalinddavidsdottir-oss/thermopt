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
    5. Turbine summary & verification (when astolfi-stacking is used)

Usage:
    from validation_checks import run_validation_checks
    issues = run_validation_checks(cycle, config_file="case.yaml")
"""

import numpy as np
import yaml


def _eval_yaml_expr(value):
    """Safely evaluate simple arithmetic expressions found in YAML config."""
    if isinstance(value, str):
        try:
            return float(eval(value, {"__builtins__": {}}, {"np": np}))
        except Exception:
            return value
    return float(value)


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
    issues += _check_energy_balance(components, energy, verbose)
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

    hx_keys = {
        k: k.replace("_heat_flow", "")
        for k in energy
        if k.endswith("_heat_flow") and "_max" not in k and k != "total_heat_input"
    }

    Q_total = abs(energy.get("total_heat_input", energy.get("heater_heat_flow", 1e6)))
    noise_threshold = max(Q_total * 1e-4, 1000)

    for key, name in sorted(hx_keys.items()):
        Q = energy[key]
        Q_kW = Q / 1e3

        if Q < -noise_threshold:
            print(
                f"    ⚠ {name:20s}: Q = {Q_kW:12.1f} kW  ← NEGATIVE (heat flows wrong way!)"
            )
            issues += 1
        elif abs(Q) < noise_threshold:
            if verbose:
                print(
                    f"    ~ {name:20s}: Q = {Q_kW:12.1f} kW  (≈ 0, effectively inactive)"
                )
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
    print(
        f"    {'Constraint':<36s} {'Value':>10s} {'Limit':>10s} {'Slack':>10s}  {'Status'}"
    )
    print("    " + "─" * 80)

    # --- Pinch points ---
    for hx_name in [
        "heater",
        "hp_evaporator",
        "lp_evaporator",
        "preheater",
        "recuperator",
        "cooler",
    ]:
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
        print(
            f"    {display + ' pinch':<36s} {min_dT:10.2f} {limit_str:>10s} {slack:10.2f}  {status}"
        )

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
            print(
                f"    {label + ' subcooling':<36s} {sc:10.2f} {limit_str:>10s} {slack:10.2f}  {status}"
            )

    # --- Superheating at expander inlet ---
    exp_name = "hp_expander" if "hp_expander" in components else "expander"
    if exp_name in components:
        exp_in = components[exp_name].get("state_in", None)
        if exp_in is not None and hasattr(exp_in, "superheating"):
            # Check if expander inlet is supercritical
            _is_supercritical = False
            if hasattr(exp_in, "p") and hasattr(exp_in, "fluid_name"):
                try:
                    import CoolProp.CoolProp as CP

                    p_crit = CP.PropsSI("pcrit", exp_in.fluid_name)
                    if exp_in.p > p_crit:
                        _is_supercritical = True
                except Exception:
                    pass

            if _is_supercritical:
                print(
                    f"    {'Expander superheating':<36s} {'N/A':>10s} {'':>10s} {'':>10s}  "
                    f"supercritical (no saturation line)"
                )
            else:
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
                print(
                    f"    {label + ' superheating':<36s} {sh:10.2f} {limit_str:>10s} {slack:10.2f}  {status}"
                )

    # --- Brine reinjection temperature (dual-pressure only) ---
    T_brine_out = energy.get("brine_exit_temperature", None)
    if T_brine_out is not None:
        T_out_C = T_brine_out - 273.15
        print(
            f"    {'Brine exit temperature':<36s} {T_out_C:10.1f} {'°C':>10s} {'':>10s}  (check vs limit)"
        )

    # --- Preheater heat flow Q > 0 (dual-pressure with recuperator) ---
    Q_pre = energy.get("preheater_heat_flow", None)
    if Q_pre is not None and "recuperator" in components:
        Q_kW = Q_pre / 1e3
        Q_total = abs(
            energy.get("total_heat_input", energy.get("heater_heat_flow", 1e6))
        )
        noise_kW = max(Q_total * 1e-4, 1000) / 1e3
        if Q_kW < -noise_kW:
            status = "⚠ VIOLATED"
            issues += 1
        elif abs(Q_kW) < noise_kW:
            status = "BINDING (≈ 0)"
        else:
            status = "slack"
        print(
            f"    {'Preheater heat flow':<36s} {Q_kW:10.1f} {'> 0 kW':>10s} {Q_kW:10.1f}  {status}"
        )

    return issues


# ══════════════════════════════════════════════════════════════════════════════
#  CHECK 3 — ENERGY BALANCE CLOSURE
# ══════════════════════════════════════════════════════════════════════════════


def _check_energy_balance(components, energy, verbose):
    """Check first-law energy balance: Q_in = W_net_system + W_aux + Q_out.

    W_net_system already has auxiliary pump power deducted, so W_aux must be
    added back explicitly to close the working-fluid-loop energy balance.
    """
    issues = 0

    print()
    print("  ── Check 3: Energy Balance (1st Law) ──")

    Q_in = energy.get("total_heat_input", energy.get("heater_heat_flow", 0))
    Q_out = abs(energy.get("cooler_heat_flow", 0))
    W_net = energy.get("net_system_power", energy.get("net_cycle_power", 0))
    hs_pump = components.get(
        "heat_source_pump", components.get("heat_source_pump_hp", None)
    )
    W_hs = hs_pump["energy_analysis"]["power"] if hs_pump else 0
    if "heat_source_pump_lp" in components:
        W_hs += components["heat_source_pump_lp"]["energy_analysis"]["power"]
    W_aux = W_hs + components["heat_sink_pump"]["energy_analysis"]["power"]

    residual = Q_in - (W_net + W_aux + Q_out)
    pct = abs(residual) / abs(Q_in) * 100 if Q_in != 0 else 0

    if verbose:
        print(f"    Q_in  (total heat input)     : {Q_in / 1e3:12.1f} kW")
        print(f"    W_net (net system power)      : {W_net / 1e3:12.1f} kW")
        print(f"    W_aux (auxiliary pumps)       : {W_aux / 1e3:12.3f} kW")
        print(f"    Q_out (cooler rejection)      : {Q_out / 1e3:12.1f} kW")
        print(
            f"    Residual (Q_in - W_net - W_aux - Q_out) : {residual / 1e3:12.3f} kW  ({pct:.4f}%)"
        )

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
        checks.append(("Pump inlet", components["compressor"]["state_in"], "liquid"))
    if "lp_pump" in components:
        checks.append(("LP pump inlet", components["lp_pump"]["state_in"], "liquid"))
    if "hp_pump" in components:
        checks.append(("HP pump inlet", components["hp_pump"]["state_in"], "liquid"))

    # Expanders (vapor expected at inlet and outlet)
    if "expander" in components:
        checks.append(("Expander inlet", components["expander"]["state_in"], "vapor"))
        checks.append(("Expander outlet", components["expander"]["state_out"], "vapor"))
    if "hp_expander" in components:
        checks.append(
            ("HP expander inlet", components["hp_expander"]["state_in"], "vapor")
        )
        checks.append(
            ("HP expander outlet", components["hp_expander"]["state_out"], "vapor")
        )
    if "lp_expander" in components:
        checks.append(
            ("LP expander inlet", components["lp_expander"]["state_in"], "vapor")
        )
        checks.append(
            ("LP expander outlet", components["lp_expander"]["state_out"], "vapor")
        )

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
        is_liquid = any(x in phase_lower for x in ["liquid", "subcooled", "compressed"])
        is_vapor = any(
            x in phase_lower
            for x in ["gas", "vapor", "superheated", "supercritical_gas"]
        )

        # Detect supercritical and near-critical states where CoolProp's
        # phase label is unreliable:
        #   1. p > p_crit: truly supercritical, no phase boundary exists
        #   2. p < p_crit but T > T_sat(p) and T > T_crit: near-critical region
        #      where CoolProp may label superheated vapor as "liquid" based on
        #      density proximity to the critical point (e.g. HP expander outlet
        #      in transcritical cycles)
        is_supercritical = False
        is_near_critical = False
        if hasattr(state, "p") and hasattr(state, "T") and hasattr(state, "fluid_name"):
            try:
                import CoolProp.CoolProp as CP

                p_crit = CP.PropsSI("pcrit", state.fluid_name)
                T_crit = CP.PropsSI("Tcrit", state.fluid_name)
                if state.p > p_crit:
                    is_supercritical = True
                elif state.p > 0.3 * p_crit and state.T > T_crit:
                    # Below p_crit but above T_crit: near-critical region
                    # Verify it's actually superheated by checking T > T_sat
                    try:
                        T_sat = CP.PropsSI(
                            "T", "P", state.p, "Q", 1.0, state.fluid_name
                        )
                        if state.T > T_sat:
                            is_near_critical = True
                    except Exception:
                        # T_sat lookup can fail very close to critical point
                        is_near_critical = True
            except Exception:
                pass

        if is_supercritical:
            T_C = state.T - 273.15 if hasattr(state, "T") else float("nan")
            p_bar = state.p / 1e5 if hasattr(state, "p") else float("nan")
            if verbose:
                print(
                    f"    ✓ {label:<28s}: supercritical        "
                    f"(T = {T_C:.1f} °C, p = {p_bar:.1f} bar > p_crit)"
                )
        elif is_near_critical:
            T_C = state.T - 273.15
            p_bar = state.p / 1e5
            if verbose:
                print(
                    f"    ✓ {label:<28s}: near-critical vapor  "
                    f"(T = {T_C:.1f} °C > T_crit, p = {p_bar:.1f} bar)"
                )
        elif expected == "liquid" and not is_liquid:
            print(f"    ⚠ {label:<28s}: phase = {phase}  (expected liquid!)")
            issues += 1
        elif expected == "vapor" and not is_vapor:
            print(f"    ⚠ {label:<28s}: phase = {phase}  (expected vapor!)")
            issues += 1
        elif verbose:
            T_C = state.T - 273.15 if hasattr(state, "T") else float("nan")
            print(f"    ✓ {label:<28s}: {str(phase):<20s} (T = {T_C:.1f} °C)")

    return issues


# ══════════════════════════════════════════════════════════════════════════════
#  CHECK 5 — TURBINE SUMMARY & VERIFICATION
# ══════════════════════════════════════════════════════════════════════════════


def _check_design_variable_bounds(cycle, config_file, verbose):
    """Check that all converged design variables are within their YAML bounds."""
    issues = 0

    print()
    print("  ── Check 5: Design Variable Bounds ──")
    print(f"    {'Variable':<40s} {'Value':>14s} {'Min':>12s} {'Max':>12s}  {'Status'}")
    print("    " + "─" * 84)

    try:
        with open(config_file, "r") as f:
            cfg = yaml.safe_load(f)
    except (OSError, yaml.YAMLError):
        print("    ⚠ Could not read config file — bounds check skipped")
        return 0

    dvs = cfg.get("problem_formulation", {}).get("design_variables", {})
    x_dict = cycle.problem.x0_dict

    for name, val in x_dict.items():
        if name not in dvs:
            continue
        entry = dvs[name]
        lo = entry.get("min", None)
        hi = entry.get("max", None)
        if lo is None and hi is None:
            continue

        lo_val = _eval_yaml_expr(lo) if lo is not None else None
        hi_val = _eval_yaml_expr(hi) if hi is not None else None
        tol = 1e-6

        lo_str = f"{lo_val:.4g}" if lo_val is not None else "—"
        hi_str = f"{hi_val:.4g}" if hi_val is not None else "—"

        if lo_val is not None and val < lo_val - tol:
            status = "⚠ BELOW MIN"
            issues += 1
        elif hi_val is not None and val > hi_val + tol:
            status = "⚠ ABOVE MAX"
            issues += 1
        elif lo_val is not None and abs(val - lo_val) < tol * max(1, abs(lo_val)):
            status = "AT LOWER BOUND"
        elif hi_val is not None and abs(val - hi_val) < tol * max(1, abs(hi_val)):
            status = "AT UPPER BOUND"
        else:
            status = "within bounds"

        if verbose or "BOUND" in status or "⚠" in status:
            print(f"    {name:<40s} {val:>14.4g} {lo_str:>12s} {hi_str:>12s}  {status}")

    return issues


def _print_expander_summary(exp, label="Expander"):
    """Print turbine summary and verification for a single expander."""
    eff_type = exp.get("efficiency_type", "isentropic")
    data_out = exp.get("data_out", {})
    state_in = exp["state_in"]
    state_out = exp["state_out"]
    m_dot = exp.get("mass_flow", float("nan"))
    n_parallel = exp.get("n_turbines_parallel", 1)

    print("\n" + "=" * 76)
    print(f"  TURBINE SUMMARY — {label}")
    print("=" * 76)
    print(f"  Fluid              : {exp.get('fluid_name', 'unknown')}")
    if n_parallel > 1:
        print(f"  Configuration      : {n_parallel} turbines in parallel")
        print(f"  Mass flow (total)  : {m_dot:.3f} kg/s")
        print(f"  Mass flow (per shaft): {m_dot / n_parallel:.3f} kg/s")
    else:
        print(f"  Mass flow rate     : {m_dot:.3f} kg/s")
    print(
        f"  Inlet  (p, T)      : {state_in.p/1e5:.2f} bar,  "
        f"{state_in.T - 273.15:.1f} °C"
    )
    print(
        f"  Outlet (p, T)      : {state_out.p/1e5:.2f} bar,  "
        f"{state_out.T - 273.15:.1f} °C"
    )
    print(f"  Pressure ratio     : {exp['pressure_ratio']:.2f}")
    print(f"  Dh_is              : {exp['isentropic_work']/1e3:.2f} kJ/kg")
    print(f"  Specific work      : {exp['specific_work']/1e3:.2f} kJ/kg")

    W_gross = exp["power"]  # already summed across all parallel shafts
    W_is = (m_dot / n_parallel) * exp["isentropic_work"] * n_parallel
    print(f"  W_is               : {W_is/1e3:.1f} kW")
    print(f"  W_actual           : {W_gross/1e3:.1f} kW")
    if n_parallel > 1:
        print(f"  W_actual (per shaft): {W_gross/n_parallel/1e3:.1f} kW")
    print("-" * 76)

    if eff_type == "astolfi-stacking":
        n_stages = data_out.get("n_stages", "?")
        RPM = data_out.get("RPM", None)
        SP = data_out.get("size_parameter", None)
        SP_eval = data_out.get("size_parameter_eval", None)
        SP_clamped = data_out.get("size_parameter_clamped", False)
        Vr = data_out.get("volume_ratio", None)
        Ns = data_out.get("specific_speed", None)
        eta_opt = exp.get("efficiency", None)

        print(
            f"  Efficiency mode    : Astolfi stage-stacking (Table 6.6, computed during optimization)"
        )
        print(f"  Stages             : {n_stages}")
        if RPM is not None:
            print(f"  RPM                : {int(RPM)}")
        print(f"  η_is (optimizer)   : {eta_opt:.4f}  ({eta_opt*100:.2f}%)")
        print("-" * 76)

        if SP is not None:
            print(f"  SP (actual)        : {SP:.4f} m")
            if SP_clamped:
                print(
                    f"  SP (used in corr.) : {SP_eval:.4f} m  "
                    f"(clamped — efficiency plateaus)"
                )
            else:
                print(f"  SP (used in corr.) : {SP:.4f} m")
            if SP < 0.02 or SP > 1.0:
                print(
                    f"  ⚠  SP = {SP:.4f} m is OUTSIDE the correlation "
                    f"range [0.02, 1.0] m"
                )

        if Vr is not None:
            print(f"  Vr                 : {Vr:.2f}")
            # Per-stage Vr limit is 5; overall Vr can be much higher
            Vr_per_stage = (
                Vr ** (1.0 / n_stages)
                if isinstance(n_stages, int) and n_stages > 0
                else Vr
            )
            print(f"  Vr per stage (est) : {Vr_per_stage:.2f}")
            if Vr_per_stage > 5.0:
                print(
                    f"  ⚠  Vr/stage ≈ {Vr_per_stage:.2f} > 5 "
                    f"(outside Table 6.6 single-stage validity)"
                )

        if Ns is not None:
            print(f"  Ns                 : {Ns:.4f}")
            if Ns < 0.05 or Ns > 0.20:
                print(
                    f"  ⚠  Ns = {Ns:.4f} is outside the typical optimal "
                    f"range [0.05, 0.20]"
                )

        # Per-stage breakdown
        stage_data = data_out.get("stage_data", [])
        if stage_data:
            print("-" * 76)
            print(f"  ── Per-Stage Breakdown ──")
            for sd in stage_data:
                eta_pct = sd["eta_stage"] * 100
                marker = ""
                if sd["eta_stage"] <= 0.0:
                    raw = sd.get("eta_raw", None)
                    if raw is not None and raw > 1.0:
                        marker = f"  ← ZERO WORK (polynomial returned {raw:.3f})"
                    elif raw is not None and raw < 0.0:
                        marker = f"  ← ZERO WORK (polynomial returned {raw:.3f})"
                    else:
                        marker = "  ← ZERO WORK"
                print(
                    f"    Stage {sd['stage']}: "
                    f"η={eta_pct:.1f}%  "
                    f"SP={sd['SP']:.4f}m  "
                    f"Vr={sd['Vr']:.2f}  "
                    f"Ns={sd['Ns']:.4f}  "
                    f"Dh_is={sd['Dh_is']/1e3:.1f} kJ/kg{marker}"
                )
            # Flag problematic stages
            for sd in stage_data:
                issues = []
                if sd["eta_stage"] <= 0.0:
                    raw = sd.get("eta_raw", None)
                    if raw is not None and raw > 1.0:
                        issues.append(
                            f"η set to 0% — polynomial returned {raw:.3f} "
                            f"(above 1.0, extrapolation artifact at Ns={sd['Ns']:.4f})"
                        )
                    elif raw is not None and raw < 0.0:
                        issues.append(
                            f"η set to 0% — polynomial returned {raw:.3f} "
                            f"(negative, extrapolation artifact at Ns={sd['Ns']:.4f})"
                        )
                    else:
                        issues.append(
                            f"η set to 0% — outside regression range "
                            f"(Ns={sd['Ns']:.4f})"
                        )
                if sd["Vr"] > 5.0:
                    issues.append(
                        f"Vr={sd['Vr']:.2f} > 5 (outside correlation validity)"
                    )
                if sd.get("SP_clamped", False):
                    issues.append(f"SP={sd['SP']:.4f} clamped to [0.02, 1.0]")
                if sd["Ns"] > 0.25 and sd["eta_stage"] > 0.0:
                    issues.append(f"Ns={sd['Ns']:.4f} above optimal range (0.10-0.15)")
                elif sd["Ns"] < 0.05 and sd["eta_stage"] > 0.0:
                    issues.append(f"Ns={sd['Ns']:.4f} below optimal range (0.10-0.15)")
                if issues:
                    print(f"  ⚠ Stage {sd['stage']}: " + "; ".join(issues))

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


# ══════════════════════════════════════════════════════════════════════════════
#  DATA EXTRACTION — for saving to Excel
# ══════════════════════════════════════════════════════════════════════════════


def get_validation_data(cycle, config_file=None):
    """
    Return validation check results as a list of (Parameter, Value) rows
    suitable for writing to an Excel sheet.
    """
    data = cycle.problem.cycle_data
    components = data["components"]
    energy = data["energy_analysis"]
    limits = _read_limits_from_yaml(config_file)

    rows = []

    # ── Energy balance ──
    Q_in = float(energy.get("total_heat_input", energy.get("heater_heat_flow", 0)))
    Q_out = float(abs(energy.get("cooler_heat_flow", 0)))
    W_net = float(energy.get("net_system_power", energy.get("net_cycle_power", 0)))
    hs_pump = components.get(
        "heat_source_pump", components.get("heat_source_pump_hp", None)
    )
    W_hs = float(hs_pump["energy_analysis"]["power"]) if hs_pump else 0.0
    if "heat_source_pump_lp" in components:
        W_hs += float(components["heat_source_pump_lp"]["energy_analysis"]["power"])
    W_aux = W_hs + float(components["heat_sink_pump"]["energy_analysis"]["power"])
    residual = Q_in - (W_net + W_aux + Q_out)
    pct = abs(residual) / abs(Q_in) * 100 if Q_in != 0 else 0

    rows += [
        ("══ ENERGY BALANCE ══", ""),
        ("Q_in [kW]", round(Q_in / 1e3, 2)),
        ("W_net system [kW]", round(W_net / 1e3, 2)),
        ("W_aux pumps [kW]", round(W_aux / 1e3, 3)),
        ("Q_out [kW]", round(Q_out / 1e3, 2)),
        ("Residual [kW]", round(residual / 1e3, 4)),
        ("Residual [%]", round(pct, 4)),
        ("Energy balance OK", "Yes" if pct <= 0.1 else "WARNING"),
        ("", ""),
    ]

    # ── Constraint activity ──
    rows.append(("══ CONSTRAINT ACTIVITY ══", ""))
    pinch_limit = limits["pinch"]
    subcooling_limit = limits["subcooling"]
    superheating_limit = limits["superheating"]

    for hx_name in [
        "heater",
        "hp_evaporator",
        "lp_evaporator",
        "preheater",
        "recuperator",
        "cooler",
    ]:
        if hx_name not in components:
            continue
        comp = components[hx_name]
        if "temperature_difference" not in comp:
            continue
        dT_array = np.array(comp["temperature_difference"])
        min_dT = float(np.min(dT_array))
        slack = min_dT - pinch_limit
        display = _HX_DISPLAY.get(hx_name, hx_name)
        if min_dT < pinch_limit - 0.1:
            status = "VIOLATED"
        elif abs(slack) < 0.1:
            status = "BINDING"
        else:
            status = "slack"
        rows += [
            (f"{display} pinch — value [K]", min_dT),
            (f"{display} pinch — limit [K]", pinch_limit),
            (f"{display} pinch — slack [K]", slack),
            (f"{display} pinch — status", status),
        ]

    pump_name = "lp_pump" if "lp_pump" in components else "compressor"
    if pump_name in components:
        pump_in = components[pump_name].get("state_in", None)
        if pump_in is not None and hasattr(pump_in, "subcooling"):
            sc = float(pump_in.subcooling)
            slack = sc - subcooling_limit
            if sc < subcooling_limit - 0.1:
                status = "VIOLATED"
            elif abs(slack) < 0.2:
                status = "BINDING"
            else:
                status = "slack"
            label = "LP pump" if pump_name == "lp_pump" else "Pump"
            rows += [
                (f"{label} subcooling — value [K]", sc),
                (f"{label} subcooling — limit [K]", subcooling_limit),
                (f"{label} subcooling — slack [K]", slack),
                (f"{label} subcooling — status", status),
            ]

    exp_name = "hp_expander" if "hp_expander" in components else "expander"
    if exp_name in components:
        exp_in = components[exp_name].get("state_in", None)
        if exp_in is not None and hasattr(exp_in, "superheating"):
            sh = float(exp_in.superheating)
            slack = sh - superheating_limit
            if sh < superheating_limit - 0.1:
                status = "VIOLATED"
            elif abs(slack) < 0.2:
                status = "BINDING"
            else:
                status = "slack"
            label = "HP expander" if exp_name == "hp_expander" else "Expander"
            rows += [
                (f"{label} superheating — value [K]", sh),
                (f"{label} superheating — limit [K]", superheating_limit),
                (f"{label} superheating — slack [K]", slack),
                (f"{label} superheating — status", status),
            ]

    # ── Phase verification ──
    rows += [("", ""), ("══ STATE POINT PHASES ══", "")]
    phase_checks = []
    if "compressor" in components:
        phase_checks.append(
            ("Pump inlet", components["compressor"]["state_in"], "liquid")
        )
    if "lp_pump" in components:
        phase_checks.append(
            ("LP pump inlet", components["lp_pump"]["state_in"], "liquid")
        )
    if "expander" in components:
        phase_checks.append(
            ("Expander inlet", components["expander"]["state_in"], "vapor")
        )
        phase_checks.append(
            ("Expander outlet", components["expander"]["state_out"], "vapor")
        )
    if "hp_expander" in components:
        phase_checks.append(
            ("HP expander inlet", components["hp_expander"]["state_in"], "vapor")
        )
        phase_checks.append(
            ("HP expander outlet", components["hp_expander"]["state_out"], "vapor")
        )
    if "lp_expander" in components:
        phase_checks.append(
            ("LP expander inlet", components["lp_expander"]["state_in"], "vapor")
        )
        phase_checks.append(
            ("LP expander outlet", components["lp_expander"]["state_out"], "vapor")
        )

    for label, state, expected in phase_checks:
        try:
            phase = str(state.phase).lower()
            # If phase label is uninformative, fall back to vapor quality Q
            if phase in ("unknown", "") or phase.startswith("0") or phase.isdigit():
                try:
                    Q = float(state.Q)
                    if Q <= 0.0:
                        phase = "liquid"
                    elif Q >= 1.0:
                        phase = "vapor"
                    else:
                        phase = f"two-phase (Q={Q:.3f})"
                except (AttributeError, ValueError, TypeError):
                    phase = "unknown"
        except AttributeError:
            phase = "unknown"
        T_C = float(state.T) - 273.15 if hasattr(state, "T") else float("nan")
        rows += [
            (f"{label} — phase", phase),
            (f"{label} — T [°C]", round(T_C, 2)),
        ]

    return rows


def get_turbine_data(cycle):
    """
    Return turbine/expander data as a list of (Parameter, Value) rows.
    Only populated for astolfi-stacking efficiency mode; returns None otherwise.
    """
    components = cycle.problem.cycle_data["components"]

    def _exp_rows(exp, label):
        eff_type = exp.get("efficiency_type", "isentropic")
        if eff_type != "astolfi-stacking":
            return None  # caller uses this to skip the sheet

        data_out = exp.get("data_out", {})
        state_in = exp["state_in"]
        state_out = exp["state_out"]
        m_dot = exp.get("mass_flow", float("nan"))
        W_gross = m_dot * exp["specific_work"]
        W_is = m_dot * exp["isentropic_work"]
        n_stages = data_out.get("n_stages", None)
        RPM = data_out.get("RPM", None)
        SP = data_out.get("size_parameter", None)
        SP_eval = data_out.get("size_parameter_eval", None)
        SP_clamped = data_out.get("size_parameter_clamped", False)
        Vr = data_out.get("volume_ratio", None)
        Ns = data_out.get("specific_speed", None)
        eta_opt = exp.get("efficiency", None)
        Vr_per_stage = (
            Vr ** (1.0 / n_stages)
            if Vr is not None and isinstance(n_stages, int) and n_stages > 0
            else None
        )

        rows = [
            (f"══ {label.upper()} ══", ""),
            ("Fluid", exp.get("fluid_name", "unknown")),
            ("Mass flow rate [kg/s]", round(m_dot, 3)),
            ("Inlet pressure [bar]", round(state_in.p / 1e5, 3)),
            ("Inlet temperature [°C]", round(state_in.T - 273.15, 2)),
            ("Outlet pressure [bar]", round(state_out.p / 1e5, 3)),
            ("Outlet temperature [°C]", round(state_out.T - 273.15, 2)),
            ("Pressure ratio [-]", round(exp["pressure_ratio"], 3)),
            ("Dh_is [kJ/kg]", round(exp["isentropic_work"] / 1e3, 3)),
            ("Specific work [kJ/kg]", round(exp["specific_work"] / 1e3, 3)),
            ("W_is [kW]", round(W_is / 1e3, 2)),
            ("W_actual [kW]", round(W_gross / 1e3, 2)),
            ("", ""),
            ("Efficiency mode", "Astolfi stage-stacking"),
            ("Stages [-]", n_stages),
            ("RPM", int(RPM) if RPM is not None else None),
            ("η_is (optimizer) [-]", round(eta_opt, 6) if eta_opt else None),
            ("η_is (optimizer) [%]", round(eta_opt * 100, 3) if eta_opt else None),
            ("SP actual [m]", round(SP, 4) if SP is not None else None),
            (
                "SP used in correlation [m]",
                round(SP_eval, 4) if SP_eval is not None else None,
            ),
            ("SP clamped", "Yes" if SP_clamped else "No"),
            (
                "SP in range [0.02, 1.0]",
                "Yes" if SP is not None and 0.02 <= SP <= 1.0 else "WARNING",
            ),
            ("Vr overall [-]", round(Vr, 3) if Vr is not None else None),
            (
                "Vr per stage (est) [-]",
                round(Vr_per_stage, 3) if Vr_per_stage is not None else None,
            ),
            (
                "Vr/stage within limit (<5)",
                (
                    "Yes"
                    if Vr_per_stage is not None and Vr_per_stage <= 5.0
                    else "WARNING"
                ),
            ),
            ("Ns [-]", round(Ns, 4) if Ns is not None else None),
            (
                "Ns in optimal range [0.05-0.20]",
                "Yes" if Ns is not None and 0.05 <= Ns <= 0.20 else "WARNING",
            ),
            ("", ""),
        ]

        # Per-stage breakdown
        stage_data = data_out.get("stage_data", [])
        if stage_data:
            rows.append(("── Per-Stage Breakdown ──", ""))
            rows.append(("Stage", "η [%] | SP [m] | Vr [-] | Ns [-] | Dh_is [kJ/kg]"))
            for sd in stage_data:
                summary = (
                    f"η={sd['eta_stage']*100:.1f}%  "
                    f"SP={sd['SP']:.4f}m  "
                    f"Vr={sd['Vr']:.2f}  "
                    f"Ns={sd['Ns']:.4f}  "
                    f"Dh_is={sd['Dh_is']/1e3:.1f} kJ/kg"
                )
                rows.append((f"Stage {sd['stage']}", summary))

        return rows

    all_rows = []
    if "hp_expander" in components:
        r1 = _exp_rows(components["hp_expander"], "HP Expander")
        r2 = _exp_rows(components["lp_expander"], "LP Expander")
        if r1 is None and r2 is None:
            return None
        all_rows = (r1 or []) + [("", "")] + (r2 or [])
    elif "expander" in components:
        all_rows = _exp_rows(components["expander"], "Expander")

    return all_rows if all_rows else None


def get_tq_summary_data(cycle):
    """
    Return T-Q diagram summary data as a list of (Parameter, Value) rows
    extracted directly from cycle component states.
    """
    components = cycle.problem.cycle_data["components"]
    rows = []

    hx_display = {
        "heater": "Evaporator",
        "hp_evaporator": "HP Evaporator",
        "lp_evaporator": "LP Evaporator",
        "preheater": "Preheater",
        "recuperator": "Recuperator",
        "cooler": "Condenser",
    }

    for hx_name, display in hx_display.items():
        if hx_name not in components:
            continue
        comp = components[hx_name]
        heat_flow = comp.get("heat_flow")
        if heat_flow is None or abs(heat_flow) < 1000:
            continue

        hot = comp["hot_side"]
        cold = comp["cold_side"]

        T_hot_in = float(hot["state_in"].T) - 273.15
        T_hot_out = float(hot["state_out"].T) - 273.15
        T_cold_in = float(cold["state_in"].T) - 273.15
        T_cold_out = float(cold["state_out"].T) - 273.15

        dT_array = comp.get("temperature_difference", None)
        if dT_array is not None:
            min_dT = float(np.min(np.array(dT_array)))
        else:
            min_dT = float("nan")

        rows += [
            (f"══ {display.upper()} T-Q SUMMARY ══", ""),
            (
                f"{display} — total heat duty [kW]",
                round(float(abs(heat_flow)) / 1e3, 2),
            ),
            (f"{display} — hot-side inlet T [°C]", round(T_hot_in, 2)),
            (f"{display} — hot-side outlet T [°C]", round(T_hot_out, 2)),
            (f"{display} — cold-side inlet T [°C]", round(T_cold_in, 2)),
            (f"{display} — cold-side outlet T [°C]", round(T_cold_out, 2)),
            (f"{display} — pinch-point ΔT [K]", round(min_dT, 3)),
            ("", ""),
        ]

    return rows
