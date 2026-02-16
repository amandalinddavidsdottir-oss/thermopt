"""
Working Fluid Sweep — Basic Subcritical ORC  (v4)

Changelog v4 (all changes marked with #v4 in the code)
-------------------------------------------------------
- FIX #4: T_cond_low / T_cond_high are now derived from YAML parameters
  (T_heat_sink_exit_max and cooler_pinch_min) instead of hardcoded offsets.
  This ensures the candidate filter is consistent with what the optimizer
  actually enforces.
- NEW: Vacuum condenser flagging — fluids with sub-atmospheric or deep-
  vacuum condensing pressures are flagged (not rejected) so the user can
  see the engineering penalty at a glance.
- NEW: Near-critical flagging — fluids whose Tc is within 30 K of the
  maximum evaporation temperature are flagged as "near_critical".
- NEW: Expanded THERMAL_STABILITY_K dictionary with aromatics and heavier
  alkanes. Fluids with Tc > 500 K and NO stability data are flagged as
  "stability_unknown" instead of silently assumed stable.

Changelog v3 (retained from previous version)
-------------------------------------------------------
- FIX #1: Config creation now uses raw string replacement instead of
  yaml.safe_load/yaml.dump. This preserves $working_fluid.* expressions,
  float formatting (45e6), and comment structure exactly.
- FIX #2: Tc filter is now relative to the *evaporation temperature*
  (T_heat_source − pinch_min) rather than T_heat_source itself.
  This is physically correct and less restrictive.
- FIX #3: Condensing pressure filter at T_cond_high is now blocking —
  fluids that fail at the upper condenser temperature are rejected.
- NEW: Thermal stability dictionary — fluids whose decomposition limit
  is below the heat source temperature are flagged or rejected.
- NEW: Post-convergence constraint validation — after the solver reports
  success, the code checks that pinch > 5 K, subcooling > 1 K,
  superheating > 5 K. Violations downgrade the result to "infeasible".
"""

import os, re, copy, yaml
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import CoolProp.CoolProp as cp
import thermopt as th
from pathlib import Path


# ══════════════════════════════════════════════════════════════════════
#  BANNED & FLAGGED FLUIDS
# ══════════════════════════════════════════════════════════════════════

BANNED = {
    # ── Ozone-depleting (ODP > 0, Montreal Protocol) ──
    "R11", "R12", "R13", "R113", "R114", "R115", "R123",
    "R141b", "R142b", "R22",
    # ── High GWP ≥ 150 (EU 517/2014 & Kigali) ────
    "R134a", "R125", "R143a", "R227EA", "R236EA", "R236FA",
    "R245fa", "R365MFC", "R32", "RC318", "R116", "R218",
    "R23", "R41", "SulfurHexafluoride",
    # ── Toxic or dangerous ────
    "CarbonMonoxide", "HydrogenSulfide", "SulfurDioxide",
    "NitrousOxide", "Methanol", "Ethanol", "Ammonia",
    # ── Cryogens ─────────
    "Helium", "Neon", "Argon", "Krypton", "Xenon",
    "Hydrogen", "Nitrogen", "Oxygen", "Fluorine",
    "ParaHydrogen", "OrthoHydrogen", "Deuterium",
    "ParaDeuterium", "OrthoDeuterium", "HeavyWater", "Air",
    # ── Other unsuitable ─────────
    "CarbonDioxide", "Acetone", "DiethylEther",
    "Ethylene", "EthyleneOxide", "CarbonylSulfide",
}

FLAGGED = {
    "Propane": "flammable", "Butane": "flammable", "Isobutane": "flammable",
    "Isopentane": "flammable", "Neopentane": "flammable", "Pentane": "flammable",
    "Isohexane": "flammable", "Hexane": "flammable", "Heptane": "flammable",
    "CycloHexane": "flammable", "CycloPropane": "flammable", "Toluene": "flammable",
    "EthylBenzene": "flammable", "m-Xylene": "flammable", "o-Xylene": "flammable",
    "p-Xylene": "flammable",
    "Benzene": "toxic",  # IARC Group 1 carcinogen
    "MDM": "siloxane", "MM": "siloxane", "MD2M": "siloxane",
    "MD3M": "siloxane", "MD4M": "siloxane",
    "D4": "siloxane", "D5": "siloxane", "D6": "siloxane",
}


# ══════════════════════════════════════════════════════════════════════
#  THERMAL STABILITY LIMITS  (v4 — expanded)
# ══════════════════════════════════════════════════════════════════════
#
# Conservative maximum operating temperatures [K] from literature.
# Sources: Ginosar et al. (2011), Pasetti et al. (2014),
#          Angelino & Invernizzi (2003), Preissinger & Brüggemann (2016).
#
# If a fluid's limit is BELOW the heat source temperature, it is
# flagged as "thermal_risk" in results. If it is more than 30 K below,
# it is rejected outright (decomposition is near-certain).
#
# v4: Fluids with Tc > 500 K and NO entry here are flagged as
# "stability_unknown" rather than silently assumed stable.

THERMAL_STABILITY_K = {
    # Hydrocarbons
    "Toluene":      623,   # ~350 °C — relatively stable aromatic
    "CycloHexane":  573,   # ~300 °C
    "Benzene":      573,   # ~300 °C
    "Pentane":      573,   # ~300 °C
    "Isopentane":   573,   # ~300 °C
    "Neopentane":   573,   # ~300 °C
    "Hexane":       573,   # ~300 °C
    "Isohexane":    573,   # ~300 °C
    "Heptane":      573,   # ~300 °C
    "Octane":       573,   # ~300 °C
    "Butane":       573,   # ~300 °C
    "Isobutane":    573,   # ~300 °C
    "Propane":      623,   # ~350 °C
    "CycloPropane": 573,   # ~300 °C
    # v4: Aromatics — generally more stable than linear alkanes
    "EthylBenzene": 623,   # ~350 °C (similar to toluene)
    "m-Xylene":     623,   # ~350 °C
    "o-Xylene":     623,   # ~350 °C
    "p-Xylene":     623,   # ~350 °C
    # v4: Heavier alkanes
    "Nonane":       573,   # ~300 °C
    "Decane":       573,   # ~300 °C
    "Undecane":     573,   # ~300 °C
    "Dodecane":     573,   # ~300 °C
    # Siloxanes (thermal degradation above ~300 °C)
    "MM":    573,  "MDM":  573,  "MD2M": 573,
    "MD3M":  573,  "MD4M": 573,
    "D4":    573,  "D5":   573,  "D6":   573,
    # Refrigerants (low-GWP alternatives)
    "R1233zdE":  473,   # ~200 °C
    "R1234zeZ":  473,   # ~200 °C
    "R1234zeE":  443,   # ~170 °C
    "R1234yf":   423,   # ~150 °C
}


# ══════════════════════════════════════════════════════════════════════
#  FLUID CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════

def classify_fluid(name):
    """
    Classifies fluid as wet, dry, or isentropic based on the
    dimensionless slope ξ = (T/s)·(ds/dT) of the saturated vapour
    line at T = 0.7·Tc.
    """
    try:
        Tc = cp.PropsSI("Tcrit", name)
        T = max(0.7 * Tc, cp.PropsSI("Tmin", name) + 20)
        dT = 2.0
        s_lo = cp.PropsSI("S", "T", T,      "Q", 1, name)
        s_hi = cp.PropsSI("S", "T", T + dT, "Q", 1, name)
        ds_dT = (s_hi - s_lo) / dT
        if abs(s_lo) < 1e-6:
            return "unknown"
        xi = (T / s_lo) * ds_dT
        # ξ > 0 means entropy INCREASES with T on the sat. vapour line
        #       → the "overhang" region → DRY fluid
        # ξ < 0 means entropy DECREASES with T → heading toward Tc → WET fluid
        if xi > 0.1:
            return "dry"
        elif xi < -0.1:
            return "wet"
        else:
            return "isentropic"
    except Exception:
        return "unknown"


# ══════════════════════════════════════════════════════════════════════
#  CANDIDATE SELECTION
# ══════════════════════════════════════════════════════════════════════

def get_candidate_fluids(config_file,
                         Tc_margin_above_evap=20,
                         near_critical_margin=30,
                         min_cond_pressure=0,
                         thermal_stability_margin=30,
                         stability_unknown_Tc_threshold=500):
    """
    Filter CoolProp fluids for subcritical ORC compatibility.

    Reads heat source/sink temperatures, design variable bounds, and
    constraint limits directly from the YAML config file.

    Parameters
    ----------
    config_file : str or Path
        Path to the thermopt YAML configuration file.
    Tc_margin_above_evap : float
        Required gap between max evaporation temperature and Tc [K].
    near_critical_margin : float
        Flag if Tc − T_evap_max < this value [K].
    min_cond_pressure : float
        Minimum acceptable condensing pressure [Pa].
    thermal_stability_margin : float
        If T_stability < T_heat_source − this margin → reject [K].
    stability_unknown_Tc_threshold : float
        If Tc > this AND no stability data → flag "stability_unknown" [K].
    """
    # ── Extract parameters from YAML ──────────────────────────────
    cfg = yaml.safe_load(Path(config_file).read_text())
    fixed = cfg["problem_formulation"]["fixed_parameters"]
    dv    = cfg["problem_formulation"]["design_variables"]

    T_heat_source      = fixed["heat_source"]["inlet_temperature"]
    T_heat_sink        = fixed["heat_sink"]["inlet_temperature"]
    T_sink_exit_max    = dv["heat_sink_exit_temperature"]["max"]

    # Extract pinch limits from constraints (with defaults)
    heater_pinch_min = 5.0
    cooler_pinch_min = 5.0
    for c in cfg["problem_formulation"]["constraints"]:
        var = c.get("variable", "")
        if "heater.temperature_difference" in var:
            heater_pinch_min = c["value"]
        elif "cooler.temperature_difference" in var:
            cooler_pinch_min = c["value"]

    # ── Derived filter temperatures ───────────────────────────────
    T_evap_max  = T_heat_source - heater_pinch_min
    Tc_min_K    = T_evap_max + Tc_margin_above_evap
    T_cond_low  = T_heat_sink + cooler_pinch_min
    T_cond_high = T_sink_exit_max - cooler_pinch_min

    print(f"\n  FILTER SETTINGS (from {Path(config_file).name})")
    print(f"    T_heat_source     = {T_heat_source - 273.15:.1f} °C")
    print(f"    T_heat_sink       = {T_heat_sink - 273.15:.1f} °C")
    print(f"    T_evap_max        = {T_evap_max - 273.15:.1f} °C  (T_hs − {heater_pinch_min} K)")
    print(f"    Tc_min            = {Tc_min_K - 273.15:.1f} °C  (T_evap_max + {Tc_margin_above_evap} K)")
    print(f"    T_cond range      = [{T_cond_low - 273.15:.1f}, {T_cond_high - 273.15:.1f}] °C")
    print(f"    min_cond_pressure = {min_cond_pressure/1e5:.4f} bar")

    candidates = []
    fluids_list = cp.FluidsList()
    if isinstance(fluids_list, str):
        fluids_list = fluids_list.split(",")

    for name in sorted(fluids_list):
        if name in BANNED:
            continue

        # ── Basic thermodynamic checks ────────────────────────────
        try:
            Tc_K = cp.PropsSI("Tcrit", name)
            p_cond_low = cp.PropsSI("P", "T", T_cond_low, "Q", 0, name)
        except Exception:
            continue

        if Tc_K < Tc_min_K:
            continue

        if p_cond_low < min_cond_pressure:
            continue

        # v3 FIX: Condensing pressure at T_cond_high is now BLOCKING
        try:
            p_cond_high = cp.PropsSI("P", "T", T_cond_high, "Q", 0, name)
        except Exception:
            print(f"    ⊘ {name}: CoolProp fails at T_cond_high={T_cond_high:.1f} K — skipped")
            continue  # v3: was `p_cond_high = np.nan` + continue to include — now reject

        if p_cond_high < min_cond_pressure:
            continue

        # ── Thermal stability check (v3 — new, v4 — expanded) ────
        T_stability = THERMAL_STABILITY_K.get(name)
        thermal_flag = None

        if T_stability is not None:
            if T_stability < T_heat_source - thermal_stability_margin:
                # Decomposition temperature is far below heat source → reject
                print(f"    ⊘ {name}: T_stability={T_stability-273.15:.0f}°C "
                      f"<< T_hs={T_heat_source-273.15:.0f}°C — rejected (thermal decomposition)")
                continue
            elif T_stability < T_heat_source:
                # Within margin — include but flag
                thermal_flag = "thermal_risk"
        else:
            # v4: No stability data — flag if Tc is high enough to worry
            if Tc_K > stability_unknown_Tc_threshold:
                thermal_flag = "stability_unknown"

        # ── Vacuum condenser flag (v4 — new) ──────────────────────
        vacuum_flag = None
        if p_cond_low < 10_000:           # < 0.1 bar → deep vacuum
            vacuum_flag = "deep_vacuum"
        elif p_cond_low < 101_325:        # < 1 bar → sub-atmospheric
            vacuum_flag = "sub_atm"

        # ── Near-critical flag (v4 — new) ─────────────────────────
        near_crit_flag = None
        if Tc_K - T_evap_max < near_critical_margin:
            near_crit_flag = "near_critical"

        # ── Combine all flags ─────────────────────────────────────
        flag = FLAGGED.get(name)
        extra_flags = [f for f in (thermal_flag, vacuum_flag, near_crit_flag) if f]
        if flag and extra_flags:
            flag = flag + "+" + "+".join(extra_flags)
        elif extra_flags:
            flag = "+".join(extra_flags)

        candidates.append({
            "name": name,
            "Tc_C": Tc_K - 273.15,
            "pc_bar": cp.PropsSI("pcrit", name) / 1e5,
            "p_cond_low_bar": p_cond_low / 1e5,
            "p_cond_high_bar": p_cond_high / 1e5,
            "T_stability_C": (T_stability - 273.15) if T_stability else None,
            "flag": flag,
            "fluid_type": classify_fluid(name),
        })

    candidates.sort(key=lambda f: f["Tc_C"])

    print(f"\n  CANDIDATES | Tc > {Tc_min_K - 273.15:.1f}°C "
          f"(T_evap_max={T_evap_max-273.15:.1f}°C + {Tc_margin_above_evap}K margin) "
          f"| {len(candidates)} fluids")
    for f in candidates:
        stab = f"  T_stab={f['T_stability_C']:.0f}°C" if f["T_stability_C"] else ""
        print(f"    {f['name']:<18} Tc={f['Tc_C']:>6.1f}°C  "
              f"p_cond=[{f['p_cond_low_bar']:.3f}, {f['p_cond_high_bar']:.3f}] bar  "
              f"{f['fluid_type']:<12} {f['flag'] or ''}{stab}")
    print()
    return candidates


# ══════════════════════════════════════════════════════════════════════
#  CONFIG FILE MANIPULATION  (v3 — fix #1, string replacement)
# ══════════════════════════════════════════════════════════════════════

def _make_tmp_config(raw_yaml, fluid_name, expander_pressure_fraction=None):
    """
    Create a modified YAML config by doing string replacement on the
    raw text. This preserves all $working_fluid.* expressions, float
    formats like 45e6, and comments exactly as written.

    v3: Replaces the old yaml.safe_load → deepcopy → yaml.dump approach,
    which could alter string quoting, float formatting, and key ordering.

    Parameters
    ----------
    raw_yaml : str
        The raw YAML text read from disk.
    fluid_name : str
        CoolProp fluid name to substitute.
    expander_pressure_fraction : float or None
        If given, replace the starting value fraction for
        expander_inlet_pressure (used for multi-start).
    """
    # ── Replace working fluid name ────────────────────────────────
    # Matches "name: <word>" under the working_fluid block.
    # The working_fluid block is the first occurrence of "name:" after
    # "working_fluid:", so we use a targeted regex with DOTALL.
    out = re.sub(
        r'(working_fluid:\s*\n\s*name:\s*)\S+',
        rf'\g<1>{fluid_name}',
        raw_yaml,
        count=1,
    )

    # ── Optionally replace expander_inlet_pressure starting value ─
    if expander_pressure_fraction is not None:
        # Match the value line inside the expander_inlet_pressure block.
        # Pattern: "expander_inlet_pressure:" ... "value:" <fraction> "*$working_fluid.critical_point.p"
        # The .*? with DOTALL bridges the lines between the block header and value.
        out = re.sub(
            r'(expander_inlet_pressure:.*?value:\s*)'
            r'[\d.]+(\s*\*\s*\$working_fluid\.critical_point\.p)',
            rf'\g<1>{expander_pressure_fraction}\2',
            out,
            count=1,
            flags=re.DOTALL,
        )

    return out


# ══════════════════════════════════════════════════════════════════════
#  POST-CONVERGENCE VALIDATION
# ══════════════════════════════════════════════════════════════════════
#
#  thermopt data structure (from inspection):
#    cycle_data["components"]["heater"]  → dict with "temperature_difference" (array), "power", etc.
#    cycle_data["components"]["compressor"]["state_in"]  → FluidState object with .subcooling, .temperature, .pressure, etc.
#    cycle_data["components"]["expander"]["state_in"]    → FluidState object with .superheating, .temperature, .pressure, etc.
#    cycle_data["energy_analysis"]  → dict with "system_efficiency", "cycle_efficiency", etc.
#

def _validate_constraints(cycle, slsqp_tol=0.5):
    """
    Check that key physical constraints hold after optimization.
    Returns (is_valid, actuals, violations).
    """
    limits = {
        "heater_pinch": 5.0,   # K
        "cooler_pinch":  5.0,   # K
        "subcooling":    1.0,   # K
        "superheating":  5.0,   # K
    }

    cd = cycle.problem.cycle_data
    actuals = {
        "heater_pinch":  float(np.min(cd["components"]["heater"]["temperature_difference"])),
        "cooler_pinch":  float(np.min(cd["components"]["cooler"]["temperature_difference"])),
        "subcooling":    float(cd["components"]["compressor"]["state_in"].subcooling),
        "superheating":  float(cd["components"]["expander"]["state_in"].superheating),
    }

    violations = {k: v for k, v in actuals.items()
                  if np.isnan(v) or v < limits[k] - slsqp_tol}

    print(f"      pinch=[{actuals['heater_pinch']:.2f}, {actuals['cooler_pinch']:.2f}] K  "
          f"subcool={actuals['subcooling']:.2f} K  "
          f"superheat={actuals['superheating']:.2f} K")

    return len(violations) == 0, actuals, violations


# ══════════════════════════════════════════════════════════════════════
#  DESIGN VARIABLE EXTRACTION
# ══════════════════════════════════════════════════════════════════════

def _extract_design_variables(cycle):
    """Pull optimised state points for the results table."""
    cd = cycle.problem.cycle_data
    exp_in  = cd["components"]["expander"]["state_in"]
    exp_out = cd["components"]["expander"]["state_out"]
    comp_in = cd["components"]["compressor"]["state_in"]

    return {
        "expander_inlet_T_K":          float(exp_in.temperature),
        "expander_inlet_p_bar":        float(exp_in.pressure) / 1e5,
        "expander_inlet_superheat_K":  float(exp_in.superheating),
        "expander_outlet_T_K":         float(exp_out.temperature),
        "expander_outlet_p_bar":       float(exp_out.pressure) / 1e5,
        "expander_outlet_superheat_K": float(exp_out.superheating),
        "pump_inlet_T_K":              float(comp_in.temperature),
        "pump_inlet_p_bar":            float(comp_in.pressure) / 1e5,
        "pump_inlet_subcooling_K":     float(comp_in.subcooling),
        "heater_pinch_K":              float(np.min(cd["components"]["heater"]["temperature_difference"])),
        "cooler_pinch_K":              float(np.min(cd["components"]["cooler"]["temperature_difference"])),
    }


# ══════════════════════════════════════════════════════════════════════
#  MAIN SWEEP  (v3 — single-start + post-convergence validation)
# ══════════════════════════════════════════════════════════════════════

def run_fluid_sweep(config_file, candidates, output_dir="results/fluid_sweep",
                    save_results=True):
    """
    For each candidate fluid, run a single optimisation using the
    default starting point from the YAML, then validate constraints.

    v3 changes
    ----------
    - Uses raw string replacement to create temp configs (fix #1).
    - Post-convergence constraint validation.
    """
    # v3 FIX #1: Read raw YAML text — do NOT parse with yaml.safe_load
    raw_yaml = Path(config_file).read_text()

    os.makedirs(output_dir, exist_ok=True)
    results = []

    for i, fluid in enumerate(candidates):
        name = fluid["name"]
        stab_str = (f", T_stab={fluid['T_stability_C']:.0f}°C"
                    if fluid.get("T_stability_C") else "")
        print(f"\n  [{i+1}/{len(candidates)}] {name}  "
              f"(Tc={fluid['Tc_C']:.1f}°C, {fluid['fluid_type']}{stab_str})")

        # v3 FIX #1: string replacement, not yaml.dump
        tmp_yaml = _make_tmp_config(raw_yaml, name)
        tmp_path = os.path.join(output_dir, f"_tmp_{name}.yaml")
        Path(tmp_path).write_text(tmp_yaml)

        # Start with a "failed" row — only update if solver succeeds
        row = {
            "fluid": name,
            "Tc [°C]": fluid["Tc_C"],
            "pc [bar]": fluid["pc_bar"],
            "p_cond_low [bar]": fluid["p_cond_low_bar"],
            "p_cond_high [bar]": fluid["p_cond_high_bar"],
            "T_stability [°C]": fluid.get("T_stability_C"),
            "fluid_type": fluid["fluid_type"],
            "flag": fluid["flag"],
            "system_efficiency [-]": np.nan,
            "status": "failed",
        }

        try:
            cycle = th.ThermodynamicCycleOptimization(
                tmp_path, out_dir=os.path.join(output_dir, name)
            )
            cycle.run_optimization()

            if not getattr(cycle.solver, "success", False):
                row["status"] = f"fail: {getattr(cycle.solver, 'message', '?')[:40]}"
                print(f"    ✗ {row['status']}")

            else:
                # ── Post-convergence constraint validation (v3 — new) ─
                is_valid, actuals, violations = _validate_constraints(cycle)

                ea = cycle.problem.cycle_data["energy_analysis"]
                eta = ea["system_efficiency"]

                if not is_valid:
                    # v4 FIX: use numbers.Number or try/except to handle
                    # numpy floats (np.float64 is NOT isinstance of float)
                    viol_parts = []
                    for k, v in violations.items():
                        try:
                            viol_parts.append(f"{k}={float(v):.2f}")
                        except (TypeError, ValueError):
                            viol_parts.append(f"{k}={v}")
                    viol_str = ", ".join(viol_parts)
                    row["status"] = f"infeasible: {viol_str}"
                    row["system_efficiency [-]"] = eta  # record but mark infeasible
                    print(f"    ⚠ converged but INFEASIBLE: {viol_str}")
                    print(f"      η_sys={eta*100:.2f}% (marked infeasible)")
                else:
                    # Valid result
                    if save_results:
                        cycle.save_results()

                    if "net_system_power" in ea:
                        net_power = ea["net_system_power"]
                    elif "net_cycle_power" in ea:
                        net_power = ea["net_cycle_power"]
                        print(f"    ⚠ using 'net_cycle_power' (no 'net_system_power')")
                    else:
                        net_power = np.nan

                    row.update({
                        "status": "converged",
                        "system_efficiency [-]": eta,
                        "cycle_efficiency [-]": ea["cycle_efficiency"],
                        "net_power [kW]": net_power / 1e3 if not np.isnan(net_power) else np.nan,
                        "mass_flow [kg/s]": ea["mass_flow_working_fluid"],
                        "backwork_ratio [-]": ea["backwork_ratio"],
                        "heater_pinch [K]": actuals["heater_pinch"],
                        "cooler_pinch [K]": actuals["cooler_pinch"],
                        "subcooling [K]": actuals["subcooling"],
                        "superheating [K]": actuals["superheating"],
                    })
                    row.update(_extract_design_variables(cycle))

                    print(f"    ✓ η_sys={eta*100:.2f}%  "
                          f"pinch=[{actuals['heater_pinch']:.1f}, {actuals['cooler_pinch']:.1f}] K  "
                          f"subcool={actuals['subcooling']:.1f} K  "
                          f"superheat={actuals['superheating']:.1f} K")

        except Exception as e:
            row["status"] = f"error: {str(e)[:60]}"
            print(f"    ✗ {row['status']}")

        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            plt.close("all")

        results.append(row)

    # ── Build DataFrame, save ─────────────────────────────────────
    df = (pd.DataFrame(results)
          .sort_values("system_efficiency [-]", ascending=False, na_position="last")
          .reset_index(drop=True))
    xlsx = os.path.join(output_dir, "fluid_sweep_results.xlsx")
    df.to_excel(xlsx, index=False)

    # ── Ranked summary ────────────────────────────────────────────
    conv = df[df["status"] == "converged"]
    print(f"\n  RESULTS | ✓ {len(conv)}  ✗ {len(df)-len(conv)}")
    for r, (_, row) in enumerate(conv.iterrows(), 1):
        flag_str = f"  ⚠ {row['flag']}" if row["flag"] else ""
        print(f"    {r}. {row['fluid']:<16} η_sys={row['system_efficiency [-]']*100:.2f}%  "
              f"{row['fluid_type']}{flag_str}")
    for _, row in df[df["status"] != "converged"].iterrows():
        print(f"    ✗ {row['fluid']:<16} {row['status']}")
    print(f"  Saved: {xlsx}\n")

    return df


# ══════════════════════════════════════════════════════════════════════
#  PLOTTING  (v4 — new flag colours for vacuum and near-critical)
# ══════════════════════════════════════════════════════════════════════

_FLAG_COLORS = {
    None:                "#2980b9",   # blue  — no flag
    "flammable":         "#e74c3c",   # red
    "toxic":             "#c0392b",   # dark red
    "siloxane":          "#f39c12",   # amber
    "thermal_risk":      "#9b59b6",   # purple
    "stability_unknown": "#8e44ad",   # dark purple
    "deep_vacuum":       "#1abc9c",   # teal
    "sub_atm":           "#85c1e9",   # light blue
    "near_critical":     "#e67e22",   # orange
}

_TYPE_HATCHES = {
    "wet":         "//",
    "dry":         "",
    "isentropic":  "...",
    "unknown":     "xx",
}


def _get_flag_color(flag):
    """
    Handle combined flags like 'flammable+thermal_risk+deep_vacuum'.

    Priority order for colour selection (first match wins):
      thermal_risk > stability_unknown > near_critical > deep_vacuum > sub_atm > flammable > siloxane
    This prioritises safety-critical flags over engineering flags.
    """
    if flag is None:
        return _FLAG_COLORS[None]
    flag_str = str(flag)
    # Check in priority order
    for key in ("toxic", "thermal_risk", "stability_unknown", "near_critical",
                "deep_vacuum", "sub_atm", "flammable", "siloxane"):
        if key in flag_str:
            return _FLAG_COLORS[key]
    return "#999999"


def plot_results(df, output_dir=None, filename="fluid_comparison.png"):
    """
    Horizontal bar chart comparing system efficiency of converged fluids.
    Colour → safety/engineering flag (highest priority) | Hatch → fluid type
    """
    df_ok = df[df["status"] == "converged"].sort_values("system_efficiency [-]")
    if len(df_ok) == 0:
        print("Nothing to plot.")
        return None

    fig, ax = plt.subplots(figsize=(10, max(4, 0.45 * len(df_ok))))

    colors  = [_get_flag_color(f) for f in df_ok["flag"]]
    hatches = [_TYPE_HATCHES.get(t, "") for t in df_ok["fluid_type"]]

    bars = ax.barh(
        range(len(df_ok)),
        df_ok["system_efficiency [-]"] * 100,
        color=colors, edgecolor="k", lw=0.6,
    )
    for bar, h in zip(bars, hatches):
        bar.set_hatch(h)

    ax.set_yticks(range(len(df_ok)))
    ax.set_yticklabels([
        f"{r['fluid']}  ({r['fluid_type']}, Tc={r['Tc [°C]']:.0f}°C)"
        for _, r in df_ok.iterrows()
    ])
    ax.set_xlabel("System Efficiency [%]")
    ax.set_title("Working Fluid Comparison — Subcritical ORC", fontweight="bold")

    from matplotlib.patches import Patch
    legend_elements = [
        # ── Colour legend (flags) ──
        Patch(facecolor="#2980b9", edgecolor="k", label="No flag"),
        Patch(facecolor="#e74c3c", edgecolor="k", label="Flammable"),
        Patch(facecolor="#f39c12", edgecolor="k", label="Siloxane"),
        Patch(facecolor="#9b59b6", edgecolor="k", label="Thermal risk"),
        Patch(facecolor="#8e44ad", edgecolor="k", label="Stability unknown"),
        Patch(facecolor="#1abc9c", edgecolor="k", label="Deep vacuum (<0.1 bar)"),
        Patch(facecolor="#85c1e9", edgecolor="k", label="Sub-atmospheric"),
        Patch(facecolor="#e67e22", edgecolor="k", label="Near critical"),
        # ── Hatch legend (fluid type) ──
        Patch(facecolor="white",  edgecolor="k", hatch="//",  label="Wet"),
        Patch(facecolor="white",  edgecolor="k", hatch="",    label="Dry"),
        Patch(facecolor="white",  edgecolor="k", hatch="...", label="Isentropic"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=7,
              title="Colour = flag (priority) | Hatch = fluid type", title_fontsize=7)

    fig.tight_layout()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, filename)
        fig.savefig(path, dpi=300)
        print(f"  Saved plot: {path}")

    return fig
