"""
Working Fluid Sweep — Transcritical ORC  (v1)

Adapted from fluid_sweep_BASIC_ORC.py (subcritical, v4).

Key differences from the subcritical sweep:
───────────────────────────────────────────
- INVERTED Tc filter: Tc must be BELOW T_heat_source (so the cycle can go
  supercritical) and ABOVE T_cond + margin (so the fluid can still condense).
  This is the opposite of the subcritical filter (Tc > T_evap + margin).
- SUPERHEAT validation removed: superheating is not meaningful at
  supercritical expander inlet pressures.
- NEAR-CRITICAL flag replaced with LOW_HEADROOM flag for fluids whose Tc
  is close to T_heat_source (limited supercritical headroom).
- HIGH_PUMP_PRESSURE flag for fluids with very high critical pressures
  (pump must compress above pc, which may be impractical).
- Plot title updated to "Transcritical ORC".
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
#  BANNED & FLAGGED FLUIDS  (same as subcritical)
# ══════════════════════════════════════════════════════════════════════

BANNED = {
    # ── Ozone-depleting (ODP > 0, Montreal Protocol) ──
    "R11",
    "R12",
    "R13",
    "R113",
    "R114",
    "R115",
    "R123",
    "R141b",
    "R142b",
    "R22",
    # ── High GWP ≥ 150 (EU 517/2014 & Kigali) ────
    "R134a",
    "R125",
    "R143a",
    "R227EA",
    "R236EA",
    "R236FA",
    "R245fa",
    "R365MFC",
    "R32",
    "RC318",
    "R116",
    "R218",
    "R23",
    "R41",
    "SulfurHexafluoride",
    # ── Toxic or dangerous ────
    "CarbonMonoxide",
    "HydrogenSulfide",
    "SulfurDioxide",
    "NitrousOxide",
    "Methanol",
    "Ethanol",
    "Ammonia",
    # ── Cryogens ─────────
    "Helium",
    "Neon",
    "Argon",
    "Krypton",
    "Xenon",
    "Hydrogen",
    "Nitrogen",
    "Oxygen",
    "Fluorine",
    "ParaHydrogen",
    "OrthoHydrogen",
    "Deuterium",
    "ParaDeuterium",
    "OrthoDeuterium",
    "HeavyWater",
    "Air",
    # ── Other unsuitable ─────────
    "CarbonDioxide",
    "Acetone",
    "DiethylEther",
    "Ethylene",
    "EthyleneOxide",
    "CarbonylSulfide",
    # ── Blends (no .critical_point in CoolProp) ──
    "R507A",
    "R410A",
    "R404A",
    "R407C",
    "SES36",
}

FLAGGED = {
    "Propane": "flammable",
    "Butane": "flammable",
    "Isobutane": "flammable",
    "Isopentane": "flammable",
    "Neopentane": "flammable",
    "Pentane": "flammable",
    "Isohexane": "flammable",
    "Hexane": "flammable",
    "Heptane": "flammable",
    "CycloHexane": "flammable",
    "CycloPropane": "flammable",
    "Toluene": "flammable",
    "EthylBenzene": "flammable",
    "m-Xylene": "flammable",
    "o-Xylene": "flammable",
    "p-Xylene": "flammable",
    "Benzene": "toxic",  # IARC Group 1 carcinogen
    "MDM": "siloxane",
    "MM": "siloxane",
    "MD2M": "siloxane",
    "MD3M": "siloxane",
    "MD4M": "siloxane",
    "D4": "siloxane",
    "D5": "siloxane",
    "D6": "siloxane",
}


# ══════════════════════════════════════════════════════════════════════
#  THERMAL STABILITY LIMITS  (same as subcritical)
# ══════════════════════════════════════════════════════════════════════

THERMAL_STABILITY_K = {
    # Hydrocarbons
    "Toluene": 623,
    "CycloHexane": 573,
    "Benzene": 573,
    "Pentane": 573,
    "Isopentane": 573,
    "Neopentane": 573,
    "Hexane": 573,
    "Isohexane": 573,
    "Heptane": 573,
    "Octane": 573,
    "Butane": 573,
    "Isobutane": 573,
    "Propane": 623,
    "CycloPropane": 573,
    "EthylBenzene": 623,
    "m-Xylene": 623,
    "o-Xylene": 623,
    "p-Xylene": 623,
    "Nonane": 573,
    "Decane": 573,
    "Undecane": 573,
    "Dodecane": 573,
    # Siloxanes
    "MM": 573,
    "MDM": 573,
    "MD2M": 573,
    "MD3M": 573,
    "MD4M": 573,
    "D4": 573,
    "D5": 573,
    "D6": 573,
    # Refrigerants
    "R1233zdE": 473,
    "R1234zeZ": 473,
    "R1234zeE": 443,
    "R1234yf": 423,
}


# ══════════════════════════════════════════════════════════════════════
#  FLUID CLASSIFICATION  (same as subcritical)
# ══════════════════════════════════════════════════════════════════════


def classify_fluid(name):
    """Classifies fluid as wet, dry, or isentropic based on sat. vapour slope."""
    try:
        Tc = cp.PropsSI("Tcrit", name)
        T = max(0.7 * Tc, cp.PropsSI("Tmin", name) + 20)
        dT = 2.0
        s_lo = cp.PropsSI("S", "T", T, "Q", 1, name)
        s_hi = cp.PropsSI("S", "T", T + dT, "Q", 1, name)
        ds_dT = (s_hi - s_lo) / dT
        if abs(s_lo) < 1e-6:
            return "unknown"
        xi = (T / s_lo) * ds_dT
        if xi > 0.1:
            return "dry"
        elif xi < -0.1:
            return "wet"
        else:
            return "isentropic"
    except Exception:
        return "unknown"


# ══════════════════════════════════════════════════════════════════════
#  CANDIDATE SELECTION — TRANSCRITICAL (inverted Tc filter)
# ══════════════════════════════════════════════════════════════════════


def get_candidate_fluids(
    config_file,
    Tc_margin_above_cond=20,
    Tc_margin_below_hs=15,
    min_cond_pressure=0,
    thermal_stability_margin=30,
    max_pump_pressure_bar=150,
    low_headroom_margin=20,
):
    """
    Filter CoolProp fluids for TRANSCRITICAL ORC compatibility.

    The Tc filter is INVERTED compared to subcritical:
      Tc > T_cond + Tc_margin_above_cond   (must be able to condense)
      Tc < T_heat_source - Tc_margin_below_hs  (must be able to go supercritical)

    Parameters
    ----------
    config_file : str or Path
        Path to the thermopt YAML configuration file.
    Tc_margin_above_cond : float
        Minimum gap between Tc and condensation temperature [K].
        Prevents near-critical condensation (poor heat transfer, CoolProp instability).
    Tc_margin_below_hs : float
        Minimum gap between Tc and heat source temperature [K].
        Ensures there is supercritical headroom.
    min_cond_pressure : float
        Minimum acceptable condensing pressure [Pa].
    thermal_stability_margin : float
        If T_stability < T_heat_source − this margin → reject [K].
    max_pump_pressure_bar : float
        Flag fluids whose critical pressure exceeds this [bar].
        (Pump must compress above pc — very high pc is impractical.)
    low_headroom_margin : float
        Flag if T_heat_source − Tc < this value [K].
    """
    # ── Extract parameters from YAML ──────────────────────────────
    cfg = yaml.safe_load(Path(config_file).read_text())
    fixed = cfg["problem_formulation"]["fixed_parameters"]
    dv = cfg["problem_formulation"]["design_variables"]

    T_heat_source = fixed["heat_source"]["inlet_temperature"]
    T_heat_sink = fixed["heat_sink"]["inlet_temperature"]
    T_sink_exit_max = dv["heat_sink_exit_temperature"]["max"]

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
    T_cond_low = T_heat_sink + cooler_pinch_min
    T_cond_high = T_sink_exit_max - cooler_pinch_min
    Tc_min_K = T_cond_high + Tc_margin_above_cond  # must condense at highest T_cond
    Tc_max_K = T_heat_source - Tc_margin_below_hs  # must go supercritical

    print(f"\n  TRANSCRITICAL FILTER SETTINGS (from {Path(config_file).name})")
    print(f"    T_heat_source     = {T_heat_source - 273.15:.1f} °C")
    print(f"    T_heat_sink       = {T_heat_sink - 273.15:.1f} °C")
    print(
        f"    T_cond range      = [{T_cond_low - 273.15:.1f}, {T_cond_high - 273.15:.1f}] °C"
    )
    print(
        f"    Tc window         = [{Tc_min_K - 273.15:.1f}, {Tc_max_K - 273.15:.1f}] °C"
    )
    print(
        f"      (Tc > T_cond_high + {Tc_margin_above_cond} K  AND  Tc < T_hs − {Tc_margin_below_hs} K)"
    )
    print(f"    min_cond_pressure = {min_cond_pressure/1e5:.4f} bar")
    print(f"    max_pump_pressure = {max_pump_pressure_bar:.0f} bar (flag threshold)")

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
            pc_Pa = cp.PropsSI("pcrit", name)
            p_cond_low = cp.PropsSI("P", "T", T_cond_low, "Q", 0, name)
        except Exception:
            continue

        # ── TRANSCRITICAL Tc window ───────────────────────────────
        # Tc must be LOW enough to go supercritical with available heat
        if Tc_K > Tc_max_K:
            continue

        # Tc must be HIGH enough to condense at the highest condenser temp
        if Tc_K < Tc_min_K:
            continue

        # ── Condensing pressure checks (same as subcritical) ──────
        if p_cond_low < min_cond_pressure:
            continue

        try:
            p_cond_high = cp.PropsSI("P", "T", T_cond_high, "Q", 0, name)
        except Exception:
            print(
                f"    ⊘ {name}: CoolProp fails at T_cond_high={T_cond_high:.1f} K — skipped"
            )
            continue

        if p_cond_high < min_cond_pressure:
            continue

        # ── Thermal stability check ───────────────────────────────
        T_stability = THERMAL_STABILITY_K.get(name)
        thermal_flag = None

        if T_stability is not None:
            if T_stability < T_heat_source - thermal_stability_margin:
                print(
                    f"    ⊘ {name}: T_stability={T_stability-273.15:.0f}°C "
                    f"<< T_hs={T_heat_source-273.15:.0f}°C — rejected (thermal decomposition)"
                )
                continue
            elif T_stability < T_heat_source:
                thermal_flag = "thermal_risk"
        else:
            if Tc_K > 500:
                thermal_flag = "stability_unknown"

        # ── Vacuum condenser flag ─────────────────────────────────
        vacuum_flag = None
        if p_cond_low < 10_000:
            vacuum_flag = "deep_vacuum"
        elif p_cond_low < 101_325:
            vacuum_flag = "sub_atm"

        # ── Low supercritical headroom flag ────────────────────────
        headroom_flag = None
        if T_heat_source - Tc_K < low_headroom_margin:
            headroom_flag = "low_headroom"

        # ── High pump pressure flag ───────────────────────────────
        pump_flag = None
        if pc_Pa / 1e5 > max_pump_pressure_bar:
            pump_flag = "high_pump_pressure"

        # ── Combine all flags ─────────────────────────────────────
        flag = FLAGGED.get(name)
        extra_flags = [
            f for f in (thermal_flag, vacuum_flag, headroom_flag, pump_flag) if f
        ]
        if flag and extra_flags:
            flag = flag + "+" + "+".join(extra_flags)
        elif extra_flags:
            flag = "+".join(extra_flags)

        candidates.append(
            {
                "name": name,
                "Tc_C": Tc_K - 273.15,
                "pc_bar": pc_Pa / 1e5,
                "p_cond_low_bar": p_cond_low / 1e5,
                "p_cond_high_bar": p_cond_high / 1e5,
                "T_stability_C": (T_stability - 273.15) if T_stability else None,
                "flag": flag,
                "fluid_type": classify_fluid(name),
            }
        )

    candidates.sort(key=lambda f: f["Tc_C"])

    print(
        f"\n  CANDIDATES | {Tc_min_K - 273.15:.1f}°C < Tc < {Tc_max_K - 273.15:.1f}°C "
        f"| {len(candidates)} fluids"
    )
    for f in candidates:
        stab = f"  T_stab={f['T_stability_C']:.0f}°C" if f["T_stability_C"] else ""
        print(
            f"    {f['name']:<18} Tc={f['Tc_C']:>6.1f}°C  pc={f['pc_bar']:>5.1f} bar  "
            f"p_cond=[{f['p_cond_low_bar']:.3f}, {f['p_cond_high_bar']:.3f}] bar  "
            f"{f['fluid_type']:<12} {f['flag'] or ''}{stab}"
        )
    print()
    return candidates


# ══════════════════════════════════════════════════════════════════════
#  CONFIG FILE MANIPULATION  (same string-replacement approach)
# ══════════════════════════════════════════════════════════════════════


def _make_tmp_config(raw_yaml, fluid_name, expander_pressure_fraction=None):
    """
    Create a modified YAML config by doing string replacement on the
    raw text. Preserves all $working_fluid.* expressions exactly.
    """
    # ── Replace working fluid name ────────────────────────────────
    out = re.sub(
        r"(working_fluid:\s*\n\s*name:\s*)\S+",
        rf"\g<1>{fluid_name}",
        raw_yaml,
        count=1,
    )

    # ── Optionally replace expander_inlet_pressure starting value ─
    if expander_pressure_fraction is not None:
        out = re.sub(
            r"(expander_inlet_pressure:.*?value:\s*)"
            r"[\d.]+(\s*\*\s*\$working_fluid\.critical_point\.p)",
            rf"\g<1>{expander_pressure_fraction}\2",
            out,
            count=1,
            flags=re.DOTALL,
        )

    return out


# ══════════════════════════════════════════════════════════════════════
#  POST-CONVERGENCE VALIDATION — TRANSCRITICAL
# ══════════════════════════════════════════════════════════════════════


def _validate_constraints(cycle, slsqp_tol=0.5):
    """
    Check that key physical constraints hold after optimization.
    Returns (is_valid, actuals, violations).

    TRANSCRITICAL change: superheat is NOT validated because at
    supercritical expander inlet pressures, CoolProp's superheating
    value is not meaningful. We still READ it for logging but do not
    treat it as a constraint violation.
    """
    limits = {
        "heater_pinch": 5.0,  # K
        "cooler_pinch": 5.0,  # K
        "subcooling": 1.0,  # K
        # No superheating limit for transcritical
    }

    cd = cycle.problem.cycle_data
    exp_in = cd["components"]["expander"]["state_in"]

    # Read superheating for logging, but handle NaN/errors gracefully
    try:
        sh_val = float(exp_in.superheating)
    except Exception:
        sh_val = np.nan

    actuals = {
        "heater_pinch": float(
            np.min(cd["components"]["heater"]["temperature_difference"])
        ),
        "cooler_pinch": float(
            np.min(cd["components"]["cooler"]["temperature_difference"])
        ),
        "subcooling": float(cd["components"]["compressor"]["state_in"].subcooling),
        "superheating": sh_val,  # logged but not enforced
    }

    # Check expander inlet pressure vs critical pressure to confirm transcritical
    try:
        p_exp_in = float(exp_in.pressure)
        fluid_name = cd["components"]["expander"]["state_in"].fluid_name
        pc = cp.PropsSI("pcrit", fluid_name)
        actuals["p_exp_in_bar"] = p_exp_in / 1e5
        actuals["pc_bar"] = pc / 1e5
        actuals["supercritical"] = p_exp_in > pc
    except Exception:
        actuals["supercritical"] = "unknown"

    # Only validate pinch and subcooling (NOT superheating)
    violations = {
        k: v
        for k, v in actuals.items()
        if k in limits and (np.isnan(v) or v < limits[k] - slsqp_tol)
    }

    sc_str = "SC" if actuals.get("supercritical") is True else "sub"
    print(
        f"      pinch=[{actuals['heater_pinch']:.2f}, {actuals['cooler_pinch']:.2f}] K  "
        f"subcool={actuals['subcooling']:.2f} K  "
        f"superheat={actuals['superheating']:.2f} K  [{sc_str}]"
    )

    return len(violations) == 0, actuals, violations


# ══════════════════════════════════════════════════════════════════════
#  DESIGN VARIABLE EXTRACTION
# ══════════════════════════════════════════════════════════════════════


def _extract_design_variables(cycle):
    """Pull optimised state points for the results table."""
    cd = cycle.problem.cycle_data
    exp_in = cd["components"]["expander"]["state_in"]
    exp_out = cd["components"]["expander"]["state_out"]
    comp_in = cd["components"]["compressor"]["state_in"]

    # Superheating may be NaN for supercritical states
    try:
        sh_in = float(exp_in.superheating)
    except Exception:
        sh_in = np.nan
    try:
        sh_out = float(exp_out.superheating)
    except Exception:
        sh_out = np.nan

    return {
        "expander_inlet_T_K": float(exp_in.temperature),
        "expander_inlet_p_bar": float(exp_in.pressure) / 1e5,
        "expander_inlet_superheat_K": sh_in,
        "expander_outlet_T_K": float(exp_out.temperature),
        "expander_outlet_p_bar": float(exp_out.pressure) / 1e5,
        "expander_outlet_superheat_K": sh_out,
        "pump_inlet_T_K": float(comp_in.temperature),
        "pump_inlet_p_bar": float(comp_in.pressure) / 1e5,
        "pump_inlet_subcooling_K": float(comp_in.subcooling),
        "heater_pinch_K": float(
            np.min(cd["components"]["heater"]["temperature_difference"])
        ),
        "cooler_pinch_K": float(
            np.min(cd["components"]["cooler"]["temperature_difference"])
        ),
    }


# ══════════════════════════════════════════════════════════════════════
#  MAIN SWEEP — TRANSCRITICAL
# ══════════════════════════════════════════════════════════════════════


def run_fluid_sweep(
    config_file,
    candidates,
    output_dir="results/fluid_sweep_TRANSCRITICAL",
    save_results=True,
):
    """
    For each candidate fluid, run a single optimisation using the
    transcritical YAML template, then validate constraints.
    """
    raw_yaml = Path(config_file).read_text()

    os.makedirs(output_dir, exist_ok=True)
    results = []

    for i, fluid in enumerate(candidates):
        name = fluid["name"]
        stab_str = (
            f", T_stab={fluid['T_stability_C']:.0f}°C"
            if fluid.get("T_stability_C")
            else ""
        )
        print(
            f"\n  [{i+1}/{len(candidates)}] {name}  "
            f"(Tc={fluid['Tc_C']:.1f}°C, pc={fluid['pc_bar']:.1f} bar, "
            f"{fluid['fluid_type']}{stab_str})"
        )

        tmp_yaml = _make_tmp_config(raw_yaml, name)
        tmp_path = os.path.join(output_dir, f"_tmp_{name}.yaml")
        Path(tmp_path).write_text(tmp_yaml)

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
                is_valid, actuals, violations = _validate_constraints(cycle)

                ea = cycle.problem.cycle_data["energy_analysis"]
                eta = ea["system_efficiency"]

                if not is_valid:
                    viol_parts = []
                    for k, v in violations.items():
                        try:
                            viol_parts.append(f"{k}={float(v):.2f}")
                        except (TypeError, ValueError):
                            viol_parts.append(f"{k}={v}")
                    viol_str = ", ".join(viol_parts)
                    row["status"] = f"infeasible: {viol_str}"
                    row["system_efficiency [-]"] = eta
                    print(f"    ⚠ converged but INFEASIBLE: {viol_str}")
                    print(f"      η_sys={eta*100:.2f}% (marked infeasible)")
                else:
                    if save_results:
                        cycle.save_results()

                    if "net_system_power" in ea:
                        net_power = ea["net_system_power"]
                    elif "net_cycle_power" in ea:
                        net_power = ea["net_cycle_power"]
                        print(f"    ⚠ using 'net_cycle_power' (no 'net_system_power')")
                    else:
                        net_power = np.nan

                    row.update(
                        {
                            "status": "converged",
                            "system_efficiency [-]": eta,
                            "cycle_efficiency [-]": ea["cycle_efficiency"],
                            "net_power [kW]": (
                                net_power / 1e3 if not np.isnan(net_power) else np.nan
                            ),
                            "mass_flow [kg/s]": ea["mass_flow_working_fluid"],
                            "backwork_ratio [-]": ea["backwork_ratio"],
                            "heater_pinch [K]": actuals["heater_pinch"],
                            "cooler_pinch [K]": actuals["cooler_pinch"],
                            "subcooling [K]": actuals["subcooling"],
                            "superheating [K]": actuals["superheating"],
                            "supercritical": actuals.get("supercritical", "unknown"),
                        }
                    )
                    row.update(_extract_design_variables(cycle))

                    sc_str = "SC" if actuals.get("supercritical") else "sub"
                    print(
                        f"    ✓ η_sys={eta*100:.2f}%  "
                        f"pinch=[{actuals['heater_pinch']:.1f}, {actuals['cooler_pinch']:.1f}] K  "
                        f"subcool={actuals['subcooling']:.1f} K  [{sc_str}]"
                    )

        except Exception as e:
            row["status"] = f"error: {str(e)[:60]}"
            print(f"    ✗ {row['status']}")

        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            plt.close("all")

        results.append(row)

    # ── Build DataFrame, save ─────────────────────────────────────
    df = (
        pd.DataFrame(results)
        .sort_values("system_efficiency [-]", ascending=False, na_position="last")
        .reset_index(drop=True)
    )
    xlsx = os.path.join(output_dir, "fluid_sweep_results.xlsx")
    df.to_excel(xlsx, index=False)

    # ── Ranked summary ────────────────────────────────────────────
    conv = df[df["status"] == "converged"]
    print(f"\n  RESULTS | ✓ {len(conv)}  ✗ {len(df)-len(conv)}")
    for r, (_, row) in enumerate(conv.iterrows(), 1):
        flag_str = f"  ⚠ {row['flag']}" if row["flag"] else ""
        sc_str = "  [SC]" if row.get("supercritical") is True else ""
        print(
            f"    {r}. {row['fluid']:<16} η_sys={row['system_efficiency [-]']*100:.2f}%  "
            f"Tc={row['Tc [°C]']:.0f}°C  pc={row['pc [bar]']:.1f}bar  "
            f"{row['fluid_type']}{flag_str}{sc_str}"
        )
    for _, row in df[df["status"] != "converged"].iterrows():
        print(f"    ✗ {row['fluid']:<16} {row['status']}")
    print(f"  Saved: {xlsx}\n")

    return df


# ══════════════════════════════════════════════════════════════════════
#  PLOTTING
# ══════════════════════════════════════════════════════════════════════

_FLAG_COLORS = {
    None: "#2980b9",  # blue  — no flag
    "flammable": "#e74c3c",  # red
    "toxic": "#c0392b",  # dark red
    "siloxane": "#f39c12",  # amber
    "thermal_risk": "#9b59b6",  # purple
    "stability_unknown": "#8e44ad",  # dark purple
    "deep_vacuum": "#1abc9c",  # teal
    "sub_atm": "#85c1e9",  # light blue
    "low_headroom": "#e67e22",  # orange
    "high_pump_pressure": "#d35400",  # dark orange
}

_TYPE_HATCHES = {
    "wet": "//",
    "dry": "",
    "isentropic": "...",
    "unknown": "xx",
}


def _get_flag_color(flag):
    """Handle combined flags — priority order for colour selection."""
    if flag is None:
        return _FLAG_COLORS[None]
    flag_str = str(flag)
    for key in (
        "toxic",
        "thermal_risk",
        "stability_unknown",
        "low_headroom",
        "high_pump_pressure",
        "deep_vacuum",
        "sub_atm",
        "flammable",
        "siloxane",
    ):
        if key in flag_str:
            return _FLAG_COLORS[key]
    return "#999999"


def plot_results(df, output_dir=None, filename="fluid_comparison_transcritical.png"):
    """
    Horizontal bar chart comparing system efficiency of converged fluids.
    """
    df_ok = df[df["status"] == "converged"].sort_values("system_efficiency [-]")
    if len(df_ok) == 0:
        print("Nothing to plot.")
        return None

    fig, ax = plt.subplots(figsize=(10, max(4, 0.45 * len(df_ok))))

    colors = [_get_flag_color(f) for f in df_ok["flag"]]
    hatches = [_TYPE_HATCHES.get(t, "") for t in df_ok["fluid_type"]]

    bars = ax.barh(
        range(len(df_ok)),
        df_ok["system_efficiency [-]"] * 100,
        color=colors,
        edgecolor="k",
        lw=0.6,
    )
    for bar, h in zip(bars, hatches):
        bar.set_hatch(h)

    ax.set_yticks(range(len(df_ok)))
    ax.set_yticklabels(
        [
            f"{r['fluid']}  ({r['fluid_type']}, Tc={r['Tc [°C]']:.0f}°C, pc={r['pc [bar]']:.0f}bar)"
            for _, r in df_ok.iterrows()
        ]
    )
    ax.set_xlabel("System Efficiency [%]")
    ax.set_title("Working Fluid Comparison — Transcritical ORC", fontweight="bold")

    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="#2980b9", edgecolor="k", label="No flag"),
        Patch(facecolor="#e74c3c", edgecolor="k", label="Flammable"),
        Patch(facecolor="#9b59b6", edgecolor="k", label="Thermal risk"),
        Patch(facecolor="#8e44ad", edgecolor="k", label="Stability unknown"),
        Patch(facecolor="#e67e22", edgecolor="k", label="Low SC headroom"),
        Patch(facecolor="#d35400", edgecolor="k", label="High pump pressure"),
        Patch(facecolor="#1abc9c", edgecolor="k", label="Deep vacuum (<0.1 bar)"),
        Patch(facecolor="#85c1e9", edgecolor="k", label="Sub-atmospheric"),
        Patch(facecolor="white", edgecolor="k", hatch="//", label="Wet"),
        Patch(facecolor="white", edgecolor="k", hatch="", label="Dry"),
        Patch(facecolor="white", edgecolor="k", hatch="...", label="Isentropic"),
    ]
    ax.legend(
        handles=legend_elements,
        loc="lower right",
        fontsize=7,
        title="Colour = flag (priority) | Hatch = fluid type",
        title_fontsize=7,
    )

    fig.tight_layout()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, filename)
        fig.savefig(path, dpi=300)
        print(f"  Saved plot: {path}")

    return fig
