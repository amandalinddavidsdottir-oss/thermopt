"""
Parametric study: Macchi & Astolfi turbine correlation
======================================================

Runs the optimizer for all combinations of:
  - n_stages: [1, 2, 3]
  - RPM: [3000, 1500]

for one or more YAML config files. Collects results into a comparison
table saved as both CSV and Excel.

Works with all cycle topologies:
  - Simple ORC
  - Recuperated ORC
  - Dual-pressure ORC (both HP and LP expanders updated)
  - Recuperated dual-pressure ORC

Usage:
    python run_parametric_study.py

Configure the YAML files and parameter grid below.

Author : Amanda (Master Thesis)
"""

import os
import re
import sys
import warnings
import tempfile
import traceback
from pathlib import Path
from datetime import datetime

import pandas as pd
import thermopt as th

# ══════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════

# YAML config files to run (one per cycle topology)
CONFIG_FILES = [
    Path(__file__).with_name("case_Toluene_simpleORC_macchi_astolfi.yaml"),
    Path(__file__).with_name("case_Toluene_recuperated_simpleORC_macchi_astolfi.yaml"),
    # Path(__file__).with_name("case_Toluene_dual_pressure_macchi_astolfi.yaml"),
    # Path(__file__).with_name("case_Toluene_recuperated_dual_pressure_macchi_astolfi.yaml"),
]

# Parameter grid
STAGES = [1, 2, 3]
RPMS = [3000, 1500]

# Output directory for the comparison table
OUTPUT_DIR = Path(__file__).parent / "results" / "parametric_study"

# ══════════════════════════════════════════════════════════════════════
#  YAML MODIFICATION
# ══════════════════════════════════════════════════════════════════════

def modify_yaml(yaml_text, n_stages, RPM):
    """
    Replace n_stages and RPM values in the raw YAML text.

    Uses regex to preserve all other content (expressions, comments, etc.).
    Works for both single-expander and dual-pressure configs (all occurrences
    of n_stages and RPM are replaced).
    """
    yaml_text = re.sub(
        r'(n_stages:\s*)\d+',
        rf'\g<1>{n_stages}',
        yaml_text,
    )
    yaml_text = re.sub(
        r'(RPM:\s*)\d+',
        rf'\g<1>{RPM}',
        yaml_text,
    )
    return yaml_text


# ══════════════════════════════════════════════════════════════════════
#  RESULT EXTRACTION
# ══════════════════════════════════════════════════════════════════════

def extract_results(cycle, config_file, n_stages, RPM):
    """
    Extract key results from a converged cycle into a flat dict.
    """
    data = cycle.problem.cycle_data
    components = data["components"]
    energy = data["energy_analysis"]

    # Detect topology from YAML filename or cycle data
    config_name = Path(config_file).stem

    result = {
        "config": config_name,
        "n_stages": n_stages,
        "RPM": RPM,
        "converged": True,
    }

    # ── System-level metrics ──
    result["eta_system"] = energy.get("system_efficiency", None)
    result["eta_cycle"] = energy.get("cycle_efficiency", None)
    result["W_net_kW"] = energy.get("net_system_power", 0) / 1e3
    result["W_gross_kW"] = energy.get("gross_power", 0) / 1e3
    result["Q_in_kW"] = energy.get("total_heat_input",
                                    energy.get("heater_heat_flow", 0)) / 1e3
    result["Q_available_kW"] = energy.get("available_heat", 0) / 1e3
    result["heat_utilization"] = (result["Q_in_kW"] / result["Q_available_kW"]
                                   if result["Q_available_kW"] > 0 else None)
    result["m_dot_wf_kg_s"] = energy.get("mass_flow_working_fluid", None)
    result["m_dot_well_kg_s"] = energy.get("mass_flow_heating_fluid", None)

    # ── Expander(s) ──
    if "hp_expander" in components:
        # Dual-pressure: extract both HP and LP
        for prefix, exp_name in [("HP", "hp_expander"), ("LP", "lp_expander")]:
            exp = components[exp_name]
            data_out = exp.get("data_out", {})
            result[f"{prefix}_eta_turbine"] = exp.get("efficiency", None)
            result[f"{prefix}_SP"] = data_out.get("size_parameter", None)
            result[f"{prefix}_SP_clamped"] = data_out.get("size_parameter_clamped", False)
            result[f"{prefix}_Vr"] = data_out.get("volume_ratio", None)
            result[f"{prefix}_Ns"] = data_out.get("specific_speed", None)
            result[f"{prefix}_Dh_is_kJ_kg"] = exp.get("isentropic_work", 0) / 1e3
            result[f"{prefix}_p_in_bar"] = exp["state_in"].p / 1e5
            result[f"{prefix}_p_out_bar"] = exp["state_out"].p / 1e5
    else:
        # Single expander
        exp = components["expander"]
        data_out = exp.get("data_out", {})
        result["eta_turbine"] = exp.get("efficiency", None)
        result["SP"] = data_out.get("size_parameter", None)
        result["SP_clamped"] = data_out.get("size_parameter_clamped", False)
        result["Vr"] = data_out.get("volume_ratio", None)
        result["Ns"] = data_out.get("specific_speed", None)
        result["Dh_is_kJ_kg"] = exp.get("isentropic_work", 0) / 1e3
        result["p_in_bar"] = exp["state_in"].p / 1e5
        result["p_out_bar"] = exp["state_out"].p / 1e5

    # ── Recuperator (if present) ──
    if "recuperator" in components:
        Q_rec = energy.get("recuperator_heat_flow", 0) / 1e3
        result["Q_recuperator_kW"] = Q_rec

    return result


# ══════════════════════════════════════════════════════════════════════
#  SINGLE RUN
# ══════════════════════════════════════════════════════════════════════

def run_single(config_file, n_stages, RPM):
    """
    Run one optimization with specific n_stages and RPM.

    Returns a result dict on success, or a failure dict on error.
    """
    # Read and modify YAML
    yaml_text = Path(config_file).read_text()
    yaml_text = modify_yaml(yaml_text, n_stages, RPM)

    # Write to temp file in same directory (so relative paths work)
    config_dir = Path(config_file).parent
    tmp_path = config_dir / f"_tmp_parametric_n{n_stages}_rpm{RPM}.yaml"

    try:
        tmp_path.write_text(yaml_text)

        # Run optimization (no interactive plot in batch mode)
        cycle = th.ThermodynamicCycleOptimization(str(tmp_path))
        cycle.run_optimization()

        # Check convergence — try multiple attribute paths
        converged = True
        for attr_path in [
            lambda: cycle.success,
            lambda: cycle.result.success,
            lambda: cycle.solver_result.success,
        ]:
            try:
                converged = attr_path()
                break
            except AttributeError:
                continue

        if not converged:
            return {
                "config": Path(config_file).stem,
                "n_stages": n_stages,
                "RPM": RPM,
                "converged": False,
            }

        return extract_results(cycle, config_file, n_stages, RPM)

    except Exception as e:
        print(f"    ✗ FAILED: {e}")
        traceback.print_exc()
        return {
            "config": Path(config_file).stem,
            "n_stages": n_stages,
            "RPM": RPM,
            "converged": False,
            "error": str(e),
        }
    finally:
        # Clean up temp file
        if tmp_path.exists():
            tmp_path.unlink()


# ══════════════════════════════════════════════════════════════════════
#  BATCH RUNNER
# ══════════════════════════════════════════════════════════════════════

def run_parametric_study(config_files, stages, rpms, output_dir):
    """
    Run the full parametric study and save results.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_runs = len(config_files) * len(stages) * len(rpms)
    results = []
    run_count = 0

    print("=" * 76)
    print("  PARAMETRIC STUDY — Macchi & Astolfi Turbine Correlation")
    print("=" * 76)
    print(f"  Configs : {len(config_files)}")
    print(f"  Stages  : {stages}")
    print(f"  RPMs    : {rpms}")
    print(f"  Total   : {total_runs} optimization runs")
    print("=" * 76 + "\n")

    for config_file in config_files:
        config_name = Path(config_file).stem
        print(f"\n{'─' * 76}")
        print(f"  Config: {config_name}")
        print(f"{'─' * 76}")

        if not Path(config_file).exists():
            print(f"  ✗ File not found: {config_file}")
            continue

        for n_stages in stages:
            for RPM in rpms:
                run_count += 1
                print(f"\n  [{run_count}/{total_runs}] "
                      f"n_stages={n_stages}, RPM={RPM} ... ",
                      end="", flush=True)

                result = run_single(config_file, n_stages, RPM)
                results.append(result)

                if result.get("converged", False):
                    eta = result.get("eta_system", result.get("HP_eta_turbine", 0))
                    eta_pct = eta * 100 if eta else 0
                    W = result.get("W_net_kW", 0)
                    print(f"✓  η_sys = {eta_pct:.2f}%,  W_net = {W:.0f} kW")
                else:
                    print(f"✗  did not converge")

    # ── Save results ──
    df = pd.DataFrame(results)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    csv_path = output_dir / f"parametric_results_{timestamp}.csv"
    xlsx_path = output_dir / f"parametric_results_{timestamp}.xlsx"

    df.to_csv(csv_path, index=False)
    df.to_excel(xlsx_path, index=False, sheet_name="Results")

    # ── Print summary table ──
    print("\n\n" + "=" * 76)
    print("  PARAMETRIC STUDY — RESULTS SUMMARY")
    print("=" * 76)

    # Select key columns that exist
    display_cols = ["config", "n_stages", "RPM", "converged"]
    for col in ["eta_system", "eta_turbine", "W_net_kW", "SP", "Vr", "Ns",
                 "HP_eta_turbine", "LP_eta_turbine"]:
        if col in df.columns:
            display_cols.append(col)

    print(df[display_cols].to_string(index=False))

    print(f"\n  Results saved to:")
    print(f"    CSV  : {csv_path}")
    print(f"    Excel: {xlsx_path}")
    print("=" * 76 + "\n")

    return df


# ══════════════════════════════════════════════════════════════════════
#  RUN
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    warnings.filterwarnings("ignore", message="FigureCanvasAgg is non-interactive")

    # Suppress the SP warning during batch runs (reported in results table)
    warnings.filterwarnings("once", message="Macchi-Astolfi: SP exceeds")

    df = run_parametric_study(
        config_files=CONFIG_FILES,
        stages=STAGES,
        rpms=RPMS,
        output_dir=OUTPUT_DIR,
    )
