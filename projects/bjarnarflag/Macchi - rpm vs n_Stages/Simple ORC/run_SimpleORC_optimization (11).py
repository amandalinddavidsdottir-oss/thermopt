import os
import re
import sys
import warnings
import traceback
import thermopt as th
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from datetime import datetime

# ---- importing post processing files:
sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent.parent / "shared_utilities")
)
# sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "shared_utilities"))
from exergy_analysis import perform_exergy_analysis, plot_heat_source_utilization
from plot_TQ_diagram import plot_TQ_diagram
from validation_checks import run_validation_checks

# -----

warnings.filterwarnings("ignore", message="FigureCanvasAgg is non-interactive")


# ══════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════
#
#  Available modes:
#
#  "optimize"    — Run a single YAML through the full optimizer and
#                  post-processing pipeline (exergy, T-Q diagrams, graphs).
#                  Use this for one-off runs or debugging a specific case.
#                  → Set CONFIG_FILE.
#
#  "sweep"       — Run a fluid sweep across candidate fluids using one YAML
#                  as a template. Fluid name is substituted for each candidate;
#                  all other settings stay fixed. One results folder per fluid.
#                  → Set CONFIG_FILE and SWEEP_OUTPUT_DIR.
#
#  "batch_yaml"  — Run a list of pre-built YAMLs sequentially, each with RPM
#                  and n_stages already baked in. Collects all results into one
#                  combined CSV + Excel. Used for the n_stages × RPM study.
#                  → Set BATCH_YAML_CONFIGS and BATCH_YAML_OUTPUT_DIR.
#
#  "pcond_sweep" — Sweep over candidate working fluids, letting the optimizer
#                  freely find p_cond, p_evap, and m_wf for each one. Plots
#                  condensing pressure vs system efficiency by fluid family.
#                  → Set PCOND_TEMPLATE, PCOND_N_STAGES, PCOND_RPM, PCOND_FLUIDS.
#
MODE = "optimize"  # "optimize" | "sweep" | "batch_yaml" | "pcond_sweep"


# ────────────────────────────────────────── optimize & sweep ─────────────────────────────────────────────────────────────────────────────────────────────────────────


# Used by: optimize, sweep
# Set this to the YAML file you want to run (optimize) or use as a
# template with the fluid name swapped out for each candidate (sweep).
# CONFIG_FILE = Path(__file__).with_name("case_Toluene_simpleORC_macchi_astolfi.yaml")
# CONFIG_FILE = Path(__file__).with_name("case_Toluene_simpleORC_macchi_astolfi_n5_rpm1000.yaml")
CONFIG_FILE = Path(__file__).with_name(
    "case_Toluene_simpleORC_macchi_astolfi_n4_rpm1000.yaml"
)
# CONFIG_FILE = Path(__file__).with_name("case_Toluene_simpleORC_macchi_astolfi_n5_rpm1500.yaml")
# CONFIG_FILE = Path(__file__).with_name("case_Toluene_simpleORC_macchi_astolfi_n5_rpm3000.yaml")


# Used by: sweep
# Folder where fluid sweep results are saved.
SWEEP_OUTPUT_DIR = "results/fluid_sweep_BASIC_ORC"

# ──────────────────────────────────────────── batch_yaml: list of pre-built YAML files to run sequentially ───────────────────────────────────────────────
# Used by: batch_yaml
# Each YAML must have RPM and n_stages already set inside the file.
# Add, remove, or reorder entries here to change which cases are run.
# Results from all files are combined into one CSV + Excel output.
BATCH_YAML_CONFIGS = [
    Path(__file__).with_name("case_Toluene_simpleORC_macchi_astolfi_n1_rpm1000.yaml"),
    Path(__file__).with_name("case_Toluene_simpleORC_macchi_astolfi_n2_rpm1000.yaml"),
    Path(__file__).with_name("case_Toluene_simpleORC_macchi_astolfi_n3_rpm1000.yaml"),
    Path(__file__).with_name("case_Toluene_simpleORC_macchi_astolfi_n4_rpm1000.yaml"),
    Path(__file__).with_name("case_Toluene_simpleORC_macchi_astolfi_n5_rpm1000.yaml"),
    Path(__file__).with_name("case_Toluene_simpleORC_macchi_astolfi_n1_rpm1500.yaml"),
    Path(__file__).with_name("case_Toluene_simpleORC_macchi_astolfi_n2_rpm1500.yaml"),
    Path(__file__).with_name("case_Toluene_simpleORC_macchi_astolfi_n3_rpm1500.yaml"),
    Path(__file__).with_name("case_Toluene_simpleORC_macchi_astolfi_n4_rpm1500.yaml"),
    Path(__file__).with_name("case_Toluene_simpleORC_macchi_astolfi_n5_rpm1500.yaml"),
    Path(__file__).with_name("case_Toluene_simpleORC_macchi_astolfi_n1_rpm3000.yaml"),
    Path(__file__).with_name("case_Toluene_simpleORC_macchi_astolfi_n2_rpm3000.yaml"),
    Path(__file__).with_name("case_Toluene_simpleORC_macchi_astolfi_n3_rpm3000.yaml"),
    Path(__file__).with_name("case_Toluene_simpleORC_macchi_astolfi_n4_rpm3000.yaml"),
    Path(__file__).with_name("case_Toluene_simpleORC_macchi_astolfi_n5_rpm3000.yaml"),
]
BATCH_YAML_OUTPUT_DIR = Path(__file__).parent / "results" / "parametric_study"

# ──────────────────────────────────────────── pcond_sweep: fluid comparison settings ────────────────────────────────────────────────────────────────────────────────────
# Used by: pcond_sweep
# Set this to any single-fluid YAML configured for the correct heat source,
# heat sink, and turbine model. The fluid name is substituted for each
# candidate fluid in PCOND_FLUIDS — everything else stays fixed.
PCOND_TEMPLATE = Path(__file__).with_name("case_Toluene_simpleORC_macchi_astolfi.yaml")

# Number of turbine stages and shaft speed used for all fluids in the sweep.
PCOND_N_STAGES = 2
PCOND_RPM = 1500

# Folder where the pcond sweep results and plot are saved.
PCOND_OUTPUT_DIR = Path(__file__).parent / "results" / "pcond_vs_efficiency"

# Fluid candidates (name → category for plot colours).
# Add or remove fluids here. Category controls marker colour and shape in the plot.
PCOND_FLUIDS = {
    # Aromatics
    "Toluene": "aromatic",
    "EthylBenzene": "aromatic",
    "m-Xylene": "aromatic",
    "o-Xylene": "aromatic",
    "p-Xylene": "aromatic",
    # Cycloalkanes
    "CycloHexane": "cycloalkane",
    "CycloPentane": "cycloalkane",
    # Linear alkanes
    "Pentane": "alkane",
    "Isopentane": "alkane",
    "Hexane": "alkane",
    "Isohexane": "alkane",
    "Heptane": "alkane",
    "Octane": "alkane",
    "Nonane": "alkane",
    "Decane": "alkane",
    # Siloxanes
    "MM": "siloxane",
    "MDM": "siloxane",
    "MD2M": "siloxane",
    "MD3M": "siloxane",
    "MD4M": "siloxane",
    "D4": "siloxane",
    "D5": "siloxane",
    "D6": "siloxane",
    # Low-GWP refrigerants
    "R1233zdE": "refrigerant",
    "R1234zeE": "refrigerant",
    "R1234yf": "refrigerant",
}

PCOND_CATEGORY_STYLE = {
    "aromatic": {"color": "#e74c3c", "marker": "o"},
    "cycloalkane": {"color": "#8e44ad", "marker": "s"},
    "alkane": {"color": "#2980b9", "marker": "^"},
    "siloxane": {"color": "#f39c12", "marker": "D"},
    "refrigerant": {"color": "#27ae60", "marker": "v"},
}

# Manual label offsets (dx, dy in points) for clustered fluids
PCOND_LABEL_OFFSETS = {
    "Toluene": (10, 8),
    "m-Xylene": (-30, 10),
    "o-Xylene": (-30, -10),
    "p-Xylene": (10, 10),
    "EthylBenzene": (10, -18),
}


# ══════════════════════════════════════════════════════════════════════
#  MODE 1: SINGLE-FLUID OPTIMIZATION
# ══════════════════════════════════════════════════════════════════════
def run_optimize(config_file):
    """Run optimization + full post-processing for a single YAML config."""

    th.print_package_info()

    cycle = th.ThermodynamicCycleOptimization(config_file)
    cycle.problem.plot_cycle_realtime(config_file, update_interval=0.1)

    cycle.run_optimization()
    cycle.save_results()

    # ──────────────────────── VALIDATION CHECKS ───────────────────────
    run_validation_checks(cycle, config_file=config_file)

    # ──────────────────────── MASS FLOW RATES ─────────────────────────
    ea = cycle.problem.cycle_data["energy_analysis"]
    print("\n" + "=" * 76)
    print("  MASS FLOW RATES")
    print("=" * 76)
    print(
        f"  Well (heating fluid)               :  {ea['mass_flow_heating_fluid']:.2f} kg/s"
    )
    print(
        f"  Working fluid                      :  {ea['mass_flow_working_fluid']:.2f} kg/s"
    )
    print(
        f"  Cooling fluid                      :  {ea['mass_flow_cooling_fluid']:.2f} kg/s"
    )
    print("=" * 76 + "\n")

    # ── Post-processing ───────────────────────────────────────────
    graph_dir = os.path.join(cycle.out_dir, "graphs")
    os.makedirs(graph_dir, exist_ok=True)

    # Exergy analysis
    exergy = perform_exergy_analysis(cycle, config_file=config_file)
    exergy.print_summary()
    exergy.to_excel(os.path.join(graph_dir, "exergy_results.xlsx"))
    exergy.plot_exergy_destruction(savefig=os.path.join(graph_dir, "exergy_bar.png"))
    try:
        exergy.plot_pie_chart(savefig=os.path.join(graph_dir, "exergy_pie.png"))
    except ValueError as e:
        print(f"Skipping exergy pie chart: {e}")
    exergy.plot_grassmann(savefig=os.path.join(graph_dir, "exergy_grassmann.png"))

    # Heat source utilization curve
    fig, axes, sweep = plot_heat_source_utilization(
        cycle,
        config_file=config_file,
        savefig=os.path.join(graph_dir, "utilization_curve.png"),
    )

    # T-Q diagrams
    plot_TQ_diagram(cycle, "heater", output_dir=graph_dir)
    plot_TQ_diagram(cycle, "cooler", output_dir=graph_dir)

    # Summary
    print("\n" + "=" * 60)
    print("  ALL RESULTS SAVED")
    print("=" * 60)
    print(f"  Thermopt results : {cycle.out_dir}")
    print(f"  Graphs & exergy  : {graph_dir}")
    print()
    for f in sorted(os.listdir(graph_dir)):
        print(f"    \u2713 {f}")
    print("=" * 60 + "\n")


# ══════════════════════════════════════════════════════════════════════
#  MODE 2: WORKING FLUID SWEEP
# ══════════════════════════════════════════════════════════════════════
def run_sweep(config_file, output_dir):
    """Run fluid sweep across all candidates using config as template."""

    from fluid_sweep_BASIC_ORC import (
        get_candidate_fluids,
        run_fluid_sweep,
        plot_results,
    )

    candidates = get_candidate_fluids(config_file)
    df = run_fluid_sweep(config_file, candidates, output_dir=output_dir)
    plot_results(df, output_dir=output_dir)


def _extract_results(cycle, config_file, n_stages, RPM):
    """Extract key results from a converged cycle into a flat dict."""
    data = cycle.problem.cycle_data
    components = data["components"]
    energy = data["energy_analysis"]
    config_name = Path(config_file).stem

    result = {
        "config": config_name,
        "n_stages": n_stages,
        "RPM": RPM,
        "converged": True,
    }

    # System-level metrics
    result["eta_system"] = energy.get("system_efficiency", None)
    result["eta_cycle"] = energy.get("cycle_efficiency", None)
    result["W_net_kW"] = energy.get("net_system_power", 0) / 1e3
    result["W_gross_kW"] = energy.get("gross_power", 0) / 1e3
    result["Q_in_kW"] = (
        energy.get("total_heat_input", energy.get("heater_heat_flow", 0)) / 1e3
    )
    result["Q_available_kW"] = energy.get("available_heat", 0) / 1e3
    result["heat_utilization"] = (
        result["Q_in_kW"] / result["Q_available_kW"]
        if result["Q_available_kW"] > 0
        else None
    )
    result["m_dot_wf_kg_s"] = energy.get("mass_flow_working_fluid", None)
    result["m_dot_well_kg_s"] = energy.get("mass_flow_heating_fluid", None)

    # Expander(s)
    if "hp_expander" in components:
        for prefix, exp_name in [("HP", "hp_expander"), ("LP", "lp_expander")]:
            exp = components[exp_name]
            data_out = exp.get("data_out", {})
            result[f"{prefix}_eta_turbine"] = exp.get("efficiency", None)
            result[f"{prefix}_SP"] = data_out.get("size_parameter", None)
            result[f"{prefix}_SP_clamped"] = data_out.get(
                "size_parameter_clamped", False
            )
            result[f"{prefix}_Vr"] = data_out.get("volume_ratio", None)
            result[f"{prefix}_Ns"] = data_out.get("specific_speed", None)
            result[f"{prefix}_Ns_out_of_range"] = data_out.get(
                "specific_speed_out_of_range", False
            )
            result[f"{prefix}_Dh_is_kJ_kg"] = exp.get("isentropic_work", 0) / 1e3
            result[f"{prefix}_p_in_bar"] = exp["state_in"].p / 1e5
            result[f"{prefix}_p_out_bar"] = exp["state_out"].p / 1e5
            # Per-stage Ns values for detailed diagnostics
            stage_data = data_out.get("stage_data", [])
            result[f"{prefix}_stage_Ns"] = [s.get("Ns") for s in stage_data]
            result[f"{prefix}_stage_Ns_flags"] = [
                s.get("Ns_out_of_range", False) for s in stage_data
            ]
            result[f"{prefix}_splitting_method"] = data_out.get(
                "splitting_method", "equal_Vr"
            )
    else:
        exp = components["expander"]
        data_out = exp.get("data_out", {})
        result["eta_turbine"] = exp.get("efficiency", None)
        result["SP"] = data_out.get("size_parameter", None)
        result["SP_clamped"] = data_out.get("size_parameter_clamped", False)
        result["Vr"] = data_out.get("volume_ratio", None)
        result["Ns"] = data_out.get("specific_speed", None)
        result["Ns_out_of_range"] = data_out.get("specific_speed_out_of_range", False)
        result["Dh_is_kJ_kg"] = exp.get("isentropic_work", 0) / 1e3
        result["p_in_bar"] = exp["state_in"].p / 1e5
        result["p_out_bar"] = exp["state_out"].p / 1e5
        # Per-stage breakdown
        stage_data = data_out.get("stage_data", [])
        result["stage_Ns"] = [s.get("Ns") for s in stage_data]
        result["stage_Ns_flags"] = [s.get("Ns_out_of_range", False) for s in stage_data]
        result["stage_Vr"] = [s.get("Vr") for s in stage_data]
        result["stage_eta"] = [s.get("eta_stage") for s in stage_data]
        result["stage_SP"] = [s.get("SP") for s in stage_data]
        result["stage_Dh_is_kJ_kg"] = [
            s.get("Dh_is", 0) / 1e3 if s.get("Dh_is") is not None else None
            for s in stage_data
        ]
        result["stage_p_in_bar"] = [
            s.get("p_in", 0) / 1e5 if s.get("p_in") is not None else None
            for s in stage_data
        ]
        result["stage_p_out_bar"] = [
            s.get("p_out", 0) / 1e5 if s.get("p_out") is not None else None
            for s in stage_data
        ]
        result["splitting_method"] = data_out.get("splitting_method", "equal_Vr")
        # Flattened per-stage columns for easy reading in Excel
        for i, s in enumerate(stage_data, 1):
            result[f"s{i}_Vr"] = s.get("Vr")
            result[f"s{i}_Ns"] = s.get("Ns")
            result[f"s{i}_eta"] = s.get("eta_stage")
            result[f"s{i}_SP"] = s.get("SP")
            result[f"s{i}_Dh_is_kJ_kg"] = (
                s.get("Dh_is", 0) / 1e3 if s.get("Dh_is") is not None else None
            )
            result[f"s{i}_p_in_bar"] = (
                s.get("p_in", 0) / 1e5 if s.get("p_in") is not None else None
            )
            result[f"s{i}_p_out_bar"] = (
                s.get("p_out", 0) / 1e5 if s.get("p_out") is not None else None
            )

    # Recuperator (if present)
    if "recuperator" in components:
        Q_rec = energy.get("recuperator_heat_flow", 0) / 1e3
        result["Q_recuperator_kW"] = Q_rec

    return result


def _check_ns_validity(result):
    """
    Check whether any stage had Ns outside [0.045, 0.20].

    Uses the specific_speed_out_of_range flag set by the Astolfi component,
    which is True if ANY single stage exceeded the regression bounds.
    Drills into per-stage data to report exactly which stages were invalid.

    Returns a human-readable problem string, or None if everything is valid.
    """
    problems = []

    # Detect which expanders are present
    expanders = []
    if "Ns_out_of_range" in result:
        expanders.append(("", "expander"))
    for prefix in ["HP", "LP"]:
        if f"{prefix}_Ns_out_of_range" in result:
            expanders.append((f"{prefix}_", prefix))

    for key_prefix, label in expanders:
        if not result.get(f"{key_prefix}Ns_out_of_range", False):
            continue  # all stages valid for this expander

        # Identify which specific stages were out of range
        stage_ns = result.get(f"{key_prefix}stage_Ns", [])
        stage_flags = result.get(f"{key_prefix}stage_Ns_flags", [])

        bad_stages = []
        for i, (ns, flag) in enumerate(zip(stage_ns, stage_flags)):
            if flag:
                ns_str = f"{ns:.3f}" if ns is not None else "?"
                bad_stages.append(f"stage {i+1} (Ns={ns_str})")

        if bad_stages:
            problems.append(f"{label}: {', '.join(bad_stages)}")
        else:
            # Top-level flag set but no stage detail available
            problems.append(f"{label}: Ns out of range (no stage detail available)")

    return "; ".join(problems) if problems else None


def _run_single(config_file, n_stages, RPM):
    """Run one optimization for a specific YAML (already correct for n_stages) at a given RPM.
    Only RPM is patched — the YAML structure (intermediate pressures, constraints) is preserved.
    """
    yaml_text = Path(config_file).read_text()
    yaml_text = re.sub(r"(RPM:\s*)\d+", rf"\g<1>{RPM}", yaml_text)

    config_dir = Path(config_file).parent
    tmp_path = config_dir / f"_tmp_run_n{n_stages}_rpm{RPM}.yaml"

    try:
        tmp_path.write_text(yaml_text)

        cycle = th.ThermodynamicCycleOptimization(str(tmp_path))
        cycle.run_optimization()

        # Check convergence
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

        return _extract_results(cycle, config_file, n_stages, RPM)

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
        if tmp_path.exists():
            tmp_path.unlink()


# ══════════════════════════════════════════════════════════════════════
#  MODE 3: BATCH OPTIMIZATION (one result file for all YAMLs)
# ══════════════════════════════════════════════════════════════════════


def run_batch_yaml(config_files, output_dir):
    """
    Run optimization for each YAML in config_files and collect all results
    into a single CSV + Excel file.

    Each YAML must already have the correct RPM and n_stages values baked in
    — no patching is done. n_stages and RPM are read directly from the YAML
    content so the result table is correctly labelled.

    This is the right mode to use when you have per-RPM warm-start YAMLs
    and want one combined output instead of 15 separate result folders.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total = len(config_files)
    print("=" * 76)
    print("  BATCH OPTIMIZATION")
    print("=" * 76)
    print(f"  Files  : {total}")
    print(f"  Output : {output_dir}")
    print(f"  Ns valid range : [0.045, 0.20]  — results outside this are flagged")
    print("=" * 76 + "\n")

    results = []

    for i, config_file in enumerate(config_files, 1):
        config_file = Path(config_file)

        # Read n_stages and RPM directly from the YAML text
        try:
            yaml_text = config_file.read_text()
            n_match = re.search(r"n_stages:\s*(\d+)", yaml_text)
            rpm_match = re.search(r"RPM:\s*(\d+)", yaml_text)
            n_stages = int(n_match.group(1)) if n_match else -1
            RPM = int(rpm_match.group(1)) if rpm_match else -1
        except Exception:
            n_stages, RPM = -1, -1

        print(
            f"  [{i:>2}/{total}] {config_file.name}  "
            f"(n_stages={n_stages}, RPM={RPM}) ... ",
            flush=True,
        )

        if not config_file.exists():
            print(f"  ✗ File not found: {config_file}")
            results.append(
                {
                    "config": config_file.stem,
                    "n_stages": n_stages,
                    "RPM": RPM,
                    "converged": False,
                    "error": "YAML file not found",
                }
            )
            continue

        # _run_single patches RPM in the YAML — passing the value already
        # present is harmless and keeps the existing result-extraction logic.
        result = _run_single(config_file, n_stages, RPM)

        if result.get("converged", False):
            ns_problem = _check_ns_validity(result)
            if ns_problem:
                result["converged"] = False
                result["skip_reason"] = f"Ns out of range: {ns_problem}"
                print(f"  ✗  INVALID — Ns out of range: {ns_problem}")
            else:
                eta = result.get("eta_system", 0) or 0
                W = result.get("W_net_kW", 0) or 0
                print(f"  ✓  η_sys = {eta*100:.2f}%,  W_net = {W:.0f} kW")
        else:
            err = result.get("error", result.get("skip_reason", "did not converge"))
            print(f"  ✗  {err}")

        results.append(result)

    # ── Save combined results ─────────────────────────────────────────
    df = (
        pd.DataFrame(results)
        .sort_values(["n_stages", "RPM"], ascending=True, na_position="last")
        .reset_index(drop=True)
    )

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    csv_path = output_dir / f"batch_yaml_results_{timestamp}.csv"
    xlsx_path = output_dir / f"batch_yaml_results_{timestamp}.xlsx"

    df.to_csv(csv_path, index=False)
    df.to_excel(xlsx_path, index=False, sheet_name="Results")

    # ── Print summary ─────────────────────────────────────────────────
    print("\n\n" + "=" * 76)
    print("  BATCH OPTIMIZATION — RESULTS SUMMARY")
    print("=" * 76)

    display_cols = ["n_stages", "RPM", "converged"]
    for col in [
        "eta_system",
        "eta_turbine",
        "W_net_kW",
        "SP",
        "Vr",
        "Ns",
        "skip_reason",
    ]:
        if col in df.columns:
            display_cols.append(col)

    print(df[display_cols].to_string(index=False))

    print(f"\n  Results saved to:")
    print(f"    CSV  : {csv_path}")
    print(f"    Excel: {xlsx_path}")
    print("=" * 76 + "\n")

    return df


# ══════════════════════════════════════════════════════════════════════
#  MODE 4: CONDENSATION PRESSURE vs EFFICIENCY — THERMOPT FLUID SWEEP
# ══════════════════════════════════════════════════════════════════════


def run_pcond_sweep(template_yaml, fluids, n_stages, RPM, output_dir):
    """
    For each candidate fluid, run a full ThermOpt optimization using the
    template YAML (same heat source, heat sink, pinch constraints, and
    Macchi-Astolfi turbine model). Only the working fluid name, n_stages,
    and RPM are substituted. The optimizer freely determines evaporation
    pressure, condensation pressure, and mass flow rate — no temperatures
    are fixed, so pinch violations on both sides of the T-Q diagram are
    properly penalised.

    This replaces the old hardcoded ORC approach where evaporation
    temperature was fixed, which masked negative pinch violations on the
    heat source side and used a constant isentropic efficiency that ignored
    fluid-specific volume ratio effects.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  CONDENSATION PRESSURE vs EFFICIENCY — ThermOpt Fluid Sweep")
    print("=" * 70)
    print(f"  Template YAML : {Path(template_yaml).name}")
    print(f"  n_stages      : {n_stages}  |  RPM : {RPM}")
    print(f"  Fluids        : {len(fluids)}")
    print("=" * 70 + "\n")

    results = []

    for i, (fluid_name, category) in enumerate(fluids.items(), 1):
        print(f"  [{i:>2}/{len(fluids)}] {fluid_name:<18} ... ", end="", flush=True)

        tmp_yaml = None
        row = {
            "fluid": fluid_name,
            "category": category,
            "converged": False,
            "ns_invalid": False,
            "skip_reason": None,
        }
        try:
            # Build temporary YAML — swap fluid name, n_stages, RPM only
            text = Path(template_yaml).read_text()
            text = re.sub(
                r"(working_fluid:.*?\n\s+name:\s*)[^\n]+",
                lambda m: m.group(1) + fluid_name,
                text,
                count=1,
                flags=re.DOTALL,
            )
            text = re.sub(r"(n_stages:\s*)\d+", rf"\g<1>{n_stages}", text)
            text = re.sub(r"(RPM:\s*)\d+", rf"\g<1>{RPM}", text)

            tmp_yaml = Path(template_yaml).parent / f"_tmp_pcond_{fluid_name}.yaml"
            tmp_yaml.write_text(text)

            cycle = th.ThermodynamicCycleOptimization(str(tmp_yaml))
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                cycle.run_optimization()

            # Check convergence robustly
            converged = False
            for attr in [
                "success",
                lambda: cycle.result.success,
                lambda: cycle.solver_result.success,
            ]:
                try:
                    converged = attr() if callable(attr) else getattr(cycle, attr)
                    break
                except AttributeError:
                    continue

            if not converged:
                row["skip_reason"] = "optimizer did not converge"
                print(f"✗  did not converge")
                results.append(row)
                continue

            data = cycle.problem.cycle_data
            energy = data["energy_analysis"]
            components = data["components"]
            exp = components["expander"]
            comp = components["compressor"]
            data_out = exp.get("data_out", {})

            row.update(
                {
                    "converged": True,
                    "eta_system": energy.get("system_efficiency"),
                    "eta_cycle": energy.get("cycle_efficiency"),
                    "p_cond_bar": comp["state_in"].p / 1e5,
                    "p_evap_bar": exp["state_in"].p / 1e5,
                    "W_net_kW": energy.get("net_system_power", 0) / 1e3,
                    "eta_turbine": exp.get("efficiency"),
                    "SP": data_out.get("size_parameter"),
                    "SP_clamped": data_out.get("size_parameter_clamped", False),
                    "Vr": data_out.get("volume_ratio"),
                    "Ns": data_out.get("specific_speed"),
                    "m_wf_kg_s": energy.get("mass_flow_working_fluid"),
                    "m_well_kg_s": energy.get("mass_flow_heating_fluid"),
                }
            )

            # Flag Ns out of range — result kept but shown as hollow marker
            if data_out.get("specific_speed_out_of_range", False):
                row["ns_invalid"] = True
                row["skip_reason"] = "Ns out of range (Astolfi extrapolation)"
                print(
                    f"⚠  Ns out of range — "
                    f"η_sys={row['eta_system']*100:.2f}%  "
                    f"p_cond={row['p_cond_bar']:.3f} bar  "
                    f"Vr={row['Vr']:.1f}"
                )
            else:
                print(
                    f"✓  η_sys={row['eta_system']*100:.2f}%  "
                    f"p_cond={row['p_cond_bar']:.3f} bar  "
                    f"Vr={row['Vr']:.1f}  "
                    f"η_turb={row['eta_turbine']:.3f}"
                )

        except Exception as e:
            row["skip_reason"] = str(e)
            print(f"✗  {e}")
            traceback.print_exc()

        finally:
            if tmp_yaml is not None and tmp_yaml.exists():
                tmp_yaml.unlink()

        results.append(row)

    # ── Save results ──────────────────────────────────────────────────
    df = (
        pd.DataFrame(results)
        .sort_values("eta_system", ascending=False, na_position="last")
        .reset_index(drop=True)
    )

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    tag = f"n{n_stages}_rpm{RPM}"
    xlsx_path = output_dir / f"pcond_vs_efficiency_{tag}_{timestamp}.xlsx"
    df.to_excel(xlsx_path, index=False)
    print(f"\n  Saved data  : {xlsx_path}")

    # ── Plot ──────────────────────────────────────────────────────────
    valid = [r for r in results if r["converged"] and not r["ns_invalid"]]
    flagged = [r for r in results if r["converged"] and r["ns_invalid"]]

    if not valid and not flagged:
        print("  No converged results to plot.")
        return df

    plot_path = _plot_pcond_efficiency(valid, flagged, n_stages, RPM, output_dir, tag)
    print(f"  Saved plot  : {plot_path}")

    n_ok = len(valid)
    n_ns = len(flagged)
    n_bad = sum(1 for r in results if not r["converged"])
    print(f"\n  Summary: {n_ok} valid | {n_ns} Ns-invalid | {n_bad} failed")
    print("=" * 70 + "\n")

    return df


def _plot_pcond_efficiency(valid, flagged, n_stages, RPM, output_dir, tag):
    """
    Thesis-quality scatter plot: condensing pressure vs system efficiency.

    Design choices:
      - No figure title (thesis figures use captions instead)
      - Clean spines — top and right removed
      - Serif font throughout for consistency with LaTeX documents
      - Toluene drawn last and highlighted with a bold ring so it stands
        out as the reference fluid from the main study
      - 1 atm reference line annotated cleanly at the top of the axis
      - Legend outside the plot area so it never overlaps data points
      - 300 dpi PNG suitable for print
    """
    import matplotlib.ticker as ticker

    # ── Typography & style ────────────────────────────────────────────
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
        }
    )

    fig, ax = plt.subplots(figsize=(13, 7))

    # Separate Toluene from the rest so it can be drawn on top
    toluene = [r for r in valid if r["fluid"] == "Toluene"]
    non_tol = [r for r in valid if r["fluid"] != "Toluene"]

    # ── Plot by fluid family ──────────────────────────────────────────
    for cat, style in PCOND_CATEGORY_STYLE.items():
        cat_r = [r for r in non_tol if r["category"] == cat]
        if not cat_r:
            continue
        ax.scatter(
            [r["p_cond_bar"] for r in cat_r],
            [r["eta_system"] * 100 for r in cat_r],
            color=style["color"],
            marker=style["marker"],
            s=90,
            edgecolors="k",
            linewidth=0.4,
            label=cat.capitalize(),
            zorder=3,
        )

    # ── Toluene — highlighted reference ──────────────────────────────
    if toluene:
        r = toluene[0]
        ax.scatter(
            r["p_cond_bar"],
            r["eta_system"] * 100,
            color=PCOND_CATEGORY_STYLE["aromatic"]["color"],
            marker=PCOND_CATEGORY_STYLE["aromatic"]["marker"],
            s=180,
            edgecolors="k",
            linewidth=1.5,
            zorder=5,
            label="Toluene (reference)",
        )

    # ── Ns-invalid results — hollow grey markers ──────────────────────
    if flagged:
        ax.scatter(
            [r["p_cond_bar"] for r in flagged],
            [r["eta_system"] * 100 for r in flagged],
            facecolors="none",
            edgecolors="#aaaaaa",
            marker="o",
            s=70,
            linewidth=0.8,
            label=r"$N_s$ out of range (extrapolation)",
            zorder=2,
        )

    ax.set_xscale("log")

    # ── Fluid name labels ─────────────────────────────────────────────
    for r in valid + flagged:
        x, y = r["p_cond_bar"], r["eta_system"] * 100
        dx, dy = PCOND_LABEL_OFFSETS.get(r["fluid"], (7, 5))
        use_arrow = abs(dx) > 15 or abs(dy) > 12
        weight = "bold" if r["fluid"] == "Toluene" else "normal"
        ax.annotate(
            r["fluid"],
            (x, y),
            textcoords="offset points",
            xytext=(dx, dy),
            fontsize=8,
            fontweight=weight,
            ha="left" if dx >= 0 else "right",
            color="#222222",
            arrowprops=(
                dict(arrowstyle="-", color="#aaaaaa", lw=0.5, shrinkA=0, shrinkB=2)
                if use_arrow
                else None
            ),
        )

    # ── 1 atm reference line ──────────────────────────────────────────
    ax.axvline(
        x=1.01325, color="#888888", linestyle="--", linewidth=0.8, alpha=0.8, zorder=1
    )
    ax.text(
        1.01325 * 1.04,
        ax.get_ylim()[1] * 0.98,
        "1 atm",
        fontsize=8,
        color="#888888",
        va="top",
        ha="left",
        style="italic",
    )

    # ── Axes formatting ───────────────────────────────────────────────
    ax.set_xlabel("Condensing Pressure [bar]")
    ax.set_ylabel(r"System Efficiency $\eta_{sys}$ [%]")

    # Clean log-scale x-axis ticks
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:g}"))

    # Remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Subtle grid on y only — avoids clutter on log x
    ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.4, color="grey")
    ax.set_axisbelow(True)

    # ── Legend outside plot, top-right ───────────────────────────────
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        frameon=True,
        framealpha=0.95,
        edgecolor="#cccccc",
        borderpad=0.8,
    )

    fig.tight_layout()
    # Make room for the legend outside the axes
    fig.subplots_adjust(right=0.82)

    plot_path = Path(output_dir) / f"pcond_vs_efficiency_{tag}.png"
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Reset rcParams to defaults so other plots in the session are unaffected
    plt.rcParams.update(plt.rcParamsDefault)

    return plot_path


# ══════════════════════════════════════════════════════════════════════
#  RUN
# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":

    print(f"Mode: {MODE}")
    print()

    if MODE == "optimize":
        if not CONFIG_FILE.exists():
            print(f"Error: config file not found: {CONFIG_FILE}")
            sys.exit(1)
        print(f"Config: {CONFIG_FILE.name}\n")
        run_optimize(CONFIG_FILE)

    elif MODE == "sweep":
        if not CONFIG_FILE.exists():
            print(f"Error: config file not found: {CONFIG_FILE}")
            sys.exit(1)
        print(f"Config: {CONFIG_FILE.name}\n")
        run_sweep(CONFIG_FILE, SWEEP_OUTPUT_DIR)

    elif MODE == "batch_yaml":
        run_batch_yaml(
            config_files=BATCH_YAML_CONFIGS,
            output_dir=BATCH_YAML_OUTPUT_DIR,
        )

    elif MODE == "pcond_sweep":
        run_pcond_sweep(
            template_yaml=PCOND_TEMPLATE,
            fluids=PCOND_FLUIDS,
            n_stages=PCOND_N_STAGES,
            RPM=PCOND_RPM,
            output_dir=PCOND_OUTPUT_DIR,
        )

    else:
        print(
            f"Error: MODE must be 'optimize', 'sweep', 'batch_yaml', or 'pcond_sweep', got '{MODE}'"
        )
        sys.exit(1)

    plt.show()
