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
from validation_checks import (
    run_validation_checks,
    get_validation_data,
    get_turbine_data,
    get_tq_summary_data,
)

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
#  "multistart"  — Latin Hypercube Sampling multistart optimisation. Generates
#                  MULTISTART_N_SAMPLES starting points spread across the design
#                  variable bounds, runs the optimizer from each, and saves the
#                  best converged result as a ready-to-use YAML. Failed/infeasible
#                  runs are logged and skipped. Use this to search for the global
#                  optimum of a basecase YAML.
#                  → Set MULTISTART_CONFIG, MULTISTART_N_SAMPLES,
#                     MULTISTART_OUTPUT_DIR.
#
MODE = (
    "pcond_sweep"  # "optimize" | "sweep" | "batch_yaml" | "pcond_sweep" | "multistart"
)


# ────────────────────────────────────────── optimize & sweep ─────────────────────────────────────────────────────────────────────────────────────────────────────────


# Used by: optimize, sweep
# Set this to the YAML file you want to run (optimize) or use as a
# template with the fluid name swapped out for each candidate (sweep).
# CONFIG_FILE = Path(__file__).with_name("case_Toluene_simpleORC_macchi_astolfi.yaml")
# CONFIG_FILE = Path(__file__).with_name("case_Toluene_simpleORC_macchi_astolfi_n5_rpm1000.yaml")

# CONFIG_FILE = Path(__file__).with_name("case_Toluene_simpleORC_macchi_astolfi_n5_rpm1500.yaml")
# CONFIG_FILE = Path(__file__).with_name("case_Toluene_simpleORC_macchi_astolfi_n5_rpm3000.yaml")

# CONFIG_FILE = Path(__file__).with_name("case_Toluene_simpleORC_macchi_astolfi_n4_rpm1000.yaml")
# CONFIG_FILE = Path(__file__).with_name("case_Toluene_recuperatedORC_macchi_astolfi_n4_rpm1000.yaml")
# CONFIG_FILE = Path(__file__).with_name("Toluene_DualPressureORC_basecase.yaml")
# CONFIG_FILE = Path(__file__).with_name("Toluene_DualPressureRecuperatedORC_basecase.yaml")
# CONFIG_FILE = Path(__file__).with_name("Cis-2-Butene_transcriticalORC_basecase.yaml")

# CONFIG_FILE = Path(__file__).with_name("Toluene_dp_twosource_basecase.yaml")
CONFIG_FILE = Path(__file__).with_name("Toluene_simple_basecase.yaml")


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
PCOND_TEMPLATE = Path(__file__).with_name("Toluene_simple_basecase.yaml")
# PCOND_TEMPLATE = Path(__file__).with_name("pcondSweep_simpleRecORC_macchi_astolfi_n4_rpm1000.yaml")


# Number of turbine stages and shaft speed used for all fluids in the sweep.
PCOND_N_STAGES = 4
PCOND_RPM = 1000

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

# ──────────────────────────────────────────── multistart: LHS settings ──────────────────────────────────────────────────────────────────────────────────────────────
# Used by: multistart
# Set MULTISTART_CONFIG to the basecase YAML you want to search globally.
# All design variable bounds (min/max) are read directly from that file —
# no manual bound specification needed.
MULTISTART_CONFIG = Path(__file__).with_name("Toluene_simple_basecase.yaml")
MULTISTART_N_SAMPLES = 3  # number of LHS starting points (50–100 recommended)
MULTISTART_OUTPUT_DIR = Path(__file__).parent / "results" / "multistart"


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
#  HELPER: APPEND EXTRA SHEETS TO POST-PROCESSING EXCEL
# ══════════════════════════════════════════════════════════════════════


def _append_extra_sheets(excel_path, cycle, config_file=None):
    """
    Append three extra sheets to the post_processing_results.xlsx:
      - validation_checks  : energy balance + constraint activity + phase check
      - turbine_summary    : only written when astolfi-stacking is active
      - TQ_summary         : heat duty, inlet/outlet temps and pinch for each HX
    """
    import openpyxl

    validation_rows = get_validation_data(cycle, config_file=config_file)
    turbine_rows = get_turbine_data(cycle)  # None if isentropic mode
    tq_rows = get_tq_summary_data(cycle)

    wb = openpyxl.load_workbook(excel_path)

    def _safe(val):
        """Convert any non-basic type to a plain Python type for openpyxl.
        Handles numpy scalars, CoolProp Array types, and anything else
        that isn't already a str/int/float/bool/None."""
        if val is None:
            return None
        if isinstance(val, (str, int, float, bool)):
            return val
        # Try .item() first (numpy scalars)
        if hasattr(val, "item"):
            try:
                return val.item()
            except Exception:
                pass
        # Fall back to float conversion (CoolProp Array, etc.)
        try:
            return float(val)
        except Exception:
            pass
        # Last resort: stringify
        return str(val)

    def _write_sheet(wb, sheet_name, rows):
        if sheet_name in wb.sheetnames:
            del wb[sheet_name]
        ws = wb.create_sheet(sheet_name)
        ws.append(["Parameter", "Value"])
        for row in rows:
            ws.append([_safe(v) for v in row])
        # Auto-width columns
        for col in ws.columns:
            max_len = max((len(str(c.value)) for c in col if c.value), default=10)
            ws.column_dimensions[col[0].column_letter].width = min(max_len + 4, 60)

    _write_sheet(wb, "validation_checks", validation_rows)
    if turbine_rows is not None:
        _write_sheet(wb, "turbine_summary", turbine_rows)
    _write_sheet(wb, "TQ_summary", tq_rows)

    wb.save(excel_path)
    print(
        f"    ✓ Extra sheets added: validation_checks"
        + (", turbine_summary" if turbine_rows is not None else "")
        + ", TQ_summary"
    )


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
    if "mass_flow_heating_fluid" in ea:
        print(
            f"  Well (heating fluid)               :  {ea['mass_flow_heating_fluid']:.2f} kg/s"
        )
    elif "mass_flow_brine_hp" in ea:
        print(
            f"  HP brine (well 2)                  :  {ea['mass_flow_brine_hp']:.2f} kg/s"
        )
        print(
            f"  LP brine (wells 1/3/4)             :  {ea['mass_flow_brine_lp']:.2f} kg/s"
        )
    print(
        f"  Working fluid                      :  {ea['mass_flow_working_fluid']:.2f} kg/s"
    )
    print(
        f"  Cooling fluid                      :  {ea['mass_flow_cooling_fluid']:.2f} kg/s"
    )
    print("=" * 76 + "\n")

    # ── Post-processing ───────────────────────────────────────────
    graph_dir = os.path.join(cycle.out_dir, "post_processing")
    os.makedirs(graph_dir, exist_ok=True)

    # Exergy analysis
    exergy = perform_exergy_analysis(cycle, config_file=config_file)
    exergy.print_summary()
    pp_excel = os.path.join(graph_dir, "post_processing_results.xlsx")
    exergy.to_excel(pp_excel)
    _append_extra_sheets(pp_excel, cycle, config_file)
    exergy.plot_exergy_destruction(savefig=os.path.join(graph_dir, "exergy_bar.png"))

    # T-Q diagrams — topology-aware
    components = cycle.problem.cycle_data["components"]
    is_dual_pressure = "hp_evaporator" in components

    if is_dual_pressure:
        # Dual-pressure topology: three brine-side heat exchangers
        plot_TQ_diagram(cycle, "hp_evaporator", output_dir=graph_dir)
        plot_TQ_diagram(cycle, "lp_evaporator", output_dir=graph_dir)
        plot_TQ_diagram(cycle, "preheater", output_dir=graph_dir)
    else:
        # Simple / recuperated topology: single heater
        plot_TQ_diagram(cycle, "heater", output_dir=graph_dir)

    plot_TQ_diagram(cycle, "cooler", output_dir=graph_dir)

    # Plot recuperator T-Q diagram only if it is active (heat flow > 1 kW)
    recup = components.get("recuperator", None)
    if recup is not None and abs(float(recup.get("heat_flow", 0))) > 1000:
        plot_TQ_diagram(cycle, "recuperator", output_dir=graph_dir)

    # Summary
    print("\n" + "=" * 60)
    print("  ALL RESULTS SAVED")
    print("=" * 60)
    print(f"  Thermopt results : {cycle.out_dir}")
    print(f"  Post-processing  : {graph_dir}")
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
        "ns_invalid": False,
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

        # Check convergence — read the flag ThermOpt/pysolver_view actually set
        converged = False
        try:
            converged = bool(cycle.solver.success)
        except AttributeError:
            pass

        result = _extract_results(cycle, config_file, n_stages, RPM)
        result["converged"] = converged
        return result

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
                result["ns_invalid"] = True
                result["skip_reason"] = f"Ns out of range: {ns_problem}"
                eta = result.get("eta_system", 0) or 0
                W = result.get("W_net_kW", 0) or 0
                print(
                    f"  ⚠  Ns out of range — η_sys = {eta*100:.2f}%,  W_net = {W:.0f} kW  [{ns_problem}]"
                )
            else:
                result["ns_invalid"] = False
                eta = result.get("eta_system", 0) or 0
                W = result.get("W_net_kW", 0) or 0
                print(f"  ✓  η_sys = {eta*100:.2f}%,  W_net = {W:.0f} kW")
        else:
            result["ns_invalid"] = False
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

    display_cols = ["n_stages", "RPM", "converged", "ns_invalid"]
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

            # cycle = th.ThermodynamicCycleOptimization(str(tmp_yaml))
            # with warnings.catch_warnings():
            #     warnings.filterwarnings("ignore")
            #     cycle.run_optimization()

            cycle = th.ThermodynamicCycleOptimization(str(tmp_yaml))
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                cycle.run_optimization()

            # Check convergence robustly
            converged = False
            for attr in [
                lambda: cycle.solver.success,
                lambda: cycle.solver.result.success,
                lambda: cycle.solver.solver_result.success,
            ]:
                try:
                    converged = attr()
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
#  MODE 5: MULTISTART — LATIN HYPERCUBE SAMPLING
# ══════════════════════════════════════════════════════════════════════


def _reconstruct_expression(converged_val, vd, refs):
    """
    Try to express converged_val back in the original coefficient format.

    Handles two patterns found in ORC basecases:

      Simple:   coeff * $ref
                e.g. "0.172758*$working_fluid.critical_point.p"
                → new_coeff = v / ref_val

      Compound: coeff * ($ref1 - $ref2) + $ref2
                e.g. "0.033007*($working_fluid.critical_point.h
                     - $working_fluid.liquid_at_ambient_temperature.h)
                     + $working_fluid.liquid_at_ambient_temperature.h"
                → new_coeff = (v - ref2_val) / (ref1_val - ref2_val)

    Returns the reconstructed expression string, or None if the pattern
    is not recognised (caller should fall back to a plain float).
    """
    import re

    # Use the min bound as the template expression to detect the pattern
    template = str(vd["min"]).strip()

    # ── Pattern 1: pure numeric bounds ────────────────────────────────
    # If both min and max look like plain numbers, just return a float.
    try:
        float(template)
        return None  # caller will write a plain float
    except ValueError:
        pass

    # ── Pattern 2: coeff * $ref  (simple scalar multiple) ─────────────
    # Matches e.g. "0.05*$working_fluid.critical_point.p"
    # Only reconstruct if min and max use the same $ref — if they differ
    # (e.g. min uses triple_point_vapor.p, max uses critical_point.p) the
    # back-calculated coefficient would be nonsensical, so fall back to float.
    m = re.match(r"^[\d.eE+-]+\*(\$working_fluid\.\w+\.\w+)$", template)
    if m:
        ref_key = m.group(1)
        max_template = str(vd["max"]).strip()
        m_max = re.match(r"^[\d.eE+-]+\*(\$working_fluid\.\w+\.\w+)$", max_template)
        if m_max and m_max.group(1) != ref_key:
            ref_key = m_max.group(
                1
            )  # switch to max bound's ref — more physically meaningful
        if ref_key in refs and refs[ref_key] != 0:
            new_coeff = converged_val / refs[ref_key]
            return f"{new_coeff:.6f}*{ref_key}"

    # ── Pattern 3: coeff * ($ref1 - $ref2) + $ref2  (affine / enthalpy)
    # Matches e.g. "0.033*($working_fluid.critical_point.h
    #               - $working_fluid.liquid_at_ambient_temperature.h)
    #               + $working_fluid.liquid_at_ambient_temperature.h"
    m = re.match(
        r"^[\d.eE+-]+\*\(\s*(\$working_fluid\.\w+\.\w+)\s*-\s*"
        r"(\$working_fluid\.\w+\.\w+)\s*\)\s*\+\s*(\$working_fluid\.\w+\.\w+)$",
        template,
    )
    if m:
        ref1_key = m.group(1)
        ref2_key = m.group(2)
        ref3_key = m.group(3)  # should equal ref2_key in all normal cases
        if all(k in refs for k in (ref1_key, ref2_key, ref3_key)):
            denom = refs[ref1_key] - refs[ref2_key]
            if abs(denom) > 1e-12:
                new_coeff = (converged_val - refs[ref3_key]) / denom
                return f"{new_coeff:.6f}*({ref1_key} - {ref2_key})" f" + {ref3_key}"

    # No pattern matched — caller falls back to plain float
    return None


def _save_optimized_yaml(base_config_file, cycle, out_path, ryaml):
    """
    Write a copy of base_config_file where every design variable's value:
    is replaced with the converged solution from cycle, expressed in the
    original coefficient format wherever possible.

    For expression-based bounds (e.g. "0.172758*$working_fluid.critical_point.p"),
    the coefficient is back-calculated from the converged value so the YAML
    stays physically readable:
        simple   → new_coeff * $working_fluid.critical_point.p
        compound → new_coeff * ($ref1 - $ref2) + $ref2

    For purely numeric bounds the converged value is written as a plain float.
    If the pattern cannot be matched the float is used as a safe fallback.
    """
    with open(base_config_file) as f:
        cfg = ryaml.load(f)

    dv = cfg["problem_formulation"]["design_variables"]
    var_names = list(dv.keys())

    # ── Get converged values from cycle.problem.x0_dict ───────────────
    # fitness() sets self.x0_dict = dict(zip(self.variable_names, x)) on
    # every call, so after optimization this holds {name: converged_float}.
    converged_values = None
    try:
        converged_values = dict(cycle.problem.x0_dict)
    except AttributeError:
        pass

    # Fallback: read directly from the configuration dict, which fitness()
    # also keeps in sync: configuration["design_variables"][k]["value"] = v
    if not converged_values:
        try:
            converged_values = {
                k: v["value"]
                for k, v in cycle.problem.configuration["design_variables"].items()
            }
        except AttributeError:
            pass

    if converged_values is None:
        print("    ⚠  Could not extract converged variable values from cycle object.")
        print("       Optimized YAML not saved.")
        return

    # ── Build fluid reference lookup table ────────────────────────────
    # cycle.problem.params["working_fluid"] holds the pre-computed special
    # points (critical_point, liquid_at_ambient_temperature, etc.) that
    # thermopt uses to evaluate the $working_fluid.X.Y expressions in the YAML.
    refs = {}
    try:
        wf_params = cycle.problem.params["working_fluid"]
        props = ["p", "h", "T", "s", "d"]
        for sp_name, sp_obj in wf_params.items():
            for prop in props:
                try:
                    val = getattr(sp_obj, prop)
                    refs[f"$working_fluid.{sp_name}.{prop}"] = float(val)
                except AttributeError:
                    continue
    except (AttributeError, KeyError, TypeError):
        print(
            "    ⚠  Could not access working fluid params from cycle — "
            "expression format unavailable, falling back to plain floats."
        )

    # ── Patch value: fields ────────────────────────────────────────────
    patched = 0
    as_expr = 0
    as_float = 0

    for name in var_names:
        if name not in converged_values:
            print(
                f"    ⚠  Variable '{name}' not found in converged solution — "
                f"value: left unchanged."
            )
            continue

        v = float(converged_values[name])
        expr = _reconstruct_expression(v, dv[name], refs)

        if expr is not None:
            dv[name]["value"] = expr
            as_expr += 1
        else:
            dv[name]["value"] = v
            as_float += 1

        patched += 1

    with open(out_path, "w") as f:
        ryaml.dump(cfg, f)

    print(
        f"    ✓ Optimized YAML saved  ({patched}/{len(var_names)} variables): "
        f"{as_expr} as expressions, {as_float} as floats  →  {out_path.name}"
    )


def run_multistart(config_file, n_samples, output_dir):
    """
    Latin Hypercube Sampling multistart optimisation.

    Generates n_samples starting points distributed evenly across the full
    design variable space using LHS (one sample per interval per variable),
    runs the optimizer from each point, logs and skips failures, and saves:
      - A CSV + Excel summary of every run (eta, W_net, convergence flag).
      - best_lhs_start_*.yaml  — the LHS starting point that led to the best
        result. Re-running this in "optimize" mode will reproduce the result.
      - optimized_*.yaml       — the converged solution with value: fields set
        to the actual optimized variable values. Use this as a warm start for
        future runs; the optimizer begins right at the optimum.

    Dependencies: scipy (>=1.7 for scipy.stats.qmc), ruamel.yaml
    """
    from scipy.stats.qmc import LatinHypercube
    from ruamel.yaml import YAML
    import shutil

    config_file = Path(config_file)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Parse YAML — extract design variable names and bounds ──────────
    ryaml = YAML()
    ryaml.preserve_quotes = True
    with open(config_file) as f:
        base_config = ryaml.load(f)

    dvars = base_config["problem_formulation"]["design_variables"]
    var_names = list(dvars.keys())
    n_vars = len(var_names)

    print("=" * 70)
    print("  MULTISTART OPTIMISATION — Latin Hypercube Sampling")
    print("=" * 70)
    print(f"  Config    : {config_file.name}")
    print(f"  Variables : {n_vars}")
    print(f"  Samples   : {n_samples}")
    print(f"  Output    : {output_dir}")
    print("=" * 70)
    print()
    print("  Design variable bounds:")
    for name in var_names:
        vd = dvars[name]
        print(f"    {name:<48}  min={vd['min']}  max={vd['max']}")
    print()

    # ── Generate LHS samples in [0, 1]^n_vars ─────────────────────────
    # seed=42 makes the sample set reproducible across runs.
    sampler = LatinHypercube(d=n_vars, seed=42)
    lhs_samples = sampler.random(n=n_samples)  # shape: (n_samples, n_vars)

    # ── Optimise from each LHS starting point ─────────────────────────
    results = []
    best_eta = -1.0
    best_run_idx = None
    best_tmp_path = None  # path to the temp YAML of the best run
    best_cycle = None  # cycle object for the best run (kept for optimized YAML)

    import time

    for i, sample in enumerate(lhs_samples):
        run_label = f"[{i+1:>3}/{n_samples}]"
        print(f"  {run_label} ", end="", flush=True)

        tmp_path = config_file.parent / f"_tmp_multistart_{i:04d}.yaml"
        row = {
            "run": i + 1,
            "converged": False,
            "eta_system": None,
            "W_net_kW": None,
            "error": None,
            "t_elapsed_s": None,
            "n_iterations": None,
            "n_func_evals": None,
            "infeasibility": None,
            "exit_message": None,
        }

        try:
            # ── Patch value: for every design variable ─────────────────
            # Re-load the config each iteration to get a fresh ruamel tree.
            with open(config_file) as f:
                cfg = ryaml.load(f)

            dv = cfg["problem_formulation"]["design_variables"]

            for j, var_name in enumerate(var_names):
                t = float(sample[j])  # LHS sample in [0, 1]
                vd = dv[var_name]
                vmin = vd["min"]
                vmax = vd["max"]

                if isinstance(vmin, (int, float)) and isinstance(vmax, (int, float)):
                    # Numeric bounds — compute value directly as a float
                    vd["value"] = float(vmin) + t * (float(vmax) - float(vmin))
                else:
                    # Expression-based bounds (e.g. "0.05*$working_fluid.critical_point.p")
                    # Write as a linear combination that thermopt can evaluate.
                    t1 = round(1.0 - t, 8)
                    t2 = round(t, 8)
                    vd["value"] = f"({t1})*({vmin}) + ({t2})*({vmax})"

                # Store the t-value in the result row for reference
                row[f"t_{var_name}"] = round(t, 6)

            with open(tmp_path, "w") as f:
                ryaml.dump(cfg, f)

            # ── Run optimiser ──────────────────────────────────────────
            t_start = time.perf_counter()
            cycle = th.ThermodynamicCycleOptimization(str(tmp_path))
            cycle.run_optimization()
            t_elapsed = time.perf_counter() - t_start
            row["t_elapsed_s"] = round(t_elapsed, 1)

            # ── Solver diagnostics ─────────────────────────────────────
            try:
                row["n_iterations"] = int(cycle.solver.nit)
            except AttributeError:
                pass
            try:
                row["n_func_evals"] = int(cycle.solver.nfev)
            except AttributeError:
                pass
            try:
                row["infeasibility"] = float(cycle.solver.infeasibility)
            except AttributeError:
                pass
            try:
                row["exit_message"] = str(cycle.solver.message)
            except AttributeError:
                pass

            # Convergence check (mirrors _run_single / run_pcond_sweep)
            converged = False
            try:
                converged = bool(cycle.solver.success)
            except AttributeError:
                pass

            if not converged:
                print(f"✗  did not converge  ({t_elapsed:.0f}s)")
                row["error"] = "optimizer did not converge"
                results.append(row)
                continue

            # ── Extract key metrics ────────────────────────────────────
            ea = cycle.problem.cycle_data["energy_analysis"]
            eta = ea.get("system_efficiency", 0) or 0
            W_net = ea.get("net_system_power", 0) / 1e3

            row["converged"] = True
            row["eta_system"] = round(eta, 8)
            row["W_net_kW"] = round(W_net, 2)

            # ── Thermodynamic outputs ──────────────────────────────────
            # Brine exit temperature(s)
            if "brine_exit_temperature" in ea:
                row["brine_exit_T_K"] = round(ea["brine_exit_temperature"], 2)
            elif "hp_brine_exit_temperature" in ea:
                row["hp_brine_exit_T_K"] = round(ea["hp_brine_exit_temperature"], 2)
                row["lp_brine_exit_T_K"] = round(ea["lp_brine_exit_temperature"], 2)

            # Mass flows
            if "mass_flow_heating_fluid" in ea:
                row["m_brine_kgs"] = round(ea["mass_flow_heating_fluid"], 2)
            elif "mass_flow_brine_hp" in ea:
                row["m_brine_hp_kgs"] = round(ea["mass_flow_brine_hp"], 2)
                row["m_brine_lp_kgs"] = round(ea["mass_flow_brine_lp"], 2)
            row["m_working_fluid_kgs"] = round(ea.get("mass_flow_working_fluid", 0), 2)

            # Expander inlet conditions (LP expander = main expander in dual pressure)
            components = cycle.problem.cycle_data.get("components", {})
            for exp_name in ["lp_expander", "expander"]:
                if exp_name in components:
                    exp = components[exp_name]
                    try:
                        row["expander_inlet_T_C"] = round(
                            exp["state_in"]["T"] - 273.15, 2
                        )
                        row["expander_inlet_p_bar"] = round(
                            exp["state_in"]["p"] / 1e5, 3
                        )
                    except (KeyError, TypeError):
                        pass
                    break

            # Condenser exit temperature
            for cool_name in ["cooler", "condenser"]:
                if cool_name in components:
                    cool = components[cool_name]
                    try:
                        row["condenser_exit_T_C"] = round(
                            cool["state_out"]["T"] - 273.15, 2
                        )
                    except (KeyError, TypeError):
                        pass
                    break

            # ── Converged variable values ──────────────────────────────
            try:
                x0_dict = cycle.problem.x0_dict
                for var_name in var_names:
                    if var_name in x0_dict:
                        row[f"x_{var_name}"] = round(float(x0_dict[var_name]), 6)
            except AttributeError:
                pass

            print(
                f"✓  η_sys = {eta * 100:.4f}%   W_net = {W_net:.1f} kW  ({t_elapsed:.0f}s)",
                end="",
            )

            # ── Track best ─────────────────────────────────────────────
            if eta > best_eta:
                best_eta = eta
                best_run_idx = i + 1
                best_cycle = cycle
                # Keep a copy of the temp YAML for this best run.
                # (Overwritten each time a better run is found.)
                _candidate = output_dir / "_best_candidate.yaml"
                shutil.copy(tmp_path, _candidate)
                print("  ← best so far", end="")

            print()

        except Exception as e:
            print(f"✗  {e}")
            traceback.print_exc()
            row["error"] = str(e)

        finally:
            if tmp_path.exists():
                tmp_path.unlink()

        results.append(row)

    # ── Save summary CSV + Excel ───────────────────────────────────────
    df = pd.DataFrame(results)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    csv_path = output_dir / f"multistart_results_{timestamp}.csv"
    xlsx_path = output_dir / f"multistart_results_{timestamp}.xlsx"
    df.to_csv(csv_path, index=False)
    df.to_excel(xlsx_path, index=False, sheet_name="Multistart")

    # ── Save LHS warm-start YAML (starting point that led to best result) ─
    # Useful for exact reproduction: running this in "optimize" mode will
    # re-run the optimizer from the same starting point.
    lhs_yaml_path = None
    candidate_path = output_dir / "_best_candidate.yaml"
    if candidate_path.exists():
        lhs_yaml_path = (
            output_dir / f"best_lhs_start_{config_file.stem}_{timestamp}.yaml"
        )
        shutil.move(str(candidate_path), str(lhs_yaml_path))

    # ── Save optimized YAML (converged solution as value: fields) ─────
    # This is the proper warm start: value: fields are set to the actual
    # converged variable values, not the LHS starting point.
    optimized_yaml_path = None
    if best_cycle is not None:
        optimized_yaml_path = (
            output_dir / f"optimized_{config_file.stem}_{timestamp}.yaml"
        )
        _save_optimized_yaml(
            base_config_file=config_file,
            cycle=best_cycle,
            out_path=optimized_yaml_path,
            ryaml=ryaml,
        )

    # ── Print summary ──────────────────────────────────────────────────
    n_ok = int(df["converged"].sum())
    n_bad = n_samples - n_ok

    print()
    print("=" * 70)
    print("  MULTISTART — SUMMARY")
    print("=" * 70)
    print(f"  Total runs : {n_samples}")
    print(f"  Converged  : {n_ok}")
    print(f"  Failed     : {n_bad}")
    print()

    converged_df = df[df["converged"] == True].copy()
    if not converged_df.empty:
        converged_df_sorted = converged_df.sort_values("eta_system", ascending=False)
        print("  Top 5 converged results:")
        print(f"    {'Run':>5}   {'η_sys [%]':>12}   {'W_net [kW]':>12}")
        print(f"    {'─'*5}   {'─'*12}   {'─'*12}")
        for _, r in converged_df_sorted.head(5).iterrows():
            marker = " ← BEST" if int(r["run"]) == best_run_idx else ""
            print(
                f"    {int(r['run']):>5}   {r['eta_system']*100:>11.4f}%   "
                f"{r['W_net_kW']:>11.1f} kW{marker}"
            )
        print()
        print(f"  Best η_sys : {best_eta * 100:.4f}%  (run {best_run_idx})")
    else:
        print("  No runs converged.")

    print()
    print("  Output files:")
    print(f"    CSV            : {csv_path}")
    print(f"    Excel          : {xlsx_path}")
    if lhs_yaml_path is not None:
        print(f"    LHS start YAML : {lhs_yaml_path}")
        print(f"                     (starting point that produced the best result)")
    if optimized_yaml_path is not None:
        print(f"    Optimized YAML : {optimized_yaml_path}")
        print(
            f"                     (converged solution as value: — use this as warm start)"
        )
        print()
        print("  To run full post-processing on the best result:")
        print(
            f'    1. Set CONFIG_FILE = Path(__file__).with_name("{optimized_yaml_path.name}")'
        )
        print('    2. Set MODE = "optimize"')
        print("    3. Run this script")
    print("=" * 70 + "\n")

    return df


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

    elif MODE == "multistart":
        if not MULTISTART_CONFIG.exists():
            print(f"Error: config file not found: {MULTISTART_CONFIG}")
            sys.exit(1)
        run_multistart(
            config_file=MULTISTART_CONFIG,
            n_samples=MULTISTART_N_SAMPLES,
            output_dir=MULTISTART_OUTPUT_DIR,
        )

    else:
        print(
            f"Error: MODE must be 'optimize', 'sweep', 'batch_yaml', 'pcond_sweep', or 'multistart', got '{MODE}'"
        )
        sys.exit(1)

    plt.show()
