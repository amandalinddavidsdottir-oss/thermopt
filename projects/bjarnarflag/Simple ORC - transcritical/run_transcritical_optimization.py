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
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "shared_utilities"))
from exergy_analysis import perform_exergy_analysis, plot_heat_source_utilization
from plot_TQ_diagram import plot_TQ_diagram
from validation_checks import run_validation_checks

# -----

warnings.filterwarnings("ignore", message="FigureCanvasAgg is non-interactive")


# ══════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════
MODE = "optimize"  # "optimize" | "sweep" | "parametric"

# CONFIG_FILE = Path(__file__).with_name("case_butane_transcritical_bjarnarflag.yaml")
CONFIG_FILE = Path(__file__).with_name(
    "case_butane_transcritical_recuperated_bjarnarflag.yaml"
)

SWEEP_OUTPUT_DIR = "results/fluid_sweep_TRANSCRITICAL_ORC"

# ── Parametric study settings ──
PARAMETRIC_CONFIGS = [
    Path(__file__).with_name("case_butane_transcritical_bjarnarflag.yaml"),
    Path(__file__).with_name("case_butane_transcritical_recuperated_bjarnarflag.yaml"),
]
PARAMETRIC_STAGES = [1, 2, 3, 4, 5, 6, 7]
PARAMETRIC_RPMS = [3000, 1500]
PARAMETRIC_OUTPUT_DIR = Path(__file__).parent / "results" / "parametric_study"


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
        f"  Well (heating fluid)               :  {ea.get('mass_flow_heating_fluid', 0):.2f} kg/s"
    )
    print(
        f"  Working fluid                      :  {ea.get('mass_flow_working_fluid', 0):.2f} kg/s"
    )
    print(
        f"  Cooling fluid                      :  {ea.get('mass_flow_cooling_fluid', 0):.2f} kg/s"
    )
    print("=" * 76 + "\n")

    # ────────────────────────────POST-PROCESSING ───────────────────────
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

    from fluid_sweep_TRANSCRITICAL_ORC import (
        get_candidate_fluids,
        run_fluid_sweep,
        plot_results,
    )

    candidates = get_candidate_fluids(config_file)
    df = run_fluid_sweep(config_file, candidates, output_dir=output_dir)
    plot_results(df, output_dir=output_dir)


# ══════════════════════════════════════════════════════════════════════
#  MODE 3: PARAMETRIC STUDY (n_stages × RPM)
# ══════════════════════════════════════════════════════════════════════


def _modify_yaml(yaml_text, n_stages, RPM):
    """Replace n_stages and RPM for the expander."""
    yaml_text = re.sub(
        r"(expander:.*?n_stages:\s*)\d+",
        rf"\g<1>{n_stages}",
        yaml_text,
        count=1,
        flags=re.DOTALL,
    )
    yaml_text = re.sub(
        r"(expander:.*?RPM:\s*)\d+",
        rf"\g<1>{RPM}",
        yaml_text,
        count=1,
        flags=re.DOTALL,
    )
    return yaml_text


def _extract_results(cycle, config_file, n_stages, RPM):
    """Extract key results from a converged cycle."""
    data = cycle.problem.cycle_data
    components = data["components"]
    energy = data["energy_analysis"]
    config_name = Path(config_file).stem

    exp = components["expander"]
    data_out = exp.get("data_out", {})

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
    result["heat_utilization"] = energy.get("heat_utilization", None)
    result["m_dot_wf_kg_s"] = energy.get("mass_flow_working_fluid", None)
    result["m_dot_well_kg_s"] = energy.get("mass_flow_heating_fluid", None)

    # Expander
    result["eta_turbine"] = exp.get("efficiency", None)
    result["SP"] = data_out.get("size_parameter", None)
    result["SP_clamped"] = data_out.get("size_parameter_clamped", False)
    result["Vr"] = data_out.get("volume_ratio", None)
    result["Ns"] = data_out.get("specific_speed", None)
    result["Dh_is_kJ_kg"] = exp.get("isentropic_work", 0) / 1e3
    result["p_in_bar"] = exp["state_in"].p / 1e5
    result["p_out_bar"] = exp["state_out"].p / 1e5

    # Per-stage diagnostics
    stage_data = data_out.get("stage_data", [])
    if stage_data:
        etas = [sd["eta_stage"] for sd in stage_data]
        ns_vals = [sd["Ns"] for sd in stage_data]
        result["eta_min_stage"] = min(etas)
        result["Ns_max"] = max(ns_vals)
        result["any_stage_zero"] = any(e <= 0.0 for e in etas)

    return result


def _run_single(config_file, n_stages, RPM):
    """Run one optimization with specific stage count and RPM."""
    yaml_text = Path(config_file).read_text()
    yaml_text = _modify_yaml(yaml_text, n_stages, RPM)

    config_dir = Path(config_file).parent
    tmp_path = config_dir / f"_tmp_parametric_{n_stages}stg_{RPM}rpm.yaml"

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


def run_parametric(config_files, stages_list, rpms, output_dir):
    """Run the full n_stages × RPM parametric study."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    warnings.filterwarnings("once", message="Astolfi-stacking: SP")

    total_runs = len(config_files) * len(stages_list) * len(rpms)
    results = []
    run_count = 0

    print("=" * 76)
    print("  PARAMETRIC STUDY — Transcritical ORC Astolfi Stage-Stacking (Table 6.6)")
    print("=" * 76)
    print(f"  Configs : {len(config_files)}")
    print(f"  Stages  : {stages_list}")
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

        for RPM in rpms:
            for n in stages_list:
                run_count += 1
                print(
                    f"\n  [{run_count}/{total_runs}] " f"{n}-stage, {RPM} RPM ... ",
                    end="",
                    flush=True,
                )

                result = _run_single(config_file, n, RPM)
                results.append(result)

                if result.get("converged", False):
                    eta = result.get("eta_system", 0)
                    eta_pct = eta * 100 if eta else 0
                    W = result.get("W_net_kW", 0)
                    print(f"✓  η_sys = {eta_pct:.2f}%,  W_net = {W:.0f} kW")
                else:
                    print("✗  did not converge")

    # Save results
    df = pd.DataFrame(results)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    csv_path = output_dir / f"parametric_transcritical_results_{timestamp}.csv"
    xlsx_path = output_dir / f"parametric_transcritical_results_{timestamp}.xlsx"

    df.to_csv(csv_path, index=False)
    df.to_excel(xlsx_path, index=False, sheet_name="Results")

    # Print summary table
    print("\n\n" + "=" * 76)
    print("  PARAMETRIC STUDY — RESULTS SUMMARY")
    print("=" * 76)

    display_cols = ["config", "n_stages", "RPM", "converged"]
    for col in [
        "eta_system",
        "W_net_kW",
        "eta_turbine",
        "Ns_max",
        "any_stage_zero",
        "SP",
        "Vr",
        "m_dot_wf_kg_s",
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

    elif MODE == "parametric":
        run_parametric(
            config_files=PARAMETRIC_CONFIGS,
            stages_list=PARAMETRIC_STAGES,
            rpms=PARAMETRIC_RPMS,
            output_dir=PARAMETRIC_OUTPUT_DIR,
        )

    else:
        print(f"Error: MODE must be 'optimize', 'sweep', or 'parametric', got '{MODE}'")
        sys.exit(1)

    plt.show()
