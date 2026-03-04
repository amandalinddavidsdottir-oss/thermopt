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

#---- importing post processing files:
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "shared_utilities"))
from exergy_analysis import perform_exergy_analysis, plot_heat_source_utilization
from plot_TQ_diagram import plot_TQ_diagram
from validation_checks import run_validation_checks
#-----

warnings.filterwarnings("ignore", message="FigureCanvasAgg is non-interactive")


# ══════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════
MODE = "optimize"  # "optimize" | "sweep" | "parametric"

#CONFIG_FILE = Path(__file__).with_name("case_Toluene_simpleORC.yaml")
#CONFIG_FILE = Path(__file__).with_name("case_Toluene_recuperated_simpleORC.yaml")
#WITH MASSFLOW RATE:
#CONFIG_FILE = Path(__file__).with_name("case_Toluene_simpleORC_mass_flow.yaml")
#CONFIG_FILE = Path(__file__).with_name("case_Toluene_recuperated_simpleORC_mass_flow.yaml")
#WITH Astolfi stacking correlations:
CONFIG_FILE = Path(__file__).with_name("case_Toluene_simpleORC_macchi_astolfi.yaml")
#CONFIG_FILE = Path(__file__).with_name("case_Toluene_recuperated_simpleORC_macchi_astolfi.yaml")

SWEEP_OUTPUT_DIR = "results/fluid_sweep_BASIC_ORC"

# ── Parametric study settings ──
PARAMETRIC_CONFIGS = [
    Path(__file__).with_name("case_Toluene_simpleORC_macchi_astolfi.yaml"),
    Path(__file__).with_name("case_Toluene_recuperated_simpleORC_macchi_astolfi.yaml"),
    # Path(__file__).with_name("case_Toluene_dual_pressure_macchi_astolfi.yaml"),
    # Path(__file__).with_name("case_Toluene_recuperated_dual_pressure_macchi_astolfi.yaml"),
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
    print(f"  Well (heating fluid)               :  {ea['mass_flow_heating_fluid']:.2f} kg/s")
    print(f"  Working fluid                      :  {ea['mass_flow_working_fluid']:.2f} kg/s")
    print(f"  Cooling fluid                      :  {ea['mass_flow_cooling_fluid']:.2f} kg/s")
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
        cycle, config_file=config_file,
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

    from fluid_sweep_BASIC_ORC import get_candidate_fluids, run_fluid_sweep, plot_results

    candidates = get_candidate_fluids(config_file)
    df = run_fluid_sweep(config_file, candidates, output_dir=output_dir)
    plot_results(df, output_dir=output_dir)


# ══════════════════════════════════════════════════════════════════════
#  MODE 3: PARAMETRIC STUDY (n_stages × RPM)
# ══════════════════════════════════════════════════════════════════════

def _modify_yaml(yaml_text, n_stages, RPM):
    """
    Replace n_stages and RPM values in raw YAML text via regex.
    Preserves all other content (expressions, comments, warm starts).
    Works for dual-pressure configs too (replaces all occurrences).
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
    result["Q_in_kW"] = energy.get("total_heat_input",
                                    energy.get("heater_heat_flow", 0)) / 1e3
    result["Q_available_kW"] = energy.get("available_heat", 0) / 1e3
    result["heat_utilization"] = (result["Q_in_kW"] / result["Q_available_kW"]
                                   if result["Q_available_kW"] > 0 else None)
    result["m_dot_wf_kg_s"] = energy.get("mass_flow_working_fluid", None)
    result["m_dot_well_kg_s"] = energy.get("mass_flow_heating_fluid", None)

    # Expander(s)
    if "hp_expander" in components:
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

    # Recuperator (if present)
    if "recuperator" in components:
        Q_rec = energy.get("recuperator_heat_flow", 0) / 1e3
        result["Q_recuperator_kW"] = Q_rec

    return result


def _run_single(config_file, n_stages, RPM):
    """Run one optimization with specific n_stages and RPM."""
    yaml_text = Path(config_file).read_text()
    yaml_text = _modify_yaml(yaml_text, n_stages, RPM)

    config_dir = Path(config_file).parent
    tmp_path = config_dir / f"_tmp_parametric_n{n_stages}_rpm{RPM}.yaml"

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


def run_parametric(config_files, stages, rpms, output_dir):
    """Run the full n_stages × RPM parametric study."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Suppress SP warning during batch (reported in results table)
    warnings.filterwarnings("once", message="Astolfi-stacking: SP")

    total_runs = len(config_files) * len(stages) * len(rpms)
    results = []
    run_count = 0

    print("=" * 76)
    print("  PARAMETRIC STUDY — Astolfi Stage-Stacking (Table 6.6)")
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

                result = _run_single(config_file, n_stages, RPM)
                results.append(result)

                if result.get("converged", False):
                    eta = result.get("eta_system", result.get("HP_eta_turbine", 0))
                    eta_pct = eta * 100 if eta else 0
                    W = result.get("W_net_kW", 0)
                    print(f"✓  η_sys = {eta_pct:.2f}%,  W_net = {W:.0f} kW")
                else:
                    print("✗  did not converge")

    # Save results
    df = pd.DataFrame(results)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    csv_path = output_dir / f"parametric_results_{timestamp}.csv"
    xlsx_path = output_dir / f"parametric_results_{timestamp}.xlsx"

    df.to_csv(csv_path, index=False)
    df.to_excel(xlsx_path, index=False, sheet_name="Results")

    # Print summary table
    print("\n\n" + "=" * 76)
    print("  PARAMETRIC STUDY — RESULTS SUMMARY")
    print("=" * 76)

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
            stages=PARAMETRIC_STAGES,
            rpms=PARAMETRIC_RPMS,
            output_dir=PARAMETRIC_OUTPUT_DIR,
        )

    else:
        print(f"Error: MODE must be 'optimize', 'sweep', or 'parametric', got '{MODE}'")
        sys.exit(1)

    plt.show()
