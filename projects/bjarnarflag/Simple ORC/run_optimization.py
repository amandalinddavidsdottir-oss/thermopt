import os
import sys
import warnings
import thermopt as th
import matplotlib.pyplot as plt
from pathlib import Path

#---- importing post processing files:
sys.path.insert(0, str(Path(__file__).resolve().parent / "shared_utilities"))
from exergy_analysis import perform_exergy_analysis, plot_heat_source_utilization
from plot_TQ_diagram import plot_TQ_diagram
from turbine_macchi_astolfi import evaluate_turbine_efficiency
from validation_checks import run_validation_checks
#-----

warnings.filterwarnings("ignore", message="FigureCanvasAgg is non-interactive")


# ══════════════════════════════════════════════════════════════════════
#  CONFIGURATION — change these two settings
# ══════════════════════════════════════════════════════════════════════
MODE = "optimize"  # "optimize" = single fluid  |  "sweep" = working fluid sweep
CONFIG_FILE = Path(__file__).with_name("case_Toluene_simpleORC.yaml")
#CONFIG_FILE = Path(__file__).with_name("case_Toluene_recuperated_simpleORC.yaml")
SWEEP_OUTPUT_DIR = "results/fluid_sweep_BASIC_ORC"


# ══════════════════════════════════════════════════════════════════════
#  SINGLE-FLUID OPTIMIZATION
# ══════════════════════════════════════════════════════════════════════
def run_optimize(config_file):
    """Run optimization + full post-processing for a single YAML config."""

    th.print_package_info()

    cycle = th.ThermodynamicCycleOptimization(config_file)
    cycle.problem.plot_cycle_realtime(config_file, update_interval=0.1)

    cycle.run_optimization()
    cycle.save_results()

    # ──────────────────────── VALIDATION CHECKS ───────────────────────
    run_validation_checks(cycle)

    # ── Post-processing ───────────────────────────────────────────
    graph_dir = os.path.join(cycle.out_dir, "graphs")
    os.makedirs(graph_dir, exist_ok=True)

    # Exergy analysis
    from exergy_analysis import perform_exergy_analysis, plot_heat_source_utilization

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
    from plot_TQ_diagram import plot_TQ_diagram
    plot_TQ_diagram(cycle, "heater", output_dir=graph_dir)
    plot_TQ_diagram(cycle, "cooler", output_dir=graph_dir)

    # Turbine efficiency estimate
    from turbine_macchi_astolfi import evaluate_turbine_efficiency
    turb = evaluate_turbine_efficiency(cycle, RPM=3000, stages=[1, 2, 3])
    turb.print_summary()

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
#  WORKING FLUID SWEEP
# ══════════════════════════════════════════════════════════════════════
def run_sweep(config_file, output_dir):
    """Run fluid sweep across all candidates using config as template."""

    from fluid_sweep_BASIC_ORC import get_candidate_fluids, run_fluid_sweep, plot_results

    candidates = get_candidate_fluids(config_file)
    df = run_fluid_sweep(config_file, candidates, output_dir=output_dir)
    plot_results(df, output_dir=output_dir)


# ══════════════════════════════════════════════════════════════════════
#  RUN
# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":

    if not CONFIG_FILE.exists():
        print(f"Error: config file not found: {CONFIG_FILE}")
        sys.exit(1)

    print(f"Config: {CONFIG_FILE.name}")
    print(f"Mode:   {MODE}")
    print()

    if MODE == "sweep":
        run_sweep(CONFIG_FILE, SWEEP_OUTPUT_DIR)
    elif MODE == "optimize":
        run_optimize(CONFIG_FILE)
    else:
        print(f"Error: MODE must be 'optimize' or 'sweep', got '{MODE}'")
        sys.exit(1)

    plt.show()
