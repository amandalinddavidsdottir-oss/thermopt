import os
import sys
import warnings
import thermopt as th
import matplotlib.pyplot as plt
from pathlib import Path


warnings.filterwarnings("ignore", message="FigureCanvasAgg is non-interactive")


# ══════════════════════════════════════════════════════════════════════
#  CONFIGURATION — change these two settings
# ══════════════════════════════════════════════════════════════════════
MODE = "optimize"  # "optimize" = single fluid  |  "sweep" = working fluid sweep
CONFIG_FILE = Path(__file__).with_name("case_Toluene_dualORC.yaml")
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

    # ────────────────────────────POST-PROCESSING ───────────────────────────────────────────
    graph_dir = os.path.join(cycle.out_dir, "graphs")
    os.makedirs(graph_dir, exist_ok=True)

    #------
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

    # Heat source utilization curve (not applicable for dual-pressure ORC
    # because brine exit temperature is computed, not a design variable)
    # fig, axes, sweep = plot_heat_source_utilization(
    #     cycle, config_file=config_file,
    #     savefig=os.path.join(graph_dir, "utilization_curve.png"),
    # )
    #-------



    #------
    #Plotting the state points on the graph:
    cycle.problem.plot_cycle()
    components = cycle.problem.cycle_data["components"]
    state_points = {
        "1":  components["lp_pump"]["state_in"],
        "2":  components["lp_pump"]["state_out"],
        "3":  components["hp_pump"]["state_in"],
        "3'": components["hp_pump"]["state_out"],
        "4":  components["lp_evaporator"]["cold_side"]["state_out"],
        "6":  components["hp_expander"]["state_in"],
        "7":  components["hp_expander"]["state_out"],
        "8":  components["lp_expander"]["state_in"],
        "9":  components["lp_expander"]["state_out"],
    }
    offsets = {
        "1":  ( -12.6,   -2.6),
        "2":  (  -6.8,    6.7),
        "3":  (  -3.2,    9.3),
        "3'": ( -11.9,    1.1),
        "4":  (  -6.6,  -12.6),
        "6":  (   5.1,   -0.6),
        "7":  (   6.6,   -5.4),
        "8":  (   5.7,   -6.4),
        "9":  (   5.0,   -4.6),
    }
    fig_ts = cycle.problem.figure
    ax_ts = fig_ts.axes[0]
    for label, st in state_points.items():
        dx, dy = offsets[label]
        ax_ts.annotate(
            label, (st.s, st.T),
            textcoords="offset points", xytext=(dx, dy),
            fontsize=9, fontweight="bold", zorder=20,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                      edgecolor="none", alpha=0.85),
        )
    fig_ts.savefig(os.path.join(graph_dir, "Ts_labeled.png"),
                   dpi=200, bbox_inches="tight")
    plt.close(fig_ts)
    #-------

    #-------
    # T-Q diagrams of the Heat Exchangers
    from plot_TQ_diagram import plot_TQ_diagram
    plot_TQ_diagram(cycle, "hp_evaporator", output_dir=graph_dir)
    plot_TQ_diagram(cycle, "lp_evaporator", output_dir=graph_dir)
    plot_TQ_diagram(cycle, "preheater", output_dir=graph_dir)
    plot_TQ_diagram(cycle, "cooler", output_dir=graph_dir)
    #-------

    # Turbine efficiency estimate (Macchi & Astolfi correlation)
    from turbine_macchi_astolfi import evaluate_turbine_efficiency
    turb = evaluate_turbine_efficiency(cycle, RPM=3000, stages=[1, 2, 3])
    # Returns a list for dual-pressure, single result for single-pressure
    if isinstance(turb, list):
        for t in turb:
            t.print_summary()
    else:
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

    from fluid_sweep_BASIC_ORC import get_candidate_fluids, run_fluid_sweep, plot_results # THIS NEEDS TO BE UPDATED LATER FOR DUAL PRESSURE ORC, FOR NOW USING THE SAME FLUID

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
