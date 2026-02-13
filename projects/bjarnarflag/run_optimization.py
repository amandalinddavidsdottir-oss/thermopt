import os
import sys
import warnings
import thermopt as th
import matplotlib.pyplot as plt
from pathlib import Path

# Suppress the non-interactive matplotlib warning
warnings.filterwarnings("ignore", message="FigureCanvasAgg is non-interactive")


# ══════════════════════════════════════════════════════════════════════
#  CORE OPTIMIZATION  (same as the original run_optimization.py)
# ══════════════════════════════════════════════════════════════════════
CONFIG_FILE = Path(__file__).with_name("case_Cyclohexane_simpleORC.yaml")

th.print_package_info()

cycle = th.ThermodynamicCycleOptimization(CONFIG_FILE)
cycle.problem.plot_cycle_realtime(CONFIG_FILE, update_interval=0.1)

cycle.run_optimization()
cycle.save_results()

# cycle.create_animation(format="mp4", fps=1)  # optional







# ══════════════════════════════════════════════════════════════════════
#  POST-PROCESSING  (additional analysis added on top of core)
# ══════════════════════════════════════════════════════════════════════
graph_dir = os.path.join(cycle.out_dir, "graphs")
os.makedirs(graph_dir, exist_ok=True)

# ── Exergy analysis ───────────────────────────────────────────────
from exergy_analysis import perform_exergy_analysis, plot_heat_source_utilization

exergy = perform_exergy_analysis(cycle, config_file=CONFIG_FILE)
exergy.print_summary()
exergy.to_excel(os.path.join(graph_dir, "exergy_results.xlsx"))
exergy.plot_exergy_destruction(savefig=os.path.join(graph_dir, "exergy_bar.png"))
try:
    exergy.plot_pie_chart(savefig=os.path.join(graph_dir, "exergy_pie.png"))
except ValueError as e:
    print(f"Skipping exergy pie chart: {e}")
exergy.plot_grassmann(savefig=os.path.join(graph_dir, "exergy_grassmann.png"))

# ── Heat source utilization curve ─────────────────────────────────
fig, axes, sweep = plot_heat_source_utilization(
    cycle, config_file=CONFIG_FILE,
    savefig=os.path.join(graph_dir, "utilization_curve.png"),
)

# ── T-Q diagrams ─────────────────────────────────────────────────
from plot_TQ_diagram import plot_TQ_diagram

plot_TQ_diagram(cycle, "heater", output_dir=graph_dir)
plot_TQ_diagram(cycle, "cooler", output_dir=graph_dir)

# ── Turbine efficiency estimate ───────────────────────────────────
from turbine_macchi_astolfi import evaluate_turbine_efficiency

turb = evaluate_turbine_efficiency(cycle, RPM=3000, stages=[1, 2, 3])
turb.print_summary()


# ══════════════════════════════════════════════════════════════════════
#  SUMMARY
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  ALL RESULTS SAVED")
print("=" * 60)
print(f"  Thermopt results : {cycle.out_dir}")
print(f"  Graphs & exergy  : {graph_dir}")
print()
for f in sorted(os.listdir(graph_dir)):
    print(f"    ✓ {f}")
print("=" * 60 + "\n")


# # ══════════════════════════════════════════════════════════════════════
# #  OPTIONAL: Working fluid sweep for BASIC SUBCRITICAL ORC
# #  Uncomment to run (takes a while — one optimization per fluid).
# #  Comment out everything above except CONFIG if you only want sweep.
# # ══════════════════════════════════════════════════════════════════════
# from fluid_sweep_BASIC_ORC import get_candidate_fluids, run_fluid_sweep, plot_results
#
# candidates = get_candidate_fluids(T_heat_source=178.1, Tc_margin=15)
# df = run_fluid_sweep("./case_bjarnarflag_ORC.yaml", candidates)
# plot_results(df, output_dir="results/fluid_sweep_BASIC_ORC")


# Keep plots open
plt.show()
