import os
import sys
import warnings
import thermopt as th
import matplotlib.pyplot as plt
from plot_TQ_diagram import plot_TQ_diagram
from fluid_sweep import get_candidate_fluids, run_fluid_sweep, plot_fluid_comparison
from exergy_analysis import perform_exergy_analysis, plot_heat_source_utilization
from turbine_macchi_astolfi import evaluate_turbine_efficiency

# Suppress the non-interactive matplotlib warning
warnings.filterwarnings("ignore", message="FigureCanvasAgg is non-interactive")

# Print package info
th.print_package_info()

# ══════════════════════════════════════════════════════════════════════
#  CONFIGURATION - Change this to switch between different setups
# ══════════════════════════════════════════════════════════════════════
#CONFIG_FILE = "./case_bjarnarflag_ORC_recuperated.yaml"
CONFIG_FILE = "./case_bjarnarflag_ORC.yaml"


# ══════════════════════════════════════════════════════════════════════
#  Option 1: Single-fluid optimization (your original workflow)
# ══════════════════════════════════════════════════════════════════════
cycle = th.ThermodynamicCycleOptimization(CONFIG_FILE)
# cycle.problem.plot_cycle_realtime(CONFIG_FILE, update_interval=0.1)

cycle.run_optimization()

# Save thermopt results (suppress its debug prints)
_stdout = sys.stdout
sys.stdout = open(os.devnull, "w")
cycle.save_results()
sys.stdout.close()
sys.stdout = _stdout

# Create output folder for all post-processing graphs
graph_dir = os.path.join(cycle.out_dir, "graphs")
os.makedirs(graph_dir, exist_ok=True)

# Exergy analysis
exergy = perform_exergy_analysis(cycle, config_file=CONFIG_FILE)
exergy.print_summary()
exergy.to_excel(os.path.join(graph_dir, "exergy_results.xlsx"))
exergy.plot_exergy_destruction(savefig=os.path.join(graph_dir, "exergy_bar.png"))
exergy.plot_pie_chart(savefig=os.path.join(graph_dir, "exergy_pie.png"))
exergy.plot_grassmann(savefig=os.path.join(graph_dir, "exergy_grassmann.png"))

# Heat source utilization curve
fig, axes, sweep = plot_heat_source_utilization(
    cycle, config_file=CONFIG_FILE,
    savefig=os.path.join(graph_dir, "utilization_curve.png"),
)

# T-Q diagrams
plot_TQ_diagram(cycle, "heater", output_dir=graph_dir)
plot_TQ_diagram(cycle, "cooler", output_dir=graph_dir)

# Post-processing to compare to the assumed efficiency
turb = evaluate_turbine_efficiency(cycle, RPM=3000, stages=[1, 2, 3])
turb.print_summary()

# ── Clean summary ─────────────────────────────────────────────────
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
# #  Option 2: Working fluid sweep
# #  Uncomment the block below to run (takes a while — one optimization
# #  per fluid). Comment out Option 1 above if you only want the sweep.
# # ══════════════════════════════════════════════════════════════════════

# #Step 1: Screen fluids by critical temperature
# #  Wide range to include all potentially viable fluids for 207°C heat source
# #  Tc_min > condensing temp (~30-40°C) with margin
# #  Tc_max ≈ heat source temp (207°C)
# candidates = get_candidate_fluids(Tc_min=60, Tc_max=300)

# # Step 2: Run optimization for each candidate (uses same CONFIG_FILE as Option 1)
# df = run_fluid_sweep(CONFIG_FILE, candidates)

# # Step 3: Plot comparison (saves into the fluid_sweep results folder)
# plot_fluid_comparison(df, output_dir="results/fluid_sweep")

# Keep plots open
plt.show()
