import os
import sys
import warnings
import thermopt as th
import matplotlib.pyplot as plt
from plot_TQ_diagram import plot_TQ_diagram
from fluid_sweep_BASIC_ORC import get_candidate_fluids, run_fluid_sweep, plot_results
from exergy_analysis import perform_exergy_analysis, plot_heat_source_utilization
from turbine_macchi_astolfi import evaluate_turbine_efficiency
from pathlib import Path

# Suppress the non-interactive matplotlib warning
warnings.filterwarnings("ignore", message="FigureCanvasAgg is non-interactive")

# Print package info
th.print_package_info()



# ══════════════════════════════════════════════════════════════════════
#  CONFIGURATION - Change this to switch between different setups
# ══════════════════════════════════════════════════════════════════════
#CONFIG_FILE = "./case_bjarnarflag_ORC_recuperated.yaml"
#CONFIG_FILE = "./case_bjarnarflag_ORC.yaml"
#CONFIG_FILE = "./case_bjarnarflag_ORC_Cyclohexane.yaml"
CONFIG_FILE = Path(__file__).with_name("case_bjarnarflag_ORC_Cyclohexane.yaml") #set config path relative to the script
#CONFIG_FILE = "./case_bjarnarflag_ORC_Cyclohexane_recuperated.yaml"

#To trouble shoot setup of Python environment:
#print("PYTHON:", sys.executable)
#print("CWD:", os.getcwd())
#print("CONFIG:", CONFIG_FILE)

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
try:
    exergy.plot_pie_chart(savefig=os.path.join(graph_dir, "exergy_pie.png"))
except ValueError as e:
    print(f"Skipping exergy pie chart: {e}")
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
# #  Option 2: Working fluid sweep for BASIC SUBCRITICAL ORC
# #  Uncomment the block below to run (takes a while — one optimization
# #  per fluid). Comment out Option 1 above if you only want the sweep.
# # ══════════════════════════════════════════════════════════════════════

# from fluid_sweep_BASIC_ORC import get_candidate_fluids, run_fluid_sweep, plot_results

# # Step 1: Get candidate fluids for subcritical ORC
# #   - Fluids with Tc < T_heat_source + Tc_margin are skipped (would need transcritical)
# #   - Tc_margin=15 means Tc must be at least 15°C above heat source
# candidates = get_candidate_fluids(T_heat_source=178.1, Tc_margin=15)

# # Step 2: Run optimization for each candidate
# #   - Bounds are calculated based on your system (heat source, heat sink, pinch)
# #   - Pinch values are read from your YAML constraints
# df = run_fluid_sweep("./case_bjarnarflag_ORC.yaml", candidates)

# # Step 3: Plot comparison
# plot_results(df, output_dir="results/fluid_sweep_BASIC_ORC")







# Keep plots open
plt.show()
