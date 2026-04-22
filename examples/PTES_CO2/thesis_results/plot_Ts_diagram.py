import os
import numpy as np
import matplotlib.pyplot as plt
import thermopt as th

# Global settings
th.set_plot_options(fontsize=16)

CONFIG_FILE = "../case_PTES_CO2.yaml"
OUT_DIR_BASE = "system_diagram"
os.makedirs(OUT_DIR_BASE, exist_ok=True)

# Parameter ranges
pinch = 20  # Fraction
eta = 0.80  # Fraction

print()
print("-" * 80)
print(f"Turbomachinery efficiency: {eta*100:.1f} %, Pinch point: {pinch:.1f} K")
print("-" * 80)

# Output directory
folder = f"eff_{int(eta*100)}_pinch_{pinch}"
out_dir = os.path.join(OUT_DIR_BASE, folder)
os.makedirs(out_dir, exist_ok=True)

# Set up cycle
cycle = th.ThermodynamicCycleOptimization(CONFIG_FILE, out_dir=out_dir)
cycle.set_config_value("solver_options.max_iterations", 200)
cycle.set_config_value("solver_options.callbacks.plot_cycle", True)

# Set all turbomachinery efficiencies
for machine in ["expander", "compressor"]:
    for side in ["charge", "discharge"]:
        var = f"problem_formulation.fixed_parameters.{machine}_{side}.efficiency"
        cycle.set_config_value(var, eta)

# Set pinch point constraints for all heat exchangers
for hx in ["heater", "cooler", "recuperator"]:
    for side in ["charge", "discharge"]:
        var = f"$components.{hx}_{side}.temperature_difference"
        cycle.set_constraint(variable=var, type=">", value=pinch, normalize=10.0)

# Run and save
cycle.run_optimization()
cycle.save_results()
th.savefig_in_formats(cycle.problem.figure, os.path.join(out_dir, "Ts_diagram"))
