import numpy as np
import matplotlib.pyplot as plt
import thermopt as th

# ------------------------------------------------------------------------------
# Part 1: Baseline optimization – Conventional Joule-Thomson expansion
# ------------------------------------------------------------------------------

# Define optimization problem
CONFIG_FILE = "./transcritical_heat_pump.yaml"
cycle_1 = th.ThermodynamicCycleOptimization(CONFIG_FILE, out_dir="results/JT_valve")

# Interactive plot to manually inspect configuration before optimization
cycle_1.problem.plot_cycle_realtime(CONFIG_FILE, update_interval=0.1)

# Run optimization and store results
cycle_1.set_config_value(
    "problem_formulation.fixed_parameters.expander.efficiency", 0.0
)
cycle_1.run_optimization()
cycle_1.save_results()
plt.close(cycle_1.problem.figure)

# Extract results
efficiency_expander_1 = cycle_1.problem.cycle_data["components"]["expander"][
    "efficiency"
]
efficiency_COP_1 = cycle_1.problem.cycle_data["energy_analysis"]["COP_heat_pump"]
