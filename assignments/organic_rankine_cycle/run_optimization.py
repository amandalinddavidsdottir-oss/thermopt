import numpy as np
import matplotlib.pyplot as plt
import thermopt as th

# ------------------------------------------------------------------------------
# Part 1: Baseline optimization – Conventional single-phase ORC
# ------------------------------------------------------------------------------

# Define optimization problem
CONFIG_FILE = "./ORC_single_phase.yaml"
cycle_1p = th.ThermodynamicCycleOptimization(
    CONFIG_FILE, out_dir="results/single_phase"
)

# Interactive plot to manually inspect configuration before optimization
cycle_1p.problem.plot_cycle_realtime(CONFIG_FILE, update_interval=0.1)

# Run optimization and store results
cycle_1p.run_optimization()
cycle_1p.save_results()
plt.close(cycle_1p.problem.figure)

# Extract results
efficiency_expander_1p = cycle_1p.problem.cycle_data["components"]["expander"][
    "efficiency"
]
efficiency_system_1p = cycle_1p.problem.cycle_data["energy_analysis"][
    "system_efficiency"
]
