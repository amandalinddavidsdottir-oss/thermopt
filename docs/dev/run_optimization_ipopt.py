import thermopt as th
import matplotlib.pyplot as plt

# Print package info
th.print_package_info()

# Define configuration filename
CONFIG_FILE = "case_sCO2_recuperated.yaml"

# Initialize Brayton cycle problem
cycle = th.ThermodynamicCycleOptimization(CONFIG_FILE)
cycle.problem.plot_cycle_realtime(CONFIG_FILE)

# Perform cycle optimization
cycle.run_optimization()
# cycle.save_results()
cycle.plot_convergence_history()

# # Create an animation of the optimization progress
# cycle.create_animation(format="mp4", fps=1.0)

# Keep plots open
plt.show()
