import thermopt as th
import matplotlib.pyplot as plt

# Print package info
th.print_package_info()

# Define configuration filename
CONFIG_FILE = "./case_R134a_refrigeration.yaml"

# Initialize Brayton cycle problem
cycle = th.ThermodynamicCycleOptimization(CONFIG_FILE)
cycle.problem.plot_cycle_realtime(CONFIG_FILE, update_interval=0.1)

# Perform cycle optimization
cycle.run_optimization()
cycle.save_results()

# Create an animation of the optimization progress
cycle.create_animation(format="mp4", fps=1)

# Keep plots open
plt.show()
