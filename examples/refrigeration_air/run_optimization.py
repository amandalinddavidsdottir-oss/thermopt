import thermopt as th
import matplotlib.pyplot as plt

# Print package info
th.print_package_info()

# Read configuration file
CONFIG_FILE = "./case_air_refrigeration_recuperated.yaml"
config = th.read_configuration_file(CONFIG_FILE)

# Initialize cycle problem
cycle = th.ThermodynamicCycleOptimization(CONFIG_FILE)
cycle.problem.plot_cycle_realtime(CONFIG_FILE, update_interval=0.1)

# Perform cycle optimization
cycle.run_optimization()
cycle.save_results()

# Keep plots open
plt.show()
