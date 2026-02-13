import thermopt as th
import matplotlib.pyplot as plt
from pathlib import Path


# Print package info
th.print_package_info()

# Define configuration filename
HERE = Path(__file__).resolve().parent
config_files = [str(HERE / "case_Cyclohexane_ORC_prufa.yaml")]
#config_files = ["./case_butane_ORC.yaml"]
# config_files = ["./case_butane_PEORC.yaml"]
# config_files = ["./case_butane_transcritical.yaml"]
# config_files = ["./case_butane_ORC.yaml", "./case_butane_PEORC.yaml","./case_butane_transcritical.yaml"]
# config_files = ["./case_Novec649_PEORC.yaml"]

# Loop over all cases
for config in config_files:

    # Initialize Brayton cycle problem
    cycle = th.ThermodynamicCycleOptimization(config)
    cycle.problem.plot_cycle_realtime(config, update_interval=0.1)

    # Perform cycle optimization
    cycle.run_optimization()
    cycle.save_results()

    # # Create an animation of the optimization progress
    # cycle.create_animation(format="mp4", fps=1)

# Keep plots open
plt.show()


