import numpy as np
import thermopt as th
import matplotlib.pyplot as plt

# Print package info
th.print_package_info()

# Initialize cycle problem
CONFIG_FILE = "../case_PTES_CO2.yaml"
# CONFIG_FILE = "./case_PTES_CO2_turbo.yaml"
cycle = th.ThermodynamicCycleOptimization(CONFIG_FILE, out_dir="results_baseline")
cycle.problem.plot_cycle_realtime(CONFIG_FILE, update_interval=0.1, write_report=True)

th.savefig_in_formats(cycle.problem.figure, "results_baseline/initial_configuration")

cycle.set_config_value("solver_options.callbacks.plot_cycle", True)
cycle.set_config_value("solver_options.callbacks.save_plot", True)
# cycle.set_config_value("solver_options.max_iterations", 5)
# cycle.set_config_value("solver_options.problem_scale", 10)

# Perform cycle optimization
cycle.run_optimization()
cycle.save_results()

# Save solver object
th.save_to_pickle(cycle.solver, "results_baseline/solver.pkl", timestamp=False)


# Create an animation of the optimization progress
cycle.create_animation(format="mp4", fps=1)

# Keep plots open
plt.show()


# # Hardcoded turbomachinery parameters
# specific_speed = 0.7         # nondimensional
# flow_coefficient = 0.06      # nondimensional
# backsweep_angle = 25.0       # degrees
# variables_to_set = {
#     # Charge side
#     "compressor_flow_coefficient_charge": flow_coefficient,
#     "compressor_backsweep_charge": backsweep_angle,
#     "expander_specific_speed_charge": specific_speed,

#     # Discharge side
#     "compressor_flow_coefficient_discharge": flow_coefficient,
#     "compressor_backsweep_discharge": backsweep_angle,
#     "expander_specific_speed_discharge": specific_speed,
# }

# # Set min, max, and value to the same number
# for var_name, value in variables_to_set.items():
#     base_key = f"problem_formulation.design_variables.{var_name}"
#     cycle.set_config_value(f"{base_key}.min", value)
#     cycle.set_config_value(f"{base_key}.max", value)
#     cycle.set_config_value(f"{base_key}.value", value)
