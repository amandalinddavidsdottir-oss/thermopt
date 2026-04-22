import os
import pathlib
import numpy as np
import matplotlib.pyplot as plt

import thermopt as th

th.set_plot_options(fontsize=16)

CONFIG_FILE = "../case_PTES_CO2.yaml"
OUT_DIR_BASE = "results"
DATA_FILE = "simulation_data.pkl"
DATA_FULLPATH = os.path.join(OUT_DIR_BASE, DATA_FILE)

# Parameter space: from high temp to low, and high to low pressure
hot_temp_range = np.linspace(800, 400, 17) + 273.15  # K
max_pressure_values = np.asarray([300, 250, 200]) * 1e5  # Pa
# hot_temp_range = [800+273.15]
# max_pressure_values = np.asarray([250])*1e5  # Pa

# --------------------------------------------------------------------- #
# Step 1: Run simulations if data does not exist
# --------------------------------------------------------------------- #
if not os.path.exists(DATA_FULLPATH):
    solvers = []
    x0 = None
    for T_hot in hot_temp_range:
        temp_solvers = []
        for p_max in max_pressure_values:
            print()
            print(80 * "-")
            print(
                f"Hot temp: {T_hot - 273.15:.2f} degC, Max pressure: {p_max / 1e5:.2f} bar"
            )
            print(80 * "-")

            # Output folder
            folder = f"T_{int(T_hot-273.15)}_P_{int(p_max/1e5)}bar"
            out_dir = os.path.join(OUT_DIR_BASE, folder)
            os.makedirs(out_dir, exist_ok=True)

            # Set up cycle and config
            cycle = th.ThermodynamicCycleOptimization(CONFIG_FILE, out_dir=out_dir)
            # cycle.set_config_value("solver_options.max_iterations", 5)
            # cycle.set_config_value("solver_options.problem_scale", 20)
            # cycle.set_config_value("solver_options.callbacks.plot_cycle", True)
            # cycle.set_config_value("solver_options.tolerance", 1e-5)

            # Extend the bounds to enable a feasible solution
            var_temp = (
                "problem_formulation.design_variables.hot_storage_upper_temperature"
            )
            cycle.set_config_value(f"{var_temp}.max", T_hot + 100)
            var_temp = (
                "problem_formulation.design_variables.hot_storage_lower_temperature"
            )
            cycle.set_config_value(f"{var_temp}.max", T_hot + 100)

            # Set temperature and pressure constraints
            cycle.set_constraint(
                variable="$components.cooler_charge.cold_side.state_out.T",
                type="=",
                value=T_hot - 0.1,
                normalize=True,
            )
            cycle.set_constraint(
                variable="$components.expander_charge.state_in.p",
                type="<",
                value=p_max,
                normalize=True,
            )
            cycle.set_constraint(
                variable="$components.expander_discharge.state_in.p",
                type="<",
                value=p_max,
                normalize=True,
            )

            # Run and save
            cycle.run_optimization(x0=x0)
            cycle.save_results()
            temp_solvers.append(cycle.solver)
            x0 = cycle.solver.x_final
        solvers.append(temp_solvers)

    th.save_to_pickle(solvers, DATA_FULLPATH, timestamp=False)
    print("All simulations completed.")
else:
    print("Loading existing simulation data...")
    solvers = th.load_from_pickle(DATA_FULLPATH)

# --------------------------------------------------------------------- #
# Step 2: Plot simulation results
# --------------------------------------------------------------------- #
fig, ax = plt.subplots(figsize=(6.0, 4.8))
colors = plt.get_cmap("magma")(np.linspace(0.2, 0.8, len(max_pressure_values)))

for j, p_max in enumerate(max_pressure_values):
    rte = [
        solver[j].problem.cycle_data["energy_analysis"]["roundtrip_efficiency"]
        for solver in solvers
    ]
    ax.plot(
        hot_temp_range - 273.15,
        100 * np.array(rte),
        label=rf"$p_\text{{max}}={p_max/1e5:.0f}$ bar",
        color=colors[j],
        marker="o",
    )

ax.set_xlabel("Hot storage maximum temperature (°C)")
ax.set_ylabel("Round-trip efficiency (%)")
ax.legend(fontsize=13, loc="lower right")
ax.grid(True)
fig.tight_layout(pad=1)

# Save
filename = f"{pathlib.Path(__file__).parent.name}_RTE"
th.savefig_in_formats(fig, os.path.join(OUT_DIR_BASE, filename))

plt.show()
