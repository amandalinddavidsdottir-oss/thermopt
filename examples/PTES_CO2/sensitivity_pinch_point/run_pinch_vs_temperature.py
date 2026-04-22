import os
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import thermopt as th

# Global settings
th.set_plot_options(fontsize=16)

CONFIG_FILE = "../case_PTES_CO2.yaml"
OUT_DIR_BASE = "results"
DATA_FILE = "simulation_data.pkl"
DATA_FULLPATH = os.path.join(OUT_DIR_BASE, DATA_FILE)

os.makedirs(OUT_DIR_BASE, exist_ok=True)

# Parameter space
pinch_deltas = np.linspace(20, 0, 21)  # K
hot_temperatures = np.array([400, 500, 600, 700, 800]) + 273.15  # K
# hot_temperatures = np.array([400, 500, 600, 700, 800]) + 273.15  # K
# hot_temperatures = np.array([600]) + 273.15  # K

# pinch_deltas = np.linspace(20, 0, 11)  # K
# hot_temperatures = np.array([400, 600, 800]) + 273.15  # K

# pinch_deltas = np.array([20])  # K
# hot_temperatures = np.array([600]) + 273.15  # K


# --------------------------------------------------------------------- #
# Step 1: Run simulations if data file does not exist
# --------------------------------------------------------------------- #
if not os.path.exists(DATA_FULLPATH):
    solvers = []
    x0_T_start = None

    for T_hot in hot_temperatures:
        T_solvers = []
        x0 = x0_T_start  # Initial guess for this temperature level
        x0 = None

        for i, pinch in enumerate(pinch_deltas):
            print()
            print("-" * 80)
            print(
                f"Hot storage temp: {T_hot - 273.15:.1f} degC, Pinch point: {pinch:.1f} degC"
            )
            print("-" * 80)

            # Output folder
            folder = f"T_{int(T_hot-273.15)}_pinsch_{pinch}"
            out_dir = os.path.join(OUT_DIR_BASE, folder)
            os.makedirs(out_dir, exist_ok=True)

            # Set up cycle
            cycle = th.ThermodynamicCycleOptimization(CONFIG_FILE, out_dir=out_dir)
            cycle.set_config_value("solver_options.max_iterations", 200)
            cycle.set_config_value("solver_options.problem_scale", 10)
            cycle.set_config_value("solver_options.callbacks.plot_cycle", False)

            # Set hot storage temperature limits and constraint for target
            cycle.set_config_values(
                {
                    "problem_formulation.design_variables.hot_storage_lower_temperature.value": T_hot
                    - 100,
                    "problem_formulation.design_variables.hot_storage_lower_temperature.min": 200
                    + 273.15,
                    "problem_formulation.design_variables.hot_storage_lower_temperature.max": 1000
                    + 273.15,
                    "problem_formulation.design_variables.hot_storage_upper_temperature.value": T_hot,
                    "problem_formulation.design_variables.hot_storage_upper_temperature.min": 200
                    + 273.15,
                    "problem_formulation.design_variables.hot_storage_upper_temperature.max": 1000
                    + 273.15,
                    "problem_formulation.design_variables.expander_inlet_enthalpy_charge.value": "1.3*$working_fluid.critical_point.h",
                    "problem_formulation.design_variables.compressor_inlet_enthalpy_discharge.value": "1.3*$working_fluid.critical_point.h",
                    # "problem_formulation.design_variables.expander_inlet_pressure_charge.max": 500e5,
                    # "problem_formulation.design_variables.expander_inlet_pressure_discharge.max": 500e5,
                }
            )
            # cycle.set_config_values({
            #     "problem_formulation.design_variables.hot_storage_lower_temperature.value": T_hot - 100,
            #     "problem_formulation.design_variables.hot_storage_lower_temperature.min": T_hot - 105,
            #     "problem_formulation.design_variables.hot_storage_lower_temperature.max": T_hot - 95,
            #     "problem_formulation.design_variables.hot_storage_upper_temperature.value": T_hot,
            #     "problem_formulation.design_variables.hot_storage_upper_temperature.min": T_hot - 10,
            #     "problem_formulation.design_variables.hot_storage_upper_temperature.max": T_hot + 10,
            # })

            cycle.set_constraint(
                variable="$components.cooler_charge.cold_side.state_out.T",
                type="=",
                value=T_hot,
                normalize=True,
            )

            # Set pinch point constraints for all heat exchangers
            for hx in ["heater", "cooler", "recuperator"]:
                for side in ["charge", "discharge"]:
                    var = f"$components.{hx}_{side}.temperature_difference"
                    cycle.set_constraint(
                        variable=var, type=">", value=pinch, normalize=10.0
                    )

            # Run and save
            cycle.run_optimization(x0=x0)
            cycle.save_results()
            T_solvers.append(cycle.solver)

            # Update initial guess
            x0 = cycle.solver.x_final

            # Save the x0 for the next T_hot loop
            if i == 0:
                x0_T_start = x0

        solvers.append(T_solvers)

    th.save_to_pickle(solvers, DATA_FULLPATH, timestamp=False)
    print("All simulations completed.")
else:
    print("Loading existing simulation data...")
    solvers = th.load_from_pickle(DATA_FULLPATH)


# --------------------------------------------------------------------- #
# Step 2: Plot round-trip efficiency vs. pinch point
# --------------------------------------------------------------------- #
fig, ax = plt.subplots(figsize=(6.0, 4.8))
colors = plt.get_cmap("magma")(np.linspace(0.2, 0.8, len(hot_temperatures)))

for i, T_hot in enumerate(hot_temperatures):
    rte = [
        solver.problem.cycle_data["energy_analysis"]["roundtrip_efficiency"]
        for solver in solvers[i]
    ]

    # Filter for pinch_deltas >= 5
    pinch_deltas_filtered = [d for d in pinch_deltas if d >= 5]
    rte_filtered = [r for d, r in zip(pinch_deltas, rte) if d >= 5]

    ax.plot(
        pinch_deltas_filtered,
        100 * np.array(rte_filtered),
        label=rf"$T_\text{{hot}} = {int(T_hot - 273.15)}^\circ$C",
        color=colors[i],
        marker="o",
        markersize=3.5,
    )

ax.set_xticks([5, 10, 15, 20])
ax.set_ylim([30, 80])
ax.set_xlabel("Pinch point temperature difference (K)")
ax.set_ylabel("Round-trip efficiency (%)")
ax.grid(True)
ax.legend(fontsize=13, loc="upper right")
fig.tight_layout(pad=1)

# Save figure
filename = f"{pathlib.Path(__file__).parent.name}_RTE"
th.savefig_in_formats(fig, os.path.join(OUT_DIR_BASE, filename))

plt.show()
