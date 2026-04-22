import os
import pathlib
import numpy as np
import matplotlib.pyplot as plt

import thermopt as th

th.set_plot_options(fontsize=16)

# Cycle config files
CONFIG_FILES = {
    # "simple": "case_sCO2_simple.yaml",
    # "recuperated": "case_sCO2_recuperated.yaml",
    "recompression": "case_sCO2_recompression.yaml",
}

OUT_DIR_BASE = "results"
DATA_FILE = "sensitivity_pressure_ratio.pkl"
DATA_FULLPATH = os.path.join(OUT_DIR_BASE, DATA_FILE)

# Parameter space
p_max_values = np.linspace(100e5, 1000e5, 19)
p_max_values = np.linspace(100e5, 300e5, 2)

efficiencies = [1.0, 0.9, 0.8]
linestyles = ["-", "-.", ":"]

# Run simulations
if not os.path.exists(DATA_FULLPATH):
    os.makedirs(OUT_DIR_BASE, exist_ok=True)
    results = {}

    for label, config_file in CONFIG_FILES.items():
        solver_lists = []
        for eta, linestyle in zip(efficiencies, linestyles):
            solvers = []
            x0 = None
            for i, p_max in enumerate(p_max_values):
                print()
                print(80 * "=")
                print(
                    f"{label.capitalize()} | efficiency = {eta:.2f} | p_max = {p_max/1e5:.1f} bar"
                )
                print(80 * "=")

                # Output dir
                folder = f"{label}_eta_{int(eta*100)}_pmax_{int(p_max/1e5)}bar"
                out_dir = os.path.join(OUT_DIR_BASE, folder)
                os.makedirs(out_dir, exist_ok=True)

                # Create cycle object
                cycle = th.ThermodynamicCycleOptimization(config_file, out_dir=out_dir)

                # Set polytropic efficiencies
                if label == "recompression":
                    cycle.set_config_values(
                        {
                            "problem_formulation.fixed_parameters.expander.efficiency": eta,
                            "problem_formulation.fixed_parameters.main_compressor.efficiency": eta,
                            "problem_formulation.fixed_parameters.split_compressor.efficiency": eta,
                        }
                    )
                else:
                    cycle.set_config_values(
                        {
                            "problem_formulation.fixed_parameters.compressor.efficiency": eta,
                            "problem_formulation.fixed_parameters.expander.efficiency": eta,
                        }
                    )

                # Widen bounds on inlet pressure if needed
                var = "problem_formulation.design_variables.expander_inlet_pressure"
                cycle.set_config_values(
                    {
                        f"{var}.min": 90e5,
                        f"{var}.max": p_max * 1.1,
                        # f"{var}.value": p_max * 1.0,
                    }
                )

                # Apply equality constraint on expander inlet pressure
                cycle.set_constraint(
                    variable="$components.expander.state_in.p",
                    type="=",
                    value=p_max,
                    normalize=True,
                )

                # Optimize
                cycle.run_optimization(x0=x0)
                cycle.save_results()
                solvers.append(cycle.solver)
                x0 = cycle.solver.x_final

            solver_lists.append(solvers)

        results[label] = solver_lists

    th.save_to_pickle(results, DATA_FULLPATH, timestamp=False)
    print("All simulations completed.")

else:
    print("Loading existing simulation data...")
    results = th.load_from_pickle(DATA_FULLPATH)

# Plot results
fig, ax = plt.subplots(figsize=(6.0, 4.8))
colors = plt.get_cmap("magma")(np.linspace(0.2, 0.8, len(CONFIG_FILES)))

for color_idx, (label, solver_lists) in enumerate(results.items()):
    for eta_idx, solvers in enumerate(solver_lists):
        eff = [
            solver.problem.cycle_data["energy_analysis"]["cycle_efficiency"]
            for solver in solvers
        ]
        ax.plot(
            p_max_values / 1e5,
            100 * np.array(eff),
            label=f"{label} ($\\eta = {int(efficiencies[eta_idx]*100)}$ %)",
            color=colors[color_idx],
            linestyle=linestyles[eta_idx],
            marker="o",
        )

ax.set_xlabel("Maximum cycle pressure (bar)")
ax.set_ylabel("Cycle efficiency (%)")
ax.legend(loc="upper left", fontsize=13)
ax.grid(True)
fig.tight_layout(pad=1)


# # Save
# filename = f"{pathlib.Path(__file__).stem}_cycle_eff_vs_pr"
# th.savefig_in_formats(fig, os.path.join(OUT_DIR_BASE, filename))

plt.show()
