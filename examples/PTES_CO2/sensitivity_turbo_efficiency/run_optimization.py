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

turb_eff_range = np.linspace(1.00, 0.70, 13)
comp_eff_values = np.linspace(1.00, 0.70, 4)

# --------------------------------------------------------------------- #
# Step 1: Run simulations if data does not exist
# --------------------------------------------------------------------- #
if not os.path.exists(DATA_FULLPATH):
    solvers = []
    x0 = None
    for comp_eff in comp_eff_values:
        comp_solvers = []
        for turb_eff in turb_eff_range:
            print()
            print(80 * "-")
            print(f"Turbine: {turb_eff*100:.1f} %, Compressor: {comp_eff*100:.1f} %")
            print(80 * "-")

            # Output folder
            folder = f"effT_{turb_eff:.2f}_effC_{comp_eff:.2f}"
            out_dir = os.path.join(OUT_DIR_BASE, folder)
            os.makedirs(out_dir, exist_ok=True)

            # Set up cycle and config
            cycle = th.ThermodynamicCycleOptimization(CONFIG_FILE, out_dir=out_dir)
            for var in [
                "problem_formulation.fixed_parameters.expander_charge.efficiency",
                "problem_formulation.fixed_parameters.expander_discharge.efficiency",
            ]:
                cycle.set_config_value(var, turb_eff)
            for var in [
                "problem_formulation.fixed_parameters.compressor_charge.efficiency",
                "problem_formulation.fixed_parameters.compressor_discharge.efficiency",
            ]:
                cycle.set_config_value(var, comp_eff)

            # Run and save
            cycle.run_optimization(x0=x0)
            cycle.save_results()
            comp_solvers.append(cycle.solver)
            x0 = cycle.solver.x_final
        solvers.append(comp_solvers)

    th.save_to_pickle(solvers, DATA_FULLPATH, timestamp=False)
    print("All simulations completed.")
else:
    print("Loading existing simulation data...")
    solvers = th.load_from_pickle(DATA_FULLPATH)

# --------------------------------------------------------------------- #
# Step 2: Plot simulation results
# --------------------------------------------------------------------- #
fig, ax = plt.subplots(figsize=(6.0, 4.8))
colors = plt.get_cmap("magma")(np.linspace(0.20, 0.80, len(comp_eff_values)))

for i, comp_eff in enumerate(comp_eff_values):
    rte = [
        solver.problem.cycle_data["energy_analysis"]["roundtrip_efficiency"]
        for solver in solvers[i]
    ]
    ax.plot(
        100 * turb_eff_range,
        100 * np.array(rte),
        label=rf"$\eta_\text{{compressor}}={100*comp_eff:.0f}$ %",
        color=colors[i],
        marker="o",
    )

ax.set_xlabel("Turbine isentropic efficiency (%)")
ax.set_ylabel("Round-trip efficiency (%)")
ax.legend(fontsize=13, loc="lower right")
ax.grid(True)
fig.tight_layout(pad=1)

# Save
filename = f"{pathlib.Path(__file__).parent.name}_RTE"
th.savefig_in_formats(fig, os.path.join(OUT_DIR_BASE, filename))

plt.show()
