import os
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import thermopt as th

# Global settings
th.set_plot_options(fontsize=16)

CONFIG_FILE = "../case_PTES_CO2.yaml"
OUT_DIR_BASE = "results"
os.makedirs(OUT_DIR_BASE, exist_ok=True)
DATA_FILE = "simulation_data.pkl"
DATA_FULLPATH = os.path.join(OUT_DIR_BASE, DATA_FILE)

# Parameter ranges
pinch_deltas = np.linspace(20, 0, 21)  # K
efficiencies = np.array([0.70, 0.80, 0.90, 0.99])  # Fraction

# --------------------------------------------------------------------- #
# Step 1: Run simulations if data file does not exist
# --------------------------------------------------------------------- #
if not os.path.exists(DATA_FULLPATH):
    solvers = []
    x0_eta_start = None  # Global x0 passed between efficiency levels

    for eta in efficiencies:
        eta_solvers = []
        x0 = x0_eta_start  # Use starting guess for this eta loop

        for i, pinch in enumerate(pinch_deltas):
            print()
            print("-" * 80)
            print(
                f"Turbomachinery efficiency: {eta*100:.1f} %, Pinch point: {pinch:.1f} K"
            )
            print("-" * 80)

            # Output directory
            folder = f"eff_{int(eta*100)}_pinch_{pinch}"
            out_dir = os.path.join(OUT_DIR_BASE, folder)
            os.makedirs(out_dir, exist_ok=True)

            # Set up cycle
            cycle = th.ThermodynamicCycleOptimization(CONFIG_FILE, out_dir=out_dir)
            cycle.set_config_value("solver_options.max_iterations", 200)

            # Set all turbomachinery efficiencies
            for machine in ["expander", "compressor"]:
                for side in ["charge", "discharge"]:
                    var = f"problem_formulation.fixed_parameters.{machine}_{side}.efficiency"
                    cycle.set_config_value(var, eta)

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
            eta_solvers.append(cycle.solver)

            # Use current x_final as next initial guess
            x0 = cycle.solver.x_final

            # Store the first x_final (at max pinch) for next efficiency loop
            if i == 0:
                x0_eta_start = x0

        solvers.append(eta_solvers)

    th.save_to_pickle(solvers, DATA_FULLPATH, timestamp=False)
    print("All simulations completed.")

else:
    print("Loading existing simulation data...")
    solvers = th.load_from_pickle(DATA_FULLPATH)

# --------------------------------------------------------------------- #
# Step 2: Plotting round-trip efficiency vs. pinch point
# --------------------------------------------------------------------- #
fig, ax = plt.subplots(figsize=(6.0, 4.8))
colors = plt.get_cmap("magma")(np.linspace(0.2, 0.8, len(efficiencies)))

for i, eta in enumerate(efficiencies):
    rte = [
        solver.problem.cycle_data["energy_analysis"]["roundtrip_efficiency"]
        for solver in solvers[i]
    ]
    ax.plot(
        pinch_deltas,
        100 * np.array(rte),
        label=rf"$\eta_\mathrm{{turbo}} = {int(eta*100)}\%$",
        color=colors[i],
        marker="o",
        markersize=3.5,
    )

ax.set_ylim([0, 110])
ax.set_xlabel("Pinch point temperature differences (K)")
ax.set_ylabel("Round-trip efficiency (%)")
ax.grid(True)
ax.legend(fontsize=13, loc="upper right")
fig.tight_layout(pad=1)

# Save figure
filename = f"{pathlib.Path(__file__).parent.name}_RTE"
th.savefig_in_formats(fig, os.path.join(OUT_DIR_BASE, filename))

plt.show()
