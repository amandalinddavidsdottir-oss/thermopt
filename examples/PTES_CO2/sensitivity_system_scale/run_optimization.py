import os
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import thermopt as th

th.set_plot_options(fontsize=16)

CONFIG_FILE = "../case_PTES_CO2_turbo.yaml"
OUT_DIR_BASE = "results"
DATA_FILE = "simulation_data.pkl"
DATA_FULLPATH = os.path.join(OUT_DIR_BASE, DATA_FILE)

# Charging power sweep from 1 MW to 1 GW (log scale)
power_values = np.logspace(6, 9, num=21)  # in watts (1e6 to 1e9)

# --------------------------------------------------------------------- #
# Step 1: Run simulations if data file does not exist
# --------------------------------------------------------------------- #
if not os.path.exists(DATA_FULLPATH):
    solvers = []
    x0 = None

    for power in power_values:
        print()
        print(80 * "-")
        print(f"Charging power: {power/1e6:.2f} MW")
        print(80 * "-")

        folder = f"P_{int(power/1e6)}MW"
        out_dir = os.path.join(OUT_DIR_BASE, folder)
        os.makedirs(out_dir, exist_ok=True)

        cycle = th.ThermodynamicCycleOptimization(CONFIG_FILE, out_dir=out_dir)
        # cycle.set_config_value("solver_options.max_iterations", 5)
        cycle.set_config_value(
            "problem_formulation.fixed_parameters.charging_power", power
        )

        # Enforce shared shaft constraints
        cycle.set_constraint(
            variable="$components.expander_charge.data_out.angular_speed - $components.compressor_charge.data_out.angular_speed",
            type="=",
            value=0.0,
            normalize=1000,
        )
        cycle.set_constraint(
            variable="$components.expander_discharge.data_out.angular_speed - $components.compressor_discharge.data_out.angular_speed",
            type="=",
            value=0.0,
            normalize=1000,
        )

        # Optionally set efficiency to ideal to isolate power scaling
        for machine in ["expander", "compressor"]:
            for side in ["charge", "discharge"]:
                var = (
                    f"problem_formulation.fixed_parameters.{machine}_{side}.efficiency"
                )
                cycle.set_config_value(var, 1.0)

        cycle.run_optimization(x0=x0)
        cycle.save_results()

        solvers.append(cycle.solver)
        x0 = cycle.solver.x_final

    th.save_to_pickle(solvers, DATA_FULLPATH, timestamp=False)
    print("All simulations completed.")

else:
    print("Loading existing simulation data...")
    solvers = th.load_from_pickle(DATA_FULLPATH)


# --------------------------------------------------------------------- #
# Step 2: Plotting round-trip and isentropic efficiencies (2 subplots)
# --------------------------------------------------------------------- #

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.0, 6.5), sharex=True)

# X values in MW
x_vals = power_values / 1e6

# --- Top plot: round-trip efficiency ---
rte = [s.problem.cycle_data["energy_analysis"]["roundtrip_efficiency"] for s in solvers]
ax1.plot(
    x_vals,
    100 * np.array(rte),
    label="Round-trip efficiency",
    color="black",
    linewidth=2,
    marker="o",
)
ax1.set_ylabel("Round-trip efficiency (%)")
ax1.grid(True, which="both", ls=":")
# ax1.legend(fontsize=12, loc="lower right")

# --- Bottom plot: isentropic efficiencies ---

keys = [
    "compressor_charge",
    "compressor_discharge",
    "expander_charge",
    "expander_discharge",
]


# Build LaTeX-style labels: η with subscript (machine) and superscript (flow direction)
def make_eta_label(key):
    machine, direction = key.split("_")
    return rf"$\eta_{{\mathrm{{{machine}}}}}^{{\mathrm{{{direction}}}}}$"


labels = [make_eta_label(k) for k in keys]

markers = ["s", "o", "^", "v"]
colors = plt.get_cmap("magma")(np.linspace(0.20, 0.80, len(keys)))

for i, (key, label) in enumerate(zip(keys, labels)):
    eta_vals = [
        solver.problem.cycle_data["components"][key]["data_out"][
            "isentropic_efficiency"
        ]
        * 100
        for solver in solvers
    ]
    ax2.plot(
        x_vals,
        eta_vals,
        label=label,
        color=colors[i],
        marker=markers[i],
        linestyle="-",
        linewidth=1.5,
    )


ax2.set_xscale("log")
ax2.set_xlabel("Charging power (MW)")
ax2.set_ylabel("Isentropic efficiency (%)")
ax2.grid(True)
# Better legend positioning in lower subplot
ax2.legend(
    fontsize=13,
    loc="lower right",
    borderaxespad=0.5,
    frameon=True,
    ncol=2,
    handlelength=1.5,
    columnspacing=0.8,  # <-- reduce horizontal gap between columns
)

fig.tight_layout(pad=1)  # Reduce vertical padding
# fig.tight_layout(pad=1.5)


filename = f"{pathlib.Path(__file__).parent.name}_power_sweep_efficiency"
th.savefig_in_formats(fig, os.path.join(OUT_DIR_BASE, filename))


# --------------------------------------------------------------------- #
# New figure: Rotational speed (compressor charge/discharge) vs. power
# --------------------------------------------------------------------- #
fig, ax = plt.subplots(figsize=(6.0, 4.0))

x_vals = power_values / 1e6  # Charging power in MW

# Components to plot
keys = ["compressor_charge", "compressor_discharge"]


# LaTeX-style labels
def make_speed_label(key):
    machine, direction = key.split("_")
    return rf"{direction.capitalize()} system"


labels = [make_speed_label(k) for k in keys]
markers = ["s", "o"]
colors = plt.get_cmap("magma")(np.linspace(0.3, 0.7, len(keys)))

# Plot loop
for i, (key, label) in enumerate(zip(keys, labels)):
    rpm_vals = [
        solver.problem.cycle_data["components"][key]["data_out"]["angular_speed"]
        * 60
        / (2 * np.pi)
        for solver in solvers
    ]
    ax.plot(
        x_vals,
        rpm_vals,
        label=label,
        color=colors[i],
        marker=markers[i],
        linestyle="-",
    )

# Axis formatting
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Charging power (MW)")
ax.set_ylabel("Rotational speed (RPM)")
ax.set_ylim(1e3, 5e5)
ax.grid(True)

# Format axes with plain integers
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x)}"))
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{int(y):,}"))

ax.legend(fontsize=12, loc="best")
fig.tight_layout(pad=0.6)

# Save
filename = f"{pathlib.Path(__file__).parent.name}_power_sweep_rpm_charge"
th.savefig_in_formats(fig, os.path.join(OUT_DIR_BASE, filename))
plt.show()

# Show figure
plt.show()
