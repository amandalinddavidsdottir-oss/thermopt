import os
import thermopt as th
import matplotlib.pyplot as plt

# Print package info
th.print_package_info()


def plot_cycle_data(ax, cycle_data, x_prop, y_prop, linestyle="-"):

    # Plot thermodynamic processes
    for name, component in cycle_data["components"].items():
        # Handle heat exchanger components in a special way
        if component["type"] == "heat_exchanger":
            for side in ["hot_side", "cold_side"]:
                _plot_cycle_process(
                    name + "_" + side,
                    cycle_data,
                    x_prop,
                    y_prop,
                    ax,
                    linestyle=linestyle,
                )
        else:
            _plot_cycle_process(
                name, cycle_data, x_prop, y_prop, ax, linestyle=linestyle
            )

    # Adjust plot limits if updating
    ax.relim(visible_only=True)
    ax.autoscale_view()


def _plot_cycle_process(name, cycle_data, x_prop, y_prop, ax, linestyle="-"):
    """
    Creates or updates the plot elements for a specific cycle process on a given axes.

    This method checks if the plot elements for the specified cycle process already exist on the given axes.
    If they exist and an axis index is provided, it updates these elements with new data. Otherwise, it creates
    new plot elements (lines and points) and stores them for future updates.

    Parameters:
    ----------
    name : str
        The name of the cycle process to plot or update.
    plot_settings : dict
        The plot settings dictionary containing settings such as x and y variables.
    ax : matplotlib.axes.Axes
        The axes on which to plot or update the cycle process.
    ax_index : int, optional
        The index of the axes in the figure, used for updating existing plots. If None, new plots are created.
    """

    # Retrieve component data
    x_data, y_data, color = _get_process_data(
        name,
        cycle_data,
        x_prop,
        y_prop,
    )

    # Create new plot elements if data is not None
    if x_data is not None and y_data is not None:
        (linestyle,) = ax.plot(
            x_data,
            y_data,
            linestyle=linestyle,
            linewidth=1.25,
            marker="none",
            markersize=4.0,
            markeredgewidth=1.25,
            markerfacecolor="w",
            color=color,
            label=name,
            zorder=1,
        )
        (points,) = ax.plot(
            [x_data[0], x_data[-1]],
            [y_data[0], y_data[-1]],
            linestyle="none",
            linewidth=1.25,
            marker="o",
            markersize=4.0,
            markeredgewidth=1.25,
            markerfacecolor="w",
            color=color,
            zorder=2,
        )


def _get_process_data(name, cycle_data, prop_x, prop_y):

    # Get heat exchanger side data
    if "_hot_side" in name or "_cold_side" in name:
        # Extract the component and side from the name
        if "_hot_side" in name:
            component_name, side_1 = name.replace("_hot_side", ""), "hot_side"
            side_2 = "cold_side"
        else:
            component_name, side_1 = name.replace("_cold_side", ""), "cold_side"
            side_2 = "hot_side"

        data = cycle_data["components"][component_name][side_1]
        data_other_side = cycle_data["components"][component_name][side_2]
        is_heat_exchanger = True

    else:  # Handle non-heat exchanger components
        data = cycle_data["components"][name]
        is_heat_exchanger = False

    # Retrieve data
    if data["states"]["identifier"][0] == "working_fluid":
        # Components of the cycle
        x_data = data["states"][prop_x]
        y_data = data["states"][prop_y]
    elif is_heat_exchanger and prop_y == "T" and prop_x in ["h", "s"]:
        # Special case for heat exchangers
        x_data = data_other_side["states"][prop_x]
        y_data = data["states"][prop_y]
    else:
        # Other cases
        x_data = None
        y_data = None

    color = data["color"]

    return x_data, y_data, color


# Optimize cycle with JT-valve
CONFIG_FILE = "./case_sCO2_heat_pump_valve.yaml"
cycle_1 = th.ThermodynamicCycleOptimization(CONFIG_FILE)
cycle_1.run_optimization()
cycle_1.save_results()
cycle_1.create_animation(format="mp4", fps=1)

# Optimize cycle with two-phase expander
CONFIG_FILE = "./case_sCO2_heat_pump_turbine.yaml"
cycle_2 = th.ThermodynamicCycleOptimization(CONFIG_FILE)
cycle_2.run_optimization()
cycle_2.save_results()
cycle_2.create_animation(format="mp4", fps=1)

# Define custom legend entries
from matplotlib.lines import Line2D

custom_legend = [
    Line2D(
        [0],
        [0],
        linestyle="-",
        marker="o",
        linewidth=1.25,
        color=th.COLORS_MATLAB[1],
        markerfacecolor="white",
        label="Joule-Thomson valve",
    ),
    Line2D(
        [0],
        [0],
        linestyle="--",
        marker="o",
        linewidth=1.25,
        color=th.COLORS_MATLAB[1],
        markerfacecolor="white",
        label="Two-phase turbine",
    ),
]

# Plot results
fig, ax = plt.subplots(figsize=(5.2, 4.8))
x_prop = "s"
y_prop = "T"
x_scale = "linear"
y_scale = "linear"
ax.set_xlabel(th.LABEL_MAPPING[x_prop])
ax.set_ylabel(th.LABEL_MAPPING[y_prop])
ax.set_xscale(x_scale)
ax.set_yscale(y_scale)
plot_cycle_data(ax, cycle_1.problem.cycle_data, x_prop, y_prop, linestyle="-")
# plot_cycle_data(ax, cycle_2.problem.cycle_data, x_prop, y_prop, linestyle="--")
cycle_2.problem.fluid.plot_phase_diagram(x_prop=x_prop, y_prop=y_prop, axes=ax)
# ax.legend(handles=custom_legend, loc="upper left")
fig.tight_layout(pad=1)
th.savefig_in_formats(fig, os.path.join("results", "valve_sCO2_heat_pump_Ts"))


# Plot results
fig, ax = plt.subplots(figsize=(5.2, 4.8))
x_prop = "s"
y_prop = "T"
x_scale = "linear"
y_scale = "linear"
ax.set_xlabel(th.LABEL_MAPPING[x_prop])
ax.set_ylabel(th.LABEL_MAPPING[y_prop])
ax.set_xscale(x_scale)
ax.set_yscale(y_scale)
plot_cycle_data(ax, cycle_1.problem.cycle_data, x_prop, y_prop, linestyle="-")
plot_cycle_data(ax, cycle_2.problem.cycle_data, x_prop, y_prop, linestyle="--")
cycle_2.problem.fluid.plot_phase_diagram(x_prop=x_prop, y_prop=y_prop, axes=ax)
ax.legend(handles=custom_legend, loc="upper left")
fig.tight_layout(pad=1)
th.savefig_in_formats(fig, os.path.join("results", "comparison_sCO2_heat_pump_Ts"))

# Plot results
fig, ax = plt.subplots(figsize=(5.2, 4.8))
x_prop = "h"
y_prop = "p"
x_scale = "linear"
y_scale = "log"
ax.set_xlabel(th.LABEL_MAPPING[x_prop])
ax.set_ylabel(th.LABEL_MAPPING[y_prop])
ax.set_xscale(x_scale)
ax.set_yscale(y_scale)
plot_cycle_data(ax, cycle_1.problem.cycle_data, x_prop, y_prop, linestyle="-")
plot_cycle_data(ax, cycle_2.problem.cycle_data, x_prop, y_prop, linestyle="--")
cycle_2.problem.fluid.plot_phase_diagram(x_prop=x_prop, y_prop=y_prop, axes=ax)
ax.legend(handles=custom_legend, loc="lower right")
fig.tight_layout(pad=1)
th.savefig_in_formats(fig, os.path.join("results", "comparison_sCO2_heat_pump_ph"))


# Keep plots open
plt.show()
