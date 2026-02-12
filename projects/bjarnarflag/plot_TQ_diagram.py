"""
T-Q diagram plotting for heat exchangers in a thermopt cycle.

Usage:
    from plot_TQ_diagram import plot_TQ_diagram
    plot_TQ_diagram(cycle, "heater")     # evaporator
    plot_TQ_diagram(cycle, "cooler")     # condenser
"""

import os
import numpy as np
import matplotlib.pyplot as plt


def plot_TQ_diagram(cycle, component_name, savefig=True, filename=None, output_dir=None):
    """
    Plot a Temperature vs Heat-duty diagram for a heat exchanger.

    Parameters
    ----------
    cycle : thermopt.ThermodynamicCycleOptimization
        A cycle object that has already been optimized.
    component_name : str
        Name of the heat exchanger in cycle_data["components"],
        e.g. "heater" (evaporator) or "cooler" (condenser).
    savefig : bool, optional
        Whether to save the figure to disk (default True).
    filename : str, optional
        Output filename.  Defaults to "TQ_<component_name>.png".
    output_dir : str, optional
        Directory to save the figure in.  If None, saves in current directory.
    """

    if filename is None:
        filename = f"TQ_{component_name}.png"

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.join(output_dir, filename)

    # Display name for titles and printout
    display_names = {"heater": "Evaporator", "cooler": "Condenser"}
    display_name = display_names.get(component_name, component_name.capitalize())

    data = cycle.problem.cycle_data
    hx   = data["components"][component_name]

    # --- Hot side ---
    # After the counter-current flip inside thermopt, hot-side states are
    # spatially aligned with the cold side (index 0 = cold end, index -1 = hot end).
    # heat_flow = m_hot * (h_hot_in - h[i])  →  large at idx 0, zero at idx -1
    # Flip Q_hot so it goes 0 → Q_total.  T_hot is already aligned → no flip.
    hot   = hx["hot_side"]
    T_hot = np.array(hot["states"]["T"])
    Q_hot = np.flip(np.array(hot["heat_flow"]))

    # --- Cold side ---
    # heat_flow = m_cold * (h[i] - h_cold_in)  →  0 → Q_total already
    cold   = hx["cold_side"]
    T_cold = np.array(cold["states"]["T"])
    Q_cold = np.array(cold["heat_flow"])

    # Convert units
    Q_hot_kW  = Q_hot  / 1e3
    Q_cold_kW = Q_cold / 1e3
    T_hot_C   = T_hot  - 273.15
    T_cold_C  = T_cold - 273.15

    # Pinch point (arrays are element-wise aligned)
    delta_T     = T_hot_C - T_cold_C
    pinch_idx   = np.argmin(delta_T)
    pinch_Q     = Q_cold_kW[pinch_idx]
    pinch_dT    = delta_T[pinch_idx]
    pinch_Thot  = T_hot_C[pinch_idx]
    pinch_Tcold = T_cold_C[pinch_idx]

    # Console summary
    eff = data["energy_analysis"]
    print("\n" + 80*"─")
    print(f"  {display_name} T–Q diagram summary")
    print(80*"─")
    print(f"  Total heat duty          : {Q_cold_kW[-1]:.1f} kW")
    print(f"  Hot-side inlet  → outlet : {T_hot_C[-1]:.1f} → {T_hot_C[0]:.1f} °C")
    print(f"  Cold-side inlet → outlet : {T_cold_C[0]:.1f} → {T_cold_C[-1]:.1f} °C")
    print(f"  Pinch-point ΔT           : {pinch_dT:.2f} °C  (at Q = {pinch_Q:.1f} kW)")
    print(f"  Cycle efficiency          : {eff['cycle_efficiency']*100:.2f} %")
    print(f"  System efficiency         : {eff['system_efficiency']*100:.2f} %")
    print(80*"─" + "\n")

    # ── Plot ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5.5))

    ax.fill_between(
        Q_cold_kW, T_cold_C, T_hot_C,
        color="#d0e8f0", alpha=0.55, label="Temperature gap",
    )
    ax.plot(Q_hot_kW,  T_hot_C,  "-", color="#c0392b", linewidth=2.0, label="Hot side")
    ax.plot(Q_cold_kW, T_cold_C, "-", color="#2980b9", linewidth=2.0, label="Cold side")

    # End-point markers
    for Q_arr, T_arr, m, c in [
        (Q_hot_kW,  T_hot_C,  "o", "#c0392b"),
        (Q_cold_kW, T_cold_C, "s", "#2980b9"),
    ]:
        ax.plot(Q_arr[0],  T_arr[0],  m, color=c, markersize=6,
                markerfacecolor="white", markeredgewidth=1.5, zorder=5)
        ax.plot(Q_arr[-1], T_arr[-1], m, color=c, markersize=6,
                markerfacecolor="white", markeredgewidth=1.5, zorder=5)

    # Pinch-point annotation
    ax.annotate(
        f"Pinch ΔT = {pinch_dT:.1f} °C",
        xy=(pinch_Q, pinch_Tcold),
        xytext=(pinch_Q + (Q_cold_kW[-1] - Q_cold_kW[0]) * 0.08, pinch_Tcold - 5),
        fontsize=9,
        arrowprops=dict(arrowstyle="->", color="black", lw=1.0),
        bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="gray", alpha=0.9),
    )
    ax.annotate(
        "", xy=(pinch_Q, pinch_Thot), xycoords="data",
        xytext=(pinch_Q, pinch_Tcold), textcoords="data",
        arrowprops=dict(arrowstyle="<->", color="black", lw=1.2),
    )

    ax.set_xlabel("Heat duty, Q [kW]", fontsize=11)
    ax.set_ylabel("Temperature [°C]", fontsize=11)
    ax.set_title(f"T–Q Diagram — {display_name}", fontsize=13, fontweight="bold")
    ax.legend(loc="best", fontsize=9, framealpha=0.9)
    ax.set_xlim(left=0)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()

    if savefig:
        fig.savefig(filename, dpi=300)
        print(f"Figure saved → {filename}")

    return fig, ax
