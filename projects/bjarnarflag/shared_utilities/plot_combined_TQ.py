"""
Combined T-Q diagram: stitch all brine-side heat exchangers into one plot.

Works with all cycle configurations:
  - Simple ORC  (heater only)
  - Dual-pressure ORC  (preheater + lp_evaporator + hp_evaporator)

Optional: overlay a second cycle (e.g. single-pressure vs dual-pressure).

Usage:
    from plot_combined_TQ import plot_combined_TQ
    plot_combined_TQ(cycle)
    plot_combined_TQ(cycle, overlay_cycle=single_cycle, overlay_label="Single-pressure")
"""

import os
import numpy as np
import matplotlib.pyplot as plt


# ══════════════════════════════════════════════════════════════════════════════
#  DATA EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def _extract_hx(components, name):
    """
    Extract T and Q arrays for one heat exchanger.
    Returns dict with Q_kW (0→Q_total), T_hot_C, T_cold_C, Q_total_kW.
    """
    hx = components[name]

    # Hot side: Q stored as m*(h_in - h[i]), large at idx 0, zero at idx -1
    # Flip so Q goes 0 → Q_total.  T is already spatially aligned (no flip).
    T_hot = np.array(hx["hot_side"]["states"]["T"]) - 273.15
    Q_hot = np.flip(np.array(hx["hot_side"]["heat_flow"])) / 1e3

    # Cold side: Q stored as m*(h[i] - h_in), already 0 → Q_total
    T_cold = np.array(hx["cold_side"]["states"]["T"]) - 273.15
    Q_cold = np.array(hx["cold_side"]["heat_flow"]) / 1e3

    return {
        "name": name,
        "Q_hot": Q_hot,
        "Q_cold": Q_cold,
        "T_hot": T_hot,
        "T_cold": T_cold,
        "Q_total": float(Q_cold[-1]),
    }


# Display names for section labels
_DISPLAY = {
    "preheater": "Preheater",
    "lp_evaporator": "LP Evaporator",
    "hp_evaporator": "HP Evaporator",
    "heater": "Evaporator",
}

# Background color (uniform light blue — same as individual T-Q diagrams)
_SECTION_COLOR = "#d0e8f0"


def _build_segments(cycle):
    """
    Build ordered list of brine-side HX segments with cumulative Q offsets.
    """
    components = cycle.problem.cycle_data["components"]
    segments = []

    # Brine-side HXs in order: preheater → LP evap → HP evap (or just heater)
    for name in ["preheater", "lp_evaporator", "hp_evaporator", "heater"]:
        if name not in components:
            continue
        seg = _extract_hx(components, name)
        # Skip preheater if heat duty ≈ 0 (recuperated case where Q_pre → 0)
        if name == "preheater" and abs(seg["Q_total"]) < 1.0:
            continue
        segments.append(seg)

    # Apply cumulative Q offsets
    Q_offset = 0.0
    for seg in segments:
        seg["Q_hot_abs"] = seg["Q_hot"] + Q_offset
        seg["Q_cold_abs"] = seg["Q_cold"] + Q_offset
        seg["Q_start"] = Q_offset
        seg["Q_end"] = Q_offset + seg["Q_total"]
        Q_offset += seg["Q_total"]

    return segments


# ══════════════════════════════════════════════════════════════════════════════
#  PLOTTING
# ══════════════════════════════════════════════════════════════════════════════

def plot_combined_TQ(cycle, overlay_cycle=None, overlay_label=None,
                     savefig=True, filename=None, output_dir=None,
                     figsize=(14, 7)):
    """
    Plot combined T-Q diagram for all brine-side heat exchangers.

    Parameters
    ----------
    cycle : ThermodynamicCycleOptimization
        Primary cycle (plotted with solid lines).
    overlay_cycle : ThermodynamicCycleOptimization, optional
        Second cycle to overlay (plotted with dashed lines).
    overlay_label : str, optional
        Legend label for overlay (e.g. "Single-pressure", "Non-recuperated").
    savefig : bool
        Whether to save the figure.
    filename : str, optional
        Output filename. Defaults to "TQ_combined.png".
    output_dir : str, optional
        Directory for the output file.
    figsize : tuple
        Figure size.

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
    """

    if filename is None:
        filename = "TQ_combined.png"
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.join(output_dir, filename)

    # ── Build segments ──
    segments = _build_segments(cycle)

    overlay_segments = None
    if overlay_cycle is not None:
        overlay_segments = _build_segments(overlay_cycle)

    # ── Create figure ──
    fig, ax = plt.subplots(figsize=figsize)

    # ── Plot main cycle segments ──
    for i, seg in enumerate(segments):
        name = seg["name"]
        color_bg = _SECTION_COLOR

        # Background shading
        ax.fill_between(
            seg["Q_cold_abs"], seg["T_cold"], seg["T_hot"],
            color=color_bg, alpha=0.5, zorder=1,
        )

        # Hot-side curve (brine)
        ax.plot(seg["Q_hot_abs"], seg["T_hot"],
                color="#c0392b", linewidth=2.0, zorder=3)

        # Cold-side curve (working fluid)
        ax.plot(seg["Q_cold_abs"], seg["T_cold"],
                color="#2980b9", linewidth=2.0, zorder=3)

        # Section boundary (vertical dashed line)
        if i > 0:
            ax.axvline(seg["Q_start"], color="gray", linestyle=":",
                       linewidth=0.8, alpha=0.7, zorder=2)

    # ── Plot overlay cycle (dashed lines, no fill) ──
    if overlay_segments is not None:
        label_done = False
        for seg in overlay_segments:
            lbl = overlay_label or "Overlay" if not label_done else None
            ax.plot(seg["Q_hot_abs"], seg["T_hot"],
                    color="#c0392b", linewidth=1.5, linestyle="--",
                    alpha=0.7, zorder=2, label=lbl)
            ax.plot(seg["Q_cold_abs"], seg["T_cold"],
                    color="#2980b9", linewidth=1.5, linestyle="--",
                    alpha=0.7, zorder=2)
            label_done = True

    # ── Pinch-point markers ──
    for seg in segments:
        dT = seg["T_hot"] - seg["T_cold"]
        idx = np.argmin(dT)
        pinch_dT = dT[idx]
        pinch_Q = seg["Q_cold_abs"][idx]
        pinch_T = seg["T_cold"][idx]

        if pinch_dT < 15 and seg["Q_total"] > 1.0:
            ax.annotate(
                f"ΔT = {pinch_dT:.1f} °C",
                xy=(pinch_Q, pinch_T),
                xytext=(pinch_Q, pinch_T - 8),
                fontsize=8, ha="center",
                arrowprops=dict(arrowstyle="->", color="black", lw=0.8),
                bbox=dict(boxstyle="round,pad=0.2", fc="lightyellow",
                          ec="gray", alpha=0.9),
                zorder=5,
            )

    # ── End-point markers ──
    if segments:
        # Brine inlet (hot end of last segment)
        last = segments[-1]
        ax.plot(last["Q_hot_abs"][-1], last["T_hot"][-1], "o",
                color="#c0392b", markersize=7, markerfacecolor="white",
                markeredgewidth=1.5, zorder=5)
        # Brine outlet (cold end of first segment)
        first = segments[0]
        ax.plot(first["Q_hot_abs"][0], first["T_hot"][0], "o",
                color="#c0392b", markersize=7, markerfacecolor="white",
                markeredgewidth=1.5, zorder=5)

    # ── Axis labels and formatting ──
    ax.set_xlabel("Cumulative heat duty, Q [kW]", fontsize=12)
    ax.set_ylabel("Temperature [°C]", fontsize=12)
    ax.set_title("Combined T–Q Diagram", fontsize=14, fontweight="bold")
    ax.set_xlim(left=0)
    ax.grid(True, linestyle="--", alpha=0.3)

    # ── Section labels ──
    y_min, y_max = ax.get_ylim()
    label_y = y_max - 0.03 * (y_max - y_min)

    for seg in segments:
        Q_mid = (seg["Q_start"] + seg["Q_end"]) / 2
        display = _DISPLAY.get(seg["name"], seg["name"])
        ax.text(Q_mid, label_y, display,
                ha="center", va="top", fontsize=9, fontweight="bold",
                color="#444444",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="#cccccc", alpha=0.9),
                zorder=6)

    # ── Legend ──
    handles = [
        plt.Line2D([0], [0], color="#c0392b", linewidth=2,
                   label="Heat source (brine)"),
        plt.Line2D([0], [0], color="#2980b9", linewidth=2,
                   label="Working fluid"),
    ]
    if overlay_segments is not None:
        handles.append(
            plt.Line2D([0], [0], color="gray", linewidth=1.5, linestyle="--",
                       label=overlay_label or "Overlay")
        )
    ax.legend(handles=handles, loc="lower right", fontsize=9, framealpha=0.9)

    fig.tight_layout()

    # ── Save ──
    if savefig:
        fig.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"Figure saved → {filename}")

    # ── Console summary ──
    print()
    print("─" * 60)
    print("  Combined T–Q diagram summary")
    print("─" * 60)
    for seg in segments:
        display = _DISPLAY.get(seg["name"], seg["name"])
        dT = seg["T_hot"] - seg["T_cold"]
        print(f"  {display:<20s}: Q = {seg['Q_total']:10.1f} kW"
              f"   pinch ΔT = {float(np.min(dT)):5.1f} °C")
    Q_total = sum(s["Q_total"] for s in segments)
    print(f"  {'Total':<20s}: Q = {Q_total:10.1f} kW")
    print("─" * 60)

    return fig, ax
