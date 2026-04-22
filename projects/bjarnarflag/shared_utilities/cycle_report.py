"""
Cycle performance reporting for ORC cycles (thermopt post-processing).

Reads energy and exergy results already computed inside thermopt and
presents them as terminal summaries, Excel exports, and plots.

Usage
-----
    from cycle_report import generate_cycle_report

    results = generate_cycle_report(cycle)
    results.print_summary()
    results.to_excel("cycle_report.xlsx")

Author : Amanda (Master Thesis)
"""

import os
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


def _eval_yaml_expr(value):
    """Safely evaluate simple arithmetic expressions found in the YAML config."""
    if isinstance(value, str):
        try:
            return float(eval(value, {"__builtins__": {}}, {"np": np}))
        except Exception:
            return value
    return float(value)


DISPLAY_NAMES = {
    "heater": "Evaporator",
    "hp_evaporator": "HP Evaporator",
    "lp_evaporator": "LP Evaporator",
    "preheater": "Preheater",
    "recuperator": "Recuperator",
    "cooler": "Condenser",
    "expander": "Expander",
    "compressor": "Compressor",
    "hp_expander": "HP Expander",
    "lp_expander": "LP Expander",
    "lp_pump": "LP Pump",
    "hp_pump": "HP Pump",
    "heat_source_pump": "Heat Source Pump",
    "heat_sink_pump": "Heat Sink Pump",
    "mixer": "Mixer",
}


def _display(name):
    """Return the human-readable display name for an internal component key."""
    return DISPLAY_NAMES.get(name, name.replace("_", " ").title())


class CycleReport:
    """Container for the full exergy-analysis output."""

    def __init__(self):
        self.T0 = None  # Dead-state temperature [K]
        self.p0 = None  # Dead-state pressure [Pa]
        self.components = {}  # Per-component results (dict of dicts)
        self.cycle = {}  # Overall cycle-level results

    def print_summary(self):
        """Print a formatted summary of the exergy analysis."""

        # ══════════════════════════════════════════════════════════════
        #  SECTION 1 — EFFICIENCY COMPARISON (the headline numbers)
        # ══════════════════════════════════════════════════════════════
        print("\n" + "=" * 76)
        print("  EFFICIENCY COMPARISON")
        print("=" * 76)
        print()
        print("  From ThermOpt (1st Law / Energy-based):")
        print(
            f"    Cycle efficiency    η_cycle   = W_net / Q_in       = "
            f"{self.cycle.get('eta_cycle', float('nan')) * 100:6.2f} %"
        )
        print(
            f"    System efficiency   η_system  = W_net / Q_avail    = "
            f"{self.cycle.get('eta_system', float('nan')) * 100:6.2f} %"
        )
        eta_sa = self.cycle.get("eta_system_ambient", None)
        if eta_sa is not None:
            print(
                f"    Ambient efficiency  η_ambient = W_net / Q_avail,0  = "
                f"{eta_sa * 100:6.2f} %"
            )
        print()
        print("  From Exergy Analysis (2nd Law / Exergy-based):")
        print(
            f"    Exergy efficiency   η_exergy  = W_net / E_fuel     = "
            f"{self.cycle.get('eta_exergy', float('nan')) * 100:6.2f} %"
        )
        print()
        print("  ── Interpretation for Geothermal ──")
        print("    • η_cycle:   How well the cycle converts received heat to work")
        print(
            "    • η_system:  How well the system extracts power from the brine (KEY METRIC)"
        )
        if eta_sa is not None:
            print(
                "    • η_ambient: η_system but referenced to ambient instead of reinjection limit"
            )
        print(
            "    • η_exergy:  How close to thermodynamic ideal (quality of conversion)"
        )
        print()

        # ── Power breakdown ──
        print("  ── Power Breakdown ──")
        print(
            f"  Gross expander power               : "
            f"{self.cycle.get('W_expander', 0) / 1e3:12.2f} kW"
        )
        print(
            f"  WF pump (compressor) power         : "
            f"{self.cycle.get('W_compressor', 0) / 1e3:12.2f} kW"
        )
        print(
            f"  Auxiliary pumps power              : "
            f"{self.cycle.get('W_aux_pumps', 0) / 1e3:12.2f} kW"
        )
        print(
            f"  Net system power                   : "
            f"{self.cycle.get('W_net_system', 0) / 1e3:12.2f} kW"
        )
        print()

        # ── Energy values (imported from ThermOpt) ──
        print("  ── Energy Analysis (from ThermOpt) ──")
        print(
            f"  Heat input Q_in                    : "
            f"{self.cycle.get('Q_in', 0) / 1e3:12.2f} kW"
        )
        print(
            f"  Available heat Q_available         : "
            f"{self.cycle.get('Q_available', 0) / 1e3:12.2f} kW"
        )
        print(
            f"  Heat utilization (Q_in/Q_avail)    : "
            f"{self.cycle.get('heat_utilization', 0) * 100:12.2f} %"
        )
        print()

        # ── Exergy values (calculated here) ──
        print("  ── Exergy Analysis (calculated here) ──")
        print(
            f"  Exergy fuel (heat source)          : "
            f"{self.cycle['E_fuel'] / 1e3:12.2f} kW"
        )
        print(
            f"  Exergy product (net power)         : "
            f"{self.cycle['E_product'] / 1e3:12.2f} kW"
        )
        print(
            f"  Exergy loss (condenser)            : "
            f"{self.cycle.get('E_loss_cooler', 0) / 1e3:12.2f} kW"
        )
        print("=" * 76)

        # ══════════════════════════════════════════════════════════════
        #  SECTION 2 — EXERGY BREAKDOWN (component-by-component)
        # ══════════════════════════════════════════════════════════════
        print("\n" + "=" * 76)
        print("  EXERGY ANALYSIS  —  Component-by-Component Results")
        print("=" * 76)
        print(
            f"  Dead state:  T0 = {self.T0:.2f} K  ({self.T0 - 273.15:.2f} °C)"
            f"  |  p0 = {self.p0:.0f} Pa"
        )
        print("-" * 76)

        header = (
            f"  {'Component':<24s} {'E_D [kW]':>10s} {'E_D [%]':>10s}"
            f" {'eta_ex [%]':>10s}"
        )
        print(header)
        print("-" * 76)

        E_D_total = self.cycle.get("E_D_total", 1.0)  # avoid /0

        for name, data in self.components.items():
            E_D_kW = data["E_D"] / 1e3
            E_D_pct = data["E_D"] / E_D_total * 100 if E_D_total != 0 else 0.0
            eta_str = (
                f"{data['eta_exergy'] * 100:10.2f}"
                if data["eta_exergy"] is not None
                else "       N/A"
            )
            n_par = data.get("n_turbines_parallel", 1)
            disp = _display(name) + (f" ({n_par}x, combined)" if n_par > 1 else "")
            print(f"  {disp:<24s} {E_D_kW:10.2f} {E_D_pct:10.2f} {eta_str}")

        print("-" * 76)
        print(f"  {'TOTAL':<24s} {E_D_total / 1e3:10.2f} {'100.00':>10s}")
        if any(d.get("n_turbines_parallel", 1) > 1 for d in self.components.values()):
            print(
                f"  * parallel machines: E_D and eta_exergy are totals across all shafts"
            )
        print()

        # Exergy balance closure
        residual = self.cycle.get("balance_residual", 0)
        E_fuel_val = self.cycle["E_fuel"]
        pct = abs(residual) / E_fuel_val * 100 if E_fuel_val != 0 else 0
        print("  ── Exergy Balance Check ──")
        print(f"  Balance: E_fuel = W_net_cycle + E_D_internal + E_loss_condenser")
        print(
            f"  Residual                           : "
            f"{residual / 1e3:12.6f} kW  ({pct:.4f}% of fuel)"
        )
        print("=" * 76 + "\n")

    def to_excel(self, filename="exergy_results.xlsx"):
        """Write the results to an Excel file with two sheets."""

        def r(val, decimals=2):
            """Round a value safely; return None if None."""
            if val is None:
                return None
            try:
                return round(float(val), decimals)
            except (TypeError, ValueError):
                return val

        # ── Sheet 1: component-level exergy ──────────────────────────────────
        E_D_total = self.cycle.get("E_D_total", 1.0)
        rows = []
        for name, data in self.components.items():
            eta = data["eta_exergy"]
            rows.append(
                {
                    "Component": _display(name),
                    "E_D [kW]": r(data["E_D"] / 1e3),
                    "E_D fraction [%]": r(
                        data["E_D"] / E_D_total * 100 if E_D_total else 0
                    ),
                    "Exergetic efficiency [%]": r(
                        eta * 100 if eta is not None else None
                    ),
                    "E_in [kW]": r(
                        data.get("E_in", 0) / 1e3
                        if data.get("E_in") is not None
                        else None
                    ),
                    "E_out [kW]": r(
                        data.get("E_out", 0) / 1e3
                        if data.get("E_out") is not None
                        else None
                    ),
                }
            )
        df_comp = pd.DataFrame(rows)

        # ── Sheet 2: cycle summary ────────────────────────────────────────────
        residual = self.cycle.get("balance_residual", 0)
        E_fuel_val = self.cycle["E_fuel"]
        resid_pct = abs(residual) / E_fuel_val * 100 if E_fuel_val != 0 else 0

        cycle_rows = [
            # Dead state
            ("══ DEAD STATE ══", ""),
            ("Dead-state temperature T0 [°C]", r(self.T0 - 273.15, 2)),
            ("Dead-state temperature T0 [K]", r(self.T0, 2)),
            ("Dead-state pressure p0 [Pa]", r(self.p0, 0)),
            ("", ""),
            # Power breakdown
            ("══ POWER BREAKDOWN ══", ""),
            ("Gross expander power [kW]", r(self.cycle.get("W_expander", 0) / 1e3)),
            (
                "WF pump (compressor) power [kW]",
                r(self.cycle.get("W_compressor", 0) / 1e3),
            ),
            (
                "Auxiliary pumps power [kW]",
                r(self.cycle.get("W_aux_pumps", 0) / 1e3, 3),
            ),
            ("Net system power [kW]", r(self.cycle.get("W_net_system", 0) / 1e3)),
            (
                "Net cycle power W_exp - W_comp [kW]",
                r(self.cycle.get("W_net_cycle", 0) / 1e3),
            ),
            ("", ""),
            # Energy analysis
            ("══ ENERGY ANALYSIS ══", ""),
            ("Heat input Q_in [kW]", r(self.cycle.get("Q_in", 0) / 1e3)),
            (
                "Available heat Q_available [kW]",
                r(self.cycle.get("Q_available", 0) / 1e3),
            ),
            (
                "Heat utilization Q_in / Q_avail [%]",
                r(self.cycle.get("heat_utilization", 0) * 100, 3),
            ),
            ("", ""),
            (
                "Cycle efficiency  η_cycle  = W_net / Q_in [%]",
                r(
                    (
                        self.cycle.get("eta_cycle", 0) * 100
                        if self.cycle.get("eta_cycle")
                        else None
                    ),
                    3,
                ),
            ),
            (
                "System efficiency η_system = W_net / Q_avail [%]",
                r(
                    (
                        self.cycle.get("eta_system", 0) * 100
                        if self.cycle.get("eta_system")
                        else None
                    ),
                    3,
                ),
            ),
            ("", ""),
            # Exergy analysis
            ("══ EXERGY ANALYSIS ══", ""),
            ("Exergy fuel E_fuel [kW]", r(E_fuel_val / 1e3)),
            ("Exergy product E_product [kW]", r(self.cycle["E_product"] / 1e3)),
            (
                "Exergy efficiency η_exergy = W_net / E_fuel [%]",
                r(
                    (
                        self.cycle.get("eta_exergy", 0) * 100
                        if self.cycle.get("eta_exergy")
                        else None
                    ),
                    3,
                ),
            ),
            ("", ""),
            (
                "Total exergy destruction E_D_total [kW]",
                r(self.cycle["E_D_total"] / 1e3),
            ),
            (
                "Internal exergy destruction E_D_internal [kW]",
                r(self.cycle.get("E_D_internal", 0) / 1e3),
            ),
            (
                "Exergy loss condenser E_loss [kW]",
                r(self.cycle.get("E_loss_cooler", 0) / 1e3),
            ),
            ("", ""),
            # Exergy balance check — placed right after the exergy numbers
            (
                "── Exergy balance check: E_fuel = W_net_cycle + E_D_internal + E_loss_condenser ──",
                "",
            ),
            ("Balance residual [kW]", r(residual / 1e3, 6)),
            ("Balance residual [%]", r(resid_pct, 4)),
            ("Balance OK", "Yes" if resid_pct < 0.01 else "WARNING"),
        ]
        df_cycle = pd.DataFrame(cycle_rows, columns=["Parameter", "Value"])

        with pd.ExcelWriter(filename, engine="openpyxl") as writer:
            df_comp.to_excel(writer, index=False, sheet_name="component_exergy")
            df_cycle.to_excel(writer, index=False, sheet_name="cycle_summary")

        print(f"    ✓ {os.path.basename(filename)}")

    def plot_exergy_destruction(self, savefig=None, figsize=(8, 5)):
        """
        Bar chart: exergy destruction per component (kW) with percentage labels.

        Parameters
        ----------
        savefig : str, optional
            If provided, save the figure to this filename (e.g. "exergy_bar.png").
        figsize : tuple
            Figure size in inches.

        Returns
        -------
        fig, ax : matplotlib Figure and Axes
        """
        # Collect data — inactive recuperators are never added to results.components
        # so all components here are always valid and should be shown
        names = []
        E_D_vals = []
        for name, data in self.components.items():
            names.append(_display(name))
            E_D_vals.append(data["E_D"] / 1e3)  # kW

        E_D_total = self.cycle["E_D_total"] / 1e3

        # Sort from largest to smallest
        order = np.argsort(E_D_vals)[::-1]
        names = [names[i] for i in order]
        E_D_vals = [E_D_vals[i] for i in order]

        # Colors
        colors = plt.cm.RdYlBu_r(np.linspace(0.15, 0.85, len(names)))

        fig, ax = plt.subplots(figsize=figsize)
        bars = ax.barh(names, E_D_vals, color=colors, edgecolor="black", linewidth=0.5)

        # Add percentage labels on each bar
        for bar, val in zip(bars, E_D_vals):
            pct = val / E_D_total * 100 if E_D_total != 0 else 0
            ax.text(
                bar.get_width() + E_D_total * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.1f} kW ({pct:.1f}%)",
                va="center",
                fontsize=9,
            )

        ax.set_xlabel("Exergy Destruction [kW]")
        ax.set_title("Exergy Destruction by Component")
        ax.invert_yaxis()
        ax.set_xlim(0, max(E_D_vals) * 1.35)
        fig.tight_layout()

        if savefig:
            fig.savefig(savefig, dpi=300, bbox_inches="tight")
            print(f"    ✓ {os.path.basename(savefig)}")

        return fig, ax

    def plot_pie_chart(self, savefig=None, figsize=(7, 7)):
        """
        Pie chart: relative exergy destruction per component.

        Parameters
        ----------
        savefig : str, optional
            If provided, save the figure to this filename.
        figsize : tuple
            Figure size in inches.

        Returns
        -------
        fig, ax : matplotlib Figure and Axes
        """
        names = []
        E_D_vals = []
        for name, data in self.components.items():
            names.append(_display(name))
            E_D_vals.append(data["E_D"] / 1e3)

        # Matplotlib requires non-negative wedge sizes. Exergy destruction should be >= 0,
        # but we guard against sign-convention issues and numerical noise.
        E_D_vals = np.asarray(E_D_vals, dtype=float)
        mask = E_D_vals > 0
        if not np.any(mask):
            raise ValueError(
                "No positive exergy destruction values available for pie chart."
            )
        E_D_vals = E_D_vals[mask]
        names = [n for n, keep in zip(names, mask) if keep]

        colors = plt.cm.Set2(np.linspace(0, 1, len(names)))

        fig, ax = plt.subplots(figsize=figsize)
        wedges, texts, autotexts = ax.pie(
            E_D_vals,
            labels=names,
            autopct="%1.1f%%",
            colors=colors,
            startangle=140,
            pctdistance=0.80,
            wedgeprops={"edgecolor": "black", "linewidth": 0.5},
        )
        for t in autotexts:
            t.set_fontsize(9)
        ax.set_title("Exergy Destruction Breakdown")
        fig.tight_layout()

        if savefig:
            fig.savefig(savefig, dpi=300, bbox_inches="tight")
            print(f"    ✓ {os.path.basename(savefig)}")

        return fig, ax

    def plot_grassmann(self, savefig=None, figsize=(10, 6)):
        """
        Stacked waterfall / Grassmann-style diagram showing how the fuel
        exergy is split into product, component destructions, and losses.

        Parameters
        ----------
        savefig : str, optional
            If provided, save the figure to this filename.
        figsize : tuple
            Figure size in inches.

        Returns
        -------
        fig, ax : matplotlib Figure and Axes
        """
        E_fuel = self.cycle["E_fuel"] / 1e3
        E_product = self.cycle["E_product"] / 1e3
        E_loss = self.cycle.get("E_loss_cooler", 0) / 1e3

        # Component destructions sorted largest first
        comp_names = []
        comp_vals = []
        for name, data in self.components.items():
            comp_names.append(_display(name))
            comp_vals.append(data["E_D"] / 1e3)
        order = np.argsort(comp_vals)[::-1]
        comp_names = [comp_names[i] for i in order]
        comp_vals = [comp_vals[i] for i in order]

        # Build waterfall: fuel → -destructions → -loss → product
        labels = (
            ["Exergy Fuel"]
            + comp_names
            + ["Exergy Loss\n(condenser)", "Net Power\n(product)"]
        )
        values = [E_fuel] + [-v for v in comp_vals] + [-E_loss, 0]

        # Compute running total for bar positioning
        cumulative = np.zeros(len(values))
        cumulative[0] = values[0]
        for i in range(1, len(values) - 1):
            cumulative[i] = cumulative[i - 1] + values[i]
        cumulative[-1] = E_product  # final bar sits at product level

        # Bottom positions for each bar
        bottoms = np.zeros(len(values))
        bottoms[0] = 0
        for i in range(1, len(values) - 1):
            bottoms[i] = cumulative[i]
        bottoms[-1] = 0

        bar_heights = np.zeros(len(values))
        bar_heights[0] = values[0]
        for i in range(1, len(values) - 1):
            bar_heights[i] = -values[i]  # positive height for display
        bar_heights[-1] = E_product

        # Colors
        bar_colors = ["#2196F3"]  # fuel = blue
        destruction_colors = plt.cm.Oranges(np.linspace(0.3, 0.8, len(comp_vals)))
        for c in destruction_colors:
            bar_colors.append(c)
        bar_colors.append("#FF9800")  # loss = orange
        bar_colors.append("#4CAF50")  # product = green

        fig, ax = plt.subplots(figsize=figsize)
        x = np.arange(len(labels))
        bars = ax.bar(
            x,
            bar_heights,
            bottom=bottoms,
            color=bar_colors,
            edgecolor="black",
            linewidth=0.5,
            width=0.65,
        )

        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, bar_heights)):
            y_pos = bottoms[i] + val / 2
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                y_pos,
                f"{val:.1f}",
                ha="center",
                va="center",
                fontsize=8,
                fontweight="bold",
            )

        # Connector lines between bars
        for i in range(len(x) - 1):
            top_current = bottoms[i] + bar_heights[i] if i == 0 else cumulative[i]
            if i < len(x) - 2:
                top_next = cumulative[i + 1] + bar_heights[i + 1]
            else:
                top_next = bottoms[-1] + bar_heights[-1]
            # Don't draw connector to the last bar if it goes negative
            if i == 0:
                y_line = cumulative[0]
            else:
                y_line = cumulative[i]
            ax.plot(
                [x[i] + 0.325, x[i + 1] - 0.325],
                [y_line, y_line],
                color="gray",
                linewidth=0.8,
                linestyle="--",
            )

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
        ax.set_ylabel("Exergy Rate [kW]")
        ax.set_title("Grassmann Diagram — Exergy Flow Through the Cycle")
        ax.set_ylim(0, E_fuel * 1.1)
        fig.tight_layout()

        if savefig:
            fig.savefig(savefig, dpi=300, bbox_inches="tight")
            print(f"    ✓ {os.path.basename(savefig)}")

        return fig, ax

    def to_dict(self):
        """Return a plain dictionary representation."""
        return {
            "T0": self.T0,
            "p0": self.p0,
            "components": self.components,
            "cycle": self.cycle,
        }


def generate_cycle_report(cycle_object):
    """
    Assemble a CycleReport object from energy and exergy data already
    computed inside thermopt during the cycle evaluation.

    Parameters
    ----------
    cycle_object : thermopt.ThermodynamicCycleOptimization
        The optimized cycle object (after ``run_optimization()``).

    Returns
    -------
    CycleReport
        Object with ``.print_summary()``, ``.to_excel()``, ``.to_dict()`` methods.
    """

    cycle_data = cycle_object.problem.cycle_data
    components = cycle_data["components"]
    energy = cycle_data["energy_analysis"]
    exergy = cycle_data["exergy_analysis"]  # computed inside thermopt

    T0 = exergy["T0"]
    p0 = exergy["p0"]

    is_dual = "hp_expander" in components
    is_two_source = "heat_source_pump_hp" in components

    eta_cycle = energy.get("cycle_efficiency", 0.0)
    eta_system = energy.get("system_efficiency", 0.0)
    eta_system_ambient = energy.get("system_efficiency_ambient", None)

    if is_dual:
        Q_in = energy.get("total_heat_input", 0.0)
        W_expander = energy.get("total_expander_power", 0.0)
        W_compressor = (
            components["lp_pump"]["energy_analysis"]["power"]
            + components["hp_pump"]["energy_analysis"]["power"]
        )
    else:
        Q_in = energy.get("heater_heat_flow", 0.0)
        W_expander = components["expander"]["energy_analysis"]["power"]
        W_compressor = components["compressor"]["energy_analysis"]["power"]

    if is_two_source:
        W_hs_pump = (
            components["heat_source_pump_hp"]["energy_analysis"]["power"]
            + components["heat_source_pump_lp"]["energy_analysis"]["power"]
        )
    else:
        W_hs_pump = components["heat_source_pump"]["energy_analysis"]["power"]

    W_hk_pump = components["heat_sink_pump"]["energy_analysis"]["power"]
    W_aux_pumps = W_hs_pump + W_hk_pump
    W_net_system = energy.get("net_system_power", 0.0)
    Q_available = energy.get("heater_heat_flow_max", 0.0)
    heat_utilization = Q_in / Q_available if Q_available != 0 else 0.0

    results = CycleReport()
    results.T0 = T0
    results.p0 = p0

    for name, comp in components.items():
        ex = comp.get("exergy_analysis")
        if ex is None:
            continue
        results.components[name] = {
            "E_D": float(ex["E_D"]),
            "eta_exergy": (
                float(ex["eta_exergy"]) if ex["eta_exergy"] is not None else None
            ),
            "E_in": float(ex["E_fuel"]),
            "E_out": float(ex["E_product"]),
            "n_turbines_parallel": comp.get("n_turbines_parallel", 1),
        }

    E_D_total = exergy["E_D_total"]
    E_D_internal = exergy["E_D_internal"]
    E_fuel = exergy["E_fuel"]
    E_product = exergy["E_product"]
    E_loss_cooler = exergy["E_loss_cooler"]
    eta_exergy = exergy["eta_exergy"]
    W_net_cycle = exergy["W_net_cycle"]
    balance_residual = exergy["balance_residual"]

    results.cycle = {
        "Q_in": float(Q_in),
        "Q_available": float(Q_available),
        "heat_utilization": float(heat_utilization),
        "eta_cycle": float(eta_cycle),
        "eta_system": float(eta_system),
        "eta_system_ambient": (
            float(eta_system_ambient) if eta_system_ambient is not None else None
        ),
        "W_expander": float(W_expander),
        "W_compressor": float(W_compressor),
        "W_aux_pumps": float(W_aux_pumps),
        "W_net_system": float(W_net_system),
        "W_net_cycle": float(W_net_cycle),
        "E_fuel": float(E_fuel),
        "E_product": float(E_product),
        "E_D_total": float(E_D_total),
        "E_D_internal": float(E_D_internal),
        "E_loss_cooler": float(E_loss_cooler),
        "eta_exergy": float(eta_exergy),
        "balance_residual": float(balance_residual),
    }

    return results


#  Parametric sweep: heat-source utilization curve
def plot_heat_source_utilization(
    cycle_object,
    config_file,
    n_points=30,
    T_exit_range=None,
    savefig=None,
    figsize=(9, 5),
):
    """
    Sweep the heat-source exit temperature and plot system efficiency,
    cycle efficiency, and net power vs. that temperature.

    This re-evaluates the cycle at the *optimal* design-variable values
    found by thermopt, changing only the heat-source exit temperature.
    It shows how deeply you can cool the heat source before performance
    degrades — directly relevant for geothermal reinjection limits.

    Parameters
    ----------
    cycle_object : thermopt.ThermodynamicCycleOptimization
        The optimized cycle object (after ``run_optimization()``).
    config_file : str
        Path to the YAML config (needed to read temperature limits).
    n_points : int
        Number of sweep points (default 30).
    T_exit_range : tuple of float, optional
        (T_min, T_max) in Kelvin for the sweep.  If None, read from the
        YAML design-variable bounds for ``heat_source_exit_temperature``.
    savefig : str, optional
        If provided, save the figure to this filename.
    figsize : tuple
        Figure size in inches.

    Returns
    -------
    fig, axes : matplotlib Figure and array of Axes
    sweep_data : dict with arrays of temperatures and performance values
    """
    import copy

    if T_exit_range is None:
        with open(config_file, "r") as f:
            cfg = yaml.safe_load(f)
        dv = cfg["problem_formulation"]["design_variables"]
        if "heat_source_exit_temperature" not in dv:
            raise ValueError(
                "Utilization curve requires 'heat_source_exit_temperature' as a "
                "design variable. The dual-pressure ORC computes brine exit "
                "temperature from the energy balance — provide T_exit_range "
                "manually or use a different sweep approach."
            )
        T_min = _eval_yaml_expr(dv["heat_source_exit_temperature"]["min"])
        T_max = _eval_yaml_expr(dv["heat_source_exit_temperature"]["max"])
    else:
        T_min, T_max = T_exit_range

    problem = cycle_object.problem
    optimal_x_dict = copy.deepcopy(problem.x0_dict)
    fixed_params = copy.deepcopy(problem.fixed_parameters)
    constraints = copy.deepcopy(problem.constraints)
    obj_func = copy.deepcopy(problem.objective_function)
    topology = problem.cycle_topology

    from thermopt.cycles import cycle_power_simple, cycle_power_recuperated

    topology_map = {
        "simple": cycle_power_simple.evaluate_cycle,
        "power_simple": cycle_power_simple.evaluate_cycle,
        "recuperated": cycle_power_recuperated.evaluate_cycle,
        "power_recuperated": cycle_power_recuperated.evaluate_cycle,
    }

    # Try to import dual pressure if available
    try:
        from thermopt.cycles import cycle_power_dual_pressure

        topology_map["dual_pressure"] = cycle_power_dual_pressure.evaluate_cycle
    except ImportError:
        pass

    if topology not in topology_map:
        raise ValueError(
            f"Utilization curve not yet supported for topology '{topology}'. "
            f"Supported: {list(topology_map.keys())}"
        )
    evaluate_fn = topology_map[topology]

    T_exit_array = np.linspace(T_min + 1.0, T_max - 1.0, n_points)

    eta_system = []
    eta_cycle = []
    W_net = []
    T_valid = []
    T_opt = float(
        optimal_x_dict.get("heat_source_exit_temperature", (T_min + T_max) / 2)
    )

    for T_exit in T_exit_array:
        try:
            variables = copy.deepcopy(optimal_x_dict)
            variables["heat_source_exit_temperature"] = float(T_exit)

            out = evaluate_fn(
                variables,
                copy.deepcopy(fixed_params),
                copy.deepcopy(constraints),
                copy.deepcopy(obj_func),
            )
            ea = out["energy_analysis"]

            eta_s = float(ea["system_efficiency"])
            eta_c = float(ea["cycle_efficiency"])
            w_net = float(ea["net_system_power"])

            # Skip clearly unphysical points
            if eta_s < 0 or eta_c < 0 or w_net < 0:
                continue

            eta_system.append(eta_s * 100)
            eta_cycle.append(eta_c * 100)
            W_net.append(w_net / 1e3)
            T_valid.append(float(T_exit) - 273.15)

        except Exception:
            # Some temperatures will cause infeasible cycles — just skip them
            continue

    T_valid = np.array(T_valid)
    eta_system = np.array(eta_system)
    eta_cycle = np.array(eta_cycle)
    W_net = np.array(W_net)

    T_opt_C = T_opt - 273.15

    fig, ax1 = plt.subplots(figsize=figsize)

    color_sys = "#2196F3"
    color_cyc = "#4CAF50"
    color_pwr = "#FF9800"

    # Left y-axis: efficiencies
    ax1.plot(
        T_valid,
        eta_system,
        "-o",
        color=color_sys,
        markersize=4,
        label="System efficiency",
        linewidth=2,
    )
    ax1.plot(
        T_valid,
        eta_cycle,
        "-s",
        color=color_cyc,
        markersize=4,
        label="Cycle efficiency",
        linewidth=2,
    )
    ax1.set_xlabel("Heat Source Exit Temperature [°C]", fontsize=11)
    ax1.set_ylabel("Efficiency [%]", fontsize=11)
    ax1.tick_params(axis="y")

    # Right y-axis: net power
    ax2 = ax1.twinx()
    ax2.plot(
        T_valid,
        W_net,
        "-^",
        color=color_pwr,
        markersize=4,
        label="Net power",
        linewidth=2,
    )
    ax2.set_ylabel("Net Power [kW]", fontsize=11, color=color_pwr)
    ax2.tick_params(axis="y", labelcolor=color_pwr)

    # Mark the optimal point
    ax1.axvline(T_opt_C, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax1.annotate(
        f"Optimum\n({T_opt_C:.1f} °C)",
        xy=(T_opt_C, ax1.get_ylim()[0]),
        xytext=(
            T_opt_C + (T_valid[-1] - T_valid[0]) * 0.05,
            ax1.get_ylim()[0] + (ax1.get_ylim()[1] - ax1.get_ylim()[0]) * 0.15,
        ),
        fontsize=9,
        color="gray",
        arrowprops=dict(arrowstyle="->", color="gray", lw=1),
    )

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best", fontsize=9)

    ax1.set_title("Heat Source Utilization Curve", fontsize=13, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()

    if savefig:
        fig.savefig(savefig, dpi=300, bbox_inches="tight")
        print(f"    ✓ {os.path.basename(savefig)}")

    sweep_data = {
        "T_exit_C": T_valid,
        "system_efficiency_pct": eta_system,
        "cycle_efficiency_pct": eta_cycle,
        "net_power_kW": W_net,
    }

    return fig, (ax1, ax2), sweep_data
