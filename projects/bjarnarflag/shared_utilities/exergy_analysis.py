"""
Exergy Analysis Module for ORC Cycles (thermopt post-processing)
================================================================

Performs a component-by-component exergy analysis on the converged thermodynamic
cycle produced by thermopt. The analysis follows the standard steady-state flow
exergy method described in most advanced thermodynamics textbooks
(Bejan et al., Çengel & Boles, Moran & Shapiro).

Methodology
-----------
Dead-state (reference environment):
    T0 = ambient temperature [K],  p0 = ambient pressure [Pa]
    Read from the YAML configuration file (special_points section).

Specific flow exergy at any state point:
    e = (h - h0) - T0 * (s - s0)                               [J/kg]
    where h0, s0 are evaluated at (T0, p0) for the respective fluid.

Component exergy-destruction rate (E_D) via exergy balance:
    - Turbomachinery  : E_D = m * [(e_in - e_out) - w]         (w > 0 for expander)
                        E_D = m * [(e_in - e_out) + w]         (w > 0 for compressor)
    - Heat exchangers : E_D = m_hot*(e_hot_in - e_hot_out)
                            + m_cold*(e_cold_in - e_cold_out)

Exergetic efficiency:
    - Expander   :  eta_ex = W_out / [m * (e_in - e_out)]
    - Compressor :  eta_ex = m * (e_out - e_in) / W_in
    - Heater     :  eta_ex = (exergy gained by cold side) / (exergy given by hot side)
    - Cooler     :  eta_ex is less meaningful; reported but optional
    - Cycle      :  eta_ex = W_net / E_fuel   where E_fuel = exergy supplied by heat source

Energy-based values (cycle_efficiency, system_efficiency, Q_in, etc.) are
IMPORTED from ThermOpt's energy_analysis to avoid redundant calculations
and ensure consistency.

Usage
-----
    from exergy_analysis import perform_exergy_analysis

    # After optimization:
    cycle = th.ThermodynamicCycleOptimization(config)
    cycle.run_optimization()
    cycle.save_results()

    results = perform_exergy_analysis(cycle, config_file="./case_butane_ORC.yaml")
    results.print_summary()
    results.to_excel("exergy_results.xlsx")

Author : Amanda (Master Thesis)
"""

import os
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import jaxprop as cpx


# ---------------------------------------------------------------------------
#  Helper: evaluate the YAML expressions that thermopt uses (e.g. "20 + 273.15")
# ---------------------------------------------------------------------------
def _eval_yaml_expr(value):
    """Safely evaluate simple arithmetic expressions found in the YAML config."""
    if isinstance(value, str):
        try:
            return float(eval(value, {"__builtins__": {}}, {"np": np}))
        except Exception:
            return value
    return float(value)


# ---------------------------------------------------------------------------
#  Helper: compute specific flow exergy
# ---------------------------------------------------------------------------
def _specific_flow_exergy(state, h0, s0, T0):
    """Return specific flow exergy  e = (h - h0) - T0*(s - s0)  in J/kg."""
    return (state.h - h0) - T0 * (state.s - s0)


# ---------------------------------------------------------------------------
#  Data class to hold all exergy results
# ---------------------------------------------------------------------------
class ExergyResults:
    """Container for the full exergy-analysis output."""

    def __init__(self):
        self.T0 = None          # Dead-state temperature [K]
        self.p0 = None          # Dead-state pressure [Pa]
        self.components = {}    # Per-component results (dict of dicts)
        self.cycle = {}         # Overall cycle-level results

    # ----- pretty-print to console ------------------------------------------
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
        print(f"    Cycle efficiency    η_cycle  = W_net / Q_in       = "
              f"{self.cycle.get('eta_cycle', float('nan')) * 100:6.2f} %")
        print(f"    System efficiency   η_system = W_net / Q_avail    = "
              f"{self.cycle.get('eta_system', float('nan')) * 100:6.2f} %")
        print()
        print("  From Exergy Analysis (2nd Law / Exergy-based):")
        print(f"    Exergy efficiency   η_exergy = W_net / E_fuel     = "
              f"{self.cycle.get('eta_exergy', float('nan')) * 100:6.2f} %")
        print()
        print("  ── Interpretation for Geothermal ──")
        print("    • η_cycle:  How well the cycle converts received heat to work")
        print("    • η_system: How well the system extracts power from the brine (KEY METRIC)")
        print("    • η_exergy: How close to thermodynamic ideal (quality of conversion)")
        print()

        # ── Power breakdown ──
        print("  ── Power Breakdown ──")
        print(f"  Gross expander power               : "
              f"{self.cycle.get('W_expander', 0) / 1e3:12.2f} kW")
        print(f"  WF pump (compressor) power         : "
              f"{self.cycle.get('W_compressor', 0) / 1e3:12.2f} kW")
        print(f"  Auxiliary pumps power              : "
              f"{self.cycle.get('W_aux_pumps', 0) / 1e3:12.2f} kW")
        print(f"  Net system power                   : "
              f"{self.cycle.get('W_net_system', 0) / 1e3:12.2f} kW")
        print()

        # ── Energy values (imported from ThermOpt) ──
        print("  ── Energy Analysis (from ThermOpt) ──")
        print(f"  Heat input Q_in                    : "
              f"{self.cycle.get('Q_in', 0) / 1e3:12.2f} kW")
        print(f"  Available heat Q_available         : "
              f"{self.cycle.get('Q_available', 0) / 1e3:12.2f} kW")
        print(f"  Heat utilization (Q_in/Q_avail)    : "
              f"{self.cycle.get('heat_utilization', 0) * 100:12.2f} %")
        print()

        # ── Exergy values (calculated here) ──
        print("  ── Exergy Analysis (calculated here) ──")
        print(f"  Exergy fuel (heat source)          : "
              f"{self.cycle['E_fuel'] / 1e3:12.2f} kW")
        print(f"  Exergy product (net power)         : "
              f"{self.cycle['E_product'] / 1e3:12.2f} kW")
        print(f"  Exergy loss (cooler)               : "
              f"{self.cycle.get('E_loss_cooler', 0) / 1e3:12.2f} kW")
        print("=" * 76)

        # ══════════════════════════════════════════════════════════════
        #  SECTION 2 — EXERGY BREAKDOWN (component-by-component)
        # ══════════════════════════════════════════════════════════════
        print("\n" + "=" * 76)
        print("  EXERGY ANALYSIS  —  Component-by-Component Results")
        print("=" * 76)
        print(f"  Dead state:  T0 = {self.T0:.2f} K  ({self.T0 - 273.15:.2f} °C)"
              f"  |  p0 = {self.p0:.0f} Pa")
        print("-" * 76)

        header = (f"  {'Component':<24s} {'E_D [kW]':>10s} {'E_D [%]':>10s}"
                  f" {'eta_ex [%]':>10s}")
        print(header)
        print("-" * 76)

        E_D_total = self.cycle.get("E_D_total", 1.0)  # avoid /0

        for name, data in self.components.items():
            E_D_kW = data["E_D"] / 1e3
            E_D_pct = data["E_D"] / E_D_total * 100 if E_D_total != 0 else 0.0
            eta_str = (f"{data['eta_exergy'] * 100:10.2f}"
                       if data["eta_exergy"] is not None else "       N/A")
            print(f"  {name:<24s} {E_D_kW:10.2f} {E_D_pct:10.2f} {eta_str}")

        print("-" * 76)
        print(f"  {'TOTAL':<24s} {E_D_total / 1e3:10.2f} {'100.00':>10s}")
        print()

        # Exergy balance closure
        residual = self.cycle.get("balance_residual", 0)
        E_fuel_val = self.cycle["E_fuel"]
        pct = abs(residual) / E_fuel_val * 100 if E_fuel_val != 0 else 0
        print("  ── Exergy Balance Check ──")
        print(f"  Balance: E_fuel = W_net_cycle + E_D_internal + E_loss_cooler")
        print(f"  Residual                           : "
              f"{residual / 1e3:12.6f} kW  ({pct:.4f}% of fuel)")
        print("=" * 76 + "\n")

    # ----- export to Excel --------------------------------------------------
    def to_excel(self, filename="exergy_results.xlsx"):
        """Write the results to an Excel file with two sheets."""
        # Sheet 1 — component-level
        rows = []
        E_D_total = self.cycle.get("E_D_total", 1.0)
        for name, data in self.components.items():
            rows.append({
                "Component": name,
                "E_D [W]": data["E_D"],
                "E_D [kW]": data["E_D"] / 1e3,
                "E_D fraction [%]": data["E_D"] / E_D_total * 100 if E_D_total else 0,
                "Exergetic efficiency [-]": data["eta_exergy"],
                "Exergetic efficiency [%]": (data["eta_exergy"] * 100
                                             if data["eta_exergy"] is not None
                                             else None),
                "E_in [W]": data.get("E_in"),
                "E_out [W]": data.get("E_out"),
            })
        df_comp = pd.DataFrame(rows)

        # Sheet 2 — cycle summary
        cycle_rows = [
            ("Dead-state temperature T0 [K]", self.T0),
            ("Dead-state temperature T0 [°C]", self.T0 - 273.15),
            ("Dead-state pressure p0 [Pa]", self.p0),
            ("", ""),
            ("══ ENERGY ANALYSIS (from ThermOpt) ══", ""),
            ("", ""),
            ("Heat input Q_in [W]", self.cycle.get("Q_in")),
            ("Heat input Q_in [kW]", self.cycle.get("Q_in", 0) / 1e3),
            ("Available heat Q_available [W]", self.cycle.get("Q_available")),
            ("Available heat Q_available [kW]", self.cycle.get("Q_available", 0) / 1e3),
            ("Heat utilization Q_in/Q_avail [-]", self.cycle.get("heat_utilization")),
            ("Heat utilization Q_in/Q_avail [%]", self.cycle.get("heat_utilization", 0) * 100),
            ("", ""),
            ("Cycle efficiency η_cycle = W_net/Q_in [-]", self.cycle.get("eta_cycle")),
            ("Cycle efficiency η_cycle = W_net/Q_in [%]", 
             self.cycle.get("eta_cycle", 0) * 100 if self.cycle.get("eta_cycle") else None),
            ("System efficiency η_system = W_net/Q_avail [-]", self.cycle.get("eta_system")),
            ("System efficiency η_system = W_net/Q_avail [%]", 
             self.cycle.get("eta_system", 0) * 100 if self.cycle.get("eta_system") else None),
            ("", ""),
            ("══ EXERGY ANALYSIS (calculated here) ══", ""),
            ("", ""),
            ("Exergy fuel E_fuel [W]", self.cycle["E_fuel"]),
            ("Exergy fuel E_fuel [kW]", self.cycle["E_fuel"] / 1e3),
            ("Exergy product E_product [W]", self.cycle["E_product"]),
            ("Exergy product E_product [kW]", self.cycle["E_product"] / 1e3),
            ("", ""),
            ("Total exergy destruction E_D_total [W]", self.cycle["E_D_total"]),
            ("Total exergy destruction E_D_total [kW]", self.cycle["E_D_total"] / 1e3),
            ("Internal exergy destruction E_D_internal [W]", self.cycle.get("E_D_internal")),
            ("Internal exergy destruction E_D_internal [kW]", self.cycle.get("E_D_internal", 0) / 1e3),
            ("Exergy loss (cooler) E_loss [W]", self.cycle.get("E_loss_cooler")),
            ("Exergy loss (cooler) E_loss [kW]", self.cycle.get("E_loss_cooler", 0) / 1e3),
            ("", ""),
            ("Exergy efficiency η_exergy = W_net/E_fuel [-]", self.cycle.get("eta_exergy")),
            ("Exergy efficiency η_exergy = W_net/E_fuel [%]", 
             self.cycle.get("eta_exergy", 0) * 100 if self.cycle.get("eta_exergy") else None),
            ("", ""),
            ("══ POWER BREAKDOWN ══", ""),
            ("", ""),
            ("Gross expander power [W]", self.cycle.get("W_expander")),
            ("Gross expander power [kW]", self.cycle.get("W_expander", 0) / 1e3),
            ("WF pump (compressor) power [W]", self.cycle.get("W_compressor")),
            ("WF pump (compressor) power [kW]", self.cycle.get("W_compressor", 0) / 1e3),
            ("Auxiliary pumps power [W]", self.cycle.get("W_aux_pumps")),
            ("Auxiliary pumps power [kW]", self.cycle.get("W_aux_pumps", 0) / 1e3),
            ("Net system power [W]", self.cycle.get("W_net_system")),
            ("Net system power [kW]", self.cycle.get("W_net_system", 0) / 1e3),
            ("Net cycle power (W_exp - W_comp) [W]", self.cycle.get("W_net_cycle")),
            ("Net cycle power (W_exp - W_comp) [kW]", self.cycle.get("W_net_cycle", 0) / 1e3),
            ("", ""),
            ("══ EXERGY BALANCE CHECK ══", ""),
            ("", ""),
            ("Balance: E_fuel = W_net_cycle + E_D_internal + E_loss_cooler", ""),
            ("Balance residual [W]", self.cycle.get("balance_residual")),
            ("Balance residual [kW]", self.cycle.get("balance_residual", 0) / 1e3),
        ]
        df_cycle = pd.DataFrame(cycle_rows, columns=["Parameter", "Value"])

        with pd.ExcelWriter(filename, engine="openpyxl") as writer:
            df_comp.to_excel(writer, index=False, sheet_name="component_exergy")
            df_cycle.to_excel(writer, index=False, sheet_name="cycle_summary")

        print(f"    ✓ {os.path.basename(filename)}")

    # ----- bar chart of exergy destruction -----------------------------------
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
        # Collect data (skip components with negligible destruction)
        names = []
        E_D_vals = []
        for name, data in self.components.items():
            names.append(name.replace("_", " ").title())
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
            ax.text(bar.get_width() + E_D_total * 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{val:.1f} kW ({pct:.1f}%)",
                    va="center", fontsize=9)

        ax.set_xlabel("Exergy Destruction [kW]")
        ax.set_title("Exergy Destruction by Component")
        ax.invert_yaxis()
        ax.set_xlim(0, max(E_D_vals) * 1.35)
        fig.tight_layout()

        if savefig:
            fig.savefig(savefig, dpi=300, bbox_inches="tight")
            print(f"    ✓ {os.path.basename(savefig)}")

        return fig, ax

    # ----- pie chart of exergy destruction -----------------------------------
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
            names.append(name.replace("_", " ").title())
            E_D_vals.append(data["E_D"] / 1e3)

        # Matplotlib requires non-negative wedge sizes. Exergy destruction should be >= 0,
        # but we guard against sign-convention issues and numerical noise.
        E_D_vals = np.asarray(E_D_vals, dtype=float)
        mask = E_D_vals > 0
        if not np.any(mask):
            raise ValueError("No positive exergy destruction values available for pie chart.")
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

    # ----- Grassmann (exergy flow) diagram -----------------------------------
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
            comp_names.append(name.replace("_", " ").title())
            comp_vals.append(data["E_D"] / 1e3)
        order = np.argsort(comp_vals)[::-1]
        comp_names = [comp_names[i] for i in order]
        comp_vals = [comp_vals[i] for i in order]

        # Build waterfall: fuel → -destructions → -loss → product
        labels = ["Exergy Fuel"] + comp_names + ["Exergy Loss\n(cooler)", "Net Power\n(product)"]
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
        bars = ax.bar(x, bar_heights, bottom=bottoms, color=bar_colors,
                       edgecolor="black", linewidth=0.5, width=0.65)

        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, bar_heights)):
            y_pos = bottoms[i] + val / 2
            ax.text(bar.get_x() + bar.get_width() / 2, y_pos,
                    f"{val:.1f}", ha="center", va="center", fontsize=8,
                    fontweight="bold")

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
            ax.plot([x[i] + 0.325, x[i + 1] - 0.325], [y_line, y_line],
                    color="gray", linewidth=0.8, linestyle="--")

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

    # ----- export to dict (for programmatic use) ----------------------------
    def to_dict(self):
        """Return a plain dictionary representation."""
        return {
            "T0": self.T0,
            "p0": self.p0,
            "components": self.components,
            "cycle": self.cycle,
        }


# ===========================================================================
#  Main function: perform_exergy_analysis
# ===========================================================================
def perform_exergy_analysis(cycle_object, config_file=None, T0=None, p0=None):
    """
    Run an exergy analysis on a converged thermopt cycle.
    
    Energy-based values (efficiencies, heat flows, power) are IMPORTED from
    ThermOpt's energy_analysis. Only exergy-specific calculations are performed
    here to avoid redundancy and ensure consistency.

    Parameters
    ----------
    cycle_object : thermopt.ThermodynamicCycleOptimization
        The optimized cycle object (after ``run_optimization()``).
    config_file : str, optional
        Path to the YAML configuration file.  Used to read T0 and p0 from
        ``special_points`` if they are not provided explicitly.
    T0 : float, optional
        Dead-state temperature [K].  Overrides the YAML value.
    p0 : float, optional
        Dead-state pressure [Pa].  Overrides the YAML value.

    Returns
    -------
    ExergyResults
        Object with ``.print_summary()``, ``.to_excel()``, ``.to_dict()`` methods.
    """

    # =========================================================================
    # 1. Access the converged cycle data from ThermOpt
    # =========================================================================
    cycle_data = cycle_object.problem.cycle_data
    components = cycle_data["components"]
    energy = cycle_data["energy_analysis"]  # ThermOpt's energy analysis results

    # =========================================================================
    # 2. IMPORT energy-based values from ThermOpt (no recalculation!)
    # =========================================================================
    
    # Auto-detect topology
    is_dual = "hp_expander" in components
    
    # Efficiencies (from ThermOpt)
    eta_cycle = energy.get("cycle_efficiency", 0.0)      # W_net / Q_in
    eta_system = energy.get("system_efficiency", 0.0)    # W_net / Q_available
    
    # Heat flows (from ThermOpt)
    if is_dual:
        Q_in = energy.get("total_heat_input", 0.0)
    else:
        Q_in = energy.get("heater_heat_flow", 0.0)
    Q_available = energy.get("heater_heat_flow_max", 0.0)
    
    # Power values (from ThermOpt)
    if is_dual:
        W_expander = energy.get("total_expander_power", 0.0)
        W_compressor = energy.get("lp_pump_power", 0.0) + energy.get("hp_pump_power", 0.0)
    else:
        W_expander = energy.get("expander_power", 0.0)
        W_compressor = energy.get("compressor_power", 0.0)
    W_hs_pump = energy.get("heat_source_pump_power", 0.0)
    W_hk_pump = energy.get("heat_sink_pump_power", 0.0)
    W_aux_pumps = W_hs_pump + W_hk_pump
    W_net_system = energy.get("net_system_power", 0.0)
    
    # Derived value
    heat_utilization = Q_in / Q_available if Q_available != 0 else 0.0

    # =========================================================================
    # 3. Determine dead-state conditions
    # =========================================================================
    if T0 is None or p0 is None:
        if config_file is None:
            raise ValueError(
                "Either provide (T0, p0) directly or supply config_file "
                "so that ambient conditions can be read from the YAML."
            )
        with open(config_file, "r") as f:
            cfg = yaml.safe_load(f)
        sp = cfg["problem_formulation"]["fixed_parameters"]["special_points"]
        if T0 is None:
            T0 = _eval_yaml_expr(sp["ambient_temperature"])
        if p0 is None:
            p0 = _eval_yaml_expr(sp["ambient_pressure"])

    # =========================================================================
    # 4. Compute dead-state properties for each fluid
    # =========================================================================
    fluids = {
        "working_fluid": cycle_data["working_fluid"],
        "heating_fluid": cycle_data["heating_fluid"],
        "cooling_fluid": cycle_data["cooling_fluid"],
    }
    dead_state = {}
    for tag, fluid in fluids.items():
        state0 = fluid.get_state(cpx.PT_INPUTS, p0, T0)
        dead_state[tag] = {"h0": state0.h, "s0": state0.s}

    # Short aliases
    h0_wf = dead_state["working_fluid"]["h0"]
    s0_wf = dead_state["working_fluid"]["s0"]
    h0_hf = dead_state["heating_fluid"]["h0"]
    s0_hf = dead_state["heating_fluid"]["s0"]
    h0_cf = dead_state["cooling_fluid"]["h0"]
    s0_cf = dead_state["cooling_fluid"]["s0"]

    # =========================================================================
    # 5. Helper functions to get exergy at a given state
    # =========================================================================
    def e_wf(state):
        """Specific flow exergy of working fluid."""
        return _specific_flow_exergy(state, h0_wf, s0_wf, T0)

    def e_hf(state):
        """Specific flow exergy of heating fluid."""
        return _specific_flow_exergy(state, h0_hf, s0_hf, T0)

    def e_cf(state):
        """Specific flow exergy of cooling fluid."""
        return _specific_flow_exergy(state, h0_cf, s0_cf, T0)

    # =========================================================================
    # 6. Component-by-component EXERGY analysis (this is what we calculate)
    # =========================================================================
    results = ExergyResults()
    results.T0 = T0
    results.p0 = p0

    if is_dual:
        # ==================================================================
        # DUAL-PRESSURE ORC
        # ==================================================================

        # ---- 6a. HP Expander (turbine) --------------------------------------
        hp_exp = components["hp_expander"]
        m_HP = hp_exp["mass_flow"]
        e_hp_exp_in = e_wf(hp_exp["state_in"])
        e_hp_exp_out = e_wf(hp_exp["state_out"])
        W_hp_exp = hp_exp["power"]
        E_D_hp_exp = m_HP * (e_hp_exp_in - e_hp_exp_out) - W_hp_exp
        eta_ex_hp_exp = (W_hp_exp / (m_HP * (e_hp_exp_in - e_hp_exp_out))
                         if (e_hp_exp_in - e_hp_exp_out) != 0 else None)

        results.components["hp_expander"] = {
            "E_D": float(E_D_hp_exp),
            "eta_exergy": float(eta_ex_hp_exp) if eta_ex_hp_exp is not None else None,
            "E_in": float(m_HP * (e_hp_exp_in - e_hp_exp_out)),
            "E_out": float(W_hp_exp),
        }

        # ---- 6b. LP Expander (turbine) --------------------------------------
        lp_exp = components["lp_expander"]
        m_total = lp_exp["mass_flow"]
        e_lp_exp_in = e_wf(lp_exp["state_in"])
        e_lp_exp_out = e_wf(lp_exp["state_out"])
        W_lp_exp = lp_exp["power"]
        E_D_lp_exp = m_total * (e_lp_exp_in - e_lp_exp_out) - W_lp_exp
        eta_ex_lp_exp = (W_lp_exp / (m_total * (e_lp_exp_in - e_lp_exp_out))
                         if (e_lp_exp_in - e_lp_exp_out) != 0 else None)

        results.components["lp_expander"] = {
            "E_D": float(E_D_lp_exp),
            "eta_exergy": float(eta_ex_lp_exp) if eta_ex_lp_exp is not None else None,
            "E_in": float(m_total * (e_lp_exp_in - e_lp_exp_out)),
            "E_out": float(W_lp_exp),
        }

        # ---- 6c. LP Pump (all fluid) ----------------------------------------
        lp_pump = components["lp_pump"]
        e_lp_pump_in = e_wf(lp_pump["state_in"])
        e_lp_pump_out = e_wf(lp_pump["state_out"])
        W_lp_pump = lp_pump["power"]
        E_D_lp_pump = W_lp_pump - m_total * (e_lp_pump_out - e_lp_pump_in)
        eta_ex_lp_pump = (m_total * (e_lp_pump_out - e_lp_pump_in) / W_lp_pump
                          if W_lp_pump != 0 else None)

        results.components["lp_pump"] = {
            "E_D": float(E_D_lp_pump),
            "eta_exergy": float(eta_ex_lp_pump) if eta_ex_lp_pump is not None else None,
            "E_in": float(W_lp_pump),
            "E_out": float(m_total * (e_lp_pump_out - e_lp_pump_in)),
        }

        # ---- 6d. HP Pump (HP fraction only) ----------------------------------
        hp_pump = components["hp_pump"]
        e_hp_pump_in = e_wf(hp_pump["state_in"])
        e_hp_pump_out = e_wf(hp_pump["state_out"])
        W_hp_pump = hp_pump["power"]
        E_D_hp_pump = W_hp_pump - m_HP * (e_hp_pump_out - e_hp_pump_in)
        eta_ex_hp_pump = (m_HP * (e_hp_pump_out - e_hp_pump_in) / W_hp_pump
                          if W_hp_pump != 0 else None)

        results.components["hp_pump"] = {
            "E_D": float(E_D_hp_pump),
            "eta_exergy": float(eta_ex_hp_pump) if eta_ex_hp_pump is not None else None,
            "E_in": float(W_hp_pump),
            "E_out": float(m_HP * (e_hp_pump_out - e_hp_pump_in)),
        }

        # ---- 6e. HP Evaporator -----------------------------------------------
        hp_evap = components["hp_evaporator"]
        m_hf = hp_evap["hot_side"]["mass_flow"]
        m_hp_cold = hp_evap["cold_side"]["mass_flow"]

        hot_in = hp_evap["hot_side"]["state_in"]
        hot_out = hp_evap["hot_side"]["state_out"]
        if hasattr(hot_in, "T") and hasattr(hot_out, "T") and (hot_in.T < hot_out.T):
            hot_in, hot_out = hot_out, hot_in

        e_hp_evap_hot_in = e_hf(hot_in)
        e_hp_evap_hot_out = e_hf(hot_out)
        e_hp_evap_cold_in = e_wf(hp_evap["cold_side"]["state_in"])
        e_hp_evap_cold_out = e_wf(hp_evap["cold_side"]["state_out"])

        E_D_hp_evap = (m_hf * (e_hp_evap_hot_in - e_hp_evap_hot_out)
                       + m_hp_cold * (e_hp_evap_cold_in - e_hp_evap_cold_out))
        E_given = m_hf * (e_hp_evap_hot_in - e_hp_evap_hot_out)
        E_gained = m_hp_cold * (e_hp_evap_cold_out - e_hp_evap_cold_in)
        eta_ex_hp_evap = E_gained / E_given if E_given != 0 else None

        results.components["hp_evaporator"] = {
            "E_D": float(E_D_hp_evap),
            "eta_exergy": float(eta_ex_hp_evap) if eta_ex_hp_evap is not None else None,
            "E_in": float(E_given),
            "E_out": float(E_gained),
        }

        # ---- 6f. LP Evaporator -----------------------------------------------
        lp_evap = components["lp_evaporator"]
        m_lp_cold = lp_evap["cold_side"]["mass_flow"]

        hot_in = lp_evap["hot_side"]["state_in"]
        hot_out = lp_evap["hot_side"]["state_out"]
        if hasattr(hot_in, "T") and hasattr(hot_out, "T") and (hot_in.T < hot_out.T):
            hot_in, hot_out = hot_out, hot_in

        e_lp_evap_hot_in = e_hf(hot_in)
        e_lp_evap_hot_out = e_hf(hot_out)
        e_lp_evap_cold_in = e_wf(lp_evap["cold_side"]["state_in"])
        e_lp_evap_cold_out = e_wf(lp_evap["cold_side"]["state_out"])

        E_D_lp_evap = (lp_evap["hot_side"]["mass_flow"] * (e_lp_evap_hot_in - e_lp_evap_hot_out)
                       + m_lp_cold * (e_lp_evap_cold_in - e_lp_evap_cold_out))
        E_given = lp_evap["hot_side"]["mass_flow"] * (e_lp_evap_hot_in - e_lp_evap_hot_out)
        E_gained = m_lp_cold * (e_lp_evap_cold_out - e_lp_evap_cold_in)
        eta_ex_lp_evap = E_gained / E_given if E_given != 0 else None

        results.components["lp_evaporator"] = {
            "E_D": float(E_D_lp_evap),
            "eta_exergy": float(eta_ex_lp_evap) if eta_ex_lp_evap is not None else None,
            "E_in": float(E_given),
            "E_out": float(E_gained),
        }

        # ---- 6g. Preheater ---------------------------------------------------
        pre = components["preheater"]
        m_pre_cold = pre["cold_side"]["mass_flow"]

        hot_in = pre["hot_side"]["state_in"]
        hot_out = pre["hot_side"]["state_out"]
        if hasattr(hot_in, "T") and hasattr(hot_out, "T") and (hot_in.T < hot_out.T):
            hot_in, hot_out = hot_out, hot_in

        e_pre_hot_in = e_hf(hot_in)
        e_pre_hot_out = e_hf(hot_out)
        e_pre_cold_in = e_wf(pre["cold_side"]["state_in"])
        e_pre_cold_out = e_wf(pre["cold_side"]["state_out"])

        E_D_pre = (pre["hot_side"]["mass_flow"] * (e_pre_hot_in - e_pre_hot_out)
                   + m_pre_cold * (e_pre_cold_in - e_pre_cold_out))
        E_given = pre["hot_side"]["mass_flow"] * (e_pre_hot_in - e_pre_hot_out)
        E_gained = m_pre_cold * (e_pre_cold_out - e_pre_cold_in)
        eta_ex_pre = E_gained / E_given if E_given != 0 else None

        results.components["preheater"] = {
            "E_D": float(E_D_pre),
            "eta_exergy": float(eta_ex_pre) if eta_ex_pre is not None else None,
            "E_in": float(E_given),
            "E_out": float(E_gained),
        }

        # Guard: if preheater heat duty is negligible, zero out E_D to avoid
        # numerical noise from discretization of near-zero enthalpy differences
        pre_Q = abs(float(pre.get("heat_flow", 0)))
        if pre_Q < 100e3:  # less than 100 kW
            results.components["preheater"]["E_D"] = 0.0
            results.components["preheater"]["eta_exergy"] = None

        # ---- 6h. Mixer (irreversible mixing of HP and LP streams) ------------
        # E_D_mixer = m_HP*e_7 + m_LP*e_4 - m_total*e_8
        m_LP = m_total - m_HP
        state_7 = components["hp_expander"]["state_out"]
        state_4 = components["lp_evaporator"]["cold_side"]["state_out"]
        state_8 = components["lp_expander"]["state_in"]

        e_7 = e_wf(state_7)
        e_4 = e_wf(state_4)
        e_8 = e_wf(state_8)

        E_D_mixer = m_HP * e_7 + m_LP * e_4 - m_total * e_8
        # Mixer has no meaningful exergetic efficiency
        results.components["mixer"] = {
            "E_D": float(E_D_mixer),
            "eta_exergy": None,
            "E_in": float(m_HP * e_7 + m_LP * e_4),
            "E_out": float(m_total * e_8),
        }

        # ---- 6i. Cooler (condenser) ------------------------------------------
        clr = components["cooler"]
        m_cf = clr["cold_side"]["mass_flow"]
        wf_hot_in_state = clr["hot_side"]["state_in"]
        wf_hot_out_state = clr["hot_side"]["state_out"]
        if hasattr(wf_hot_in_state, "T") and hasattr(wf_hot_out_state, "T") and (wf_hot_in_state.T < wf_hot_out_state.T):
            wf_hot_in_state, wf_hot_out_state = wf_hot_out_state, wf_hot_in_state

        cf_in_state = clr["cold_side"]["state_in"]
        cf_out_state = clr["cold_side"]["state_out"]
        if hasattr(cf_in_state, "T") and hasattr(cf_out_state, "T") and (cf_in_state.T > cf_out_state.T):
            cf_in_state, cf_out_state = cf_out_state, cf_in_state

        e_clr_hot_in = e_wf(wf_hot_in_state)
        e_clr_hot_out = e_wf(wf_hot_out_state)
        e_clr_cold_in = e_cf(cf_in_state)
        e_clr_cold_out = e_cf(cf_out_state)

        E_D_clr = (m_total * (e_clr_hot_in - e_clr_hot_out)
                   + m_cf * (e_clr_cold_in - e_clr_cold_out))
        E_given_wf = m_total * (e_clr_hot_in - e_clr_hot_out)
        E_gained_cf = m_cf * (e_clr_cold_out - e_clr_cold_in)
        eta_ex_clr = E_gained_cf / E_given_wf if E_given_wf != 0 else None

        results.components["cooler"] = {
            "E_D": float(E_D_clr),
            "eta_exergy": float(eta_ex_clr) if eta_ex_clr is not None else None,
            "E_in": float(E_given_wf),
            "E_out": float(E_gained_cf),
        }

        # ---- 6j. Recuperator (if present in dual-pressure) -------------------
        recup = components.get("recuperator", None)
        if recup is not None:
            e_rec_hot_in = e_wf(recup["hot_side"]["state_in"])
            e_rec_hot_out = e_wf(recup["hot_side"]["state_out"])
            e_rec_cold_in = e_wf(recup["cold_side"]["state_in"])
            e_rec_cold_out = e_wf(recup["cold_side"]["state_out"])
            m_rec_hot = recup["hot_side"]["mass_flow"]
            m_rec_cold = recup["cold_side"]["mass_flow"]

            E_D_rec = (m_rec_hot * (e_rec_hot_in - e_rec_hot_out)
                       + m_rec_cold * (e_rec_cold_in - e_rec_cold_out))
            E_given = m_rec_hot * (e_rec_hot_in - e_rec_hot_out)
            E_gained = m_rec_cold * (e_rec_cold_out - e_rec_cold_in)
            eta_ex_rec = E_gained / E_given if E_given != 0 else None

            results.components["recuperator"] = {
                "E_D": float(E_D_rec),
                "eta_exergy": float(eta_ex_rec) if eta_ex_rec is not None else None,
                "E_in": float(E_given),
                "E_out": float(E_gained),
            }

    else:
        # ==================================================================
        # SINGLE-PRESSURE ORC (original code)
        # ==================================================================

        # ---- 6a. Expander (turbine) ------------------------------------------
        exp = components["expander"]
        m_wf = exp["mass_flow"]
        e_exp_in = e_wf(exp["state_in"])
        e_exp_out = e_wf(exp["state_out"])
        W_exp = exp["power"]
        E_D_exp = m_wf * (e_exp_in - e_exp_out) - W_exp
        eta_ex_exp = W_exp / (m_wf * (e_exp_in - e_exp_out)) if (e_exp_in - e_exp_out) != 0 else None

        results.components["expander"] = {
            "E_D": float(E_D_exp),
            "eta_exergy": float(eta_ex_exp) if eta_ex_exp is not None else None,
            "E_in": float(m_wf * (e_exp_in - e_exp_out)),
            "E_out": float(W_exp),
        }

        # ---- 6b. Compressor (pump) -------------------------------------------
        comp = components["compressor"]
        e_comp_in = e_wf(comp["state_in"])
        e_comp_out = e_wf(comp["state_out"])
        W_comp = comp["power"]
        E_D_comp = W_comp - m_wf * (e_comp_out - e_comp_in)
        eta_ex_comp = (m_wf * (e_comp_out - e_comp_in) / W_comp
                       if W_comp != 0 else None)

        results.components["compressor"] = {
            "E_D": float(E_D_comp),
            "eta_exergy": float(eta_ex_comp) if eta_ex_comp is not None else None,
            "E_in": float(W_comp),
            "E_out": float(m_wf * (e_comp_out - e_comp_in)),
        }

        # ---- 6c. Heater (heat source HX) ------------------------------------
        htr = components["heater"]
        m_hf = htr["hot_side"]["mass_flow"]
        hot_in_state = htr["hot_side"]["state_in"]
        hot_out_state = htr["hot_side"]["state_out"]
        if hasattr(hot_in_state, "T") and hasattr(hot_out_state, "T") and (hot_in_state.T < hot_out_state.T):
            hot_in_state, hot_out_state = hot_out_state, hot_in_state

        e_htr_hot_in = e_hf(hot_in_state)
        e_htr_hot_out = e_hf(hot_out_state)
        e_htr_cold_in = e_wf(htr["cold_side"]["state_in"])
        e_htr_cold_out = e_wf(htr["cold_side"]["state_out"])

        E_D_htr = (m_hf * (e_htr_hot_in - e_htr_hot_out)
                   + m_wf * (e_htr_cold_in - e_htr_cold_out))
        E_given_hot = m_hf * (e_htr_hot_in - e_htr_hot_out)
        E_gained_cold = m_wf * (e_htr_cold_out - e_htr_cold_in)
        eta_ex_htr = E_gained_cold / E_given_hot if E_given_hot != 0 else None

        results.components["heater"] = {
            "E_D": float(E_D_htr),
            "eta_exergy": float(eta_ex_htr) if eta_ex_htr is not None else None,
            "E_in": float(E_given_hot),
            "E_out": float(E_gained_cold),
        }

        # ---- 6d. Cooler (condenser) ------------------------------------------
        clr = components["cooler"]
        m_wf_cooler = m_wf
        m_cf = clr["cold_side"]["mass_flow"]
        wf_hot_in_state = clr["hot_side"]["state_in"]
        wf_hot_out_state = clr["hot_side"]["state_out"]
        if hasattr(wf_hot_in_state, "T") and hasattr(wf_hot_out_state, "T") and (wf_hot_in_state.T < wf_hot_out_state.T):
            wf_hot_in_state, wf_hot_out_state = wf_hot_out_state, wf_hot_in_state

        cf_in_state = clr["cold_side"]["state_in"]
        cf_out_state = clr["cold_side"]["state_out"]
        if hasattr(cf_in_state, "T") and hasattr(cf_out_state, "T") and (cf_in_state.T > cf_out_state.T):
            cf_in_state, cf_out_state = cf_out_state, cf_in_state

        e_clr_hot_in = e_wf(wf_hot_in_state)
        e_clr_hot_out = e_wf(wf_hot_out_state)
        e_clr_cold_in = e_cf(cf_in_state)
        e_clr_cold_out = e_cf(cf_out_state)

        E_D_clr = (m_wf_cooler * (e_clr_hot_in - e_clr_hot_out)
                   + m_cf * (e_clr_cold_in - e_clr_cold_out))
        E_given_wf = m_wf_cooler * (e_clr_hot_in - e_clr_hot_out)
        E_gained_cf = m_cf * (e_clr_cold_out - e_clr_cold_in)
        eta_ex_clr = E_gained_cf / E_given_wf if E_given_wf != 0 else None

        results.components["cooler"] = {
            "E_D": float(E_D_clr),
            "eta_exergy": float(eta_ex_clr) if eta_ex_clr is not None else None,
            "E_in": float(E_given_wf),
            "E_out": float(E_gained_cf),
        }

        # ---- 6e. Recuperator (if present) ------------------------------------
        recup = components.get("recuperator", None)
        if recup is not None:
            e_rec_hot_in = e_wf(recup["hot_side"]["state_in"])
            e_rec_hot_out = e_wf(recup["hot_side"]["state_out"])
            e_rec_cold_in = e_wf(recup["cold_side"]["state_in"])
            e_rec_cold_out = e_wf(recup["cold_side"]["state_out"])
            m_rec_hot = recup["hot_side"]["mass_flow"]
            m_rec_cold = recup["cold_side"]["mass_flow"]

            E_D_rec = (m_rec_hot * (e_rec_hot_in - e_rec_hot_out)
                       + m_rec_cold * (e_rec_cold_in - e_rec_cold_out))
            E_given = m_rec_hot * (e_rec_hot_in - e_rec_hot_out)
            E_gained = m_rec_cold * (e_rec_cold_out - e_rec_cold_in)
            eta_ex_rec = E_gained / E_given if E_given != 0 else None

            results.components["recuperator"] = {
                "E_D": float(E_D_rec),
                "eta_exergy": float(eta_ex_rec) if eta_ex_rec is not None else None,
                "E_in": float(E_given),
                "E_out": float(E_gained),
            }

    # ---- 6f. Heat-source pump -----------------------------------------------
    hsp = components["heat_source_pump"]
    m_hsp = hsp["mass_flow"]
    e_hsp_in = e_hf(hsp["state_in"])
    e_hsp_out = e_hf(hsp["state_out"])
    W_hsp = hsp["power"]  # From ThermOpt
    E_D_hsp = W_hsp - m_hsp * (e_hsp_out - e_hsp_in)
    eta_ex_hsp = (m_hsp * (e_hsp_out - e_hsp_in) / W_hsp
                  if W_hsp != 0 else None)

    results.components["heat_source_pump"] = {
        "E_D": float(E_D_hsp),
        "eta_exergy": float(eta_ex_hsp) if eta_ex_hsp is not None else None,
        "E_in": float(W_hsp),
        "E_out": float(m_hsp * (e_hsp_out - e_hsp_in)),
    }

    # ---- 6g. Heat-sink pump -------------------------------------------------
    hkp = components["heat_sink_pump"]
    m_hkp = hkp["mass_flow"]
    e_hkp_in = e_cf(hkp["state_in"])
    e_hkp_out = e_cf(hkp["state_out"])
    W_hkp = hkp["power"]  # From ThermOpt
    E_D_hkp = W_hkp - m_hkp * (e_hkp_out - e_hkp_in)
    eta_ex_hkp = (m_hkp * (e_hkp_out - e_hkp_in) / W_hkp
                  if W_hkp != 0 else None)

    results.components["heat_sink_pump"] = {
        "E_D": float(E_D_hkp),
        "eta_exergy": float(eta_ex_hkp) if eta_ex_hkp is not None else None,
        "E_in": float(W_hkp),
        "E_out": float(m_hkp * (e_hkp_out - e_hkp_in)),
    }

    # =========================================================================
    # 7. Cycle-level EXERGY results
    # =========================================================================
    E_D_total = sum(c["E_D"] for c in results.components.values())

    # Exergy fuel = exergy change of the heat source across ALL brine-side HXs
    # For dual pressure: brine enters HP evaporator and exits preheater
    # For single pressure: brine enters and exits the heater
    if is_dual:
        # Brine inlet = HP evaporator hot-side inlet (hottest point)
        brine_in = components["hp_evaporator"]["hot_side"]["state_in"]
        if hasattr(brine_in, "T"):
            brine_out_state = components["preheater"]["hot_side"]["state_out"]
            if brine_in.T < brine_out_state.T:
                brine_in = components["hp_evaporator"]["hot_side"]["state_out"]
                brine_out_state = components["preheater"]["hot_side"]["state_in"]
        else:
            brine_out_state = components["preheater"]["hot_side"]["state_out"]
        
        # Use preheater hot-side outlet as brine exit (coldest brine point)
        E_fuel = m_hf * (e_hf(brine_in) - e_hf(brine_out_state))
    else:
        htr = components["heater"]
        E_fuel = m_hf * (e_hf(htr["hot_side"]["state_in"])
                         - e_hf(htr["hot_side"]["state_out"]))

    # Exergy product = net power output (from ThermOpt)
    E_product = W_net_system

    # Exergy loss = exergy gained by cooling water in condenser
    clr = components["cooler"]
    m_cf = clr["cold_side"]["mass_flow"]
    E_loss_cooler = m_cf * (e_cf(clr["cold_side"]["state_out"])
                            - e_cf(clr["cold_side"]["state_in"]))

    # Internal exergy destruction (excluding auxiliary pumps for balance)
    E_D_internal = sum(
        c["E_D"] for name, c in results.components.items()
        if name not in ("heat_source_pump", "heat_sink_pump")
    )
    
    # Net cycle power (for exergy balance check)
    W_net_cycle = W_expander - W_compressor

    # Exergy balance residual (should be ~0)
    balance_residual = E_fuel - (W_net_cycle + E_D_internal + E_loss_cooler)

    # =========================================================================
    # 8. EXERGY efficiency (the only efficiency calculated here)
    # =========================================================================
    eta_exergy = E_product / E_fuel if E_fuel != 0 else 0.0

    # =========================================================================
    # 9. Store all results
    # =========================================================================
    results.cycle = {
        # ── Imported from ThermOpt (energy-based) ──
        "Q_in": float(Q_in),
        "Q_available": float(Q_available),
        "heat_utilization": float(heat_utilization),
        "eta_cycle": float(eta_cycle),
        "eta_system": float(eta_system),
        
        # ── Power breakdown (from ThermOpt) ──
        "W_expander": float(W_expander),
        "W_compressor": float(W_compressor),
        "W_aux_pumps": float(W_aux_pumps),
        "W_net_system": float(W_net_system),
        "W_net_cycle": float(W_net_cycle),
        
        # ── Calculated here (exergy-based) ──
        "E_fuel": float(E_fuel),
        "E_product": float(E_product),
        "E_D_total": float(E_D_total),
        "E_D_internal": float(E_D_internal),
        "E_loss_cooler": float(E_loss_cooler),
        "eta_exergy": float(eta_exergy),
        
        # ── Balance check ──
        "balance_residual": float(balance_residual),
    }

    return results


# ===========================================================================
#  Parametric sweep: heat-source utilization curve
# ===========================================================================
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

    # ---- Read bounds from YAML if not provided ------------------------------
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

    # ---- Get the optimal design variables from the converged solution -------
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

    # ---- Parametric sweep ---------------------------------------------------
    T_exit_array = np.linspace(T_min + 1.0, T_max - 1.0, n_points)

    eta_system = []
    eta_cycle = []
    W_net = []
    T_valid = []
    T_opt = float(optimal_x_dict.get("heat_source_exit_temperature", (T_min + T_max) / 2))

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

    # ---- Find optimal point in Celsius for marking --------------------------
    T_opt_C = T_opt - 273.15

    # ---- Plot ---------------------------------------------------------------
    fig, ax1 = plt.subplots(figsize=figsize)

    color_sys = "#2196F3"
    color_cyc = "#4CAF50"
    color_pwr = "#FF9800"

    # Left y-axis: efficiencies
    ax1.plot(T_valid, eta_system, "-o", color=color_sys, markersize=4,
             label="System efficiency", linewidth=2)
    ax1.plot(T_valid, eta_cycle, "-s", color=color_cyc, markersize=4,
             label="Cycle efficiency", linewidth=2)
    ax1.set_xlabel("Heat Source Exit Temperature [°C]", fontsize=11)
    ax1.set_ylabel("Efficiency [%]", fontsize=11)
    ax1.tick_params(axis="y")

    # Right y-axis: net power
    ax2 = ax1.twinx()
    ax2.plot(T_valid, W_net, "-^", color=color_pwr, markersize=4,
             label="Net power", linewidth=2)
    ax2.set_ylabel("Net Power [kW]", fontsize=11, color=color_pwr)
    ax2.tick_params(axis="y", labelcolor=color_pwr)

    # Mark the optimal point
    ax1.axvline(T_opt_C, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax1.annotate(f"Optimum\n({T_opt_C:.1f} °C)",
                 xy=(T_opt_C, ax1.get_ylim()[0]),
                 xytext=(T_opt_C + (T_valid[-1] - T_valid[0]) * 0.05,
                         ax1.get_ylim()[0] + (ax1.get_ylim()[1] - ax1.get_ylim()[0]) * 0.15),
                 fontsize=9, color="gray",
                 arrowprops=dict(arrowstyle="->", color="gray", lw=1))

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
