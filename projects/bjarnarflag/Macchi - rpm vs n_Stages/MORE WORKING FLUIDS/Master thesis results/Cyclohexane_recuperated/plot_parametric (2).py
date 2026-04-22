"""
Parametric sweep plots for the Macchi-Astolfi expander study.

Usage
-----
Drop this script in the folder that contains the parametric .xlsx files produced
by the sweep runner. The files should be named with the pattern:

    parametric_nstages_rpm_<Fluid>_recup_basecase_optimized_Macchi*.xlsx

Run:
    python plot_parametric.py

Outputs (written to ./figures/):
    fig1_eta_vs_nstages_<Fluid>.png         — headline multi-line plot
    fig2_constraints_<Fluid>.png            — binding-constraint map
    fig3_per_stage_best_<Fluid>.png         — per-stage Ns/Vr/η at best config
    fig4_cross_fluid_comparison.png         — bar chart across all fluids (if >1 fluid)
    table_best_configs.csv                  — one-line summary per fluid

Constants
---------
Macchi-Astolfi geometric envelope used throughout:
    Ns:   0.045 <= Ns <= 0.195
    Vr:   1.2   <= Vr <= 5.0
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────────────────────
# Constants (Macchi-Astolfi envelope)
# ──────────────────────────────────────────────────────────────────────────────
NS_LO, NS_HI = 0.045, 0.195
VR_LO, VR_HI = 1.2, 5.0

# Consistent color cycle for the 5 (RPM, n_turb) combinations
# (shared across figures so a reader can track one combo across plots)
COMBO_COLORS = {
    (1000, 1): "#1f77b4",  # blue
    (1500, 2): "#ff7f0e",  # orange
    (1500, 4): "#2ca02c",  # green
    (3000, 4): "#d62728",  # red
    (3000, 6): "#9467bd",  # purple
}
COMBO_MARKERS = {
    (1000, 1): "o",
    (1500, 2): "s",
    (1500, 4): "^",
    (3000, 4): "D",
    (3000, 6): "v",
}

FLUID_COLORS = {
    "Toluene":    "#6a5acd",  # slate blue
    "Cyclohexane":"#d62728",  # red
    "Isopentane": "#2ca02c",  # green
}


# ──────────────────────────────────────────────────────────────────────────────
# File discovery
# ──────────────────────────────────────────────────────────────────────────────
def discover_files(folder: Path) -> dict[str, Path]:
    """Find parametric .xlsx files in folder, keyed by fluid name.
    If multiple files for the same fluid exist, pick the most recent by mtime."""
    pattern = re.compile(
        r"parametric_nstages_rpm_(?P<fluid>[A-Za-z]+)_recup_basecase_optimized_Macchi.*\.xlsx$"
    )
    candidates: dict[str, list[Path]] = {}
    for f in folder.glob("*.xlsx"):
        m = pattern.match(f.name)
        if m:
            candidates.setdefault(m.group("fluid"), []).append(f)
    # Pick newest for each fluid
    return {
        fluid: max(files, key=lambda p: p.stat().st_mtime)
        for fluid, files in candidates.items()
    }


def load_sweep(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, engine="openpyxl")
    # Keep only converged rows (all should be, but be safe)
    df = df[df["converged"] == True].copy()  # noqa: E712
    df["combo"] = list(zip(df["RPM"], df["n_turbines_parallel"]))
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Figure 1 — η_sys vs. n_stages, one line per (RPM, n_turb)
# ──────────────────────────────────────────────────────────────────────────────
def plot_fig1_eta_vs_nstages(df: pd.DataFrame, fluid: str, outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.8))

    combos = sorted(df["combo"].unique())
    for combo in combos:
        sub = df[df["combo"] == combo].sort_values("n_stages")
        rpm, nt = combo
        color = COMBO_COLORS.get(combo, "gray")
        marker = COMBO_MARKERS.get(combo, "o")
        ax.plot(
            sub["n_stages"],
            sub["eta_system"] * 100,
            marker=marker, color=color, linewidth=1.8, markersize=7,
            label=f"RPM={rpm}, $n_{{turb}}$={nt}",
        )

    # Mark the overall best point
    best = df.loc[df["eta_system"].idxmax()]
    ax.scatter(
        [best["n_stages"]], [best["eta_system"] * 100],
        s=220, facecolor="none", edgecolor="black", linewidth=1.5,
        zorder=10,
    )
    ax.annotate(
        f"best: {best['eta_system']*100:.2f}%\n"
        f"(n={int(best['n_stages'])}, "
        f"RPM={int(best['RPM'])}, "
        f"$n_{{turb}}$={int(best['n_turbines_parallel'])})",
        xy=(best["n_stages"], best["eta_system"] * 100),
        xytext=(best["n_stages"] - 1.8, best["eta_system"] * 100 + 0.15),
        fontsize=8.5, ha="left", va="bottom",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor="gray", alpha=0.95),
        arrowprops=dict(arrowstyle="->", color="black", lw=0.8),
    )

    ax.set_xlabel("Number of stages $n_s$", fontsize=11)
    ax.set_ylabel(r"System efficiency $\eta_\mathrm{sys}$ (%)", fontsize=11)
    ax.set_title(f"{fluid}: system efficiency vs. expander configuration", fontsize=11.5)
    ax.set_xticks(sorted(df["n_stages"].unique()))
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.95)

    # Headroom for annotation box above peak
    y_min, y_max = ax.get_ylim()
    ax.set_ylim(y_min, y_max + 0.8)

    fig.tight_layout()
    out = outdir / f"fig1_eta_vs_nstages_{fluid}.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"  → {out.name}")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 2 — Binding constraint map
# ──────────────────────────────────────────────────────────────────────────────
def get_stage_cols(n: int, field: str) -> list[str]:
    """Return column names s1_<field>, s2_<field>, ..., sn_<field>."""
    return [f"s{i}_{field}" for i in range(1, n + 1)]


def get_all_bindings(row: pd.Series, tol: float = 1e-3) -> list[str]:
    """For one run, return a list of ALL constraints binding to within `tol`.
    Each returned code is one of 'Ns_lo', 'Ns_hi', 'Vr_lo', 'Vr_hi'.
    Empty list means no Macchi envelope constraint is binding
    (interior optimum — something else is limiting the cycle)."""
    n = int(row["n_stages"])
    ns_vals = [row.get(c, np.nan) for c in get_stage_cols(n, "Ns")]
    vr_vals = [row.get(c, np.nan) for c in get_stage_cols(n, "Vr")]
    ns_vals = [v for v in ns_vals if pd.notna(v)]
    vr_vals = [v for v in vr_vals if pd.notna(v)]

    codes: list[str] = []
    if ns_vals:
        if min(ns_vals) - NS_LO < tol:
            codes.append("Ns_lo")
        if NS_HI - max(ns_vals) < tol:
            codes.append("Ns_hi")
    if vr_vals:
        if min(vr_vals) - VR_LO < tol:
            codes.append("Vr_lo")
        if VR_HI - max(vr_vals) < tol:
            codes.append("Vr_hi")
    return codes


def plot_fig2_constraints(df: pd.DataFrame, fluid: str, outdir: Path) -> None:
    df = df.copy()
    df["bindings"] = df.apply(get_all_bindings, axis=1)

    n_stages_list = sorted(df["n_stages"].unique())
    combos = sorted(df["combo"].unique())

    label_map = {
        "Ns_hi": r"$N_s\!=\!0.195$",
        "Ns_lo": r"$N_s\!=\!0.045$",
        "Vr_hi": r"$V_r\!=\!5.0$",
        "Vr_lo": r"$V_r\!=\!1.2$",
    }
    code_color = {
        "Ns_hi": "#d62728",   # red
        "Ns_lo": "#ff7f0e",   # orange
        "Vr_hi": "#1f77b4",   # blue
        "Vr_lo": "#9467bd",   # purple
    }
    # Color for cells where no Macchi envelope constraint is binding
    interior_color = "#e8e8e8"  # light gray

    fig, ax = plt.subplots(figsize=(8.5, 4.8))

    for i, combo in enumerate(combos):
        for j, n in enumerate(n_stages_list):
            row = df[(df["combo"] == combo) & (df["n_stages"] == n)]
            if row.empty:
                continue
            codes = row["bindings"].iloc[0]
            eta = row["eta_system"].iloc[0] * 100

            x0, y0 = j - 0.48, i - 0.48
            w, h = 0.96, 0.96

            # Background fill: single color if 0 or 1 binding; split if 2+
            if not codes:
                # Interior optimum — no Macchi envelope constraint binding
                ax.add_patch(plt.Rectangle(
                    (x0, y0), w, h,
                    facecolor=interior_color, edgecolor="white", linewidth=1.5,
                ))
            elif len(codes) == 1:
                ax.add_patch(plt.Rectangle(
                    (x0, y0), w, h,
                    facecolor=code_color[codes[0]], edgecolor="white", linewidth=1.5,
                ))
            else:
                # Multiple constraints binding: split cell into vertical stripes
                stripe_w = w / len(codes)
                for k, code in enumerate(codes):
                    ax.add_patch(plt.Rectangle(
                        (x0 + k * stripe_w, y0), stripe_w, h,
                        facecolor=code_color[code],
                        edgecolor="white", linewidth=0.5,
                    ))
                # Outer border to make split cells look like one unit
                ax.add_patch(plt.Rectangle(
                    (x0, y0), w, h, facecolor="none",
                    edgecolor="white", linewidth=1.5,
                ))

            # Text: η at top
            ax.text(j, i + 0.22, f"η={eta:.2f}%",
                    ha="center", va="center", fontsize=9, color="black",
                    fontweight="bold")
            # Text: constraint labels stacked underneath
            if not codes:
                ax.text(j, i - 0.14, "interior\noptimum",
                        ha="center", va="center", fontsize=7.5, color="#505050",
                        style="italic")
            else:
                label_txt = "\n".join(label_map[c] for c in codes)
                ax.text(j, i - 0.14, label_txt,
                        ha="center", va="center", fontsize=7.8, color="black")

    ax.set_xlim(-0.5, len(n_stages_list) - 0.5)
    ax.set_ylim(-0.5, len(combos) - 0.5)
    ax.set_xticks(range(len(n_stages_list)))
    ax.set_xticklabels([f"$n_s$={n}" for n in n_stages_list])
    ax.set_yticks(range(len(combos)))
    ax.set_yticklabels([f"RPM={r}, $n_t$={t}" for (r, t) in combos])
    ax.set_title(f"{fluid}: binding Macchi–Astolfi constraints at each converged run",
                 fontsize=11)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(left=False, bottom=False)

    # Legend
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=code_color[c], edgecolor="white")
        for c in ["Ns_hi", "Ns_lo", "Vr_hi", "Vr_lo"]
    ]
    labels = [label_map[c] for c in ["Ns_hi", "Ns_lo", "Vr_hi", "Vr_lo"]]
    handles.append(plt.Rectangle((0, 0), 1, 1, facecolor=interior_color,
                                 edgecolor="white"))
    labels.append("no Macchi constraint\nbinding (interior)")
    ax.legend(handles, labels, loc="center left", bbox_to_anchor=(1.02, 0.5),
              fontsize=8.5, frameon=False, title="Binding constraint",
              title_fontsize=9.5)

    # Footnote explaining stripes
    fig.text(0.5, -0.02,
             "Split cells indicate two constraints binding simultaneously "
             "(typically $N_s$ lower bound on early stages + $V_r$ upper bound on later stages).",
             ha="center", fontsize=8.5, style="italic", color="#404040")

    fig.tight_layout()
    out = outdir / f"fig2_constraints_{fluid}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out.name}")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 3 — Per-stage Ns, Vr, η for the best configuration
# ──────────────────────────────────────────────────────────────────────────────
def plot_fig3_per_stage_best(df: pd.DataFrame, fluid: str, outdir: Path) -> None:
    best = df.loc[df["eta_system"].idxmax()]
    n = int(best["n_stages"])
    stages = list(range(1, n + 1))

    ns = [best[f"s{i}_Ns"] for i in stages]
    vr = [best[f"s{i}_Vr"] for i in stages]
    eta = [best[f"s{i}_eta"] * 100 for i in stages]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(11, 4))

    # Panel 1: Ns per stage with envelope
    ax1.axhspan(NS_LO, NS_HI, color="#2ca02c", alpha=0.15, label="allowed")
    ax1.axhline(NS_LO, color="#2ca02c", linestyle="--", linewidth=0.8)
    ax1.axhline(NS_HI, color="#2ca02c", linestyle="--", linewidth=0.8)
    ax1.plot(stages, ns, marker="o", color="#d62728", linewidth=2, markersize=8)
    ax1.set_xlabel("Stage number")
    ax1.set_ylabel(r"Specific speed $N_s$")
    ax1.set_title(r"Per-stage $N_s$")
    ax1.set_xticks(stages)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper left", fontsize=9)
    y_lo = min(NS_LO * 0.8, min(ns) * 0.9)
    y_hi = max(NS_HI * 1.1, max(ns) * 1.1)
    ax1.set_ylim(y_lo, y_hi)

    # Panel 2: Vr per stage with envelope
    ax2.axhspan(VR_LO, VR_HI, color="#2ca02c", alpha=0.15, label="allowed")
    ax2.axhline(VR_LO, color="#2ca02c", linestyle="--", linewidth=0.8)
    ax2.axhline(VR_HI, color="#2ca02c", linestyle="--", linewidth=0.8)
    ax2.plot(stages, vr, marker="s", color="#1f77b4", linewidth=2, markersize=8)
    ax2.set_xlabel("Stage number")
    ax2.set_ylabel(r"Volumetric expansion ratio $V_r$")
    ax2.set_title(r"Per-stage $V_r$")
    ax2.set_xticks(stages)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper left", fontsize=9)
    y_lo = min(VR_LO * 0.8, min(vr) * 0.9)
    y_hi = max(VR_HI * 1.1, max(vr) * 1.1)
    ax2.set_ylim(y_lo, y_hi)

    # Panel 3: η per stage
    ax3.plot(stages, eta, marker="D", color="#6a5acd", linewidth=2, markersize=8)
    ax3.set_xlabel("Stage number")
    ax3.set_ylabel(r"Stage efficiency $\eta$ (%)")
    ax3.set_title(r"Per-stage $\eta$")
    ax3.set_xticks(stages)
    ax3.grid(True, alpha=0.3)
    ax3.axhline(best["eta_turbine_overall"] * 100, color="black",
                linestyle=":", linewidth=1,
                label=f"overall: {best['eta_turbine_overall']*100:.2f}%")
    ax3.legend(loc="lower right", fontsize=9)

    title = (f"{fluid}: best configuration  (n_s={n}, "
             f"RPM={int(best['RPM'])}, $n_{{turb}}$={int(best['n_turbines_parallel'])}, "
             f"η_sys={best['eta_system']*100:.2f}%)")
    fig.suptitle(title, fontsize=12, y=1.02)

    fig.tight_layout()
    out = outdir / f"fig3_per_stage_best_{fluid}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out.name}")


def plot_fig5_ns_per_stage(df: pd.DataFrame, fluid: str, outdir: Path) -> None:
    """Per-stage specific speed Ns for every run, faceted by n_stages.
    One subplot per n_stages value; each subplot has one line per (RPM, n_turb) combo."""
    n_stages_list = sorted(df["n_stages"].unique())
    ncols = len(n_stages_list)

    fig, axes = plt.subplots(1, ncols, figsize=(3.2 * ncols, 4.2),
                             sharey=True)
    if ncols == 1:
        axes = [axes]

    for ax, n in zip(axes, n_stages_list):
        # Envelope shading
        ax.axhspan(NS_LO, NS_HI, color="#2ca02c", alpha=0.12, zorder=0)
        ax.axhline(NS_LO, color="#2ca02c", linestyle="--", linewidth=0.8, zorder=1)
        ax.axhline(NS_HI, color="#2ca02c", linestyle="--", linewidth=0.8, zorder=1)

        sub = df[df["n_stages"] == n]
        for _, row in sub.iterrows():
            stages = list(range(1, n + 1))
            ns_vals = [row[f"s{i}_Ns"] for i in stages]
            combo = (int(row["RPM"]), int(row["n_turbines_parallel"]))
            ax.plot(
                stages, ns_vals,
                marker=COMBO_MARKERS.get(combo, "o"),
                color=COMBO_COLORS.get(combo, "gray"),
                linewidth=1.6, markersize=7, zorder=3,
                label=f"RPM={combo[0]}, $n_t$={combo[1]}",
            )

        ax.set_title(f"$n_s$ = {n}", fontsize=10.5)
        ax.set_xlabel("Stage number", fontsize=10)
        ax.set_xticks(range(1, n + 1))
        ax.grid(True, alpha=0.3, zorder=0)

    axes[0].set_ylabel(r"Specific speed $N_s$", fontsize=11)

    # Single legend on the right
    handles, labels = axes[0].get_legend_handles_labels()
    # Add an envelope marker to the legend
    env_patch = plt.Rectangle((0, 0), 1, 1, facecolor="#2ca02c", alpha=0.25,
                              edgecolor="#2ca02c", linestyle="--")
    handles.append(env_patch)
    labels.append("allowed envelope")
    fig.legend(handles, labels, loc="center right", fontsize=9, frameon=False,
               bbox_to_anchor=(1.0, 0.5))

    fig.suptitle(f"{fluid}: per-stage specific speed $N_s$ for every converged run",
                 fontsize=11.5, y=1.0)
    fig.tight_layout(rect=[0, 0, 0.86, 0.97])
    out = outdir / f"fig5_Ns_per_stage_{fluid}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out.name}")


def plot_fig6_vr_per_stage(df: pd.DataFrame, fluid: str, outdir: Path) -> None:
    """Per-stage volumetric expansion ratio Vr for every run, faceted by n_stages."""
    n_stages_list = sorted(df["n_stages"].unique())
    ncols = len(n_stages_list)

    fig, axes = plt.subplots(1, ncols, figsize=(3.2 * ncols, 4.2),
                             sharey=True)
    if ncols == 1:
        axes = [axes]

    for ax, n in zip(axes, n_stages_list):
        ax.axhspan(VR_LO, VR_HI, color="#2ca02c", alpha=0.12, zorder=0)
        ax.axhline(VR_LO, color="#2ca02c", linestyle="--", linewidth=0.8, zorder=1)
        ax.axhline(VR_HI, color="#2ca02c", linestyle="--", linewidth=0.8, zorder=1)

        sub = df[df["n_stages"] == n]
        for _, row in sub.iterrows():
            stages = list(range(1, n + 1))
            vr_vals = [row[f"s{i}_Vr"] for i in stages]
            combo = (int(row["RPM"]), int(row["n_turbines_parallel"]))
            ax.plot(
                stages, vr_vals,
                marker=COMBO_MARKERS.get(combo, "o"),
                color=COMBO_COLORS.get(combo, "gray"),
                linewidth=1.6, markersize=7, zorder=3,
                label=f"RPM={combo[0]}, $n_t$={combo[1]}",
            )

        ax.set_title(f"$n_s$ = {n}", fontsize=10.5)
        ax.set_xlabel("Stage number", fontsize=10)
        ax.set_xticks(range(1, n + 1))
        ax.grid(True, alpha=0.3, zorder=0)

    axes[0].set_ylabel(r"Volumetric expansion ratio $V_r$", fontsize=11)

    handles, labels = axes[0].get_legend_handles_labels()
    env_patch = plt.Rectangle((0, 0), 1, 1, facecolor="#2ca02c", alpha=0.25,
                              edgecolor="#2ca02c", linestyle="--")
    handles.append(env_patch)
    labels.append("allowed envelope")
    fig.legend(handles, labels, loc="center right", fontsize=9, frameon=False,
               bbox_to_anchor=(1.0, 0.5))

    fig.suptitle(f"{fluid}: per-stage volumetric expansion ratio $V_r$ for every converged run",
                 fontsize=11.5, y=1.0)
    fig.tight_layout(rect=[0, 0, 0.86, 0.97])
    out = outdir / f"fig6_Vr_per_stage_{fluid}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out.name}")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 4 — Cross-fluid comparison
# ──────────────────────────────────────────────────────────────────────────────
def plot_fig4_cross_fluid(best_per_fluid: dict[str, pd.Series], outdir: Path) -> None:
    if len(best_per_fluid) < 2:
        return

    fluids = list(best_per_fluid.keys())
    etas = [best_per_fluid[f]["eta_system"] * 100 for f in fluids]
    labels = [
        f"{f}\n(n={int(best_per_fluid[f]['n_stages'])}, "
        f"RPM={int(best_per_fluid[f]['RPM'])}, "
        f"$n_t$={int(best_per_fluid[f]['n_turbines_parallel'])})"
        for f in fluids
    ]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = [FLUID_COLORS.get(f, "gray") for f in fluids]
    bars = ax.bar(range(len(fluids)), etas, color=colors, edgecolor="black",
                  linewidth=0.8, width=0.62)

    for bar, eta in zip(bars, etas):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
                f"{eta:.2f}%", ha="center", va="bottom",
                fontsize=10.5, fontweight="bold")

    ax.set_xticks(range(len(fluids)))
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_ylabel(r"Best system efficiency $\eta_\mathrm{sys,max}$ (%)", fontsize=11)
    ax.set_title("Cross-fluid comparison: best expander configuration per fluid",
                 fontsize=11.5)
    ax.set_ylim(0, max(etas) * 1.15)
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    out = outdir / "fig4_cross_fluid_comparison.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"  → {out.name}")


# ──────────────────────────────────────────────────────────────────────────────
# Summary table
# ──────────────────────────────────────────────────────────────────────────────
def write_summary_table(best_per_fluid: dict[str, pd.Series], outdir: Path) -> None:
    rows = []
    for fluid, best in best_per_fluid.items():
        rows.append({
            "fluid": fluid,
            "n_stages": int(best["n_stages"]),
            "RPM": int(best["RPM"]),
            "n_turbines_parallel": int(best["n_turbines_parallel"]),
            "eta_system_%": round(best["eta_system"] * 100, 3),
            "eta_turbine_overall_%": round(best["eta_turbine_overall"] * 100, 3),
            "expander_p_in_bar": round(best["expander_p_in_bar"], 4),
            "expander_p_out_bar": round(best["expander_p_out_bar"], 4),
            "m_dot_wf_kg_s": round(best["m_dot_wf_kg_s"], 2),
        })
    df_summary = pd.DataFrame(rows)
    out = outdir / "table_best_configs.csv"
    df_summary.to_csv(out, index=False)
    print(f"  → {out.name}")
    print()
    print(df_summary.to_string(index=False))


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    here = Path(__file__).parent
    outdir = here / "figures"
    outdir.mkdir(exist_ok=True)

    print("Parametric Macchi-Astolfi sweep — figure generation")
    print("=" * 60)
    print(f"Working folder: {here}")
    print(f"Output folder:  {outdir}")
    print()

    files = discover_files(here)
    if not files:
        print("No parametric sweep files found.")
        print("Expected filename pattern:")
        print("  parametric_nstages_rpm_<Fluid>_recup_basecase_optimized_Macchi*.xlsx")
        return

    print(f"Found {len(files)} fluid(s):")
    for fluid, path in files.items():
        print(f"  {fluid}: {path.name}")
    print()

    best_per_fluid: dict[str, pd.Series] = {}

    for fluid, path in files.items():
        print(f"[{fluid}]")
        df = load_sweep(path)
        if df.empty:
            print("  No converged runs — skipping.")
            continue

        plot_fig1_eta_vs_nstages(df, fluid, outdir)
        plot_fig2_constraints(df, fluid, outdir)
        plot_fig3_per_stage_best(df, fluid, outdir)
        plot_fig5_ns_per_stage(df, fluid, outdir)
        plot_fig6_vr_per_stage(df, fluid, outdir)

        best_per_fluid[fluid] = df.loc[df["eta_system"].idxmax()]
        print()

    if best_per_fluid:
        print("[Cross-fluid comparison]")
        plot_fig4_cross_fluid(best_per_fluid, outdir)
        write_summary_table(best_per_fluid, outdir)

    print()
    print("Done.")


if __name__ == "__main__":
    main()
