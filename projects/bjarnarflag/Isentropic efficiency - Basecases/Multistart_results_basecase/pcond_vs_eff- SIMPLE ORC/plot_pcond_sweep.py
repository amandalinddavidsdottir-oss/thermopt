"""
plot_pcond_sweep.py
===================
Reads pcond sweep results from an Excel file in the same folder and
produces three publication-quality figures following the chemical family
classification of Astolfi (2014), Tables 4.3 / 4.4:

  Figure 1 — Subcritical fluids only   (coloured by chemical family)
  Figure 2 — Transcritical fluids only (coloured by chemical family)
  Figure 3 — Combined (all fluids coloured by family; SC=filled, TC=hollow)

Usage:  python plot_pcond_sweep.py

Customisation
-------------
All user-facing settings are in the CONFIGURATION block below:

  EXCEL_FILE       — name of the results file (same folder as script)
  EXCLUDE_FLUIDS   — fluids to hide from all plots
  LABEL_SC         — which fluids get a text label in subcritical plots
  LABEL_TC         — which fluids get a text label in transcritical plots
  LABEL_OFFSETS    — (dx, dy) in points to nudge individual labels
                     positive dx = right, positive dy = up
  OUTPUT_FORMAT    — "png" | "pdf" | "svg"
  DPI              — output resolution (use 300 for print)
"""

import sys
import warnings
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from adjustText import adjust_text
from pathlib import Path

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION — edit this block
# ══════════════════════════════════════════════════════════════════════════════

EXCEL_FILE = "pcond_vs_efficiency_simple_isentropic_2026-04-12_11-27-41.xlsx"
OUT_DIR = Path(__file__).parent
OUTPUT_FORMAT = "png"
DPI = 180

REFERENCE_SC = None  # no reference fluid
REFERENCE_TC = None  # no reference fluid

EXCLUDE_FLUIDS = {
    "Dichloroethane",
    "R40",
    "R11",
    "R12",
    "R22",
    "R113",
    "R114",
    "R141b",
    "R124",
    "R142b",
    "HeavyWater",
    "HydrogenChloride",
    "NitrousOxide",
    "HydrogenSulfide",
    "CarbonylSulfide",
    "EthyleneOxide",
    "R41",
    "DimethylCarbonate",
}

ASTOLFI_CATEGORY = {
    "n-Propane": "alkane",
    "IsoButane": "alkane",
    "n-Butane": "alkane",
    "Neopentane": "alkane",
    "Isopentane": "alkane",
    "Pentane": "alkane",
    "n-Pentane": "alkane",
    "Isohexane": "alkane",
    "n-Hexane": "alkane",
    "Hexane": "alkane",
    "Heptane": "alkane",
    "n-Heptane": "alkane",
    "Octane": "alkane",
    "n-Octane": "alkane",
    "Nonane": "alkane",
    "n-Nonane": "alkane",
    "Decane": "alkane",
    "n-Decane": "alkane",
    "n-Undecane": "alkane",
    "n-Dodecane": "alkane",
    "CycloPentane": "cycloalkane",
    "Cyclopentane": "cycloalkane",
    "CycloHexane": "cycloalkane",
    "CycloPropane": "cycloalkane",
    # Alkenes + aromatics = "Aliene and alkyne" (Astolfi Table 4.3)
    "1-Butene": "aliene_alkyne",
    "IsoButene": "aliene_alkyne",
    "trans-2-Butene": "aliene_alkyne",
    "cis-2-Butene": "aliene_alkyne",
    "Propylene": "aliene_alkyne",
    "Benzene": "aliene_alkyne",
    "Toluene": "aliene_alkyne",
    "o-Xylene": "aliene_alkyne",
    "m-Xylene": "aliene_alkyne",
    "p-Xylene": "aliene_alkyne",
    "EthylBenzene": "aliene_alkyne",
    "DimethylEther": "alcohol_ketone",
    "Acetone": "alcohol_ketone",
    "Methanol": "alcohol_ketone",
    "Ethanol": "alcohol_ketone",
    "R125": "refrigerant",
    "R218": "refrigerant",
    "R143a": "refrigerant",
    "R32": "refrigerant",
    "R1234yf": "refrigerant",
    "R134a": "refrigerant",
    "R227EA": "refrigerant",
    "R161": "refrigerant",
    "R1234zeE": "refrigerant",
    "R152A": "refrigerant",
    "R236FA": "refrigerant",
    "R236EA": "refrigerant",
    "R245fa": "refrigerant",
    "RC318": "refrigerant",
    "R365MFC": "refrigerant",
    "R245ca": "refrigerant",
    "R1233zd(E)": "refrigerant",
    "R1234ze(Z)": "refrigerant",
    "R1234ze(E)": "refrigerant",
    "HFE143m": "refrigerant",
    "R1336mzz(E)": "refrigerant",
    "R1243zf": "refrigerant",
    "Novec649": "refrigerant",
    "MM": "siloxane",
    "MDM": "siloxane",
    "D4": "siloxane",
    "MD2M": "siloxane",
    "D5": "siloxane",
    "MD3M": "siloxane",
    "D6": "siloxane",
    "MD4M": "siloxane",
    "Ammonia": "inorganic",
    "SulfurDioxide": "inorganic",
}

LABEL_SC = {
    "Toluene",
    "Benzene",
    "Methanol",
    "Ethanol",
    "Acetone",
    "CycloHexane",
    "Cyclopentane",
    "o-Xylene",
    "EthylBenzene",
    "m-Xylene",
    "p-Xylene",
    "n-Dodecane",
    "n-Undecane",
    "n-Decane",
    "n-Nonane",
    "n-Octane",
    "n-Heptane",
    "n-Hexane",
    "n-Pentane",
    "Isohexane",
    "MM",
    "MDM",
    "D4",
    "MD2M",
    "MD3M",
    "D5",
    "MD4M",
    "D6",
}
LABEL_TC = {
    "cis-2-Butene",
    "Isopentane",
    "n-Butane",
    "IsoButane",
    "Neopentane",
    "n-Propane",
    "Ammonia",
    "SulfurDioxide",
    "R245fa",
    "R365MFC",
    "R1233zd(E)",
    "R1234ze(Z)",
    "R152A",
    "R236EA",
    "R236FA",
    "R161",
    "R32",
    "R134a",
    "R1234yf",
    "Propylene",
    "RC318",
    "R227EA",
    "R143a",
    "R125",
    "Novec649",
    "DimethylEther",
    "1-Butene",
    "IsoButene",
    "trans-2-Butene",
    "CycloPropane",
    "HFE143m",
    "R1234ze(E)",
    "R1243zf",
    "R1336mzz(E)",
    "R245ca",
}
# Reduced TC label set for the combined figure — only the most distinct fluids
# to avoid overcrowding the 1–20 bar region where all TC fluids cluster
LABEL_COMBINED_TC = {
    "cis-2-Butene",
    "n-Butane",
    "Isopentane",
    "n-Propane",
    "Ammonia",
    "SulfurDioxide",
    "R245fa",
    "R1233zd(E)",
    "R365MFC",
    "R125",
    "R1234yf",
    "Novec649",
    "R32",
    "DimethylEther",
}

LABEL_OFFSETS = {
    "1-Butene": (3.9, 8.6),
    "Acetone": (4.1, 6.9),
    "Ammonia": (-16.1, 8.5),
    "Benzene": (6, 4),
    "CycloHexane": (-29.1, 9.2),
    "CycloPropane": (-27.9, 9.1),
    "Cyclopentane": (-28.0, -15.2),
    "D4": (-4.8, 9.0),
    "D5": (-5.5, -18.1),
    "D6": (-4.8, -15.9),
    "DimethylEther": (6.0, -1.5),
    "Ethanol": (6, -8),
    "EthylBenzene": (4.8, -10.2),
    "HFE143m": (-18.5, 11.2),
    "IsoButane": (-17.7, 7.5),
    "IsoButene": (1.0, -12.3),
    "Isohexane": (-20.9, -15.2),
    "Isopentane": (-19.5, 9.8),
    "MD2M": (-10.6, 9.3),
    "MD3M": (-8.9, 7.8),
    "MD4M": (-13.4, 9.8),
    "MDM": (-7.7, 9.0),
    "MM": (-4.1, 9.0),
    "Methanol": (6, 4),
    "Neopentane": (-20.6, 8.3),
    "Novec649": (-14.2, 9.0),
    "Propylene": (-20.7, 8.5),
    "R1233zd(E)": (-18.5, 9.0),
    "R1234yf": (-17.0, 6.9),
    "R1234ze(E)": (-40.1, 8.6),
    "R1234ze(Z)": (-21.3, 10.5),
    "R1243zf": (-10.6, -15.5),
    "R125": (-14.1, 9.1),
    "R1336mzz(E)": (-27.8, -17.3),
    "R134a": (6.7, -3.0),
    "R143a": (-14.2, 9.1),
    "R152A": (-5.5, -13.3),
    "R161": (5.3, -3.2),
    "R227EA": (-14.9, 9.3),
    "R236EA": (-10.6, 7.8),
    "R236FA": (-5.5, 7.8),
    "R245ca": (-12.0, -15.4),
    "R245fa": (-14.1, -15.8),
    "R32": (-8.4, 8.3),
    "R365MFC": (-20.2, 7.6),
    "RC318": (-12.7, 7.6),
    "SulfurDioxide": (-27.8, 6.6),
    "Toluene": (6, 4),
    "cis-2-Butene": (-24.5, 7.6),
    "m-Xylene": (0.2, 9.0),
    "n-Butane": (-18.4, -16.2),
    "n-Decane": (-17.0, 7.4),
    "n-Dodecane": (-28.2, 7.4),
    "n-Heptane": (-21.8, 7.4),
    "n-Hexane": (-22.8, 6.9),
    "n-Nonane": (-20.9, 7.8),
    "n-Octane": (-18.5, 8.8),
    "n-Pentane": (-32.6, 8.8),
    "n-Propane": (-15.6, -16.6),
    "n-Undecane": (-20.9, 8.8),
    "o-Xylene": (-55, 4),
    "p-Xylene": (12.7, -3.2),
    "trans-2-Butene": (-13.5, 9.1),
}

# Category names follow Astolfi (2014) Tables 4.3/4.4 exactly
CAT_STYLE = {
    "aliene_alkyne": {"color": "#C0392B", "marker": "o", "label": "Aliene and Alkyne"},
    "cycloalkane": {"color": "#8E44AD", "marker": "s", "label": "Cycloalkane"},
    "alkane": {"color": "#2980B9", "marker": "^", "label": "Linear Alkane"},
    "alcohol_ketone": {
        "color": "#D35400",
        "marker": "X",
        "label": "Alcohol and Ketone",
    },
    "refrigerant": {"color": "#27AE60", "marker": "v", "label": "Refrigerant"},
    "siloxane": {"color": "#F39C12", "marker": "D", "label": "Siloxane"},
    "inorganic": {"color": "#7F8C8D", "marker": "*", "label": "Inorganic"},
    "other": {"color": "#BDC3C7", "marker": "o", "label": "Other"},
}
TC_TEAL = "#1ABC9C"

# ══════════════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════


def load_data(excel_path):
    df = pd.read_excel(excel_path)
    df = df[df["converged"] == True].copy()
    df["eta_pct"] = df["eta_system"] * 100
    for fluid, cat in ASTOLFI_CATEGORY.items():
        df.loc[df["fluid"] == fluid, "category"] = cat
    df = df[~df["fluid"].isin(EXCLUDE_FLUIDS)].copy()
    sc = df[df["cycle_type"] == "subcritical"].copy()
    tc = df[df["cycle_type"] == "transcritical"].copy()
    return df, sc, tc


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════


def _scatter(
    ax,
    row,
    color=None,
    ref_fluid=None,
    labels_set=None,
    hollow=False,
    fontsize=8.5,
    ms=60,
    ms_ref=160,
):
    cat = row["category"] if row["category"] in CAT_STYLE else "other"
    style = CAT_STYLE[cat]
    c = color if color else style["color"]
    is_ref = row["fluid"] == ref_fluid
    if hollow and not is_ref:
        ax.scatter(
            row["p_cond_bar"],
            row["eta_pct"],
            facecolors="none",
            edgecolors=c,
            marker=style["marker"],
            s=ms,
            zorder=3,
            linewidths=1.3,
            alpha=0.88,
        )
    else:
        ax.scatter(
            row["p_cond_bar"],
            row["eta_pct"],
            c=c,
            marker=style["marker"],
            s=ms_ref if is_ref else ms,
            zorder=5 if is_ref else 3,
            edgecolors="black" if is_ref else c,
            linewidths=1.8 if is_ref else 0.5,
            alpha=0.88,
        )
    if labels_set and row["fluid"] in labels_set:
        t = ax.text(
            row["p_cond_bar"],
            row["eta_pct"],
            row["fluid"],
            fontsize=fontsize,
            color="#2C3E50",
            fontweight="normal",
        )
        return t
    return None


def _handle(marker, facecolor, edgecolor="none", size=7, edgewidth=0.8, hollow=False):
    """Legend handle — linestyle=None prevents the dash rendering artifact."""
    if hollow:
        return Line2D(
            [0],
            [0],
            linestyle="None",
            marker=marker,
            markerfacecolor="none",
            markeredgecolor=facecolor,
            markeredgewidth=1.3,
            markersize=size,
        )
    return Line2D(
        [0],
        [0],
        linestyle="None",
        marker=marker,
        markerfacecolor=facecolor,
        markeredgecolor=edgecolor,
        markeredgewidth=edgewidth,
        markersize=size,
    )


def _fix_labels(ax, texts):
    """Reposition labels to avoid overlaps, drawing arrows back to data points."""
    if not texts:
        return
    adjust_text(
        texts,
        ax=ax,
        expand=(1.15, 1.3),
        arrowprops=dict(arrowstyle="-", color="#95A5A6", lw=0.5, shrinkA=5),
        force_text=(0.4, 0.55),
        force_points=(0.15, 0.2),
        force_static=(0.08, 0.12),
        min_arrow_len=10,
        lim=800,
    )


def _widen_x(ax, factor=0.18):
    """Extend x-axis by factor on each side in log space, called after adjust_text."""
    import math

    xlo, xhi = ax.get_xlim()
    ax.set_xlim(10 ** (math.log10(xlo) - factor), 10 ** (math.log10(xhi) + factor))


def _style(ax, title, labelsize=13, ticksize=11, titlesize=13):
    ax.set_xscale("log")
    ax.set_xlabel("Condensing pressure [bar]", fontsize=labelsize)
    ax.set_ylabel(r"System efficiency $\eta_{sys}$ [%]", fontsize=labelsize)
    ax.grid(True, which="both", ls=":", lw=0.4, color="#D5D8DC", alpha=0.8)
    ax.tick_params(labelsize=ticksize)
    ax.axvline(1.01325, color="#95A5A6", lw=1.0, ls="--", zorder=1)
    ax.text(
        1.01325 * 1.07,
        ax.get_ylim()[0] + 0.2,
        "1 atm",
        color="#95A5A6",
        fontsize=9,
        va="bottom",
    )


def _ref_hline(ax, data, fluid):
    eta = data.loc[data["fluid"] == fluid, "eta_pct"]
    if not eta.empty:
        ax.axhline(eta.values[0], color="#C0392B", lw=0.8, ls=":", alpha=0.5)


def _save(fig, name):
    p = OUT_DIR / f"{name}.{OUTPUT_FORMAT}"
    fig.savefig(p, dpi=DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"  v {p.name}")
    plt.close(fig)


SUBTITLE = (
    r"Simple ORC — Bjarnarflag geothermal  "
    r"($\dot{W}_{net}$=45 MW,  $T_{source}$=207 C,  "
    r"$T_{sink}$=22 C,  $\eta_{is}$=0.90)"
)

# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 1 — SUBCRITICAL
# ══════════════════════════════════════════════════════════════════════════════


def fig_subcritical(sc):
    fig, ax = plt.subplots(figsize=(12, 6.5))
    fig.patch.set_facecolor("#FDFEFE")
    texts = []
    for _, r in sc.iterrows():
        t = _scatter(ax, r, ref_fluid=REFERENCE_SC, labels_set=LABEL_SC)
        if t is not None:
            texts.append(t)
    _style(ax, "")
    _fix_labels(ax, texts)
    _widen_x(ax)
    handles, labels = [], []
    for cat, s in CAT_STYLE.items():
        if cat == "other":
            continue
        if sc[sc["category"] == cat].empty:
            continue
        handles.append(_handle(s["marker"], s["color"]))
        labels.append(s["label"])
    fig.tight_layout()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        fontsize=11,
        frameon=True,
        bbox_to_anchor=(0.5, 0.01),
        edgecolor="#BDC3C7",
        facecolor="white",
        handletextpad=0.4,
        columnspacing=1.0,
    )
    fig.subplots_adjust(bottom=0.18)
    _save(fig, "fig1_subcritical")


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 2 — TRANSCRITICAL
# ══════════════════════════════════════════════════════════════════════════════


def fig_transcritical(tc):
    fig, ax = plt.subplots(figsize=(14, 7))
    fig.patch.set_facecolor("#FDFEFE")
    texts = []
    for _, r in tc.iterrows():
        t = _scatter(ax, r, ref_fluid=REFERENCE_TC, labels_set=LABEL_TC)
        if t is not None:
            texts.append(t)
    _style(ax, "")
    _fix_labels(ax, texts)
    _widen_x(ax)
    tc_cats = [
        "aliene_alkyne",
        "alkane",
        "cycloalkane",
        "alcohol_ketone",
        "refrigerant",
        "inorganic",
    ]
    handles, labels = [], []
    for c in tc_cats:
        if tc[tc["category"] == c].empty:
            continue
        handles.append(_handle(CAT_STYLE[c]["marker"], CAT_STYLE[c]["color"]))
        labels.append(CAT_STYLE[c]["label"])
    fig.tight_layout()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        fontsize=11,
        frameon=True,
        bbox_to_anchor=(0.5, 0.01),
        edgecolor="#BDC3C7",
        facecolor="white",
        handletextpad=0.4,
        columnspacing=1.0,
    )
    fig.subplots_adjust(bottom=0.18)
    _save(fig, "fig2_transcritical")


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 3 — COMBINED
# ══════════════════════════════════════════════════════════════════════════════


def fig_combined(sc, tc):
    fig, ax = plt.subplots(figsize=(18, 10))
    fig.patch.set_facecolor("#FDFEFE")
    # TC: hollow markers coloured by chemical family (same palette as SC)
    texts = []
    for _, r in tc.iterrows():
        t = _scatter(
            ax,
            r,
            hollow=True,
            ref_fluid=REFERENCE_TC,
            labels_set=LABEL_TC,
            fontsize=8.5,
            ms=80,
        )
        if t is not None:
            texts.append(t)
    # SC: filled markers coloured by chemical family
    for _, r in sc.iterrows():
        t = _scatter(
            ax, r, ref_fluid=REFERENCE_SC, labels_set=LABEL_SC, fontsize=8.5, ms=80
        )
        if t is not None:
            texts.append(t)
    _style(ax, "", labelsize=13, ticksize=11, titlesize=14)
    _fix_labels(ax, texts)
    _widen_x(ax)

    # Legend: one filled + one hollow entry per family present in SC or TC
    all_cats = [
        "aliene_alkyne",
        "cycloalkane",
        "alkane",
        "alcohol_ketone",
        "siloxane",
        "refrigerant",
        "inorganic",
    ]
    handles, labels = [], []
    for c in all_cats:
        in_sc = not sc[sc["category"] == c].empty
        in_tc = not tc[tc["category"] == c].empty
        if not (in_sc or in_tc):
            continue
        s = CAT_STYLE[c]
        if in_sc:
            handles.append(_handle(s["marker"], s["color"]))
            labels.append(s["label"] + " — SC")
        if in_tc:
            handles.append(_handle(s["marker"], s["color"], hollow=True))
            labels.append(s["label"] + " — TC")

    fig.tight_layout()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=5,
        fontsize=11,
        frameon=True,
        bbox_to_anchor=(0.5, 0.01),
        edgecolor="#BDC3C7",
        facecolor="white",
        handletextpad=0.5,
        columnspacing=1.2,
    )
    fig.subplots_adjust(bottom=0.16)
    # Save combined at higher DPI for readability
    p_out = OUT_DIR / f"fig3_combined.{OUTPUT_FORMAT}"
    fig.savefig(p_out, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"  v {p_out.name}")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Resolve strips the \\?\  long-path prefix VS Code adds
    script_dir = Path(__file__).resolve().parent
    excel_path = script_dir / EXCEL_FILE
    # If not found, also try .xlsx / .xls variants automatically
    if not excel_path.exists():
        for ext in [".xlsx", ".xls", ".XLS", ".XLSX"]:
            candidate = script_dir / (Path(EXCEL_FILE).stem + ext)
            if candidate.exists():
                excel_path = candidate
                break
    if not excel_path.exists():
        sys.exit(
            f"ERROR: '{EXCEL_FILE}' not found in {script_dir}\n"
            f"Files in that folder:\n"
            + "\n".join(
                f"  {f.name}"
                for f in script_dir.iterdir()
                if f.suffix.lower() in (".xlsx", ".xls")
            )
        )
    print(f"Reading: {excel_path.name}")
    df, sc, tc = load_data(excel_path)
    print(f"  Subcritical : {len(sc)} fluids")
    print(f"  Transcritical: {len(tc)} fluids\n")
    fig_subcritical(sc)
    fig_transcritical(tc)
    fig_combined(sc, tc)
    print("\nDone.")
