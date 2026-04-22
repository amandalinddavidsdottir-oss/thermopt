"""
label_editor.py
===============
Interactive label position editor for the fluid screening plots.

Run this script to drag fluid name labels to wherever you want them.
When you are happy, press  S  to save the positions into
plot_pcond_sweep.py automatically.

Controls
--------
  Left-click + drag  : move a label
  S                  : save all current positions to LABEL_OFFSETS and close
  Q / close window   : quit without saving

Usage
-----
  1. Set FIGURE below to "subcritical", "transcritical", or "combined"
  2. Run:  poetry run python label_editor.py
  3. Drag labels until happy, press S
  4. Re-run plot_pcond_sweep.py to see the final figures

The script updates LABEL_OFFSETS inside plot_pcond_sweep.py in the
same folder.  A backup is written as plot_pcond_sweep.py.bak first.
"""

import re
import sys
import shutil
import warnings
import pandas as pd
import matplotlib

# Backend is set AFTER importing the plot script,
# because the plot script calls matplotlib.use("Agg") and would override it.
# We store the desired backend here and apply it later.
_INTERACTIVE_BACKEND = "TkAgg"
from matplotlib.lines import Line2D
from pathlib import Path

warnings.filterwarnings("ignore")

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
FIGURE = (
    "fig1_subcritical"  # "fig1_subcritical" | "fig2_transcritical" | "fig3_combined"
)
EXCEL_FILE = "pcond_vs_efficiency_simple_isentropic_2026-04-12_11-27-41.xlsx"
PLOT_SCRIPT = "plot_pcond_sweep.py"  # file whose LABEL_OFFSETS gets updated
# ─────────────────────────────────────────────────────────────────────────────

# Pull config from the plot script so we stay in sync
script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(script_dir))

# ── import everything we need from the plot script ───────────────────────────
import importlib, types

spec = importlib.util.spec_from_file_location("plot_mod", script_dir / PLOT_SCRIPT)
mod = types.ModuleType("plot_mod")
mod.__file__ = str(
    script_dir / PLOT_SCRIPT
)  # needed for Path(__file__) inside the script
spec.loader.exec_module(mod)
# The plot script calls matplotlib.use("Agg") — switch back to interactive now
matplotlib.use(_INTERACTIVE_BACKEND, force=True)
import matplotlib.pyplot as plt

print(f"Using backend: {_INTERACTIVE_BACKEND}")

ASTOLFI_CATEGORY = mod.ASTOLFI_CATEGORY
EXCLUDE_FLUIDS = mod.EXCLUDE_FLUIDS
CAT_STYLE = mod.CAT_STYLE
LABEL_SC = mod.LABEL_SC
LABEL_TC = mod.LABEL_TC
LABEL_OFFSETS = dict(mod.LABEL_OFFSETS)  # mutable copy

# ── load data ─────────────────────────────────────────────────────────────────
excel_path = script_dir / EXCEL_FILE
df = pd.read_excel(excel_path)
df = df[df["converged"] == True].copy()
df["eta_pct"] = df["eta_system"] * 100
for fluid, cat in ASTOLFI_CATEGORY.items():
    df.loc[df["fluid"] == fluid, "category"] = cat
df = df[~df["fluid"].isin(EXCLUDE_FLUIDS)].copy()
sc = df[df["cycle_type"] == "subcritical"].copy()
tc = df[df["cycle_type"] == "transcritical"].copy()

if FIGURE == "fig1_subcritical":
    data_rows = sc
    labels_set = LABEL_SC
elif FIGURE == "fig2_transcritical":
    data_rows = tc
    labels_set = LABEL_TC
elif FIGURE == "fig3_combined":
    data_rows = pd.concat([sc, tc])
    labels_set = LABEL_SC | LABEL_TC
else:
    raise ValueError(
        f"Unknown FIGURE: {FIGURE!r}. Choose fig1_subcritical, fig2_transcritical, or fig3_combined."
    )

# ── build figure using the EXACT same functions as the plot script ────────────
# Figure size matches the relevant figure function
_figsize_map = {
    "fig1_subcritical": (12, 6.5),
    "fig2_transcritical": (14, 7),
    "fig3_combined": (18, 10),
}
fig, ax = plt.subplots(figsize=_figsize_map.get(FIGURE, (14, 8)))
fig.patch.set_facecolor("#FDFEFE")

# Step 1: scatter points only (no labels) using the plot script's _scatter
is_combined = FIGURE == "fig3_combined"
for _, row in data_rows.iterrows():
    hollow = is_combined and (row["fluid"] in tc["fluid"].values)
    ms = 80 if is_combined else 60
    ref = (
        mod.REFERENCE_TC
        if (hollow or FIGURE == "fig2_transcritical")
        else mod.REFERENCE_SC
    )
    mod._scatter(
        ax, row, hollow=hollow, ref_fluid=ref, labels_set=None, ms=ms
    )  # labels_set=None → no labels placed

# Step 2: place ALL labels ourselves so the editor holds references to every one
annotations = {}  # fluid → (Annotation, x_data, y_data)
texts_auto = {}  # fluid → (Text,       x_data, y_data)
_fontsize = 9.5 if is_combined else 8.5  # match plot script fontsize defaults

for _, row in data_rows.iterrows():
    fluid = row["fluid"]
    if fluid not in labels_set:
        continue
    x, y = row["p_cond_bar"], row["eta_pct"]
    if fluid in mod.LABEL_OFFSETS:
        dx, dy = mod.LABEL_OFFSETS[fluid]
        ann = ax.annotate(
            fluid,
            xy=(x, y),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=_fontsize,
            color="#2C3E50",
            arrowprops=(
                dict(arrowstyle="-", color="#95A5A6", lw=0.5)
                if (abs(dx) > 15 or abs(dy) > 15)
                else None
            ),
        )
        annotations[fluid] = (ann, x, y)
    else:
        t = ax.text(x, y, fluid, fontsize=_fontsize, color="#2C3E50")
        texts_auto[fluid] = (t, x, y)

# Apply the same _style as the plot script
if FIGURE == "fig3_combined":
    mod._style(ax, "", labelsize=13, ticksize=11, titlesize=14)
else:
    mod._style(ax, "")

plt.tight_layout()

# ── Add legend so it can be dragged ──────────────────────────────────────────
_cat_handles, _cat_labels = [], []
for cat, s in mod.CAT_STYLE.items():
    if cat == "other":
        continue
    if data_rows[data_rows["category"] == cat].empty:
        continue
    _cat_handles.append(
        Line2D(
            [0],
            [0],
            linestyle="None",
            marker=s["marker"],
            markerfacecolor=s["color"],
            markeredgecolor=s["color"],
            markersize=7,
        )
    )
    _cat_labels.append(s["label"])
_legend = ax.legend(
    _cat_handles,
    _cat_labels,
    loc="lower center",
    fontsize=9,
    frameon=True,
    ncol=3,
    edgecolor="#BDC3C7",
    facecolor="white",
)
_legend.set_draggable(True)  # ← drag the legend anywhere you like


# ── draggable label implementation ────────────────────────────────────────────
class DraggableLabels:
    def __init__(self, fig, ax, annotations, texts_auto):
        self.fig = fig
        self.ax = ax
        self.annotations = annotations  # fluid → (ann, x_data, y_data)
        self.texts_auto = texts_auto  # fluid → (txt, x_data, y_data)
        self._dragging = None  # (fluid, kind, obj, x_data, y_data)
        self._all = {}  # fluid → (artist, x_data, y_data, kind)
        for fluid, (ann, xd, yd) in annotations.items():
            self._all[fluid] = (ann, xd, yd, "ann")
        for fluid, (txt, xd, yd) in texts_auto.items():
            self._all[fluid] = (txt, xd, yd, "txt")

        fig.canvas.mpl_connect("button_press_event", self._on_press)
        fig.canvas.mpl_connect("motion_notify_event", self._on_motion)
        fig.canvas.mpl_connect("button_release_event", self._on_release)
        fig.canvas.mpl_connect("key_press_event", self._on_key)

    def _hit(self, event):
        """Return fluid name if the click lands on a label, else None."""
        if event.inaxes != self.ax:
            return None
        renderer = self.fig.canvas.get_renderer()
        for fluid, (artist, *_) in self._all.items():
            bbox = artist.get_window_extent(renderer=renderer)
            if bbox.contains(event.x, event.y):
                return fluid
        return None

    def _on_press(self, event):
        if event.button != 1:
            return
        fluid = self._hit(event)
        if fluid:
            artist, xd, yd, kind = self._all[fluid]
            self._dragging = (fluid, kind, artist, xd, yd, event.x, event.y)

    def _on_motion(self, event):
        if self._dragging is None or event.inaxes != self.ax:
            return
        fluid, kind, artist, xd, yd, x0, y0 = self._dragging
        dx_disp = event.x - x0
        dy_disp = event.y - y0
        # convert display-pixel delta to points (1 pt = dpi/72 px)
        dpi = self.fig.dpi
        dx_pt = dx_disp / dpi * 72
        dy_pt = dy_disp / dpi * 72
        # store running total of movement so far
        self._dragging = (fluid, kind, artist, xd, yd, event.x, event.y)

        if kind == "ann":
            # move the annotation's text position
            off = artist.xyann
            artist.xyann = (off[0] + dx_disp / dpi * 72, off[1] + dy_disp / dpi * 72)
        else:
            # move the plain text object in data coordinates
            # convert display coords back to data coords
            inv = self.ax.transData.inverted()
            cur_disp = self.ax.transData.transform((artist.get_position()))
            new_disp = (cur_disp[0] + dx_disp, cur_disp[1] + dy_disp)
            new_data = inv.transform(new_disp)
            artist.set_position(new_data)

        self.fig.canvas.draw_idle()

    def _on_release(self, event):
        self._dragging = None

    def _on_key(self, event):
        if event.key in ("s", "S"):
            self._save_and_close()
        elif event.key in ("q", "Q"):
            print("Quit without saving.")
            plt.close(self.fig)

    def _save_and_close(self):
        """Compute final offsets in points and write to LABEL_OFFSETS."""
        renderer = self.fig.canvas.get_renderer()
        new_offsets = {}
        dpi = self.fig.dpi

        for fluid, (artist, xd, yd, kind) in self._all.items():
            # data-point position in display coords
            pt_disp = self.ax.transData.transform((xd, yd))

            if kind == "ann":
                # annotation stores offset in points directly
                dx_pt, dy_pt = artist.xyann
            else:
                # plain text: compute offset from data point in display, then to pts
                txt_disp = self.ax.transData.transform(artist.get_position())
                dx_pt = (txt_disp[0] - pt_disp[0]) / dpi * 72
                dy_pt = (txt_disp[1] - pt_disp[1]) / dpi * 72

            new_offsets[fluid] = (round(dx_pt, 1), round(dy_pt, 1))

        _write_offsets(new_offsets)
        plt.close(self.fig)


def _write_offsets(new_offsets):
    """Merge new_offsets into LABEL_OFFSETS in the plot script, and optionally update legend bbox_to_anchor."""
    plot_file = script_dir / PLOT_SCRIPT
    backup = plot_file.with_suffix(".py.bak")
    shutil.copy(plot_file, backup)
    print(f"  Backup saved: {backup.name}")

    src = plot_file.read_text(encoding="utf-8")

    # Build the new LABEL_OFFSETS block
    lines = ["LABEL_OFFSETS = {\n"]
    # existing entries not in new_offsets stay untouched
    merged = dict(mod.LABEL_OFFSETS)  # original from script
    merged.update(new_offsets)  # overwrite with new positions
    for fluid, (dx, dy) in sorted(merged.items()):
        lines.append(f'    "{fluid}":({dx},{dy}),\n')
    lines.append("}\n")
    new_block = "".join(lines)

    # Replace the existing LABEL_OFFSETS block
    pattern = r"LABEL_OFFSETS\s*=\s*\{[^}]*\}"
    new_src, n = re.subn(pattern, new_block.rstrip("\n"), src, flags=re.DOTALL)
    if n == 0:
        print(
            "  ERROR: could not find LABEL_OFFSETS block in script — nothing written."
        )
        return

    plot_file.write_text(new_src, encoding="utf-8")
    print(f"  Saved {len(new_offsets)} label positions → {PLOT_SCRIPT}")
    print("  Re-run plot_pcond_sweep_recup.py to regenerate figures.")


dl = DraggableLabels(fig, ax, annotations, texts_auto)

print(f"Label editor open — {FIGURE}")
print("  Left-click + drag to move labels")
print("  S = save positions and close")
print("  Q = quit without saving")

plt.draw()
plt.pause(0.1)
# Use the Tk mainloop directly — guaranteed to block until window closes
try:
    fig.canvas.manager.window.mainloop()
except Exception:
    plt.show(block=True)
