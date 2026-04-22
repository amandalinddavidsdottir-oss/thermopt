import numpy as np
import matplotlib.pyplot as plt
from CoolProp.CoolProp import PropsSI

# ── Fluid ──────────────────────────────────────────────────────────────────
fluid = "Toluene"

# ── Get critical and triple point ─────────────────────────────────────────
T_crit = PropsSI("Tcrit", fluid)
T_min = PropsSI("Tmin", fluid) + 1

# ── Get critical point entropy ────────────────────────────────────────────
s_crit = PropsSI("S", "T", T_crit, "Q", 0.5, fluid) / 1000

# ── Build saturation dome ─────────────────────────────────────────────────
T_range = np.linspace(T_min, T_crit - 0.1, 500)

s_liq = []
s_vap = []

for T in T_range:
    try:
        s_liq.append(PropsSI("S", "T", T, "Q", 0, fluid) / 1000)
        s_vap.append(PropsSI("S", "T", T, "Q", 1, fluid) / 1000)
    except:
        pass

T_plot = T_range[: len(s_liq)] - 273.15

# Add critical point to close the dome at the top
s_liq.append(s_crit)
s_vap.append(s_crit)
T_plot = np.append(T_plot, T_crit - 273.15)

# ── Plot ──────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))

ax.plot(s_liq, T_plot, color="black", linewidth=2)
ax.plot(s_vap, T_plot, color="black", linewidth=2)

ax.set_xlabel("Entropy, s", fontsize=17)
ax.set_ylabel("Temperature, T", fontsize=17)
ax.grid(False)

ax.set_xticks([])
ax.set_yticks([])

plt.tight_layout()

# ── Save ──────────────────────────────────────────────────────────────────
plt.savefig(
    r"C:\Users\asdis\OneDrive\Documents\Amanda\Master Thesis\PYTHON_Code\ORC-project\Poetry - Working code\thermopt_repo\projects\bjarnarflag\Plot figures for theory\toluene_dome.svg",
    format="svg",
    bbox_inches="tight",
)
print("Saved!")

plt.show()
