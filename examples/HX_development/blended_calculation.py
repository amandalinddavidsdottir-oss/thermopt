import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

import thermopt as th


def get_htc_1phase(state):
    return 500


def get_htc_2phase(state):
    return 5000


def get_htc_blend(state, Q_start=0.0, Q_end=0.05):
    Q = state["Q"]
    x = (Q - Q_start) / (Q_end - Q_start)  # Normalize Q to [0, 1]
    sigma = th.sigmoid_smoothstep(x)  # 3rd order polynomial
    # sigma = th.sigmoid_smootherstep(x)  # 5th order polynomial
    return get_htc_1phase(state) * (1 - sigma) + get_htc_2phase(state) * sigma


# Create the folder to save figures
th.set_plot_options()


# --------------------------------------------------------------------------- #
# Calculculate states and heat transfer coefficients
# --------------------------------------------------------------------------- #

# Create fluid
fluid = th.Fluid(name="butane", exceptions=True)

# Compute subcritical state in two-phase region
quality = 0.50
pressure = (
    0.25 * fluid.critical_point.p
)  # Access property using object-like notation (".")
state_1 = fluid.get_state(th.PQ_INPUTS, pressure, quality)
print(state_1)

# Compute subcritical state with subcooling
subcooling = 50.0
pressure = 0.95 * state_1["p"]  # Access property using dict-like notation ("[]")
temperature = fluid.get_state(th.PQ_INPUTS, pressure, 0.0).T - subcooling
state_2 = fluid.get_state(th.PT_INPUTS, pressure, temperature)
print(state_2)

# Calculate states
points = 300
enthalpies = np.linspace(state_1.h, state_2.h, points)
pressures = np.linspace(state_1.p, state_2.p, points)
states = []
htc_1phase_list = []
htc_2phase_list = []
htc_blend_list = []
for p, h in zip(pressures, enthalpies):

    # Compute thermodynamic states
    state = fluid.get_state(
        th.HmassP_INPUTS, h, p, generalize_quality=True, supersaturation=True
    )
    states.append(state)

    # Compute heat transfer coefficients
    htc_1phase = get_htc_1phase(state)
    htc_2phase = get_htc_2phase(state)
    htc_blend = get_htc_blend(state, Q_end=0.15)
    htc_1phase_list.append(htc_1phase)
    htc_2phase_list.append(htc_2phase)
    htc_blend_list.append(htc_blend)

# Convert from list of states to dict of arrays
states = th.states_to_dict(states)


# --------------------------------------------------------------------------- #
# Plot thermodynamic states
# --------------------------------------------------------------------------- #

# Plot phase diagram
prop_x = "s"
prop_y = "T"
fig, ax = fluid.plot_phase_diagram(
    prop_x,
    prop_y,
    plot_critical_point=True,
    plot_quality_isolines=True,
)

# Plot states
ax.plot(state_1[prop_x], state_1[prop_y], marker="o", color=th.COLORS_MATLAB[0])
ax.plot(state_2[prop_x], state_2[prop_y], marker="o", color=th.COLORS_MATLAB[0])
ax.plot(states[prop_x], states[prop_y], color=th.COLORS_MATLAB[0])
th.savefig_in_formats(fig, "Ts_diagram", formats=[".png"])
fig.tight_layout(pad=1)


# --------------------------------------------------------------------------- #
# Plot heat transfer coefficient
# --------------------------------------------------------------------------- #

# Plot the heat transfer coefficient
fig2, ax2 = plt.subplots(figsize=(6.0, 4.8))
ax2.set_ylim([0000, 10000])
ax2.set_xlabel("Enthalpy (J/kg)")
ax2.set_ylabel("Heat trransfer coefficient (W/m2/K)")
ax2.plot(states["h"], htc_1phase_list, label="Single-phase")
ax2.plot(states["h"], htc_2phase_list, label="Two-phase")
ax2.plot(states["h"], htc_blend_list, color="k", linestyle="--", label="Blended")

# Plot vertical line at phase change onset
ax2.axvline(
    x=states["h"][np.argmin(np.abs(states["Q"]))],
    color="k",
    linestyle=":",
    label="Q ≈ 0",
)
ax2.legend(loc="upper right", fontsize=11)
fig2.tight_layout(pad=1)
th.savefig_in_formats(fig2, "heat_transfer_coefficient", formats=[".png"])

# Show figures
plt.show()


# # Create entropy range
# s1 = fluid.triple_point_liquid.s
# s2 = fluid.triple_point_vapor.s
# delta_s = s2 - s1
# s_array = np.linspace(s1 + delta_s / 8, s2 + delta_s / 16, 100)

# # Subcritical cases
# p_array = np.asarray([0.5, 0.6, 0.7, 0.8, 0.9, 0.99]) * fluid.critical_point.p
# states = bpy.compute_property_grid(fluid, bpy.PSmass_INPUTS, p_array, s_array, generalize_quality=True)
# colormap = cm.magma(np.linspace(0.1, 0.7, len(p_array)))
# for i in range(states[prop_x].shape[-1]):
#     ax1.plot(
#         states[prop_x][:, i],
#         states[prop_y][:, i],
#         color=colormap[i],
#         label=f"$p/p_{{crit}}={p_array[i]/fluid.critical_point.p:0.2f}$",
#     )
#     ax2.plot(
#         states[prop_x][:, i],
#         states["Q"][:, i],
#         color=colormap[i],
#         label=f"$p/p_{{crit}}={p_array[i]/fluid.critical_point.p:0.2f}$",
#     )

# # Supercritical cases
# p_array = np.asarray([1.01, 1.2, 1.4, 1.6, 1.8, 2.0]) * fluid.critical_point.p
# states = bpy.compute_property_grid(fluid, bpy.PSmass_INPUTS, p_array, s_array, generalize_quality=True)
# colormap = cm.magma(np.linspace(0.7, 0.1, len(p_array)))
# for i in range(states[prop_x].shape[-1]):
#     ax1.plot(
#         states[prop_x][:, i],
#         states[prop_y][:, i],
#         color=colormap[i],
#         linestyle="--",
#         label=f"$p/p_{{crit}}={p_array[i]/fluid.critical_point.p:0.2f}$",
#     )
#     ax2.plot(
#         states[prop_x][:, i],
#         states["Q"][:, i],
#         color=colormap[i],
#         linestyle="--",
#         label=f"$p/p_{{crit}}={p_array[i]/fluid.critical_point.p:0.2f}$",
#     )


# ax2.legend(loc="upper left", fontsize=10)
# fig.tight_layout(pad=2)
# bpy.savefig_in_formats(fig, os.path.join(fig_dir, "generalized_vapor_quality_isobars"))

# # p_array1 = np.asarray(np.linspace(0.5, 0.99, 100)) * fluid.critical_point.p
# # p_array2 = np.asarray(np.linspace(1.01, 2.00, 100)) * fluid.critical_point.p
# # p_array = np.concatenate([p_array1, p_array2])
# # states = bpy.compute_properties_meshgrid(fluid, bpy.PSmass_INPUTS, p_array, s_array)
# # contour = ax1.contour(
# #     states[prop_x],
# #     states[prop_y],
# #     states["Q"],
# #     np.linspace(-1, 2, 31),
# #     linewidths=0.5,
# # )


# # --------------------------------------------------------------------------- #
# # Plot iso-temperature lines
# # --------------------------------------------------------------------------- #

# # Create figure
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
# ax1.set_xlabel("Entropy (J/kg/K)")
# ax1.set_ylabel("Temperature (K)")
# ax2.set_xlabel("Entropy (J/kg/K)")
# ax2.set_ylabel("Vapor quality (-)")
# # ax1.set_ylim([0.5 * fluid.critical_point.T, 1.5 * fluid.critical_point.T])
# prop_x = "s"
# prop_y = "T"

# # Create entropy range
# s1 = fluid.triple_point_liquid.s
# s2 = fluid.triple_point_vapor.s
# delta_s = s2 - s1
# s_array = np.linspace(s1 + delta_s / 8, s2 + delta_s / 16, 100)

#
