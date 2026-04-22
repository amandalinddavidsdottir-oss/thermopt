import numpy as np
import matplotlib.pyplot as plt

import thermopt as th


# -------------------------------
# Muley and Manglik correlation
# -------------------------------


def get_muley_manglik_nusselt(Re, Pr, mu_ratio, beta, phi):
    """
    Compute the Nusselt number in a plate heat exchanger using the Muley and Manglik correlation.

    Parameters
    ----------
    Re : float or ndarray
        Reynolds number based on equivalent diameter [-]
    Pr : float
        Prandtl number [-]
    mu_ratio : float
        Viscosity ratio μ/μ_w, where μ is bulk viscosity and μ_w is wall viscosity [-]
    beta : float
        Chevron angle in degrees [deg]
    phi : float
        Surface enlargement factor due to corrugations [-]

    Returns
    -------
    Nu : float or ndarray
        Nusselt number [-]
    """
    term1 = 0.2668 - 0.006967 * (90 - beta) + 7.244e-5 * (90 - beta) ** 2
    term2 = 20.78 - 50.94 * phi + 41.16 * phi**2 - 10.51 * phi**3
    term3 = Re ** (0.728 + 0.0543 * np.sin(np.pi * (90 - beta) / 45 + 3.7))
    term4 = Pr ** (1 / 3) * (mu_ratio) ** 0.14
    return term1 * term2 * term3 * term4


def get_muley_manglik_friction_factor(Re, beta, phi):
    """
    Compute the Darcy friction factor in a plate heat exchanger using the Muley and Manglik correlation.

    The Darcy friction factor (also known as the Moody friction factor) is defined from the pressure drop
    as:

        Δp = f * (L/D_h) * (rho * u² / 2)

    where Δp is the pressure drop, L is the flow length, D_h is the hydraulic diameter, rho is the fluid density,
    and u is the bulk velocity.

    Note
    ----
    This is the **Darcy friction factor**, which is four times larger than the **Fanning friction factor**:

        f_Darcy = 4 * f_Fanning

    Parameters
    ----------
    Re : float or ndarray
        Reynolds number based on hydraulic diameter [-]
    beta : float
        Chevron angle in degrees [deg]
    phi : float
        Surface enlargement factor due to corrugations [-]

    Returns
    -------
    f : float or ndarray
        Darcy friction factor [-]
    """
    term1 = 2.917 - 0.1277 * (90 - beta) + 2.016e-3 * (90 - beta) ** 2
    term2 = 5.474 - 19.02 * phi + 18.93 * phi**2 - 5.341 * phi**3
    term3 = Re ** (-0.2 - 0.0577 * np.sin(np.pi * (90 - beta) / 45 + 2.1))
    return term1 * term2 * term3


def check_validity(Re, beta, phi):
    if Re < 1e3:
        raise ValueError("Reynolds number must be ≥ 10³")
    if not (30 <= beta <= 60):
        raise ValueError("Chevron angle β must be between 30 and 60 degrees")
    if not (1.0 <= phi <= 1.5):
        raise ValueError("Enlargement factor φ must be between 1.0 and 1.5")


# ----------------------------------
# Amalfi-Vakili-Thome correlation
# ----------------------------------


# Define baseline correlation functions from Amalfi et al. (2016)
def get_amalfi_friction_factor(We_m, Bd, rho_star, beta_star):
    """
    Friction factor correlation from Amalfi et al. (2016) based on nondimensional groups.
    """
    C = 2.125 * beta_star**9.993 + 0.955
    f_tp = C * 15.698 * We_m**-0.475 * Bd**0.255 * rho_star**-0.571
    return f_tp


def get_amalfi_nusselt(We_m, Bd, rho_star, beta_star, Re_lo, Re_v, Bo):
    """
    Two-phase Nusselt number based on Amalfi et al. (2016), distinguishing between
    macro and microscale flow using Bond number.

    Parameters
    ----------
    We_m : float
        Homogeneous Weber number [-]
    Bd : float
        Bond number [-]
    rho_star : float
        Liquid to vapor density ratio [-]
    beta_star : float
        Chevron angle normalized by 60 degrees [-]
    Re_lo : float
        Reynolds number with liquid-only properties [-]
    Re_v : float
        Reynolds number with vapor-only properties [-]
    Bo : float
        Boiling number [-]

    Returns
    -------
    Nu_tp : float
        Two-phase Nusselt number [-]
    """
    # TODO: Add smooth blending between the microscale and macroscale functions
    if Bd < 4:
        # Microscale correlation
        Nu_tp = 982 * beta_star**1.101 * We_m**0.315 * Bo**0.320 * rho_star**-0.224
    else:
        # Macroscale correlation
        Nu_tp = (
            18.495
            * beta_star**0.248
            * Re_v**0.135
            * Re_lo**0.351
            * Bd**0.235
            * Bo**0.198
            * rho_star**-0.223
        )
    return Nu_tp


def get_amalfi_nondimensional_groups(geometry, mass_flow, p, h, Fluid):
    """
    Compute nondimensional groups for Amalfi et al. (2016) correlations based on pressure, enthalpy,
    quality, geometry, and mass flow rate.

    Parameters
    ----------
    p : float
        Pressure [Pa]
    h : float
        Specific enthalpy [J/kg]
    x : float
        Vapor quality (mass fraction of vapor) [-]
    geometry : dict
        Dictionary with geometry keys: 'b', 'pitch', 'beta' (deg)
    mass_flow : float
        Mass flow rate [kg/s]
    Fluid : object
        CoolProp AbstractState object (or equivalent) with .get_state(...)
    th : module
        Thermodynamic input key (e.g., CoolProp wrapper with PHmass_inputs)

    Returns
    -------
    dict
        Dictionary of nondimensional groups:
        - We_m, Bd, rho_star, beta_star, Re_lo, Re_v, Bo
    """

    # TODO: 22.06.2025 to be continued. Skeleton generated with AI

    # --- Geometry ---
    b = geometry["b"]
    pitch = geometry["pitch"]
    beta = geometry["beta"]
    d_h = 2 * b
    A_flow = b * geometry["w_p"]  # single channel flow area

    # --- Thermodynamic states ---
    # Saturated liquid and vapor
    Fluid.update(th.PQ_INPUTS, p, 0)
    rho_l = Fluid.rhomass()
    mu_l = Fluid.viscosity()
    h_l = Fluid.hmass()

    Fluid.update(th.PQ_INPUTS, p, 1)
    rho_v = Fluid.rhomass()
    mu_v = Fluid.viscosity()
    h_v = Fluid.hmass()

    sigma = Fluid.surface_tension()  # [N/m]

    # Mixture (inlet) properties
    Fluid.update(th.PHmass_INPUTS, p, h)
    rho_m = Fluid.rhomass()
    k_l = Fluid.conductivity()  # Use liquid for Pr
    cp_l = Fluid.cpmass()
    Pr_l = mu_l * cp_l / k_l

    # --- Mean velocity ---
    G = mass_flow / A_flow  # mass flux [kg/(m²·s)]
    u_m = G / rho_m

    # --- Dimensionless groups ---
    We_m = G**2 * d_h / (rho_m * sigma)
    Bd = (rho_l - rho_v) * 9.81 * d_h**2 / sigma
    rho_star = rho_l / rho_v
    beta_star = beta / 60
    Re_lo = G * d_h / mu_l
    Re_v = G * x * d_h / mu_v
    Bo = G * (1 - x) * (h_v - h_l) / (sigma * Pr_l)

    return {
        "We_m": We_m,
        "Bd": Bd,
        "rho_star": rho_star,
        "beta_star": beta_star,
        "Re_lo": Re_lo,
        "Re_v": Re_v,
        "Bo": Bo,
        "Pr_l": Pr_l,
        "G": G,
        "d_h": d_h,
        "rho_m": rho_m,
    }


# -------------------------------
# Plate heat exchanger
# -------------------------------


def compute_sinusoidal_enlargement_factor(amplitude, pitch):
    """
    Approximate the surface enlargement factor Φ for sinusoidal corrugations.

    This formula is from Holger Martin (1996), "A theoretical approach to predict the performance of
    chevron-type plate heat exchangers" [Chemical Engineering and Processing, Vol. 35, pp. 301-310].
    It estimates the surface area enlargement factor Φ as the ratio of the actual (developed) area
    to the projected flat area of a sinusoidally corrugated plate.

    Using the dimensionless corrugation parameter:
        X = 2π * amplitude / pitch

    The developed surface area ratio is approximated by:
        Φ(X) ≈ (1/6) * [1 + √(1 + X²) + 4√(1 + X² / 2)]

    This is a three-point quadrature-based approximation of the surface integral over one wave period.

    Examples:
        - For X = 1 (Λ/â = 2π), Φ ≈ 1.22
        - For Φ = 2, Λ/â ≈ 2.46

    Parameters
    ----------
    amplitude : float
        Corrugation amplitude â (m)
    pitch : float
        Corrugation wavelength Λ (m)

    Returns
    -------
    phi : float
        Estimated surface enlargement factor Φ [-]
    """
    X = 2 * np.pi * amplitude / pitch
    return (1 / 6) * (1 + np.sqrt(1 + X**2) + 4 * np.sqrt(1 + X**2 / 2))


def evaluate_plate_hex(state, geometry, mass_flow):
    """
    Evaluate performanceof a single-channel plate heat exchanger using
    the Muley and Manglik correlations for Nusselt number and friction factor.

    Parameters
    ----------
    state : thermopt.State
        Thermodynamic state at the channel inlet. Must provide viscosity [Pa·s],
        thermal conductivity [W/(m·K)], density [kg/m³], and specific heat capacity [J/(kg·K)].

    geometry : dict
        Dictionary containing the geometric configuration of the plate heat exchanger.

        Required keys:
        - "L_p" : float
            Plate length in the flow direction [m]. Typical values: 0.5-3.5 m.
        - "w_p" : float
            Plate width perpendicular to flow [m]. Typical values: 0.1-1.0 m.
        - "t" : float
            Plate thickness [m]. Typical values: 0.0004-0.0012 m.
        - "b" : float
            Corrugation amplitude, i.e., half the channel height [m]. Typical values: 0.001-0.005 m.
        - "pitch" : float
            Corrugation wavelength (peak-to-peak distance) [m]. Typical values: 0.005-0.02 m.
        - "beta" : float
            Chevron (corrugation inclination) angle [degrees]. Typical values: 30-60°.
            TODO we should decide the convention for bete. Against the flow or aligned with the flow?

        Optional keys:
        - "D_port" : float
            Port diameter [m] (default: 0.03 m).
        - "phi" : float
            Surface enlargement factor [-]. If not provided, it is estimated from pitch and amplitude
            using a sinusoidal corrugation model.

    mass_flow : float
        Mass flow rate through the exchanger [kg/s].

    Returns
    -------
    dict
        Dictionary containing computed performance parameters:
        - "Re"        : Reynolds number [-]
        - "Pr"        : Prandtl number [-]
        - "Nu"        : Nusselt number [-]
        - "h"         : Heat transfer coefficient [W/(m²·K)]
        - "f"         : Fanning friction factor [-]
        - "delta_p"   : Pressure drop [Pa]
        - "phi"       : Surface enlargement factor [-]
        - "v_avg"     : Average velocity [m/s]
        - "N_channels": Number of flow channels [-]
        - "A_total_flow": Total flow area [m²]

    Notes
    -----
    - The equivalent diameter is defined as two times the amplitude.
    - The total flow area assumes N_channels = w_p / pitch.
    - Assumes the same viscosity at the wall and bulk fluid (μ/μ_w = 1).
    - Correlations valid for:
        Re ≥ 1000, 30° ≤ β ≤ 60°, 1.0 ≤ φ ≤ 1.5.
    - The enlargement factor φ is calculated using:
        φ ≈ (1/6) · [1 + √(1 + X²) + 4√(1 + X²/2)], with X = 2*pi*a / pitch.

    Example
    -------
    >>> geometry = {
    >>>     "L_p": 1.0,
    >>>     "w_p": 0.5,
    >>>     "t": 0.0005,
    >>>     "amplitude": 0.002,
    >>>     "pitch": 0.01,
    >>>     "beta": 45
    >>> }
    >>> results = evaluate_plate_hex(state, geometry, m_dot=0.5)
    >>> print(results["Nu"], results["h"])
    """
    # Unpack required geometry
    L_p = geometry["L_p"]
    w_p = geometry["w_p"]
    t = geometry["t"]
    b = geometry["b"]  # [m]
    pitch = geometry["pitch"]
    beta = geometry["beta"]  # [deg]
    # D_port = geometry.get("D_port", 0.03)  # optional
    phi = geometry.get("phi", compute_sinusoidal_enlargement_factor(b, pitch))

    # Fluid properties
    mu = state.viscosity
    rho = state.density
    k = state.conductivity
    cp = state.cp
    mu_ratio = 1.0  # assume same temp wall for now

    # Derived geometry
    N_channels = w_p / pitch
    A_flow_channel = b * w_p  # cross-section area per channel
    A_total_flow = A_flow_channel * N_channels
    d_e = 2 * b  # equivalent diameter (2*b)

    # Flow parameters
    v_avg = mass_flow / (rho * A_total_flow)
    Re = rho * v_avg * d_e / mu
    Pr = mu * cp / k

    Nu = get_muley_manglik_nusselt(Re, Pr, mu_ratio, beta, phi)
    h = Nu * k / d_e

    f = get_muley_manglik_friction_factor(Re, beta, phi)
    delta_p = 4 * f * 0.5 * rho * v_avg**2 * (L_p / d_e)

    return {
        "Re": Re,
        "Pr": Pr,
        "Nu": Nu,
        "h": h,
        "f": f,
        "delta_p": delta_p,
        "phi": phi,
        "v_avg": v_avg,
        "N_channels": N_channels,
        "A_total_flow": A_total_flow,
    }


fluid = th.Fluid(name="water")
pressure = 2e5
temperature = 293.15
mass_flow = 1.5
state = fluid.get_state(th.PT_INPUTS, pressure, temperature)

geometry = {
    "L_p": 0.25,  # Plate length in flow direction [m]
    "w_p": 0.1,  # Plate width perpendicular to flow [m]
    "t": 0.0005,  # Plate thickness [m]
    "b": 0.002,  # Corrugation amplitude (half channel height) [m]
    "pitch": 2 * np.pi * 0.002,  # Corrugation wavelength (pitch) [m]
    "beta": 45,  # Chevron (corrugation) angle [deg]
    # Optional:
    # "D_port": 0.03, # Port diameter [m]
    # "phi": 1.2      # Surface enlargement factor [-]
}

results = evaluate_plate_hex(state, geometry, mass_flow)

for k, v in results.items():
    print(f"{k}: {v:.3f}")


# -------------------------------
# Amplitude to pitch sensitivity
# -------------------------------

pitch = 1.00
X_values = [1.0, 1.5, 2.0, 2.5]
x = np.linspace(0, 3 * pitch, 100)
fig, ax = plt.subplots(figsize=(6, 3))
colors = plt.get_cmap("magma")(np.linspace(0.8, 0.2, len(X_values)))
for i, X in enumerate(X_values):
    b = X * pitch / 2 / np.pi
    y = b * np.cos(2 * np.pi * x / pitch)
    ax.plot(x, y, label=f"X = {X}", color=colors[i])

ax.set_aspect("equal")
ax.set_ylabel("Channel amplitude")
ax.set_xlabel("Number of pitches")
ax.legend(loc="right")
ax.set_title("Corrugated channel shape")


# -------------------------------
# Sensitivity to chevron angle β
# -------------------------------
Re_values = np.logspace(np.log10(600), np.log10(20000), 200)
Pr = 5.0
mu_ratio = 1.0
phi = 1.25
betas = [30, 40, 50, 60]

nu_beta = {}
f_beta = {}
for beta in betas:
    nu_beta[beta] = get_muley_manglik_nusselt(Re_values, Pr, mu_ratio, beta, phi)
    f_beta[beta] = get_muley_manglik_friction_factor(Re_values, beta, phi)

fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 6), sharex=True)
colors = plt.get_cmap("magma")(np.linspace(0.8, 0.2, len(betas)))

for i, beta in enumerate(betas):
    ax1.plot(Re_values, nu_beta[beta], label=f"β = {beta}°", color=colors[i])
ax1.set_ylabel("Nusselt number (Nu)")
ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.set_ylim([10, 1000])
ax1.grid(True)
ax1.legend()
ax1.set_title("Sensitivity to chevron angle")

for i, beta in enumerate(betas):
    ax2.plot(Re_values, f_beta[beta], label=f"β = {beta}°", color=colors[i])
ax2.set_xlabel("Reynolds number (Re)")
ax2.set_ylabel("Friction factor (f)")
ax2.set_xscale("log")
ax2.set_ylim([0, 0.8])
ax2.grid(True)

plt.tight_layout(pad=1)


# -------------------------------
# Sensitivity to enlargement factor φ
# -------------------------------
Re_values = np.logspace(np.log10(600), np.log10(20000), 200)
Pr = 5.0
mu_ratio = 1.0
beta = 45
phis = [1.1, 1.2, 1.3, 1.4]

nu_phi = {}
f_phi = {}
for phi in phis:
    nu_phi[phi] = get_muley_manglik_nusselt(Re_values, Pr, mu_ratio, beta, phi)
    f_phi[phi] = get_muley_manglik_friction_factor(Re_values, beta, phi)

fig2, (ax3, ax4) = plt.subplots(2, 1, figsize=(6, 6), sharex=True)
colors = plt.get_cmap("magma")(np.linspace(0.8, 0.2, len(phis)))

for i, phi in enumerate(phis):
    ax3.plot(Re_values, nu_phi[phi], label=f"φ = {phi}", color=colors[i])
ax3.set_ylabel("Nusselt number (Nu)")
ax3.set_xscale("log")
ax3.set_yscale("log")
ax3.set_ylim([10, 1000])
ax3.grid(True)
ax3.legend()
ax3.set_title("Sensitivity to enlargement factor")

for i, phi in enumerate(phis):
    ax4.plot(Re_values, f_phi[phi], label=f"φ = {phi}", color=colors[i])
ax4.set_xlabel("Reynolds number (Re)")
ax4.set_ylabel("Friction factor (f)")
ax4.set_xscale("log")
ax4.set_ylim([0, 0.8])
ax4.grid(True)

plt.tight_layout(pad=1)


# Show figures
plt.show()
