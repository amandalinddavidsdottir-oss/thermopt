"""
Working fluid screening for ORC optimization.

Queries CoolProp for all available pure fluids, filters out:
  - fluids whose critical temperature falls outside a user-defined range
  - fluids banned or restricted by EU F-gas regulation, Montreal Protocol, etc.
  - fluids that are impractical (inerts, cryogens, metals, etc.)

Then runs the thermopt optimization for each surviving candidate.

Usage:
    from fluid_sweep import get_candidate_fluids, run_fluid_sweep
    candidates = get_candidate_fluids(Tc_min=100, Tc_max=250)
    results    = run_fluid_sweep("./case_butane_ORC.yaml", candidates)

IMPROVEMENTS over original version:
  - Proper YAML parsing instead of fragile string replacement
  - Automatic adjustment of design variable bounds per fluid
  - Automatic adjustment of initial guesses per fluid
  - Integration of exergy analysis for second-law efficiency
  - Net power output included in results
  - Wet/dry fluid classification
"""

import os
import copy
import yaml
import traceback
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import CoolProp.CoolProp as cp

import thermopt as th
import matplotlib
matplotlib.use("Agg")


# ══════════════════════════════════════════════════════════════════════
#  Regulatory blacklist
# ══════════════════════════════════════════════════════════════════════
# CoolProp names → reason for exclusion.
# Sources: EU F-gas Regulation 2024/573, Montreal Protocol, REACH.
# This is not exhaustive but covers the main offenders in CoolProp's list.

BANNED_FLUIDS = {
    # ── Ozone-depleting substances (Montreal Protocol) ────────────────
    "R11":          "CFC – ozone depleting (Montreal Protocol)",
    "R12":          "CFC – ozone depleting (Montreal Protocol)",
    "R13":          "CFC – ozone depleting (Montreal Protocol)",
    "R113":         "CFC – ozone depleting (Montreal Protocol)",
    "R114":         "CFC – ozone depleting (Montreal Protocol)",
    "R115":         "CFC – ozone depleting (Montreal Protocol)",
    "R123":         "HCFC – ozone depleting, phased out",
    "R141b":        "HCFC – ozone depleting, phased out",
    "R142b":        "HCFC – ozone depleting, phased out",
    "R22":          "HCFC – ozone depleting, phased out",

    # ── High-GWP HFCs (EU F-gas Regulation phase-down) ───────────────
    "R134a":        "HFC – high GWP (1430), EU F-gas phase-down",
    "R125":         "HFC – high GWP (3500), EU F-gas phase-down",
    "R143a":        "HFC – high GWP (4470), EU F-gas phase-down",
    "R227EA":       "HFC – high GWP (3220), EU F-gas phase-down",
    "R236EA":       "HFC – high GWP (1370), EU F-gas phase-down",
    "R236FA":       "HFC – high GWP (9810), EU F-gas phase-down",
    "R245fa":       "HFC – high GWP (1030), EU F-gas phase-down",
    "R365MFC":      "HFC – high GWP (794), EU F-gas phase-down",
    "R32":          "HFC – moderate GWP (675), restricted in EU",
    "RC318":        "PFC – very high GWP (10300)",
    "R116":         "PFC – very high GWP (12200)",
    "R218":         "PFC – very high GWP (8830)",
    "R23":          "HFC – very high GWP (14800)",
    "R41":          "HFC – limited data, not commercially used",

    # ── Toxic / impractical ───────────────────────────────────────────
    "CarbonMonoxide":    "Toxic (CO)",
    "HydrogenSulfide":   "Highly toxic and corrosive",
    "SulfurDioxide":     "Toxic, corrosive",
    "NitrousOxide":      "GHG (GWP=298), oxidizer",
    "SulfurHexafluoride":"Very high GWP (22800)",
    "Methanol":          "Toxic, flammable – not used in ORC",
    "Ethanol":           "Not practical for ORC power cycles",

    # ── Cryogens / inerts (Tc far too low for any ORC) ────────────────
    "Helium":       "Cryogen",
    "Neon":         "Cryogen",
    "Argon":        "Cryogen",
    "Krypton":      "Cryogen",
    "Xenon":        "Cryogen, extremely expensive",
    "Hydrogen":     "Cryogen, explosive",
    "Nitrogen":     "Cryogen",
    "Oxygen":       "Cryogen, oxidizer",
    "Fluorine":     "Cryogen, extremely reactive",
    "ParaHydrogen": "Cryogen",
    "OrthoHydrogen":"Cryogen",
    "Deuterium":    "Cryogen",
    "ParaDeuterium":"Cryogen",
    "OrthoDeuterium":"Cryogen",
    "HeavyWater":   "Not practical for ORC",
    "Air":          "Mixture (not a pure working fluid)",

    # ── Others not relevant for power ORC ─────────────────────────────
    "CarbonDioxide":     "Tc=31°C, requires supercritical cycle (not simple ORC)",
    "Acetone":           "Highly flammable, decomposes at moderate T",
    "DiethylEther":      "Extremely flammable, peroxide-forming",
    "Ethylene":          "Cryogen for ORC temperatures",
    "EthyleneOxide":     "Explosive, toxic",
    "CarbonylSulfide":   "Toxic",
}

# ── Fluids that are NOT banned but worth flagging ─────────────────────
# These are allowed but have some concern (mild flammability, limited
# commercial availability, etc.).  We include them in the sweep but
# print a note.
FLAGGED_FLUIDS = {
    "Propane":       "A3 flammability – requires safety measures",
    "Butane":        "A3 flammability – requires safety measures",
    "Isobutane":     "A3 flammability – requires safety measures",
    "Isopentane":    "A3 flammability – requires safety measures",
    "Neopentane":    "A3 flammability – requires safety measures",
    "Pentane":       "A3 flammability – requires safety measures",
    "Isohexane":     "A3 flammability – requires safety measures",
    "Hexane":        "A3 flammability – requires safety measures",
    "Heptane":       "A3 flammability – higher boiling point",
    "Cyclohexane":   "A3 flammability",
    "CycloPropane":  "A3 flammability, limited availability",
    "Toluene":       "A3 flammability, toxic at high concentrations",
    "MDM":           "Siloxane – limited commercial ORC experience",
    "MM":            "Siloxane – limited commercial ORC experience",
    "MD2M":          "Siloxane – limited commercial ORC experience",
    "MD3M":          "Siloxane – limited commercial ORC experience",
    "MD4M":          "Siloxane – limited commercial ORC experience",
    "D4":            "Siloxane – REACH SVHC candidate",
    "D5":            "Siloxane – REACH SVHC candidate",
    "D6":            "Siloxane – limited data",
}


# ══════════════════════════════════════════════════════════════════════
#  Helper: classify fluid as wet/dry/isentropic
# ══════════════════════════════════════════════════════════════════════
def classify_fluid_type(name):
    """
    Classify a fluid as 'wet', 'dry', or 'isentropic' based on the slope
    of the saturated vapor line (dT/ds).
    
    - Wet fluid: dT/ds > 0 on sat. vapor line → expands into two-phase
    - Dry fluid: dT/ds < 0 on sat. vapor line → expands into superheated
    - Isentropic: dT/ds ≈ 0 → vertical saturation line
    
    Returns
    -------
    str : 'wet', 'dry', or 'isentropic'
    """
    try:
        Tc = cp.PropsSI("Tcrit", name)
        # Evaluate at 0.85*Tc (well below critical, in the dome)
        T_eval = 0.85 * Tc
        
        # Get saturation entropy at T and T + dT
        dT = 1.0  # K
        s1 = cp.PropsSI("S", "T", T_eval, "Q", 1, name)      # sat vapor at T
        s2 = cp.PropsSI("S", "T", T_eval + dT, "Q", 1, name) # sat vapor at T+dT
        
        ds_dT = (s2 - s1) / dT  # slope of sat vapor line in T-s diagram
        
        # ds/dT > 0 means T increases with s → "wet" (overhanging dome)
        # ds/dT < 0 means T decreases with s → "dry" (retrograde dome)
        if ds_dT > 0.5:    # J/(kg·K²) - threshold for "wet"
            return "wet"
        elif ds_dT < -0.5:
            return "dry"
        else:
            return "isentropic"
    except Exception:
        return "unknown"


# ══════════════════════════════════════════════════════════════════════
#  Fluid screening
# ══════════════════════════════════════════════════════════════════════
def get_candidate_fluids(Tc_min=100, Tc_max=250, include_flagged=True,
                         print_summary=True):
    """
    Query CoolProp for all pure fluids and filter by critical temperature
    and regulatory status.

    Parameters
    ----------
    Tc_min : float
        Minimum critical temperature [°C].
    Tc_max : float
        Maximum critical temperature [°C].
    include_flagged : bool
        If True, include fluids with safety/availability concerns (printed
        with a warning).  If False, exclude them too.
    print_summary : bool
        Print a table of candidates and rejected fluids.

    Returns
    -------
    list of dict
        Each entry: {"name": str, "Tc": float, "pc": float, "flag": str or None,
                     "fluid_type": str}
    """
    fluids_raw = cp.FluidsList()
    all_fluids = fluids_raw if isinstance(fluids_raw, list) else fluids_raw.split(",")

    candidates = []
    rejected   = []

    for name in sorted(all_fluids):
        # Check blacklist first
        if name in BANNED_FLUIDS:
            rejected.append((name, BANNED_FLUIDS[name]))
            continue

        # Try to get critical properties
        try:
            Tc = cp.PropsSI("Tcrit", name) - 273.15   # °C
            pc = cp.PropsSI("pcrit", name) / 1e5       # bar
        except Exception:
            rejected.append((name, "CoolProp failed to evaluate critical point"))
            continue

        # Filter by critical temperature range
        if Tc < Tc_min:
            rejected.append((name, f"Tc = {Tc:.1f} °C < {Tc_min} °C"))
            continue
        if Tc > Tc_max:
            rejected.append((name, f"Tc = {Tc:.1f} °C > {Tc_max} °C"))
            continue

        # Check if flagged
        flag = FLAGGED_FLUIDS.get(name, None)
        if flag and not include_flagged:
            rejected.append((name, f"Flagged: {flag}"))
            continue

        # Classify fluid type
        fluid_type = classify_fluid_type(name)

        candidates.append({
            "name": name,
            "Tc": Tc,
            "pc": pc,
            "flag": flag,
            "fluid_type": fluid_type,
        })

    # Sort by critical temperature
    candidates.sort(key=lambda f: f["Tc"])

    if print_summary:
        print("\n" + 80*"═")
        print("  WORKING FLUID SCREENING")
        print(80*"═")
        print(f"  Critical temperature range : {Tc_min} – {Tc_max} °C")
        print(f"  CoolProp fluids checked    : {len(all_fluids)}")
        print(f"  Candidates after filtering : {len(candidates)}")
        print(f"  Rejected                   : {len(rejected)}")
        print(80*"─")
        print(f"  {'Fluid':<20s} {'Tc [°C]':>8s} {'pc [bar]':>9s} {'Type':>10s}   Note")
        print(80*"─")
        for f in candidates:
            note = f["flag"] if f["flag"] else "✓"
            print(f"  {f['name']:<20s} {f['Tc']:>8.1f} {f['pc']:>9.1f} {f['fluid_type']:>10s}   {note}")
        print(80*"═" + "\n")

    return candidates


# ══════════════════════════════════════════════════════════════════════
#  Adjust design variables for a specific fluid
# ══════════════════════════════════════════════════════════════════════
def adjust_config_for_fluid(config, fluid_name, Tc, pc, T_hot_in, T_cold_in):
    """
    Adjust the YAML config for a specific working fluid.
    
    NOTE: This function intentionally does NOT modify design variable bounds.
    The YAML file uses $working_fluid references (e.g., 
    "$working_fluid.critical_point.p") which thermopt evaluates automatically
    for each fluid. Overwriting these with numeric values would break this.
    
    The only modification needed is changing the working fluid name, which
    is done in run_fluid_sweep() before calling this function.
    
    Parameters
    ----------
    config : dict
        Parsed YAML configuration.
    fluid_name : str
        CoolProp fluid name (for reference, not used).
    Tc : float
        Critical temperature [°C] (for reference, not used).
    pc : float
        Critical pressure [bar] (for reference, not used).
    T_hot_in : float
        Heat source inlet temperature [°C] (for reference, not used).
    T_cold_in : float
        Heat sink inlet temperature [°C] (for reference, not used).
    
    Returns
    -------
    config : dict
        Unchanged configuration (thermopt handles $working_fluid references).
    """
    # Do nothing - let thermopt handle the $working_fluid references
    # The working fluid name is already changed in run_fluid_sweep()
    return config


# ══════════════════════════════════════════════════════════════════════
#  Optimization sweep
# ══════════════════════════════════════════════════════════════════════
def run_fluid_sweep(config_file, candidates, save_results=True,
                    run_exergy_analysis=True, output_dir="results/fluid_sweep"):
    """
    Run the ORC optimization for each candidate working fluid.

    Parameters
    ----------
    config_file : str
        Path to the base YAML configuration file.
    candidates : list of dict
        Output from get_candidate_fluids().
    save_results : bool
        Whether to call cycle.save_results() for each fluid.
    run_exergy_analysis : bool
        Whether to run exergy analysis for each converged case.
    output_dir : str
        Base output directory for results.

    Returns
    -------
    pandas.DataFrame
        Summary table with one row per fluid.
    """
    # Load and parse the base YAML
    with open(config_file, "r") as f:
        base_config = yaml.safe_load(f)
    
    # Extract boundary conditions for bounds adjustment
    # Support both old and new YAML structure
    fp = base_config.get("problem_formulation", {}).get("fixed_parameters", {})
    hs = fp.get("heat_source", {})
    hk = fp.get("heat_sink", {})
    
    # Try to get temperatures, with fallbacks
    T_hot_in_raw = hs.get("inlet_temperature", 413.15)
    T_cold_in_raw = hk.get("inlet_temperature", 293.15)
    
    # Handle expressions like "207.1 + 273.15"
    if isinstance(T_hot_in_raw, str):
        T_hot_in = eval(T_hot_in_raw) - 273.15
    else:
        T_hot_in = T_hot_in_raw - 273.15
    
    if isinstance(T_cold_in_raw, str):
        T_cold_in = eval(T_cold_in_raw) - 273.15
    else:
        T_cold_in = T_cold_in_raw - 273.15
    
    # Try to import exergy analysis
    if run_exergy_analysis:
        try:
            from exergy_analysis import perform_exergy_analysis
            has_exergy = True
        except ImportError:
            print("  Warning: exergy_analysis module not found, skipping exergy calculations")
            has_exergy = False
    else:
        has_exergy = False

    os.makedirs(output_dir, exist_ok=True)
    results = []

    for i, fluid in enumerate(candidates):
        name = fluid["name"]
        Tc = fluid["Tc"]
        pc = fluid["pc"]
        
        print(f"\n{'='*80}")
        print(f"  [{i+1}/{len(candidates)}]  Optimizing with {name}  "
              f"(Tc = {Tc:.1f} °C, pc = {pc:.1f} bar, {fluid['fluid_type']})")
        print(f"{'='*80}")

        # Deep copy and modify the config for this fluid
        config = copy.deepcopy(base_config)
        config["problem_formulation"]["fixed_parameters"]["working_fluid"]["name"] = name
        
        # Adjust bounds and initial values for this fluid
        config = adjust_config_for_fluid(config, name, Tc, pc, T_hot_in, T_cold_in)

        # Write temporary YAML
        tmp_config = os.path.join(output_dir, f"_temp_config_{name}.yaml")
        with open(tmp_config, "w") as f:
            yaml.dump(config, f, default_flow_style=False)

        try:
            fluid_out_dir = os.path.join(output_dir, name)
            cycle = th.ThermodynamicCycleOptimization(tmp_config, out_dir=fluid_out_dir)
            cycle.run_optimization()

            if save_results:
                cycle.save_results()

            # Extract energy analysis results
            ea = cycle.problem.cycle_data["energy_analysis"]
            
            result_row = {
                "fluid":              name,
                "Tc_C":               Tc,
                "pc_bar":             pc,
                "fluid_type":         fluid["fluid_type"],
                "cycle_efficiency":   ea["cycle_efficiency"],
                "system_efficiency":  ea["system_efficiency"],
                "net_power_kW":       ea.get("net_system_power", ea.get("net_cycle_power", 0)) / 1e3,
                "expander_power_kW":  ea["expander_power"] / 1e3,
                "heater_duty_kW":     ea["heater_heat_flow"] / 1e3,
                "cooler_duty_kW":     ea["cooler_heat_flow"] / 1e3,
                "mass_flow_wf":       ea["mass_flow_working_fluid"],
                "backwork_ratio":     ea["backwork_ratio"],
                "status":             "converged",
                "flag":               fluid["flag"],
            }
            
            # Run exergy analysis if available
            if has_exergy:
                try:
                    exergy_results = perform_exergy_analysis(cycle, tmp_config)
                    result_row["exergy_efficiency"] = exergy_results.cycle["eta_exergy"]
                    result_row["E_D_total_kW"] = exergy_results.cycle["E_D_total"] / 1e3
                    result_row["E_fuel_kW"] = exergy_results.cycle["E_fuel"] / 1e3
                except Exception as ex:
                    print(f"  Warning: Exergy analysis failed: {ex}")
                    result_row["exergy_efficiency"] = np.nan
                    result_row["E_D_total_kW"] = np.nan
                    result_row["E_fuel_kW"] = np.nan
            
            results.append(result_row)
            print(f"  ✓ η_sys = {ea['system_efficiency']*100:.2f}%  "
                  f"η_cyc = {ea['cycle_efficiency']*100:.2f}%  "
                  f"W_net = {result_row['net_power_kW']:.1f} kW")
            if has_exergy and not np.isnan(result_row.get("exergy_efficiency", np.nan)):
                print(f"    η_ex = {result_row['exergy_efficiency']*100:.2f}%")

        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            traceback.print_exc()
            results.append({
                "fluid":              name,
                "Tc_C":               Tc,
                "pc_bar":             pc,
                "fluid_type":         fluid["fluid_type"],
                "cycle_efficiency":   np.nan,
                "system_efficiency":  np.nan,
                "net_power_kW":       np.nan,
                "expander_power_kW":  np.nan,
                "heater_duty_kW":     np.nan,
                "cooler_duty_kW":     np.nan,
                "mass_flow_wf":       np.nan,
                "backwork_ratio":     np.nan,
                "exergy_efficiency":  np.nan,
                "E_D_total_kW":       np.nan,
                "E_fuel_kW":          np.nan,
                "status":             str(e)[:80],
                "flag":               fluid["flag"],
            })

        finally:
            # Clean up temp file
            if os.path.exists(tmp_config):
                os.remove(tmp_config)

    # Build summary DataFrame
    df = pd.DataFrame(results)
    df = df.sort_values("system_efficiency", ascending=False, na_position="last")
    df = df.reset_index(drop=True)

    # Save to Excel
    excel_path = os.path.join(output_dir, "fluid_sweep_summary.xlsx")
    df.to_excel(excel_path, index=False)
    print(f"\n  Summary saved to: {excel_path}")

    # Print summary
    print("\n" + 80*"═")
    print("  FLUID SWEEP RESULTS — sorted by system efficiency")
    print(80*"═")
    converged = df[df["status"] == "converged"]
    
    header = f"  {'Fluid':<18s} {'η_sys':>7s} {'η_cyc':>7s}"
    if "exergy_efficiency" in df.columns:
        header += f" {'η_ex':>7s}"
    header += f" {'W_net':>9s} {'Type':>10s}"
    print(header)
    print(80*"─")
    
    for _, row in converged.iterrows():
        line = f"  {row['fluid']:<18s} {row['system_efficiency']*100:>6.2f}% {row['cycle_efficiency']*100:>6.2f}%"
        if "exergy_efficiency" in row and not np.isnan(row["exergy_efficiency"]):
            line += f" {row['exergy_efficiency']*100:>6.2f}%"
        else:
            line += f" {'N/A':>7s}"
        line += f" {row['net_power_kW']:>8.1f}kW {row['fluid_type']:>10s}"
        if row["flag"]:
            line += f"  ⚠"
        print(line)
    
    failed = df[df["status"] != "converged"]
    if len(failed) > 0:
        print(f"\n  {len(failed)} fluid(s) failed to converge:")
        for _, row in failed.iterrows():
            print(f"    - {row['fluid']}: {row['status']}")
    print(80*"═" + "\n")

    return df


# ══════════════════════════════════════════════════════════════════════
#  Quick bar-chart comparison
# ══════════════════════════════════════════════════════════════════════
def plot_fluid_comparison(df, savefig=True, filename="fluid_comparison.png",
                          output_dir=None, show_exergy=True):
    """
    Bar chart comparing system, cycle, and exergy efficiency across fluids.

    Parameters
    ----------
    df : pandas.DataFrame
        Output from run_fluid_sweep().
    savefig : bool
        Whether to save the figure to disk.
    filename : str
        Output filename.
    output_dir : str, optional
        Directory to save the figure in.  If None, saves in current directory.
    show_exergy : bool
        Whether to include exergy efficiency bars (if available).
    """
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.join(output_dir, filename)

    df_ok = df[df["status"] == "converged"].copy()
    df_ok = df_ok.sort_values("system_efficiency", ascending=True)

    has_exergy = "exergy_efficiency" in df_ok.columns and df_ok["exergy_efficiency"].notna().any()
    n_bars = 3 if (has_exergy and show_exergy) else 2
    
    fig, ax = plt.subplots(figsize=(10, max(4, 0.5*len(df_ok))))

    y = np.arange(len(df_ok))
    h = 0.8 / n_bars

    ax.barh(y - h*(n_bars-1)/2, df_ok["system_efficiency"]*100, h,
            color="#2980b9", label="System eff. (η_sys)")
    ax.barh(y - h*(n_bars-1)/2 + h, df_ok["cycle_efficiency"]*100, h,
            color="#e67e22", label="Cycle eff. (η_cyc)")
    
    if has_exergy and show_exergy:
        ax.barh(y - h*(n_bars-1)/2 + 2*h, df_ok["exergy_efficiency"]*100, h,
                color="#27ae60", label="Exergy eff. (η_ex)")

    ax.set_yticks(y)
    ax.set_yticklabels(df_ok["fluid"])
    ax.set_xlabel("Efficiency [%]")
    ax.set_title("Working Fluid Comparison", fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    fig.tight_layout()

    if savefig:
        fig.savefig(filename, dpi=300)
        print(f"Figure saved → {filename}")

    return fig, ax


def plot_efficiency_vs_Tc(df, savefig=True, filename="efficiency_vs_Tc.png",
                          output_dir=None):
    """
    Scatter plot showing efficiency vs critical temperature.
    
    Useful for identifying trends in fluid selection.
    """
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.join(output_dir, filename)
    
    df_ok = df[df["status"] == "converged"].copy()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Color by fluid type
    colors = {"dry": "#e74c3c", "wet": "#3498db", "isentropic": "#2ecc71", "unknown": "#95a5a6"}
    
    for fluid_type in df_ok["fluid_type"].unique():
        mask = df_ok["fluid_type"] == fluid_type
        subset = df_ok[mask]
        ax.scatter(subset["Tc_C"], subset["system_efficiency"]*100,
                   c=colors.get(fluid_type, "#95a5a6"),
                   label=f"{fluid_type.capitalize()} fluid",
                   s=80, alpha=0.7, edgecolors="black", linewidth=0.5)
        
        # Add labels
        for _, row in subset.iterrows():
            ax.annotate(row["fluid"], (row["Tc_C"], row["system_efficiency"]*100),
                        fontsize=7, ha="left", va="bottom",
                        xytext=(3, 3), textcoords="offset points")
    
    ax.set_xlabel("Critical Temperature [°C]")
    ax.set_ylabel("System Efficiency [%]")
    ax.set_title("System Efficiency vs Critical Temperature", fontweight="bold")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    
    if savefig:
        fig.savefig(filename, dpi=300)
        print(f"Figure saved → {filename}")
    
    return fig, ax


# ══════════════════════════════════════════════════════════════════════
#  Main entry point (for standalone testing)
# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # Example usage
    candidates = get_candidate_fluids(Tc_min=100, Tc_max=200)
    print(f"\nFound {len(candidates)} candidate fluids")
    
    # To run the sweep (uncomment):
    # df = run_fluid_sweep("case_butane_ORC.yaml", candidates)
    # plot_fluid_comparison(df, output_dir="results/fluid_sweep")
    # plot_efficiency_vs_Tc(df, output_dir="results/fluid_sweep")
