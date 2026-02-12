"""
Working Fluid Sweep for Basic Subcritical ORC

Finds the best working fluid for a SIMPLE subcritical ORC given your system.

The key to getting convergence: bounds are calculated based on YOUR SYSTEM
(heat source temp, heat sink temp, pinch points) — not arbitrary fluid fractions.

Assumptions:
  - Your YAML config has cycle_topology: simple
  - Subcritical operation only (Tc > T_heat_source)

Usage:
    from fluid_sweep_simple import get_candidate_fluids, run_fluid_sweep
    candidates = get_candidate_fluids(T_heat_source=178.1)
    df = run_fluid_sweep("./case_bjarnarflag_ORC.yaml", candidates)
"""

import os
import copy
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import CoolProp.CoolProp as cp

import thermopt as th
import matplotlib
matplotlib.use("Agg")


# ══════════════════════════════════════════════════════════════════════
#  Regulatory exclusions
# ══════════════════════════════════════════════════════════════════════
BANNED_FLUIDS = {
    # Ozone-depleting (Montreal Protocol)
    "R11", "R12", "R13", "R113", "R114", "R115",
    "R123", "R141b", "R142b", "R22",
    # High-GWP (EU F-gas)
    "R134a", "R125", "R143a", "R227EA", "R236EA", "R236FA",
    "R245fa", "R365MFC", "R32", "RC318", "R116", "R218", "R23", "R41",
    # Toxic / dangerous
    "CarbonMonoxide", "HydrogenSulfide", "SulfurDioxide", "NitrousOxide",
    "SulfurHexafluoride", "Methanol", "Ethanol", "Ammonia",
    # Cryogens
    "Helium", "Neon", "Argon", "Krypton", "Xenon", "Hydrogen", "Nitrogen",
    "Oxygen", "Fluorine", "ParaHydrogen", "OrthoHydrogen", "Deuterium",
    "ParaDeuterium", "OrthoDeuterium", "HeavyWater", "Air",
    # Other
    "CarbonDioxide", "Acetone", "DiethylEther", "Ethylene",
    "EthyleneOxide", "CarbonylSulfide",
}

FLAGGED_FLUIDS = {
    "Propane": "flammable", "Butane": "flammable", "Isobutane": "flammable",
    "Isopentane": "flammable", "Neopentane": "flammable", "Pentane": "flammable",
    "Isohexane": "flammable", "Hexane": "flammable", "Heptane": "flammable",
    "Cyclohexane": "flammable", "CycloPropane": "flammable", "Toluene": "flammable",
    "MDM": "siloxane", "MM": "siloxane", "MD2M": "siloxane",
    "MD3M": "siloxane", "MD4M": "siloxane", "D4": "siloxane", "D5": "siloxane", "D6": "siloxane",
}


# ══════════════════════════════════════════════════════════════════════
#  Fluid classification (wet/dry/isentropic)
#  
#  Based on the slope of saturated vapor curve on T-s diagram:
#    - Wet:        ds/dT > 0  → expansion goes INTO two-phase region
#    - Dry:        ds/dT < 0  → expansion stays in superheated region  
#    - Isentropic: ds/dT ≈ 0  → expansion follows saturation curve
#
#  For ORC, DRY fluids are preferred because:
#    - No liquid droplets in turbine (erosion, efficiency loss)
#    - No need for superheating
#    - Recuperator can recover exhaust heat
# ══════════════════════════════════════════════════════════════════════

# Known classifications from literature (more reliable than calculation)
KNOWN_FLUID_TYPES = {
    # Wet fluids (expansion into two-phase)
    "Water": "wet",
    "Ammonia": "wet",
    "R32": "wet",
    "R152A": "wet",
    
    # Isentropic fluids (nearly vertical saturation curve)
    "R134a": "isentropic",
    "R1234yf": "isentropic",
    "R1234ze(E)": "isentropic",
    "R1234ze(Z)": "isentropic",
    "Benzene": "isentropic",
    "R11": "isentropic",
    "R123": "isentropic",
    "R1233zd(E)": "isentropic",
    
    # Dry fluids (expansion stays superheated)
    "n-Pentane": "dry",
    "Pentane": "dry",  # Alias
    "Isopentane": "dry",
    "Neopentane": "dry",
    "n-Butane": "dry",
    "Butane": "dry",
    "Isobutane": "dry",
    "IsoButane": "dry",
    "n-Hexane": "dry",
    "Hexane": "dry",
    "Isohexane": "dry",
    "n-Heptane": "dry",
    "Heptane": "dry",
    "n-Octane": "dry",
    "Octane": "dry",
    "n-Nonane": "dry",
    "Nonane": "dry",
    "n-Decane": "dry",
    "Decane": "dry",
    "Cyclopentane": "dry",
    "CycloHexane": "dry",
    "Cyclohexane": "dry",
    "Toluene": "dry",
    "p-Xylene": "dry",
    "m-Xylene": "dry",
    "o-Xylene": "dry",
    "EthylBenzene": "dry",
    "MM": "dry",
    "MDM": "dry",
    "MD2M": "dry",
    "MD3M": "dry",
    "MD4M": "dry",
    "D4": "dry",
    "D5": "dry",
    "D6": "dry",
    "R245fa": "dry",
    "R236fa": "dry",
    "R227ea": "dry",
    "R365mfc": "dry",
    "RC318": "dry",
    "Propane": "isentropic",  # Slightly isentropic
    "n-Propane": "isentropic",
    "DimethylCarbonate": "dry",
    "Dichloroethane": "isentropic",
}

def classify_fluid_type(name):
    """
    Classify fluid as wet/dry/isentropic based on saturated vapor curve slope.
    
    Uses literature values for known ORC fluids, calculates for unknown fluids.
    
    Returns
    -------
    str: 'wet', 'dry', 'isentropic', or 'unknown'
    """
    # Check known fluids first (most reliable)
    if name in KNOWN_FLUID_TYPES:
        return KNOWN_FLUID_TYPES[name]
    
    # Calculate for unknown fluids
    try:
        Tc = cp.PropsSI("Tcrit", name)
        
        # Evaluate at reduced temperature ~0.7 (away from critical point)
        # Near critical point, all fluids behave similarly
        T_eval = 0.7 * Tc
        
        # Ensure we're above minimum temperature
        try:
            T_min = cp.PropsSI("Tmin", name)
            T_eval = max(T_eval, T_min + 20)
        except:
            pass
        
        # Calculate ds/dT on saturated vapor curve
        dT = 5.0  # K - larger step for numerical stability
        s1 = cp.PropsSI("S", "T", T_eval, "Q", 1, name)
        s2 = cp.PropsSI("S", "T", T_eval + dT, "Q", 1, name)
        
        ds_dT = (s2 - s1) / dT  # J/(kg·K) per K
        
        # Classification thresholds
        # Typical values: wet ~ +1 to +5, dry ~ -1 to -10, isentropic ~ -0.5 to +0.5
        if ds_dT > 0.3:
            return "wet"
        elif ds_dT < -0.3:
            return "dry"
        else:
            return "isentropic"
    except:
        return "unknown"


# ══════════════════════════════════════════════════════════════════════
#  Get candidate fluids (subcritical only)
# ══════════════════════════════════════════════════════════════════════
def get_candidate_fluids(T_heat_source, T_heat_sink=20, pinch_cooler=5,
                         Tc_margin=15, Tc_min_margin=10, Tc_max_margin=250,
                         include_flagged=True, print_summary=True):
    """
    Get candidate fluids for SUBCRITICAL ORC.
    
    Fluids with Tc < T_heat_source + Tc_margin are skipped (would need transcritical).
    
    Parameters
    ----------
    T_heat_source : float
        Heat source inlet temperature [°C]
    T_heat_sink : float
        Heat sink inlet temperature [°C] (default: 20°C)
    pinch_cooler : float
        Minimum temperature difference in cooler [°C] (default: 5°C)
    Tc_margin : float
        Required margin between Tc and T_heat_source for subcritical operation [°C]
    Tc_min_margin : float
        Margin above minimum condensing temperature for Tc_min [°C] (default: 10°C)
        Kept small to preserve low-Tc fluid options
    Tc_max_margin : float
        Margin above T_heat_source for Tc_max [°C] (default: 250°C)
        Kept large to preserve high-Tc fluid options (siloxanes, etc.)
    
    Tc bounds are calculated as:
        Tc_min = T_heat_sink + pinch_cooler + Tc_min_margin
                 (fluid must be able to condense above heat sink)
        Tc_max = T_heat_source + Tc_max_margin
                 (generous upper limit to include high-Tc fluids)
    """
    fluids_raw = cp.FluidsList()
    all_fluids = fluids_raw if isinstance(fluids_raw, list) else fluids_raw.split(",")

    # Calculate Tc bounds from system parameters
    Tc_min = T_heat_sink + pinch_cooler + Tc_min_margin  # Must condense above heat sink
    Tc_max = T_heat_source + Tc_max_margin               # Wide upper limit
    Tc_subcritical_min = T_heat_source + Tc_margin       # Must stay subcritical

    candidates = []
    rejected_transcritical = []
    rejected_Tc_low = []
    rejected_Tc_high = []

    for name in sorted(all_fluids):
        if name in BANNED_FLUIDS:
            continue

        try:
            Tc = cp.PropsSI("Tcrit", name) - 273.15
            pc = cp.PropsSI("pcrit", name) / 1e5
        except:
            continue

        # Track rejections for summary
        if Tc < Tc_min:
            rejected_Tc_low.append((name, Tc))
            continue
        if Tc > Tc_max:
            rejected_Tc_high.append((name, Tc))
            continue

        # Skip transcritical
        if Tc < Tc_subcritical_min:
            rejected_transcritical.append((name, Tc))
            continue

        flag = FLAGGED_FLUIDS.get(name, None)
        if flag and not include_flagged:
            continue

        fluid_type = classify_fluid_type(name)
        candidates.append({
            "name": name, "Tc": Tc, "pc": pc,
            "flag": flag, "fluid_type": fluid_type,
        })

    candidates.sort(key=lambda f: f["Tc"])

    if print_summary:
        print("\n" + "═"*75)
        print("  WORKING FLUID CANDIDATES — SUBCRITICAL ORC")
        print("═"*75)
        print(f"  Heat source temperature: {T_heat_source}°C")
        print(f"  Heat sink temperature:   {T_heat_sink}°C")
        print(f"  Cooler pinch point:      {pinch_cooler}°C")
        print("─"*75)
        print(f"  Calculated Tc bounds (from system parameters):")
        print(f"      Tc_min = {T_heat_sink} + {pinch_cooler} + {Tc_min_margin} = {Tc_min:.1f}°C  (must condense)")
        print(f"      Tc_max = {T_heat_source} + {Tc_max_margin} = {Tc_max:.1f}°C  (wide upper limit)")
        print(f"      Tc_subcritical_min = {T_heat_source} + {Tc_margin} = {Tc_subcritical_min:.1f}°C  (stay subcritical)")
        print("─"*75)
        
        if rejected_Tc_low:
            print(f"\n  Skipped (Tc < {Tc_min:.1f}°C, cannot condense above heat sink):")
            for name, tc in sorted(rejected_Tc_low, key=lambda x: x[1])[:10]:
                print(f"      {name:<18} Tc = {tc:.1f}°C")
            if len(rejected_Tc_low) > 10:
                print(f"      ... and {len(rejected_Tc_low) - 10} more")
        
        if rejected_Tc_high:
            print(f"\n  Skipped (Tc > {Tc_max:.1f}°C, above upper limit):")
            for name, tc in sorted(rejected_Tc_high, key=lambda x: x[1]):
                print(f"      {name:<18} Tc = {tc:.1f}°C")
        
        if rejected_transcritical:
            print(f"\n  Skipped (would need transcritical, Tc < {Tc_subcritical_min:.1f}°C):")
            for name, tc in sorted(rejected_transcritical, key=lambda x: x[1]):
                print(f"      {name:<18} Tc = {tc:.1f}°C")
        
        print(f"\n  Candidates ({len(candidates)} fluids):")
        print(f"    {'Fluid':<18} {'Tc [°C]':>8} {'pc [bar]':>9} {'Type':>11}  Note")
        print("    " + "─"*55)
        for f in candidates:
            note = f["flag"] if f["flag"] else "—"
            print(f"    {f['name']:<18} {f['Tc']:>8.1f} {f['pc']:>9.1f} {f['fluid_type']:>11}  {note}")
        print("═"*75 + "\n")

    return candidates


# ══════════════════════════════════════════════════════════════════════
#  Calculate bounds based on SYSTEM CONSTRAINTS
#  
#  This is the key function. The bounds define where the optimizer can
#  search. If bounds are wrong, optimization will fail or give bad results.
#  
#  Physical reasoning for each bound:
#  
#  EVAPORATOR (high pressure side):
#    - T_evap cannot exceed T_heat_source - pinch (heat transfer limit)
#    - T_evap cannot exceed Tc - margin (must stay subcritical)
#    - T_evap must be well above T_cond (need pressure ratio for turbine)
#  
#  CONDENSER (low pressure side):
#    - T_cond cannot go below T_heat_sink + pinch (heat transfer limit)
#    - T_cond should not be too high (reduces cycle efficiency)
#  
# ══════════════════════════════════════════════════════════════════════
def calculate_cycle_bounds(fluid_name, T_heat_source, T_heat_sink,
                           T_min_reinjection, pinch_heater, pinch_cooler):
    """
    Calculate physically meaningful bounds for ORC design variables.
    
    Parameters
    ----------
    fluid_name : str
        CoolProp fluid name
    T_heat_source : float
        Heat source inlet temperature [°C]
    T_heat_sink : float
        Heat sink inlet temperature [°C]
    T_min_reinjection : float
        Minimum heat source exit temperature [°C]
    pinch_heater : float
        Minimum ΔT in evaporator [°C] - from your YAML constraints
    pinch_cooler : float
        Minimum ΔT in condenser [°C] - from your YAML constraints
    
    Returns
    -------
    dict with 'success', 'bounds', 'initial', 'info', 'error'
    """
    result = {"success": False, "error": None}

    try:
        # ══════════════════════════════════════════════════════════════
        # FLUID PROPERTIES
        # ══════════════════════════════════════════════════════════════
        Tc_K = cp.PropsSI("Tcrit", fluid_name)
        pc = cp.PropsSI("pcrit", fluid_name)
        Tc_C = Tc_K - 273.15

        # ══════════════════════════════════════════════════════════════
        # EVAPORATOR SIDE (high pressure)
        # 
        # The evaporating temperature is limited by:
        #   1. Heat source temperature minus pinch point
        #   2. Critical temperature minus safety margin (stay subcritical)
        # ══════════════════════════════════════════════════════════════
        
        T_evap_max_C = min(
            T_heat_source - pinch_heater,  # Can't exceed heat source - pinch
            Tc_C - 10                       # Stay 10°C below critical
        )
        
        T_evap_min_C = T_heat_sink + 40  # Must be well above condensing temp
        
        # Initial guess: 80% of the way from min to max
        T_evap_init_C = T_evap_min_C + 0.8 * (T_evap_max_C - T_evap_min_C)

        # Convert temperatures to pressures (saturation)
        p_evap_max = cp.PropsSI("P", "T", T_evap_max_C + 273.15, "Q", 1, fluid_name)
        p_evap_min = cp.PropsSI("P", "T", T_evap_min_C + 273.15, "Q", 1, fluid_name)
        p_evap_init = cp.PropsSI("P", "T", T_evap_init_C + 273.15, "Q", 1, fluid_name)

        # Turbine inlet enthalpy: superheated vapor
        # Superheat by 5°C, but don't exceed heat source - pinch
        T_turb_in_C = min(T_evap_init_C + 5, T_heat_source - pinch_heater)
        h_turb_init = cp.PropsSI("H", "T", T_turb_in_C + 273.15, "P", p_evap_init, fluid_name)
        
        # Enthalpy bounds for expander inlet
        h_sat_vap_at_min = cp.PropsSI("H", "T", T_evap_min_C + 273.15, "Q", 1, fluid_name)
        h_at_max_temp = cp.PropsSI("H", "T", T_heat_source - pinch_heater + 273.15, 
                                    "P", p_evap_max * 0.95, fluid_name)

        # ══════════════════════════════════════════════════════════════
        # CONDENSER SIDE (low pressure)
        # 
        # The condensing temperature is limited by:
        #   1. Heat sink temperature plus pinch point (minimum)
        #   2. Should not be too high or efficiency suffers
        # ══════════════════════════════════════════════════════════════
        
        T_cond_min_C = T_heat_sink + pinch_cooler  # Can't go below heat sink + pinch
        T_cond_max_C = T_heat_sink + 30            # Upper limit for condensing
        T_cond_init_C = T_cond_min_C + 5           # Start near minimum (better efficiency)

        # Convert to pressures
        p_cond_min = cp.PropsSI("P", "T", T_cond_min_C + 273.15, "Q", 0, fluid_name)
        p_cond_max = cp.PropsSI("P", "T", T_cond_max_C + 273.15, "Q", 0, fluid_name)
        p_cond_init = cp.PropsSI("P", "T", T_cond_init_C + 273.15, "Q", 0, fluid_name)

        # Pump inlet enthalpy: subcooled liquid (3°C subcooling)
        T_pump_in_C = T_cond_init_C - 3
        h_pump_init = cp.PropsSI("H", "T", T_pump_in_C + 273.15, "P", p_cond_init * 1.01, fluid_name)
        
        # Enthalpy bounds for compressor inlet
        h_sat_liq_cold = cp.PropsSI("H", "T", T_cond_min_C + 273.15, "Q", 0, fluid_name)
        h_sat_liq_warm = cp.PropsSI("H", "T", T_cond_max_C + 273.15, "Q", 0, fluid_name)

        # ══════════════════════════════════════════════════════════════
        # HEAT SOURCE/SINK EXIT TEMPERATURES
        # ══════════════════════════════════════════════════════════════
        
        T_hs_exit_min_K = T_min_reinjection + 273.15
        T_hs_exit_max_K = T_heat_source + 273.15 - 5
        T_hs_exit_init_K = (T_hs_exit_min_K + T_hs_exit_max_K) / 2

        T_hk_exit_min_K = T_heat_sink + 273.15 + 3
        T_hk_exit_max_K = T_heat_sink + 273.15 + 15
        T_hk_exit_init_K = T_heat_sink + 273.15 + 8

        # ══════════════════════════════════════════════════════════════
        # VALIDATION: Check that bounds make physical sense
        # ══════════════════════════════════════════════════════════════
        
        pressure_ratio = p_evap_init / p_cond_init
        if pressure_ratio < 2:
            result["error"] = f"Pressure ratio too low ({pressure_ratio:.1f})"
            return result
        
        if p_cond_min < 500:  # Less than 5 mbar - extreme vacuum
            result["error"] = f"Condensing pressure too low ({p_cond_min:.0f} Pa)"
            return result

        # ══════════════════════════════════════════════════════════════
        # STORE RESULTS
        # ══════════════════════════════════════════════════════════════
        
        result["bounds"] = {
            "expander_inlet_pressure": {
                "min": p_evap_min,
                "max": p_evap_max,
            },
            "expander_inlet_enthalpy": {
                "min": h_sat_vap_at_min,
                "max": h_at_max_temp,
            },
            "compressor_inlet_pressure": {
                "min": p_cond_min * 0.95,
                "max": p_cond_max * 1.05,
            },
            "compressor_inlet_enthalpy": {
                "min": h_sat_liq_cold - 30000,  # Allow some subcooling
                "max": h_sat_liq_warm,
            },
            "heat_source_exit_temperature": {
                "min": T_hs_exit_min_K,
                "max": T_hs_exit_max_K,
            },
            "heat_sink_exit_temperature": {
                "min": T_hk_exit_min_K,
                "max": T_hk_exit_max_K,
            },
        }

        result["initial"] = {
            "expander_inlet_pressure": p_evap_init,
            "expander_inlet_enthalpy": h_turb_init,
            "compressor_inlet_pressure": p_cond_init,
            "compressor_inlet_enthalpy": h_pump_init,
            "heat_source_exit_temperature": T_hs_exit_init_K,
            "heat_sink_exit_temperature": T_hk_exit_init_K,
        }

        result["info"] = {
            "Tc_C": Tc_C,
            "pc_bar": pc / 1e5,
            "T_evap_C": T_evap_init_C,
            "T_evap_max_C": T_evap_max_C,
            "T_cond_C": T_cond_init_C,
            "p_evap_bar": p_evap_init / 1e5,
            "p_cond_bar": p_cond_init / 1e5,
            "pressure_ratio": pressure_ratio,
        }

        result["success"] = True

    except Exception as e:
        result["error"] = str(e)

    return result


# ══════════════════════════════════════════════════════════════════════
#  Check TRUE convergence (solver success + constraints satisfied)
# ══════════════════════════════════════════════════════════════════════
def check_convergence(cycle, tol=0.1):
    """
    Check if optimization truly converged.
    
    Returns: (converged: bool, message: str)
    """
    try:
        if not hasattr(cycle, 'solver'):
            return False, "no solution"

        solver = cycle.solver

        # New ThermOpt (v0.2.3+): success/message are directly on solver
        success = getattr(solver, 'success', False)
        message = getattr(solver, 'message', 'unknown')

        # Get constraint violation (infeasibility)
        infeas = float('inf')
        if hasattr(solver, 'convergence_history'):
            hist = solver.convergence_history
            if hasattr(hist, 'infeasibility') and len(hist.infeasibility) > 0:
                infeas = hist.infeasibility[-1]
            elif hasattr(hist, 'constraint_violation') and len(hist.constraint_violation) > 0:
                infeas = hist.constraint_violation[-1]

        # Converged only if solver succeeded AND constraints satisfied
        if success and infeas < tol:
            return True, "converged"
        elif not success:
            return False, f"solver failed: {message[:40]}"
        else:
            return False, f"constraints violated (infeas={infeas:.1f})"

    except Exception as e:
        return False, f"error: {str(e)[:30]}"


# ══════════════════════════════════════════════════════════════════════
#  Main sweep function
# ══════════════════════════════════════════════════════════════════════
def run_fluid_sweep(config_file, candidates, output_dir="results/fluid_sweep",
                    save_results=True, run_exergy=True):
    """
    Run ORC optimization for each candidate fluid.
    
    Parameters
    ----------
    config_file : str
        Path to YAML config (must have cycle_topology: simple)
    candidates : list
        From get_candidate_fluids()
    output_dir : str
        Where to save results
    save_results : bool
        Save detailed results per fluid
    run_exergy : bool
        Run exergy analysis for converged cases
    """
    # Load config
    with open(config_file, "r") as f:
        base_config = yaml.safe_load(f)
    
    # ─── Extract system parameters ────────────────────────────────────
    pf = base_config["problem_formulation"]
    fp = pf["fixed_parameters"]
    hs = fp["heat_source"]
    hk = fp["heat_sink"]
    
    def eval_temp(val):
        """Convert temperature value (may be string expression) to °C"""
        if isinstance(val, str):
            return eval(val) - 273.15
        return val - 273.15
    
    T_heat_source = eval_temp(hs["inlet_temperature"])
    T_heat_sink = eval_temp(hk["inlet_temperature"])
    T_min_reinj = eval_temp(hs.get("minimum_temperature", 373.15))
    
    # ─── Extract pinch points from constraints ────────────────────────
    pinch_heater = 5.0  # Default
    pinch_cooler = 5.0  # Default
    
    for constraint in pf.get("constraints", []):
        var = constraint.get("variable", "")
        val = constraint.get("value", 5.0)
        if "heater" in var and "temperature_difference" in var:
            pinch_heater = val
        elif "cooler" in var and "temperature_difference" in var:
            pinch_cooler = val
    
    print(f"\n  System parameters (from {config_file}):")
    print(f"    T_heat_source:  {T_heat_source:.1f}°C")
    print(f"    T_heat_sink:    {T_heat_sink:.1f}°C")
    print(f"    T_min_reinject: {T_min_reinj:.1f}°C")
    print(f"    Pinch (heater): {pinch_heater}°C")
    print(f"    Pinch (cooler): {pinch_cooler}°C\n")
    
    # Check for exergy module
    has_exergy = False
    if run_exergy:
        try:
            from exergy_analysis import perform_exergy_analysis
            has_exergy = True
        except ImportError:
            print("  Note: exergy_analysis module not available\n")
    
    os.makedirs(output_dir, exist_ok=True)
    results = []
    stats = {"converged": 0, "failed_bounds": 0, "failed_optim": 0}

    # ══════════════════════════════════════════════════════════════════
    # Loop through candidates
    # ══════════════════════════════════════════════════════════════════
    for i, fluid in enumerate(candidates):
        name = fluid["name"]
        
        print(f"{'─'*70}")
        print(f"  [{i+1}/{len(candidates)}] {name}  (Tc={fluid['Tc']:.1f}°C, {fluid['fluid_type']})")
        
        # ─── Calculate bounds for this fluid ──────────────────────────
        calc = calculate_cycle_bounds(
            name, T_heat_source, T_heat_sink, T_min_reinj,
            pinch_heater, pinch_cooler
        )
        
        if not calc["success"]:
            print(f"      ✗ Bounds failed: {calc['error']}")
            results.append({
                "fluid": name,
                "Tc_C": fluid["Tc"],
                "pc_bar": fluid["pc"],
                "fluid_type": fluid["fluid_type"],
                "status": f"bounds: {calc['error']}",
                "system_efficiency": np.nan,
                "flag": fluid["flag"],
            })
            stats["failed_bounds"] += 1
            continue
        
        info = calc["info"]
        print(f"      T_evap={info['T_evap_C']:.0f}°C (max {info['T_evap_max_C']:.0f}°C), "
              f"T_cond={info['T_cond_C']:.0f}°C, PR={info['pressure_ratio']:.1f}")
        
        # ─── Build config with calculated bounds ──────────────────────
        config = copy.deepcopy(base_config)
        config["problem_formulation"]["fixed_parameters"]["working_fluid"]["name"] = name
        
        # Apply bounds and initial values
        dvars = config["problem_formulation"]["design_variables"]
        for var_name, bounds in calc["bounds"].items():
            if var_name in dvars:
                dvars[var_name]["min"] = bounds["min"]
                dvars[var_name]["max"] = bounds["max"]
                dvars[var_name]["value"] = calc["initial"][var_name]
        
        # Write temp config
        tmp_file = os.path.join(output_dir, f"_temp_{name}.yaml")
        with open(tmp_file, "w") as f:
            yaml.dump(config, f, default_flow_style=False)
        
        # ─── Run optimization ─────────────────────────────────────────
        try:
            fluid_dir = os.path.join(output_dir, name)
            cycle = th.ThermodynamicCycleOptimization(tmp_file, out_dir=fluid_dir)
            cycle.run_optimization()
            
            converged, msg = check_convergence(cycle)
            
            if converged:
                print(f"      ✓ Converged")
                
                if save_results:
                    cycle.save_results()
                
                ea = cycle.problem.cycle_data["energy_analysis"]
                
                row = {
                    "fluid": name,
                    "Tc_C": fluid["Tc"],
                    "pc_bar": fluid["pc"],
                    "fluid_type": fluid["fluid_type"],
                    "T_evap_C": info["T_evap_C"],
                    "T_cond_C": info["T_cond_C"],
                    "p_evap_bar": info["p_evap_bar"],
                    "p_cond_bar": info["p_cond_bar"],
                    "pressure_ratio": info["pressure_ratio"],
                    "system_efficiency": ea["system_efficiency"],
                    "cycle_efficiency": ea["cycle_efficiency"],
                    "net_power_kW": ea.get("net_system_power", ea.get("net_cycle_power", 0)) / 1e3,
                    "expander_power_kW": ea["expander_power"] / 1e3,
                    "heater_duty_kW": ea["heater_heat_flow"] / 1e3,
                    "mass_flow_kg_s": ea["mass_flow_working_fluid"],
                    "backwork_ratio": ea["backwork_ratio"],
                    "status": "converged",
                    "flag": fluid["flag"],
                }
                
                # Exergy analysis
                if has_exergy:
                    try:
                        ex = perform_exergy_analysis(cycle, tmp_file)
                        row["exergy_efficiency"] = ex.cycle["eta_exergy"]
                    except:
                        row["exergy_efficiency"] = np.nan
                
                results.append(row)
                print(f"        η_sys = {row['system_efficiency']*100:.2f}%  "
                      f"η_cyc = {row['cycle_efficiency']*100:.2f}%  "
                      f"W = {row['net_power_kW']:.0f} kW")
                stats["converged"] += 1
                
            else:
                print(f"      ✗ {msg}")
                results.append({
                    "fluid": name,
                    "Tc_C": fluid["Tc"],
                    "pc_bar": fluid["pc"],
                    "fluid_type": fluid["fluid_type"],
                    "status": msg,
                    "system_efficiency": np.nan,
                    "flag": fluid["flag"],
                })
                stats["failed_optim"] += 1
                
        except Exception as e:
            print(f"      ✗ Error: {str(e)[:50]}")
            results.append({
                "fluid": name,
                "Tc_C": fluid["Tc"],
                "pc_bar": fluid["pc"],
                "fluid_type": fluid["fluid_type"],
                "status": f"error: {str(e)[:40]}",
                "system_efficiency": np.nan,
                "flag": fluid["flag"],
            })
            stats["failed_optim"] += 1
            
        finally:
            if os.path.exists(tmp_file):
                os.remove(tmp_file)

    # ══════════════════════════════════════════════════════════════════
    # Summary
    # ══════════════════════════════════════════════════════════════════
    df = pd.DataFrame(results)
    df = df.sort_values("system_efficiency", ascending=False, na_position="last")
    df = df.reset_index(drop=True)
    
    excel_file = os.path.join(output_dir, "fluid_sweep_results.xlsx")
    df.to_excel(excel_file, index=False)
    
    print("\n" + "═"*70)
    print("  FLUID SWEEP RESULTS — BASIC SUBCRITICAL ORC")
    print("═"*70)
    print(f"  Converged:     {stats['converged']}")
    print(f"  Failed bounds: {stats['failed_bounds']}")
    print(f"  Failed optim:  {stats['failed_optim']}")
    print("─"*70)
    
    conv_df = df[df["status"] == "converged"]
    if len(conv_df) > 0:
        print(f"  {'Fluid':<14} {'Tc':>6} {'η_sys':>7} {'η_cyc':>7} {'W_net':>9} {'Type':>10}")
        print("─"*70)
        for _, r in conv_df.iterrows():
            flag = " ⚠" if r["flag"] else ""
            print(f"  {r['fluid']:<14} {r['Tc_C']:>5.0f}°C {r['system_efficiency']*100:>6.2f}% "
                  f"{r['cycle_efficiency']*100:>6.2f}% {r['net_power_kW']:>8.0f}kW "
                  f"{r['fluid_type']:>10}{flag}")
    else:
        print("  No fluids converged.")
    
    print("═"*70)
    print(f"  Results saved: {excel_file}")
    print("═"*70 + "\n")
    
    return df


# ══════════════════════════════════════════════════════════════════════
#  Plotting
# ══════════════════════════════════════════════════════════════════════
def plot_results(df, output_dir=None, filename="fluid_comparison.png"):
    """Bar chart comparing system efficiency of converged fluids."""
    df_ok = df[df["status"] == "converged"].copy()
    if len(df_ok) == 0:
        print("No converged results to plot.")
        return None
    
    df_ok = df_ok.sort_values("system_efficiency", ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, max(4, 0.4*len(df_ok))))
    y = np.arange(len(df_ok))
    
    # Color: red for flammable, blue for non-flammable
    colors = []
    for f in df_ok["flag"]:
        if f and "flammable" in str(f):
            colors.append("#e74c3c")
        else:
            colors.append("#2980b9")
    
    ax.barh(y, df_ok["system_efficiency"]*100, color=colors, edgecolor="black", linewidth=0.5)
    
    # Labels with Tc
    labels = [f"{r['fluid']} (Tc={r['Tc_C']:.0f}°C)" for _, r in df_ok.iterrows()]
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("System Efficiency [%]")
    ax.set_title("Working Fluid Comparison — Basic Subcritical ORC", fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    
    # Legend
    from matplotlib.patches import Patch
    legend = [Patch(color="#2980b9", label="Non-flammable"),
              Patch(color="#e74c3c", label="Flammable")]
    ax.legend(handles=legend, loc="lower right")
    
    fig.tight_layout()
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, filename)
        fig.savefig(path, dpi=300)
        print(f"Figure saved: {path}")
    
    return fig


# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # Example: Finding subcritical fluids for Bjarnarflag case
    print("\n  Example: Finding subcritical fluids for 178°C heat source\n")
    
    # Tc bounds are now calculated from system parameters:
    #   Tc_min = T_heat_sink + pinch_cooler + Tc_min_margin (must be able to condense)
    #   Tc_max = T_heat_source + Tc_max_margin (wide upper limit for high-Tc fluids)
    #   Tc_subcritical_min = T_heat_source + Tc_margin (must stay subcritical)
    
    candidates = get_candidate_fluids(
        T_heat_source=178.1,      # Heat source inlet [°C]
        T_heat_sink=20,           # Heat sink inlet [°C]
        pinch_cooler=5,           # Cooler pinch point [°C]
        Tc_margin=15,             # Margin for subcritical operation [°C]
        Tc_min_margin=10,         # Small margin to keep low-Tc options [°C]
        Tc_max_margin=250,        # Large margin to keep high-Tc options [°C]
    )
    
    # To run:
    # df = run_fluid_sweep("./case_bjarnarflag_ORC.yaml", candidates)
    # plot_results(df, output_dir="results/fluid_sweep")
