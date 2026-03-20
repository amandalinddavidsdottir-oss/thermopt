#!/usr/bin/env python3
"""
══════════════════════════════════════════════════════════════════════════════
  MULTI-START SCREENING FOR THERMOPT ORC OPTIMIZATIONS
══════════════════════════════════════════════════════════════════════════════

Purpose:
    Gradient-based optimizers (SLSQP) can converge to local optima.
    This script generates multiple random starting points, runs the optimizer
    from each, and reports the best result found. If any perturbed start
    beats the baseline, the baseline was a local optimum.

Usage:
    1. Set CONFIG_FILES to your YAML config(s)
    2. Set N_STARTS (recommended: 15-20 per config)
    3. Run:  python run_multistart.py

Works for ALL topologies: simple, recuperated, dual_pressure, transcritical.
══════════════════════════════════════════════════════════════════════════════
"""

import os
import re
import sys
import yaml
import warnings
import traceback
import numpy as np
import pandas as pd
import thermopt as th
import matplotlib

matplotlib.use("Agg")  # non-interactive backend for batch runs
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "shared_utilities"))

warnings.filterwarnings("ignore", message="FigureCanvasAgg is non-interactive")


# ══════════════════════════════════════════════════════════════════════════════
#  USER CONFIGURATION — edit this section
# ══════════════════════════════════════════════════════════════════════════════

CONFIG_FILES = [
    # Uncomment / add the configs you want to screen:
    # Path(__file__).with_name("case_Toluene_simpleORC_macchi_astolfi.yaml"),
    # Path(__file__).with_name("case_Toluene_recuperated_simpleORC_macchi_astolfi.yaml"),
    # Path(__file__).with_name("case_Toluene_dualORC_mass_flow_macchi_astolfi.yaml"),
    # Path(__file__).with_name("case_butane_transcritical_bjarnarflag.yaml"),
    # Path(__file__).with_name("case_butane_transcritical_recuperated_bjarnarflag.yaml"),
    Path(__file__).with_name("case_butane_transcritical_dualORC_bjarnarflag.yaml"),
]

N_STARTS = 15  # Total starting points per config (including baseline)
N_NEAR = 10  # Of those, how many are "near" perturbations (rest are fully random)
SEED = 42  # Random seed for reproducibility
PERTURBATION_FRAC = 0.25  # ±fraction of [min,max] range for "near" starts

OUTPUT_DIR = Path(__file__).parent / "results" / "multistart_screening"


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1: EVALUATE THERMOPT EXPRESSION BOUNDS USING COOLPROP
# ══════════════════════════════════════════════════════════════════════════════


def compute_reference_points(fluid_name, T_ambient_K, T_max_K):
    """
    Compute the ThermOpt reference state points using CoolProp.
    These are the $working_fluid.XXX values used in YAML expressions.
    """
    import CoolProp.CoolProp as CP

    refs = {}

    # ── Critical point ──
    p_crit = CP.PropsSI("pcrit", fluid_name)
    T_crit = CP.PropsSI("Tcrit", fluid_name)
    # At critical point, use Q=0.5 to get critical enthalpy
    h_crit = CP.PropsSI("H", "T", T_crit, "Q", 0, fluid_name)
    refs["critical_point.p"] = p_crit
    refs["critical_point.T"] = T_crit
    refs["critical_point.h"] = h_crit

    # ── Triple point ──
    p_triple = CP.PropsSI("ptriple", fluid_name)
    T_triple = CP.PropsSI("Ttriple", fluid_name)
    refs["triple_point_vapor.p"] = p_triple
    refs["triple_point_vapor.T"] = T_triple

    # ── Liquid at ambient temperature ──
    p_sat_amb = CP.PropsSI("P", "T", T_ambient_K, "Q", 0, fluid_name)
    h_liq_amb = CP.PropsSI("H", "T", T_ambient_K, "Q", 0, fluid_name)
    refs["liquid_at_ambient_temperature.p"] = p_sat_amb
    refs["liquid_at_ambient_temperature.h"] = h_liq_amb
    refs["liquid_at_ambient_temperature.T"] = T_ambient_K

    # ── Gas at maximum temperature (low-pressure superheated vapor) ──
    p_low = max(p_triple * 1.5, 500.0)  # avoid numerical issues at very low p
    h_gas_max = CP.PropsSI("H", "T", T_max_K, "P", p_low, fluid_name)
    refs["gas_at_maximum_temperature.h"] = h_gas_max
    refs["gas_at_maximum_temperature.T"] = T_max_K
    refs["gas_at_maximum_temperature.p"] = p_low

    return refs


def evaluate_expression(expr, refs):
    """
    Evaluate a ThermOpt YAML expression like:
      '0.50*$working_fluid.critical_point.p'
      '0.0*($working_fluid.critical_point.h - $working_fluid.liquid_at_ambient_temperature.h) + ...'
      373.15  (plain number)
    """
    if isinstance(expr, (int, float)):
        return float(expr)

    s = str(expr)

    # Replace all $working_fluid.XXX references with numeric values
    for key, val in refs.items():
        s = s.replace(f"$working_fluid.{key}", f"{val:.10f}")

    # Safety check: only allow numbers and basic arithmetic
    allowed = set("0123456789.+-*/() eE")
    cleaned = s.strip()
    if not all(c in allowed for c in cleaned):
        raise ValueError(f"Cannot evaluate expression: '{expr}' → '{cleaned}'")

    return float(eval(cleaned))


def extract_fluid_and_temps(config_file):
    """Extract fluid name, ambient temperature, and max temperature from YAML."""
    with open(config_file) as f:
        config = yaml.safe_load(f)

    pf = config.get("problem_formulation", config)
    fp = pf.get("fixed_parameters", {})

    fluid_name = fp["working_fluid"]["name"]

    sp = fp.get("special_points", {})
    T_ambient_raw = sp.get("ambient_temperature", 293.15)
    T_max_raw = sp.get("maximum_temperature", 470)

    # Evaluate if expression (e.g., "20.0 + 273.15")
    T_ambient_K = float(eval(str(T_ambient_raw)))
    T_max_K = float(T_max_raw)

    return fluid_name, T_ambient_K, T_max_K


def extract_design_variable_bounds(config_file):
    """
    Parse YAML and evaluate all min/max/value expressions using CoolProp.
    Returns: dict of {var_name: (lower, upper, baseline_value)}
    """
    fluid_name, T_ambient_K, T_max_K = extract_fluid_and_temps(config_file)
    refs = compute_reference_points(fluid_name, T_ambient_K, T_max_K)

    with open(config_file) as f:
        config = yaml.safe_load(f)

    pf = config.get("problem_formulation", config)
    dv_section = pf.get("design_variables", {})

    bounds = {}
    print(f"\n  Design variables for: {Path(config_file).name}")
    print(
        f"  Fluid: {fluid_name}  |  T_amb = {T_ambient_K:.2f} K  |  T_max = {T_max_K:.2f} K"
    )
    print(f"  {'Variable':<40s} {'Lower':>14s} {'Value':>14s} {'Upper':>14s}")
    print(f"  {'─' * 84}")

    for var_name, var_def in dv_section.items():
        try:
            lb = evaluate_expression(var_def["min"], refs)
            ub = evaluate_expression(var_def["max"], refs)
            x0 = evaluate_expression(var_def["value"], refs)
        except Exception as e:
            print(f"  ⚠ Could not evaluate bounds for '{var_name}': {e}")
            print(f"    min = {var_def['min']}")
            print(f"    max = {var_def['max']}")
            print(f"    value = {var_def['value']}")
            continue

        # Clip baseline to bounds (may be slightly off due to floating point)
        x0 = np.clip(x0, lb, ub)
        bounds[var_name] = (lb, ub, x0)

        print(f"  {var_name:<40s} {lb:>14.2f} {x0:>14.2f} {ub:>14.2f}")

    print()
    return bounds


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2: GENERATE RANDOM STARTING POINTS
# ══════════════════════════════════════════════════════════════════════════════


def generate_starting_points(bounds, n_starts, n_near, perturbation_frac, seed):
    """
    Generate starting points for multi-start screening.

    Returns list of (label, {var_name: value}) tuples.
      - 1 baseline (unchanged)
      - n_near "near" perturbations (±perturbation_frac of range around baseline)
      - remainder are fully random within bounds
    """
    rng = np.random.default_rng(seed)
    var_names = list(bounds.keys())
    starts = []

    # ── Baseline (run 0) ──
    baseline = {v: bounds[v][2] for v in var_names}
    starts.append(("baseline", baseline))

    # ── Near perturbations ──
    for i in range(n_near):
        point = {}
        for v in var_names:
            lb, ub, x0 = bounds[v]
            span = ub - lb
            if span < 1e-12:
                point[v] = x0
            else:
                delta = rng.uniform(-perturbation_frac, perturbation_frac) * span
                point[v] = float(np.clip(x0 + delta, lb, ub))
        starts.append((f"near_{i+1:02d}", point))

    # ── Fully random starts ──
    n_random = max(0, n_starts - n_near - 1)
    for i in range(n_random):
        point = {}
        for v in var_names:
            lb, ub, _ = bounds[v]
            if ub - lb < 1e-12:
                point[v] = lb
            else:
                point[v] = float(rng.uniform(lb, ub))
        starts.append((f"random_{i+1:02d}", point))

    return starts


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 3: MODIFY YAML AND RUN OPTIMIZER
# ══════════════════════════════════════════════════════════════════════════════


def modify_yaml_values(yaml_text, var_values):
    """
    Replace the 'value:' field for each design variable in the YAML text.
    Keeps min/max expressions intact. Only changes the numeric value.
    """
    lines = yaml_text.split("\n")

    for var_name, new_val in var_values.items():
        in_var_block = False
        for i, line in enumerate(lines):
            stripped = line.strip()

            # Detect start of this variable's block
            if stripped.startswith(f"{var_name}:"):
                in_var_block = True
                continue

            if in_var_block:
                if stripped.startswith("value:"):
                    # Replace value line, preserving indentation
                    indent = line[: len(line) - len(line.lstrip())]
                    lines[i] = f"{indent}value: {new_val}"
                    in_var_block = False
                    break
                # If we hit another key at the same or higher indent level, stop
                if ":" in stripped and not stripped.startswith(("min:", "max:", "#")):
                    in_var_block = False
                    break

    return "\n".join(lines)


def extract_results(cycle, config_file, label):
    """
    Extract key results from a converged cycle.
    Handles all topologies: simple, recuperated, dual_pressure.
    """
    data = cycle.problem.cycle_data
    components = data["components"]
    energy = data["energy_analysis"]

    result = {
        "config": Path(config_file).stem,
        "start_label": label,
        "converged": True,
    }

    # ── System-level metrics ──
    result["eta_system"] = energy.get("system_efficiency", None)
    result["eta_cycle"] = energy.get("cycle_efficiency", None)
    result["W_net_kW"] = energy.get("net_system_power", 0) / 1e3
    result["W_gross_kW"] = energy.get("gross_power", 0) / 1e3
    result["Q_in_kW"] = (
        energy.get("total_heat_input", energy.get("heater_heat_flow", 0)) / 1e3
    )
    result["Q_available_kW"] = energy.get("available_heat", 0) / 1e3
    Q_avail = result["Q_available_kW"]
    result["heat_utilization"] = (
        result["Q_in_kW"] / Q_avail if Q_avail and Q_avail > 0 else None
    )
    result["m_dot_wf_kg_s"] = energy.get("mass_flow_working_fluid", None)

    # ── Expander(s) ──
    if "hp_expander" in components:
        # Dual-pressure topology
        for prefix, exp_name in [("HP", "hp_expander"), ("LP", "lp_expander")]:
            exp = components[exp_name]
            data_out = exp.get("data_out", {})
            result[f"{prefix}_eta_turbine"] = exp.get("efficiency", None)
            result[f"{prefix}_SP"] = data_out.get("size_parameter", None)
            result[f"{prefix}_Vr"] = data_out.get("volume_ratio", None)
            result[f"{prefix}_Ns"] = data_out.get("specific_speed", None)
            result[f"{prefix}_p_in_bar"] = exp["state_in"].p / 1e5
            result[f"{prefix}_p_out_bar"] = exp["state_out"].p / 1e5
            stage_data = data_out.get("stage_data", [])
            if stage_data:
                result[f"{prefix}_Ns_max"] = max(sd["Ns"] for sd in stage_data)
                result[f"{prefix}_eta_min_stage"] = min(
                    sd["eta_stage"] for sd in stage_data
                )
        result["split_fraction"] = energy.get("split_fraction", None)
    else:
        # Single-expander topology (simple or recuperated)
        exp = components["expander"]
        data_out = exp.get("data_out", {})
        result["eta_turbine"] = exp.get("efficiency", None)
        result["SP"] = data_out.get("size_parameter", None)
        result["Vr"] = data_out.get("volume_ratio", None)
        result["Ns"] = data_out.get("specific_speed", None)
        result["p_in_bar"] = exp["state_in"].p / 1e5
        result["p_out_bar"] = exp["state_out"].p / 1e5
        stage_data = data_out.get("stage_data", [])
        if stage_data:
            result["Ns_max"] = max(sd["Ns"] for sd in stage_data)
            result["eta_min_stage"] = min(sd["eta_stage"] for sd in stage_data)

    # ── Recuperator (if present) ──
    if "recuperator" in components:
        result["Q_recuperator_kW"] = energy.get("recuperator_heat_flow", 0) / 1e3

    # ── Converged design variable values ──
    # Store for comparison between runs
    try:
        opt_config = cycle.problem.cycle_data.get("config", {})
        opt_dv = opt_config.get("design_variables", {})
        for var_name, var_def in opt_dv.items():
            if isinstance(var_def, dict) and "value" in var_def:
                result[f"dv_{var_name}"] = var_def["value"]
    except Exception:
        pass

    return result


def run_single_start(config_file, var_values, label, tmp_dir):
    """Run one optimization from a specific starting point."""
    yaml_text = Path(config_file).read_text()
    yaml_text = modify_yaml_values(yaml_text, var_values)

    tmp_path = tmp_dir / f"_tmp_multistart_{label}.yaml"

    try:
        tmp_path.write_text(yaml_text)

        cycle = th.ThermodynamicCycleOptimization(str(tmp_path))
        cycle.run_optimization()

        # Check convergence (try multiple attribute paths)
        converged = True
        for accessor in [
            lambda: cycle.success,
            lambda: cycle.result.success,
            lambda: cycle.solver_result.success,
        ]:
            try:
                converged = accessor()
                break
            except AttributeError:
                continue

        if not converged:
            return {
                "config": Path(config_file).stem,
                "start_label": label,
                "converged": False,
            }

        return extract_results(cycle, config_file, label)

    except Exception as e:
        print(f"  ✗ EXCEPTION: {e}")
        traceback.print_exc()
        return {
            "config": Path(config_file).stem,
            "start_label": label,
            "converged": False,
            "error": str(e),
        }
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 4: MAIN ORCHESTRATION
# ══════════════════════════════════════════════════════════════════════════════


def run_multistart(config_files, n_starts, n_near, perturbation_frac, seed, output_dir):
    """Run multi-start screening for all specified configs."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    print("=" * 80)
    print("  MULTI-START SCREENING — Local Optima Detection")
    print("=" * 80)
    print(f"  Configs        : {len(config_files)}")
    print(
        f"  Starts/config  : {n_starts} ({n_near} near + {n_starts - n_near - 1} random + 1 baseline)"
    )
    print(f"  Perturbation   : ±{perturbation_frac * 100:.0f}% of feasible range")
    print(f"  Random seed    : {seed}")
    print(f"  Total runs     : {len(config_files) * n_starts}")
    print("=" * 80)

    for config_file in config_files:
        config_file = Path(config_file)
        config_name = config_file.stem

        if not config_file.exists():
            print(f"\n  ✗ File not found: {config_file}")
            continue

        print(f"\n{'━' * 80}")
        print(f"  Config: {config_name}")
        print(f"{'━' * 80}")

        # ── Extract bounds ──
        try:
            bounds = extract_design_variable_bounds(config_file)
        except Exception as e:
            print(f"  ✗ Could not extract bounds: {e}")
            traceback.print_exc()
            continue

        if not bounds:
            print(f"  ✗ No design variables found — skipping")
            continue

        # ── Generate starting points ──
        starts = generate_starting_points(
            bounds, n_starts, n_near, perturbation_frac, seed
        )

        # ── Run each starting point ──
        config_results = []
        tmp_dir = config_file.parent
        for i, (label, var_values) in enumerate(starts):
            print(
                f"\n  [{i + 1}/{len(starts)}] {label:>12s} ... ",
                end="",
                flush=True,
            )

            result = run_single_start(config_file, var_values, label, tmp_dir)
            config_results.append(result)

            if result.get("converged", False):
                eta = result.get("eta_system", 0) or 0
                W = result.get("W_net_kW", 0) or 0
                print(f"✓  η_sys = {eta * 100:.2f}%,  W_net = {W:.0f} kW")
            else:
                err = result.get("error", "did not converge")
                print(f"✗  {err}")

        # ── Analyse results for this config ──
        df_config = pd.DataFrame(config_results)
        converged = df_config[df_config["converged"] == True].copy()

        if len(converged) == 0:
            print(f"\n  ⚠ No converged runs for {config_name}")
            all_results.extend(config_results)
            continue

        # Sort by eta_system descending
        converged = converged.sort_values("eta_system", ascending=False)
        best = converged.iloc[0]
        baseline_row = converged[converged["start_label"] == "baseline"]

        print(f"\n  {'─' * 74}")
        print(f"  RESULTS FOR: {config_name}")
        print(f"  {'─' * 74}")
        print(f"  Converged    : {len(converged)} / {len(starts)}")
        print(
            f"  Best η_system: {best['eta_system'] * 100:.4f}%  (start: {best['start_label']})"
        )

        if len(baseline_row) > 0:
            eta_base = baseline_row.iloc[0]["eta_system"]
            eta_best = best["eta_system"]
            print(f"  Baseline     : {eta_base * 100:.4f}%")
            delta = (eta_best - eta_base) * 100
            if delta > 0.01:  # more than 0.01 percentage point improvement
                print(
                    f"  ⚠ IMPROVEMENT FOUND: +{delta:.4f} pp  →  baseline was a LOCAL OPTIMUM"
                )
            else:
                print(
                    f"  ✓ Baseline is the global optimum (within screening resolution)"
                )

        # Show top 5
        print(f"\n  Top 5 results:")
        print(
            f"  {'Rank':<6s} {'Label':<14s} {'η_system':>10s} {'W_net [kW]':>12s} {'η_turbine':>10s}"
        )
        print(f"  {'─' * 54}")
        for rank, (_, row) in enumerate(converged.head(5).iterrows(), 1):
            eta = row.get("eta_system", 0) or 0
            W = row.get("W_net_kW", 0) or 0
            # Turbine efficiency: handle both single and dual-expander
            eta_t = row.get("eta_turbine", row.get("LP_eta_turbine", None))
            eta_t_str = f"{eta_t * 100:.2f}%" if eta_t else "N/A"
            print(
                f"  {rank:<6d} {row['start_label']:<14s} "
                f"{eta * 100:>9.4f}% {W:>11.0f}  {eta_t_str:>10s}"
            )

        # Show spread
        eta_all = converged["eta_system"].dropna()
        if len(eta_all) > 1:
            print(
                f"\n  η_system spread: {eta_all.min() * 100:.4f}% – {eta_all.max() * 100:.4f}%"
                f"  (range: {(eta_all.max() - eta_all.min()) * 100:.4f} pp)"
            )
            print(f"  η_system stdev : {eta_all.std() * 100:.4f} pp")

        all_results.extend(config_results)

    # ══════════════════════════════════════════════════════════════════════
    #  SAVE ALL RESULTS
    # ══════════════════════════════════════════════════════════════════════
    df_all = pd.DataFrame(all_results)
    csv_path = output_dir / f"multistart_results_{timestamp}.csv"
    xlsx_path = output_dir / f"multistart_results_{timestamp}.xlsx"

    df_all.to_csv(csv_path, index=False)
    df_all.to_excel(xlsx_path, index=False, sheet_name="MultiStart")

    print(f"\n\n{'=' * 80}")
    print(f"  ALL MULTI-START RESULTS SAVED")
    print(f"{'=' * 80}")
    print(f"  CSV  : {csv_path}")
    print(f"  Excel: {xlsx_path}")
    print(f"{'=' * 80}\n")

    return df_all


# ══════════════════════════════════════════════════════════════════════════════
#  RUN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    run_multistart(
        config_files=CONFIG_FILES,
        n_starts=N_STARTS,
        n_near=N_NEAR,
        perturbation_frac=PERTURBATION_FRAC,
        seed=SEED,
        output_dir=OUTPUT_DIR,
    )
