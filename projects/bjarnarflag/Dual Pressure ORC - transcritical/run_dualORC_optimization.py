import os
import re
import sys
import warnings
import traceback
import thermopt as th
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from datetime import datetime

# ---- importing post processing files:
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "shared_utilities"))
from exergy_analysis import perform_exergy_analysis, plot_heat_source_utilization
from plot_TQ_diagram import plot_TQ_diagram
from validation_checks import run_validation_checks

# -----

warnings.filterwarnings("ignore", message="FigureCanvasAgg is non-interactive")


# ══════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════
MODE = "optimize"  # "optimize" | "sweep" | "parametric"


# WITH Astolfi stacking correlations:
CONFIG_FILE = Path(__file__).with_name(
    "case_butane_transcritical_dualORC_bjarnarflag.yaml"
)

SWEEP_OUTPUT_DIR = "results/fluid_sweep_DUAL_ORC"

# ── Parametric study settings ──
PARAMETRIC_CONFIGS = [
    Path(__file__).with_name("case_Toluene_dualORC_mass_flow_macchi_astolfi.yaml"),
    # Path(__file__).with_name("case_Toluene_recuperated_dualORC_mass_flow_macchi_astolfi.yaml"),
]
PARAMETRIC_HP_STAGES = [2, 3]
PARAMETRIC_LP_STAGES = [2, 3, 5, 7]
PARAMETRIC_HP_RPMS = [1500]
PARAMETRIC_LP_RPMS = [1500, 1000, 750]
PARAMETRIC_OUTPUT_DIR = Path(__file__).parent / "results" / "parametric_study"


# ══════════════════════════════════════════════════════════════════════
#  MODE 1: SINGLE-FLUID OPTIMIZATION
# ══════════════════════════════════════════════════════════════════════
def run_optimize(config_file):
    """Run optimization + full post-processing for a single YAML config."""

    th.print_package_info()

    cycle = th.ThermodynamicCycleOptimization(config_file)
    cycle.problem.plot_cycle_realtime(config_file, update_interval=0.1)

    cycle.run_optimization()
    cycle.save_results()

    # ──────────────────────── VALIDATION CHECKS ───────────────────────
    run_validation_checks(cycle, config_file=config_file)

    # ──────────────────────── MASS FLOW RATES ─────────────────────────
    ea = cycle.problem.cycle_data["energy_analysis"]
    print("\n" + "=" * 76)
    print("  MASS FLOW RATES")
    print("=" * 76)
    print(
        f"  Well (heating fluid)               :  {ea['mass_flow_heating_fluid']:.2f} kg/s"
    )
    print(
        f"  Working fluid (total)              :  {ea['mass_flow_working_fluid']:.2f} kg/s"
    )
    print(f"    HP branch                        :  {ea['mass_flow_hp']:.2f} kg/s")
    print(f"    LP branch                        :  {ea['mass_flow_lp']:.2f} kg/s")
    print(f"  Split fraction (x)                 :  {ea['split_fraction']:.4f}")
    print(
        f"  Cooling fluid                      :  {ea['mass_flow_cooling_fluid']:.2f} kg/s"
    )
    print("=" * 76 + "\n")

    # ────────────────────────────POST-PROCESSING ───────────────────────
    graph_dir = os.path.join(cycle.out_dir, "graphs")
    os.makedirs(graph_dir, exist_ok=True)

    # Exergy analysis
    exergy = perform_exergy_analysis(cycle, config_file=config_file)
    exergy.print_summary()
    exergy.to_excel(os.path.join(graph_dir, "exergy_results.xlsx"))
    exergy.plot_exergy_destruction(savefig=os.path.join(graph_dir, "exergy_bar.png"))
    try:
        exergy.plot_pie_chart(savefig=os.path.join(graph_dir, "exergy_pie.png"))
    except ValueError as e:
        print(f"Skipping exergy pie chart: {e}")
    exergy.plot_grassmann(savefig=os.path.join(graph_dir, "exergy_grassmann.png"))

    # Heat source utilization curve (not applicable for dual-pressure ORC
    # because brine exit temperature is computed, not a design variable)
    # fig, axes, sweep = plot_heat_source_utilization(
    #     cycle, config_file=config_file,
    #     savefig=os.path.join(graph_dir, "utilization_curve.png"),
    # )

    # Plotting the state points on the graph
    cycle.problem.plot_cycle()
    components = cycle.problem.cycle_data["components"]
    state_points = {
        "1": components["lp_pump"]["state_in"],
        "2": components["lp_pump"]["state_out"],
        "3": components["hp_pump"]["state_in"],
        "3'": components["hp_pump"]["state_out"],
        "4": components["lp_evaporator"]["cold_side"]["state_out"],
        "6": components["hp_expander"]["state_in"],
        "7": components["hp_expander"]["state_out"],
        "8": components["lp_expander"]["state_in"],
        "9": components["lp_expander"]["state_out"],
    }
    offsets = {
        "1": (-12.6, -2.6),
        "2": (-6.8, 6.7),
        "3": (-3.2, 9.3),
        "3'": (-11.9, 1.1),
        "4": (-6.6, -12.6),
        "6": (5.1, -0.6),
        "7": (6.6, -5.4),
        "8": (5.7, -6.4),
        "9": (5.0, -4.6),
    }
    fig_ts = cycle.problem.figure
    ax_ts = fig_ts.axes[0]
    for label, st in state_points.items():
        dx, dy = offsets[label]
        ax_ts.annotate(
            label,
            (st.s, st.T),
            textcoords="offset points",
            xytext=(dx, dy),
            fontsize=9,
            fontweight="bold",
            zorder=20,
            bbox=dict(
                boxstyle="round,pad=0.2",
                facecolor="white",
                edgecolor="none",
                alpha=0.85,
            ),
        )
    fig_ts.savefig(
        os.path.join(graph_dir, "Ts_labeled.png"), dpi=200, bbox_inches="tight"
    )
    plt.close(fig_ts)

    # T-Q diagrams of the Heat Exchangers
    plot_TQ_diagram(cycle, "hp_evaporator", output_dir=graph_dir)
    plot_TQ_diagram(cycle, "lp_evaporator", output_dir=graph_dir)
    plot_TQ_diagram(cycle, "preheater", output_dir=graph_dir)
    plot_TQ_diagram(cycle, "cooler", output_dir=graph_dir)

    # Summary
    print("\n" + "=" * 60)
    print("  ALL RESULTS SAVED")
    print("=" * 60)
    print(f"  Thermopt results : {cycle.out_dir}")
    print(f"  Graphs & exergy  : {graph_dir}")
    print()
    for f in sorted(os.listdir(graph_dir)):
        print(f"    \u2713 {f}")
    print("=" * 60 + "\n")


# ══════════════════════════════════════════════════════════════════════
#  MODE 2: WORKING FLUID SWEEP
# ══════════════════════════════════════════════════════════════════════
def run_sweep(config_file, output_dir):
    """Run fluid sweep across all candidates using config as template."""

    from fluid_sweep_BASIC_ORC import (
        get_candidate_fluids,
        run_fluid_sweep,
        plot_results,
    )

    candidates = get_candidate_fluids(config_file)
    df = run_fluid_sweep(config_file, candidates, output_dir=output_dir)
    plot_results(df, output_dir=output_dir)


# ══════════════════════════════════════════════════════════════════════
#  MODE 3: PARAMETRIC STUDY (HP_stages × LP_stages)
# ══════════════════════════════════════════════════════════════════════


def _modify_yaml_dual(yaml_text, hp_stages, lp_stages, hp_RPM=None, lp_RPM=None):
    """
    Replace n_stages and RPM for HP and LP expanders independently.
    Each expander section is targeted separately using regex.
    """
    # Replace hp_expander n_stages
    yaml_text = re.sub(
        r"(hp_expander:.*?n_stages:\s*)\d+",
        rf"\g<1>{hp_stages}",
        yaml_text,
        count=1,
        flags=re.DOTALL,
    )
    # Replace lp_expander n_stages
    yaml_text = re.sub(
        r"(lp_expander:.*?n_stages:\s*)\d+",
        rf"\g<1>{lp_stages}",
        yaml_text,
        count=1,
        flags=re.DOTALL,
    )
    # Replace hp_expander RPM
    if hp_RPM is not None:
        yaml_text = re.sub(
            r"(hp_expander:.*?RPM:\s*)\d+",
            rf"\g<1>{hp_RPM}",
            yaml_text,
            count=1,
            flags=re.DOTALL,
        )
    # Replace lp_expander RPM
    if lp_RPM is not None:
        yaml_text = re.sub(
            r"(lp_expander:.*?RPM:\s*)\d+",
            rf"\g<1>{lp_RPM}",
            yaml_text,
            count=1,
            flags=re.DOTALL,
        )
    return yaml_text


def _extract_results_dual(cycle, config_file, hp_stages, lp_stages, hp_RPM, lp_RPM):
    """Extract key results from a converged dual-pressure cycle."""
    data = cycle.problem.cycle_data
    components = data["components"]
    energy = data["energy_analysis"]
    config_name = Path(config_file).stem

    result = {
        "config": config_name,
        "HP_n_stages": hp_stages,
        "LP_n_stages": lp_stages,
        "HP_RPM": hp_RPM,
        "LP_RPM": lp_RPM,
        "shared_shaft": (hp_RPM == lp_RPM),
        "converged": True,
    }

    # System-level metrics
    result["eta_system"] = energy.get("system_efficiency", None)
    result["eta_cycle"] = energy.get("cycle_efficiency", None)
    result["W_net_kW"] = energy.get("net_system_power", 0) / 1e3
    result["W_gross_kW"] = energy.get("gross_power", 0) / 1e3
    result["Q_in_kW"] = (
        energy.get("total_heat_input", energy.get("heater_heat_flow", 0)) / 1e3
    )
    result["m_dot_wf_kg_s"] = energy.get("mass_flow_working_fluid", None)
    result["m_dot_hp_kg_s"] = energy.get("mass_flow_hp", None)
    result["m_dot_lp_kg_s"] = energy.get("mass_flow_lp", None)
    result["split_fraction"] = energy.get("split_fraction", None)

    # Both expanders
    for prefix, exp_name in [("HP", "hp_expander"), ("LP", "lp_expander")]:
        exp = components[exp_name]
        data_out = exp.get("data_out", {})
        result[f"{prefix}_eta_turbine"] = exp.get("efficiency", None)
        result[f"{prefix}_SP"] = data_out.get("size_parameter", None)
        result[f"{prefix}_SP_clamped"] = data_out.get("size_parameter_clamped", False)
        result[f"{prefix}_Vr"] = data_out.get("volume_ratio", None)
        result[f"{prefix}_Ns"] = data_out.get("specific_speed", None)
        result[f"{prefix}_Dh_is_kJ_kg"] = exp.get("isentropic_work", 0) / 1e3
        result[f"{prefix}_p_in_bar"] = exp["state_in"].p / 1e5
        result[f"{prefix}_p_out_bar"] = exp["state_out"].p / 1e5
        result[f"{prefix}_W_kW"] = (
            exp.get("mass_flow", 0) * exp.get("specific_work", 0) / 1e3
        )

        # Per-stage diagnostics (astolfi-stacking)
        stage_data = data_out.get("stage_data", [])
        if stage_data:
            etas = [sd["eta_stage"] for sd in stage_data]
            ns_vals = [sd["Ns"] for sd in stage_data]
            result[f"{prefix}_eta_min_stage"] = min(etas)
            result[f"{prefix}_Ns_max"] = max(ns_vals)
            result[f"{prefix}_any_stage_zero"] = any(e <= 0.0 for e in etas)

    # Recuperator (if present)
    if "recuperator" in components:
        Q_rec = energy.get("recuperator_heat_flow", 0) / 1e3
        result["Q_recuperator_kW"] = Q_rec

    return result


def _run_single_dual(config_file, hp_stages, lp_stages, hp_RPM, lp_RPM):
    """Run one dual-pressure optimization with specific HP/LP stage counts and RPMs."""
    yaml_text = Path(config_file).read_text()
    yaml_text = _modify_yaml_dual(yaml_text, hp_stages, lp_stages, hp_RPM, lp_RPM)

    config_dir = Path(config_file).parent
    tmp_path = (
        config_dir
        / f"_tmp_parametric_hp{hp_stages}_lp{lp_stages}_hprpm{hp_RPM}_lprpm{lp_RPM}.yaml"
    )

    try:
        tmp_path.write_text(yaml_text)

        cycle = th.ThermodynamicCycleOptimization(str(tmp_path))
        cycle.run_optimization()

        # Check convergence
        converged = True
        for attr_path in [
            lambda: cycle.success,
            lambda: cycle.result.success,
            lambda: cycle.solver_result.success,
        ]:
            try:
                converged = attr_path()
                break
            except AttributeError:
                continue

        if not converged:
            return {
                "config": Path(config_file).stem,
                "HP_n_stages": hp_stages,
                "LP_n_stages": lp_stages,
                "HP_RPM": hp_RPM,
                "LP_RPM": lp_RPM,
                "converged": False,
            }

        return _extract_results_dual(
            cycle, config_file, hp_stages, lp_stages, hp_RPM, lp_RPM
        )

    except Exception as e:
        print(f"    ✗ FAILED: {e}")
        traceback.print_exc()
        return {
            "config": Path(config_file).stem,
            "HP_n_stages": hp_stages,
            "LP_n_stages": lp_stages,
            "HP_RPM": hp_RPM,
            "LP_RPM": lp_RPM,
            "converged": False,
            "error": str(e),
        }
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def run_parametric(
    config_files, hp_stages_list, lp_stages_list, hp_rpms, lp_rpms, output_dir
):
    """Run the full HP_stages × LP_stages × HP_RPM × LP_RPM parametric study."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    warnings.filterwarnings("once", message="Astolfi-stacking: SP")

    total_runs = (
        len(config_files)
        * len(hp_stages_list)
        * len(lp_stages_list)
        * len(hp_rpms)
        * len(lp_rpms)
    )
    results = []
    run_count = 0

    print("=" * 76)
    print("  PARAMETRIC STUDY — Dual-Pressure Astolfi Stage-Stacking (Table 6.6)")
    print("=" * 76)
    print(f"  Configs   : {len(config_files)}")
    print(f"  HP stages : {hp_stages_list}")
    print(f"  LP stages : {lp_stages_list}")
    print(f"  HP RPMs   : {hp_rpms}")
    print(f"  LP RPMs   : {lp_rpms}")
    print(f"  Total     : {total_runs} optimization runs")
    print("=" * 76 + "\n")

    for config_file in config_files:
        config_name = Path(config_file).stem
        print(f"\n{'─' * 76}")
        print(f"  Config: {config_name}")
        print(f"{'─' * 76}")

        if not Path(config_file).exists():
            print(f"  ✗ File not found: {config_file}")
            continue

        for hp_rpm in hp_rpms:
            for lp_rpm in lp_rpms:
                shaft_label = "shared" if hp_rpm == lp_rpm else "separate"
                for hp_n in hp_stages_list:
                    for lp_n in lp_stages_list:
                        run_count += 1
                        print(
                            f"\n  [{run_count}/{total_runs}] "
                            f"HP={hp_n}stg@{hp_rpm}, LP={lp_n}stg@{lp_rpm} "
                            f"({shaft_label}) ... ",
                            end="",
                            flush=True,
                        )

                        result = _run_single_dual(
                            config_file, hp_n, lp_n, hp_rpm, lp_rpm
                        )
                        results.append(result)

                        if result.get("converged", False):
                            eta = result.get("eta_system", 0)
                            eta_pct = eta * 100 if eta else 0
                            W = result.get("W_net_kW", 0)
                            print(f"✓  η_sys = {eta_pct:.2f}%,  W_net = {W:.0f} kW")
                        else:
                            print("✗  did not converge")

    # Save results
    df = pd.DataFrame(results)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    csv_path = output_dir / f"parametric_dual_results_{timestamp}.csv"
    xlsx_path = output_dir / f"parametric_dual_results_{timestamp}.xlsx"

    df.to_csv(csv_path, index=False)
    df.to_excel(xlsx_path, index=False, sheet_name="Results")

    # Print summary table
    print("\n\n" + "=" * 76)
    print("  PARAMETRIC STUDY — RESULTS SUMMARY")
    print("=" * 76)

    display_cols = [
        "config",
        "HP_n_stages",
        "LP_n_stages",
        "HP_RPM",
        "LP_RPM",
        "shared_shaft",
        "converged",
    ]
    for col in [
        "eta_system",
        "W_net_kW",
        "HP_eta_turbine",
        "HP_Ns_max",
        "HP_any_stage_zero",
        "LP_eta_turbine",
        "LP_Ns_max",
        "LP_any_stage_zero",
        "split_fraction",
    ]:
        if col in df.columns:
            display_cols.append(col)

    print(df[display_cols].to_string(index=False))

    print(f"\n  Results saved to:")
    print(f"    CSV  : {csv_path}")
    print(f"    Excel: {xlsx_path}")
    print("=" * 76 + "\n")

    return df


# ══════════════════════════════════════════════════════════════════════
#  RUN
# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":

    print(f"Mode: {MODE}")
    print()

    if MODE == "optimize":
        if not CONFIG_FILE.exists():
            print(f"Error: config file not found: {CONFIG_FILE}")
            sys.exit(1)
        print(f"Config: {CONFIG_FILE.name}\n")
        run_optimize(CONFIG_FILE)

    elif MODE == "sweep":
        if not CONFIG_FILE.exists():
            print(f"Error: config file not found: {CONFIG_FILE}")
            sys.exit(1)
        print(f"Config: {CONFIG_FILE.name}\n")
        run_sweep(CONFIG_FILE, SWEEP_OUTPUT_DIR)

    elif MODE == "parametric":
        run_parametric(
            config_files=PARAMETRIC_CONFIGS,
            hp_stages_list=PARAMETRIC_HP_STAGES,
            lp_stages_list=PARAMETRIC_LP_STAGES,
            hp_rpms=PARAMETRIC_HP_RPMS,
            lp_rpms=PARAMETRIC_LP_RPMS,
            output_dir=PARAMETRIC_OUTPUT_DIR,
        )

    else:
        print(f"Error: MODE must be 'optimize', 'sweep', or 'parametric', got '{MODE}'")
        sys.exit(1)

    plt.show()
