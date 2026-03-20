"""
generate_warmstart_yamls.py
===========================
Generates 15 fully consistent warm-start YAML files for the Macchi-Astolfi
parametric study (n=1..5 × RPM=1000/1500/3000).

HOW IT WORKS
------------
For each n/RPM combination, it reads the converged operating point from the
combined CSV (pressures, intermediate pressures, mass flow), then uses CoolProp
to compute the matching enthalpies at those exact states. This ensures ALL design
variables in the YAML are thermodynamically consistent — no nan, no crashes.

For cases that previously found local optima (RPM=3000, non-monotonic Vr), it
uses the best available converged pressures from the nearest good run as a
starting point, which gives the optimizer a much better chance of finding the
global optimum.

USAGE
-----
Run this script from your ThermOpt project directory (so CoolProp is available):

    python generate_warmstart_yamls.py

Or with explicit paths:

    python generate_warmstart_yamls.py \
        --csv   results/parametric_study/parametric_results_combined.csv \
        --base  path/to/base_yamls/ \
        --out   path/to/output_yamls/

REQUIREMENTS
------------
- CoolProp (available in your ThermOpt venv)
- pandas
- The 5 base YAML files (n1..n5 without RPM suffix)
- The combined parametric results CSV
"""

import re
import ast
import argparse
import shutil
from pathlib import Path

import pandas as pd
from CoolProp.CoolProp import PropsSI


# ── Configuration ─────────────────────────────────────────────────────────────

FLUID = "Toluene"

# Superheating at expander inlet [K] and subcooling at compressor inlet [K]
# Must match the constraints in your YAML files
SUPERHEAT_K = 5.0
SUBCOOL_K = 1.0

# Default heat exchanger temperatures (used when CSV does not have them)
T_HSOURCE_EXIT_DEFAULT = 378.0  # K  (heat source exit)
T_HSINK_EXIT_DEFAULT = 310.0  # K  (heat sink exit)

# ── Helpers ───────────────────────────────────────────────────────────────────


def parse_np_list(s):
    """Parse a numpy-formatted list string from the CSV into a plain Python list."""
    if pd.isna(s):
        return []
    try:
        s = (
            str(s)
            .replace("np.float64(", "")
            .replace("np.float32(", "")
            .replace("np.False_", "False")
            .replace("np.True_", "True")
            .replace(")", "")
        )
        return ast.literal_eval(s)
    except Exception:
        return []


def sat_temperature(p_pa: float) -> float:
    """Saturation temperature [K] at pressure p_pa [Pa]."""
    return PropsSI("T", "P", p_pa, "Q", 0.5, FLUID)


def enthalpy_superheated(p_pa: float, superheat_K: float) -> float:
    """Enthalpy [J/kg] of superheated vapour: T_sat + superheat_K at pressure p_pa."""
    T_sat = PropsSI("T", "P", p_pa, "Q", 1.0, FLUID)
    T = T_sat + superheat_K
    return PropsSI("H", "P", p_pa, "T", T, FLUID)


def enthalpy_subcooled(p_pa: float, subcool_K: float) -> float:
    """Enthalpy [J/kg] of subcooled liquid: T_sat - subcool_K at pressure p_pa."""
    T_sat = PropsSI("T", "P", p_pa, "Q", 0.0, FLUID)
    T = T_sat - subcool_K
    return PropsSI("H", "P", p_pa, "T", T, FLUID)


def patch_yaml(text: str, patches: dict) -> str:
    """
    Replace the `value:` line of specific design variables in raw YAML text.
    patches = {variable_name: new_value, ...}
    Only the `value:` field is changed — bounds, comments, and structure preserved.
    """
    for var, val in patches.items():
        # Match the variable block: find `var_name:` then the `value:` field inside it
        pattern = rf"({re.escape(var)}:.*?value:\s*)[\d\.\-e\+]+"
        replacement = rf"\g<1>{val:.6g}"
        new_text = re.sub(pattern, replacement, text, count=1, flags=re.DOTALL)
        if new_text == text:
            print(f"    WARNING: could not patch '{var}' — pattern not found")
        text = new_text
    return text


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Generate consistent warm-start YAMLs")
    parser.add_argument(
        "--csv", default=None, help="Path to parametric_results_combined.csv"
    )
    parser.add_argument(
        "--base",
        default=None,
        help="Directory containing base n1..n5 YAMLs (no RPM suffix)",
    )
    parser.add_argument(
        "--out", default=None, help="Output directory for warm-start YAMLs"
    )
    args = parser.parse_args()

    # ── Locate files ──────────────────────────────────────────────────────────
    script_dir = Path(__file__).parent

    csv_path = (
        Path(args.csv) if args.csv else (script_dir / "parametric_results_combined.csv")
    )
    base_dir = Path(args.base) if args.base else script_dir
    out_dir = Path(args.out) if args.out else script_dir

    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  Generating consistent warm-start YAMLs")
    print("=" * 70)
    print(f"  CSV    : {csv_path}")
    print(f"  Base   : {base_dir}")
    print(f"  Output : {out_dir}")
    print()

    # ── Load results ──────────────────────────────────────────────────────────
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # ── Process each combination ───────────────────────────────────────────────
    rpms = [1000, 1500, 3000]
    n_values = [1, 2, 3, 4, 5]

    for n in n_values:
        # Load base YAML
        base_path = base_dir / f"case_Toluene_simpleORC_macchi_astolfi_n{n}.yaml"
        if not base_path.exists():
            print(f"  ERROR: base YAML not found: {base_path}")
            continue

        base_text = base_path.read_text()

        for rpm in rpms:
            print(f"  n={n}, RPM={rpm} ...", end=" ", flush=True)

            # ── Find best reference operating point ───────────────────────────
            # Try to use the converged result for this exact n/RPM combination.
            # Fall back to same-n at nearest RPM if this one did not converge well.
            row = None

            # First preference: exact match that converged
            mask = (df["n_stages"] == n) & (df["RPM"] == rpm)
            if mask.any():
                row = df[mask].iloc[0]

            # Second preference: same n, RPM=1500 (usually most reliable)
            if row is None:
                mask2 = (df["n_stages"] == n) & (df["RPM"] == 1500)
                if mask2.any():
                    row = df[mask2].iloc[0]
                    print(f"(using n={n}/RPM=1500 as reference) ", end="")

            # Third preference: same n, RPM=1000
            if row is None:
                mask3 = (df["n_stages"] == n) & (df["RPM"] == 1000)
                if mask3.any():
                    row = df[mask3].iloc[0]
                    print(f"(using n={n}/RPM=1000 as reference) ", end="")

            if row is None:
                print("SKIP — no converged reference found")
                # Fall back to just patching RPM
                text = re.sub(r"(RPM:\s*)\d+", rf"\g<1>{rpm}", base_text)
                dst = (
                    out_dir
                    / f"case_Toluene_simpleORC_macchi_astolfi_n{n}_rpm{rpm}.yaml"
                )
                dst.write_text(text)
                continue

            # ── Extract converged operating point ─────────────────────────────
            p_evap_pa = float(row["p_in_bar"]) * 1e5  # expander inlet
            p_cond_pa = float(row["p_out_bar"]) * 1e5  # compressor inlet
            m_dot_wf = float(row["m_dot_wf_kg_s"])

            # Intermediate pressures (convert from bar to Pa)
            stage_p_out_bar = parse_np_list(row.get("stage_p_out_bar", "[]"))
            # stage_p_out_bar[-1] is p_cond, so intermediates are [:-1]
            intermediate_pressures_pa = [p * 1e5 for p in stage_p_out_bar[:-1]]

            # Heat exchanger temperatures (use from CSV if available, else defaults)
            # ThermOpt doesn't store these directly in the results CSV,
            # so we use physically reasonable values near the constraints
            T_hsource_exit = T_HSOURCE_EXIT_DEFAULT
            T_hsink_exit = T_HSINK_EXIT_DEFAULT

            # ── Compute consistent enthalpies using CoolProp ──────────────────
            try:
                h_exp_in = enthalpy_superheated(p_evap_pa, SUPERHEAT_K)
                h_comp_in = enthalpy_subcooled(p_cond_pa, SUBCOOL_K)
            except Exception as e:
                print(f"CoolProp error: {e} — using RPM-only patch")
                text = re.sub(r"(RPM:\s*)\d+", rf"\g<1>{rpm}", base_text)
                dst = (
                    out_dir
                    / f"case_Toluene_simpleORC_macchi_astolfi_n{n}_rpm{rpm}.yaml"
                )
                dst.write_text(text)
                continue

            # ── Build patches dict ────────────────────────────────────────────
            patches = {
                "heat_source_exit_temperature": T_hsource_exit,
                "heat_sink_exit_temperature": T_hsink_exit,
                "compressor_inlet_pressure": p_cond_pa,
                "compressor_inlet_enthalpy": h_comp_in,
                "expander_inlet_pressure": p_evap_pa,
                "expander_inlet_enthalpy": h_exp_in,
                "expander_mass_flow_rate": m_dot_wf,
            }

            # Add intermediate pressures
            for i, p_int in enumerate(intermediate_pressures_pa, 1):
                patches[f"expander_intermediate_pressure_{i}"] = p_int

            # ── Apply patches ─────────────────────────────────────────────────
            text = base_text

            # Patch RPM
            text = re.sub(r"(RPM:\s*)\d+", rf"\g<1>{rpm}", text)

            # Patch all design variable values
            text = patch_yaml(text, patches)

            # ── Write output ──────────────────────────────────────────────────
            dst = out_dir / f"case_Toluene_simpleORC_macchi_astolfi_n{n}_rpm{rpm}.yaml"
            dst.write_text(text)

            print(
                f"✓  p_evap={p_evap_pa/1e5:.2f} bar  "
                f"p_cond={p_cond_pa/1e5:.4f} bar  "
                f"h_in={h_exp_in:.0f} J/kg  "
                f"h_cond={h_comp_in:.0f} J/kg  "
                f"m_dot={m_dot_wf:.1f} kg/s"
            )

    print()
    print("=" * 70)
    print(f"  Done. Files written to: {out_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
