import os
import re
import sys
import shutil
import warnings
import traceback
import thermopt as th
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from datetime import datetime

# ---- importing post processing files:
sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent.parent / "shared_utilities")
)
# sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "shared_utilities"))
from cycle_report import generate_cycle_report, plot_heat_source_utilization
from plot_TQ_diagram import plot_TQ_diagram
from validation_checks import (
    run_validation_checks,
    get_validation_data,
    get_turbine_data,
    get_tq_summary_data,
)

# -----

warnings.filterwarnings("ignore", message="FigureCanvasAgg is non-interactive")


# ══════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════
#
#  Available modes:
#
#  "optimize"    — Run a single YAML through the full optimizer and
#                  post-processing pipeline (exergy, T-Q diagrams, graphs).
#                  Use this for one-off runs or debugging a specific case.
#                  → Set CONFIG_FILE.
#
#  "sweep"       — Run a fluid sweep across candidate fluids using one YAML
#                  as a template. Fluid name is substituted for each candidate;
#                  all other settings stay fixed. One results folder per fluid.
#                  → Set CONFIG_FILE and SWEEP_OUTPUT_DIR.
#
#  "pcond_sweep" — Working fluid screening sweep: condensing pressure vs
#                  system efficiency. Every fluid is run through a single
#                  wide-bounds YAML template whose design-variable bounds
#                  span both subcritical and transcritical operation — the
#                  optimizer is free to settle in either regime.
#
#                  For each fluid, two optimizations are run from different
#                  initial guesses (dual warm start): one from a subcritical
#                  starting point and one from a transcritical starting point.
#                  The result with the higher converged system efficiency is
#                  kept.  This prevents a gradient-based optimizer from being
#                  locked into one operating regime by the choice of initial
#                  guess alone.
#
#                  After convergence, the cycle is classified as subcritical
#                  or transcritical by comparing the optimized expander inlet
#                  pressure to the fluid's critical pressure:
#                    p_exp_in > p_crit  ->  transcritical
#                    p_exp_in <= p_crit ->  subcritical
#
#                  Results are saved to Excel and three scatter plots are
#                  produced (condensing pressure vs system efficiency):
#                    1. All fluids combined — subcritical coloured by chemical
#                       family, transcritical in teal.
#                    2. Subcritical fluids only, coloured by chemical family.
#                    3. Transcritical fluids only, coloured by chemical family.
#
#                  Set PCOND_USE_ALL_COOLPROP = True to screen every pure
#                  fluid in the CoolProp database (minus exclusions), or
#                  False to use only the curated PCOND_FLUIDS list.
#                  -> Set PCOND_CYCLE, PCOND_USE_ALL_COOLPROP,
#                     PCOND_DUAL_WARMSTART, PCOND_OUTPUT_DIR.
#
#  "k_sensitivity" — Sweep transcritical fluids across multiple reduced-pressure
#                   ratios k = P_max/P_crit (e.g. 1.05, 1.10, 1.20, 1.30, 1.50).
#                   Produces a scatter plot with one colour per k value, matching
#                   the sensitivity analysis in ORC fluid screening literature.
#                   → Set PCOND_CYCLE, PCOND_TC_FLUIDS, KSENS_K_VALUES.
#
#  "multistart"  — Latin Hypercube Sampling multistart optimisation. Generates
#                  MULTISTART_N_SAMPLES starting points spread across the design
#                  variable bounds, runs the optimizer from each, and saves the
#                  best converged result as a ready-to-use YAML. Failed/infeasible
#                  runs are logged and skipped. Use this to search for the global
#                  optimum of one or more basecase YAMLs.
#
#                  If MULTISTART_YAML_FILES contains ONE entry, the optimizer
#                  runs sequentially in the main process (easier to debug).
#                  If it contains MORE THAN ONE entry, all (entry × sample)
#                  tasks are dispatched to a parallel worker pool so multiple
#                  YAMLs are optimized simultaneously.
#
#                  → Set MULTISTART_YAML_FILES, MULTISTART_N_SAMPLES,
#                     MULTISTART_OUTPUT_DIR.
#
#  "parametric_study" — Nested n_stages × RPM sweep with automatic warm start
#                  management. Outer loop = n_stages (structural change), inner
#                  loop = RPM (simple value change). ThermOpt is instantiated
#                  ONCE; the config dict is modified in-place via the native API.
#                  One button press runs all 15 combinations automatically.
#                  → Set PARAMETRIC_BASE_YAML, PARAMETRIC_N_STAGES_LIST,
#                     PARAMETRIC_RPM_LIST, PARAMETRIC_OUTPUT_DIR.
#
#  "batch"        — Run optimization on multiple YAML files in sequence.
#                  Each file is optimized independently with full post-processing.
#                  Results are saved in each YAML's own output folder so nothing
#                  is overwritten. A summary table is printed at the end.
#                  → Set BATCH_YAML_FILES.
#
MODE = "multistart"  # "optimize" | "sweep" | "pcond_sweep" | "k_sensitivity" | "multistart" | "parametric_study" | "batch"


# ────────────────────────────────────────── optimize & sweep ─────────────────────────────────────────────────────────────────────────────────────────────────────────


# Used by: optimize, sweep
# Set this to the YAML file you want to run (optimize) or use as a
# template with the fluid name swapped out for each candidate (sweep).
# CONFIG_FILE = Path(__file__).with_name("case_Toluene_simpleORC_macchi_astolfi.yaml")
# CONFIG_FILE = Path(__file__).with_name("case_Toluene_simpleORC_macchi_astolfi_n5_rpm1000.yaml")

# CONFIG_FILE = Path(__file__).with_name("case_Toluene_simpleORC_macchi_astolfi_n5_rpm1500.yaml")
# CONFIG_FILE = Path(__file__).with_name("case_Toluene_simpleORC_macchi_astolfi_n5_rpm3000.yaml")

# CONFIG_FILE = Path(__file__).with_name("case_Toluene_simpleORC_macchi_astolfi_n4_rpm1000.yaml")
# CONFIG_FILE = Path(__file__).with_name("case_Toluene_recuperatedORC_macchi_astolfi_n4_rpm1000.yaml")
# CONFIG_FILE = Path(__file__).with_name("Toluene_DualPressureORC_basecase.yaml")
# CONFIG_FILE = Path(__file__).with_name("Toluene_DualPressureRecuperatedORC_basecase.yaml")
# CONFIG_FILE = Path(__file__).with_name("Cis-2-Butene_transcriticalORC_basecase.yaml")

# CONFIG_FILE = Path(__file__).with_name("Toluene_dp_twosource_basecase.yaml")
# CONFIG_FILE = Path(__file__).with_name("Toluene_simple_basecase.yaml")
# CONFIG_FILE = Path(__file__).with_name("Toluene_recup_basecase.yaml")
# CONFIG_FILE = Path(__file__).with_name("Toluene_dp_basecase.yaml")

CONFIG_FILE = Path(__file__).with_name("Toluene_recup_basecase.yaml")

# ──────────────────────────────────────────── batch ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────
# Used by: batch
# List of YAML files to optimize in sequence. Each file is run independently
# with full post-processing. Results go into each YAML's own output folder.
BATCH_YAML_FILES = [
    Path(__file__).with_name("Toluene_simple_basecase.yaml"),
    Path(__file__).with_name("Toluene_recup_basecase.yaml"),
    Path(__file__).with_name("Cis2Butene_tc_simple_basecase.yaml"),
    Path(__file__).with_name("Cis2Butene_tc_recup_basecase.yaml"),
    Path(__file__).with_name("Toluene_dp_basecase.yaml"),
    Path(__file__).with_name("Toluene_dp_recup_basecase.yaml"),
    Path(__file__).with_name("Toluene_dp_twosource_basecase.yaml"),
    Path(__file__).with_name("Toluene_dp_twosource_recup_basecase.yaml"),
]


# ── batch output root ────────────────────────────────────────────────────────
# Used by: batch
# All batch results are written here. Each run creates a new timestamped
# folder (e.g. batch_2025-06-01_14-30-00) inside this root, and inside that
# folder one subfolder per YAML file (named after the YAML stem).
BATCH_OUTPUT_ROOT = Path(
    r"C:\Users\asdis\OneDrive\Documents\Amanda\Master Thesis"
    r"\PYTHON_Code\ORC-project\Poetry - Working code"
    r"\thermopt_repo\projects\bjarnarflag"
    r"\Isentropic efficiency - Basecases\Simple ORC - basecase\results"
)


# Used by: sweep
# Folder where fluid sweep results are saved.
SWEEP_OUTPUT_DIR = "results/fluid_sweep_BASIC_ORC"


# ──────────────────────────────────────────── parametric_study: n_stages × RPM ──────────────────────────────────────────────────────────────────────────────────────
#
#  Used by: parametric_study
#
#  The base YAML is loaded ONCE as a Python dictionary.  The script then
#  manages everything in-place — no per-case YAML files are needed.
#
#  Loop structure (Roberto's recommendation):
#    Outer loop : n_stages  — structural change: adds/removes intermediate
#                             pressure design variables from the dict.
#    Inner loop : RPM       — simple value change: warm start transfers well.
#
#  Warm start strategy:
#    Within RPM loop (same n_stages):
#        Full converged x0_dict is written back into the dict value: fields
#        before the next run.  The cycle barely changes, so convergence is fast.
#    Across n_stages boundary:
#        Only non-intermediate-pressure variables are carried over.
#        New intermediate pressure variables get a first-guess from equal
#        pressure-ratio spacing, computed from the current warm inlet/outlet
#        pressures.
#
#  The base YAML should be your n=1 case (no intermediate pressures).
#  The script adds expander_intermediate_pressure_1 ... _(n-1) as needed.
#  n_stages and RPM are updated in the dict each run.

# Single base YAML — loaded once, then modified in Python as a dictionary.
# Use your n=1 file (no intermediate pressures yet).
PARAMETRIC_BASE_YAML = Path(__file__).with_name(
    "case_Toluene_simpleORC_macchi_astolfi_n1_rpm1000.yaml"
)

# n_stages outer loop — goes from 1 upward (variables are ADDED each step).
PARAMETRIC_N_STAGES_LIST = [1, 2, 3, 4, 5]

# RPM values for the inner loop.
PARAMETRIC_RPM_LIST = [1000, 1500, 3000]

# Where to save the combined CSV + Excel results.
PARAMETRIC_OUTPUT_DIR = Path(__file__).parent / "results" / "parametric_study"

# Prefix used for intermediate pressure design variable names.
# Variables are managed as  <prefix>_1, <prefix>_2, ... automatically.
PARAMETRIC_INTERP_PREFIX = "expander_intermediate_pressure"

# Bounds written into the YAML dict when a new intermediate pressure variable
# is created for the first time.  Set these to match the pressure range of
# your cycle (same units as your other pressure design variables — Pa).
PARAMETRIC_INTERP_P_MIN_EXPR = "0.9*$working_fluid.liquid_at_ambient_temperature.p"
PARAMETRIC_INTERP_P_MAX_EXPR = "0.8*$working_fluid.critical_point.p"

# Variable name in x0_dict for expander inlet/outlet pressure (used to compute
# equal pressure-ratio spacing when a new intermediate pressure variable is
# created). These match the design variable names in your n=1 YAML.
PARAMETRIC_EXPANDER_INLET_P_VAR = "expander_inlet_pressure"
PARAMETRIC_EXPANDER_OUTLET_P_VAR = "compressor_inlet_pressure"  # = p_cond


# ──────────────────────────────────────────── pcond_sweep & k_sensitivity settings ───────────────────────────────────────────────────────────────────────────────────
# Used by: pcond_sweep, k_sensitivity
#
# CYCLE TYPE — selects which YAML template to use:
#   "simple"      — no recuperator   (Toluene_simple_basecase.yaml)
#   "recuperated" — with recuperator  (Toluene_recup_basecase.yaml)
#
# Both YAMLs have wide enough bounds to cover subcritical and transcritical
# operation.  The optimizer decides which regime to operate in; the result is
# classified post-convergence by checking expander inlet pressure vs p_crit.
PCOND_CYCLE = "simple"  # "simple" | "recuperated"

# Single YAML template per cycle topology — used for ALL fluids in pcond_sweep.
PCOND_TEMPLATES = {
    "simple": Path(__file__).with_name("Toluene_simple_basecase.yaml"),
    "recuperated": Path(__file__).with_name("Toluene_recup_basecase.yaml"),
}

# Transcritical templates — used only by k_sensitivity mode (unchanged).
PCOND_TRANSCRITICAL_TEMPLATES = {
    "simple": Path(__file__).with_name("Cis2Butene_tc_simple_basecase.yaml"),
    "recuperated": Path(__file__).with_name("Cis2Butene_tc_recup_basecase.yaml"),
}

# ── Fluid selection ───────────────────────────────────────────────────────────
# Set PCOND_USE_ALL_COOLPROP = True  to run every pure fluid in the CoolProp
# database (except those listed in PCOND_EXCLUDE_FLUIDS).
# Set PCOND_USE_ALL_COOLPROP = False to use only the curated PCOND_FLUIDS dict.
PCOND_USE_ALL_COOLPROP = True

# Fluids to skip when PCOND_USE_ALL_COOLPROP = True.
# Water is a heat-source/sink fluid, not a working fluid.
# Air and typical pseudo-pure fluids are excluded by default.
PCOND_EXCLUDE_FLUIDS = {
    "Water",
    "Air",
    "Air.mix",
    "R404A",
    "R407C",
    "R410A",
    "R507A",  # mixtures that CoolProp may list
}

# T_crit threshold used ONLY for the warm-start heuristic inside
# _compute_fluid_warmstart — NOT for YAML routing.
# Fluids with T_crit below this get a transcritical-style warm start guess.
PCOND_WARMSTART_TCRIT_K = 202.15 + 273.15  # 475.30 K

# Turbine model used in the sweep.
#   "isentropic" — uses the efficiency value already set in the YAML (no YAML patching).
#   "astolfi"    — patches n_stages and RPM into the YAML and uses the
#                  Macchi-Astolfi correlation; Ns-invalid results are flagged.
PCOND_TURBINE_MODEL = "isentropic"  # "isentropic" | "astolfi"

# Only used when PCOND_TURBINE_MODEL = "astolfi"
PCOND_N_STAGES = 4
PCOND_RPM = 1000

# Folder where the pcond sweep results and plots are saved.
PCOND_OUTPUT_DIR = Path(__file__).parent / "results" / "pcond_vs_efficiency"

# Run each fluid twice — once from a subcritical warm start and once from a
# transcritical warm start — and keep whichever converged solution has the
# higher system efficiency.  This avoids the gradient-based optimizer being
# locked into one regime by the warm-start heuristic, at the cost of roughly
# doubling computation time.  Set to False to use the single T_crit heuristic.
PCOND_DUAL_WARMSTART = True

# Number of parallel worker processes for pcond_sweep.
# Each worker runs one fluid (both SC + TC attempts) independently.
# Recommended: 4 on a memory-constrained machine (<=16 GB RAM).
# Set to 1 to disable parallelism (useful for debugging).
PCOND_N_WORKERS = 4

# Debug fluid list for pcond_sweep.
# When non-empty, ONLY these fluids are run — ignores PCOND_USE_ALL_COOLPROP
# and PCOND_FLUIDS entirely.  Use this to test the mode quickly with 1-2
# fluids before committing to the full run.  Set PCOND_N_WORKERS = 1 as well
# for clean sequential output that is easy to read.
# Set to [] (empty list) for the normal full run.
PCOND_DEBUG_FLUIDS = []  # []  # e.g. ["Toluene", "R134a"]

# ──────────────────────────────────────────── k_sensitivity settings ─────────────────────────────────────────────────────────────────────────────────────────────────
# Used by: k_sensitivity
# Transcritical fluids for the reduced-pressure sensitivity analysis.
# Matches all fluids in PCOND_FLUIDS that have T_crit below the brine cutoff.
# The code will automatically skip any fluid whose T_crit is above the cutoff.
PCOND_TC_FLUIDS = {
    # Linear alkanes (transcritical at Bjarnarflag conditions)
    "n-Propane": "alkane",
    "IsoButane": "alkane",
    "n-Butane": "alkane",
    "Neopentane": "alkane",
    "Isopentane": "alkane",
    "Pentane": "alkane",
    # Alkenes
    "1-Butene": "alkene_alkyne",
    "IsoButene": "alkene_alkyne",
    "trans-2-Butene": "alkene_alkyne",
    "cis-2-Butene": "alkene_alkyne",
    # Alcohols / ketones (low T_crit)
    "Acetone": "alcohol_ketone",
    "Methanol": "alcohol_ketone",
    "Ethanol": "alcohol_ketone",
    # Refrigerants
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
    # Siloxanes (all below cutoff)
    "MDM": "siloxane",
    "MD2M": "siloxane",
    "MD3M": "siloxane",
    "MD4M": "siloxane",
    "D4": "siloxane",
    "D5": "siloxane",
    "D6": "siloxane",
    # Inorganics
    "Ammonia": "inorganic",
}

# Reduced pressure ratios k = P_max / P_crit to sweep.
KSENS_K_VALUES = [1.05, 1.10, 1.20, 1.30, 1.50]

# Folder where k-sensitivity results and plot are saved.
KSENS_OUTPUT_DIR = Path(__file__).parent / "results" / "k_sensitivity"

# ── Fluid candidates for pcond_sweep ────────────────────────────────────────
# Based on Tables 4.3 and 4.4 from the ORC fluid screening reference.
# Routing to sub/transcritical template happens automatically based on T_crit.
# T_crit cutoff = 202.15°C — fluids below go to transcritical template.
PCOND_FLUIDS = {
    # ── SUBCRITICAL (T_crit > 202.15°C) ─────────────────────────────────────
    # Aromatic hydrocarbons
    "Benzene": "aromatic",
    "Toluene": "aromatic",
    # Cycloalkanes
    "CycloPentane": "cycloalkane",
    "CycloHexane": "cycloalkane",
    # Linear alkanes (subcritical)
    "Isohexane": "alkane",
    "Hexane": "alkane",
    "Heptane": "alkane",
    "Octane": "alkane",
    "Nonane": "alkane",
    "Decane": "alkane",
    "n-Dodecane": "alkane",
    # Siloxanes (subcritical)
    "MM": "siloxane",
    # ── TRANSCRITICAL (T_crit < 202.15°C) ───────────────────────────────────
    # Linear alkanes (transcritical)
    "n-Propane": "alkane",
    "IsoButane": "alkane",
    "n-Butane": "alkane",
    "Neopentane": "alkane",
    "Isopentane": "alkane",
    "Pentane": "alkane",
    # Alkenes
    "1-Butene": "alkene_alkyne",
    "IsoButene": "alkene_alkyne",
    "trans-2-Butene": "alkene_alkyne",
    "cis-2-Butene": "alkene_alkyne",
    # Alcohols and ketones
    "DimethylEther": "alcohol_ketone",
    "Acetone": "alcohol_ketone",
    "Methanol": "alcohol_ketone",
    "Ethanol": "alcohol_ketone",
    # Refrigerants
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
    # Siloxanes (transcritical)
    "MDM": "siloxane",
    "MD2M": "siloxane",
    "MD3M": "siloxane",
    "MD4M": "siloxane",
    "D4": "siloxane",
    "D5": "siloxane",
    "D6": "siloxane",
    # Inorganics
    "Ammonia": "inorganic",
}

# ──────────────────────────────────────────── multistart: LHS settings ──────────────────────────────────────────────────────────────────────────────────────────────
# Used by: multistart
# List of YAML files to run LHS multistart on.
#   - ONE entry   → runs sequentially in the main process (easy to debug).
#   - MANY entries → dispatches all (entry × sample) tasks to a parallel
#                    worker pool so multiple YAMLs run simultaneously.
# Each entry is one of:
#   Path                          — fluid from YAML,  n = MULTISTART_N_SAMPLES
#   (Path, fluid_name)            — fluid override,   n = MULTISTART_N_SAMPLES
#   (Path, fluid_name, n_samples) — fluid override,   n = n_samples  (per-entry override)
#   (Path, None,       n_samples) — fluid from YAML,  n = n_samples  (per-entry override)
# Each YAML gets its own subfolder inside MULTISTART_OUTPUT_DIR.
MULTISTART_YAML_FILES = [
    (Path(__file__).with_name("Toluene_simple_basecase.yaml"), None, 2),
    (Path(__file__).with_name("Toluene_recup_basecase.yaml"), None, 2),
    (Path(__file__).with_name("Simple_basecase - transcritical.yaml"), None, 2),
    (Path(__file__).with_name("Recup_basecase - transcritical.yaml"), None, 2),
]

# MULTISTART_YAML_FILES = [
#     Path(__file__).with_name("Toluene_simple_basecase.yaml"),
#     Path(__file__).with_name("Toluene_recup_basecase.yaml"),
#     (Path(__file__).with_name("Simple_basecase - transcritical.yaml"),     None, 100),
#     (Path(__file__).with_name("Recup_basecase - transcritical.yaml"),      None, 100),
#     (Path(__file__).with_name("Toluene_dp_basecase.yaml"), None, 500),
#     (Path(__file__).with_name("Toluene_dp_recup_basecase.yaml"), None, 500),
#     Path(__file__).with_name("Toluene_dp_twosource_basecase.yaml"),
#     Path(__file__).with_name("Toluene_dp_twosource_recup_basecase.yaml"),
# ]

MULTISTART_N_SAMPLES = 300  # number of LHS starting points (50–100 recommended)
MULTISTART_OUTPUT_DIR = Path(__file__).parent / "results" / "multistart"


# Categories match Tables 4.3 and 4.4 from the ORC fluid screening reference:
#   Table 4.3 — Hydrocarbons:
#     aromatic      : Aromatic Hydrocarbons
#     cycloalkane   : Cycloalkanes
#     alkane        : Linear Alkanes
#     alkene_alkyne : Alkenes and Alkynes
#     alcohol_ketone: Alcohols and Ketones  (incl. DimethylEther)
#   Table 4.4 — Refrigerants, Siloxanes, and Inorganics:
#     refrigerant   : Refrigerant Fluids
#     siloxane      : Siloxanes
#     inorganic     : Other Inorganic Fluids
PCOND_CATEGORY_STYLE = {
    "aromatic": {"color": "#e74c3c", "marker": "o"},  # red circles
    "cycloalkane": {"color": "#8e44ad", "marker": "s"},  # purple squares
    "alkane": {"color": "#2980b9", "marker": "^"},  # blue triangles
    "alkene_alkyne": {"color": "#16a085", "marker": "P"},  # teal plus
    "alcohol_ketone": {"color": "#c0392b", "marker": "X"},  # dark red X
    "siloxane": {"color": "#f39c12", "marker": "D"},  # orange diamonds
    "refrigerant": {"color": "#27ae60", "marker": "v"},  # green inv-triangles
    "inorganic": {"color": "#7f8c8d", "marker": "*"},  # grey stars
    "other": {
        "color": "#bdc3c7",
        "marker": "o",
    },  # light grey — unknown CoolProp fluids
}

# Manual label offsets (dx, dy in points) for clustered fluids
PCOND_LABEL_OFFSETS = {
    # Subcritical aromatics
    "Benzene": (10, 8),
    "Toluene": (10, 8),
    # Subcritical cycloalkanes
    "CycloPentane": (10, 6),
    "CycloHexane": (-60, -6),
    # Subcritical alkanes
    "Isohexane": (10, -10),
    "Hexane": (10, -10),
    "Heptane": (10, 6),
    "Octane": (-45, 6),
    "Nonane": (10, 6),
    "Decane": (10, 6),
    "n-Dodecane": (10, 6),
    # Subcritical siloxane
    "MM": (10, 6),
    # Transcritical alkanes
    "n-Propane": (-8, -12),
    "n-Butane": (10, 8),
    "IsoButane": (-50, -8),
    "Isopentane": (-55, 8),
    "Pentane": (10, -10),
    "Neopentane": (10, -10),
    # Transcritical alkenes
    "1-Butene": (10, 8),
    "IsoButene": (10, -10),
    "cis-2-Butene": (-70, 8),
    "trans-2-Butene": (10, 8),
    # Alcohols / ketones
    "DimethylEther": (8, -10),
    "Acetone": (10, 8),
    "Methanol": (-50, 8),
    "Ethanol": (10, -8),
    # Transcritical siloxanes
    "MDM": (10, 6),
    "MD2M": (10, -8),
    "MD3M": (10, 4),
    "MD4M": (10, 4),
    "D4": (-8, 8),
    "D5": (10, -8),
    "D6": (10, 4),
    # Refrigerants
    "R125": (8, 8),
    "R218": (8, -10),
    "R143a": (8, 8),
    "R32": (8, -10),
    "R1234yf": (8, -10),
    "R134a": (8, 8),
    "R227EA": (8, 6),
    "R161": (8, -10),
    "R1234zeE": (8, 8),
    "R152A": (8, 8),
    "R236FA": (-45, -8),
    "R236EA": (8, -10),
    "R245fa": (-60, -6),
    "R365MFC": (8, 8),
    # Inorganics
    "Ammonia": (10, 8),
}


# ══════════════════════════════════════════════════════════════════════
#  HELPER: APPEND EXTRA SHEETS TO POST-PROCESSING EXCEL
# ══════════════════════════════════════════════════════════════════════


def _append_extra_sheets(excel_path, cycle, config_file=None):
    """
    Append three extra sheets to the post_processing_results.xlsx:
      - validation_checks  : energy balance + constraint activity + phase check
      - turbine_summary    : only written when astolfi-stacking is active
      - TQ_summary         : heat duty, inlet/outlet temps and pinch for each HX
    """
    import openpyxl

    validation_rows = get_validation_data(cycle, config_file=config_file)
    turbine_rows = get_turbine_data(cycle)  # None if isentropic mode
    tq_rows = get_tq_summary_data(cycle)

    wb = openpyxl.load_workbook(excel_path)

    def _safe(val):
        """Convert any non-basic type to a plain Python type for openpyxl.
        Handles numpy scalars, CoolProp Array types, and anything else
        that isn't already a str/int/float/bool/None."""
        if val is None:
            return None
        if isinstance(val, (str, int, float, bool)):
            return val
        # Try .item() first (numpy scalars)
        if hasattr(val, "item"):
            try:
                return val.item()
            except Exception:
                pass
        # Fall back to float conversion (CoolProp Array, etc.)
        try:
            return float(val)
        except Exception:
            pass
        # Last resort: stringify
        return str(val)

    def _write_sheet(wb, sheet_name, rows):
        if sheet_name in wb.sheetnames:
            del wb[sheet_name]
        ws = wb.create_sheet(sheet_name)
        ws.append(["Parameter", "Value"])
        for row in rows:
            ws.append([_safe(v) for v in row])
        # Auto-width columns
        for col in ws.columns:
            max_len = max((len(str(c.value)) for c in col if c.value), default=10)
            ws.column_dimensions[col[0].column_letter].width = min(max_len + 4, 60)

    _write_sheet(wb, "validation_checks", validation_rows)
    if turbine_rows is not None:
        _write_sheet(wb, "turbine_summary", turbine_rows)
    _write_sheet(wb, "TQ_summary", tq_rows)

    wb.save(excel_path)
    print(
        f"    ✓ Extra sheets added: validation_checks"
        + (", turbine_summary" if turbine_rows is not None else "")
        + ", TQ_summary"
    )


# ══════════════════════════════════════════════════════════════════════
#  MODE 1: SINGLE-FLUID OPTIMIZATION
# ══════════════════════════════════════════════════════════════════════
def run_optimize(config_file, batch=False, dest_dir=None):
    """Run optimization + full post-processing for a single YAML config.

    Parameters
    ----------
    batch : bool
        When True, the realtime cycle plot is skipped so the run does not
        pause waiting for the window to be closed between YAML files.
    dest_dir : Path or str, optional
        If provided, the entire contents of cycle.out_dir (thermopt results +
        post-processing) are copied here after all processing is done.
        Used by batch mode to redirect results into the batch folder structure.
    """

    th.print_package_info()

    cycle = th.ThermodynamicCycleOptimization(config_file)
    if not batch:
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
    if "mass_flow_heating_fluid" in ea:
        print(
            f"  Well (heating fluid)               :  {ea['mass_flow_heating_fluid']:.2f} kg/s"
        )
    elif "mass_flow_brine_hp" in ea:
        print(
            f"  HP brine (well 2)                  :  {ea['mass_flow_brine_hp']:.2f} kg/s"
        )
        print(
            f"  LP brine (wells 1/3/4)             :  {ea['mass_flow_brine_lp']:.2f} kg/s"
        )
    print(
        f"  Working fluid                      :  {ea['mass_flow_working_fluid']:.2f} kg/s"
    )
    print(
        f"  Cooling fluid                      :  {ea['mass_flow_cooling_fluid']:.2f} kg/s"
    )
    print("=" * 76 + "\n")

    # ── Post-processing ───────────────────────────────────────────
    graph_dir = os.path.join(cycle.out_dir, "post_processing")
    os.makedirs(graph_dir, exist_ok=True)

    # Exergy analysis
    exergy = generate_cycle_report(cycle)
    exergy.print_summary()
    pp_excel = os.path.join(graph_dir, "post_processing_results.xlsx")
    exergy.to_excel(pp_excel)
    _append_extra_sheets(pp_excel, cycle, config_file)
    exergy.plot_exergy_destruction(savefig=os.path.join(graph_dir, "exergy_bar.png"))

    # T-Q diagrams — topology-aware
    components = cycle.problem.cycle_data["components"]
    is_dual_pressure = "hp_evaporator" in components

    if is_dual_pressure:
        # Dual-pressure topology: three brine-side heat exchangers
        plot_TQ_diagram(cycle, "hp_evaporator", output_dir=graph_dir)
        plot_TQ_diagram(cycle, "lp_evaporator", output_dir=graph_dir)
        plot_TQ_diagram(cycle, "preheater", output_dir=graph_dir)
    else:
        # Simple / recuperated topology: single heater
        plot_TQ_diagram(cycle, "heater", output_dir=graph_dir)

    plot_TQ_diagram(cycle, "cooler", output_dir=graph_dir)

    # Plot recuperator T-Q diagram only if it is active (heat flow > 1 kW)
    recup = components.get("recuperator", None)
    if (
        recup is not None
        and abs(float(recup.get("energy_analysis", {}).get("Q_hot", 0))) > 1000
    ):
        plot_TQ_diagram(cycle, "recuperator", output_dir=graph_dir)

    # Summary
    print("\n" + "=" * 60)
    print("  ALL RESULTS SAVED")
    print("=" * 60)
    print(f"  Thermopt results : {cycle.out_dir}")
    print(f"  Post-processing  : {graph_dir}")
    print()
    for f in sorted(os.listdir(graph_dir)):
        print(f"    \u2713 {f}")
    print("=" * 60 + "\n")

    # ── Copy to batch destination folder (batch mode only) ────────────
    if dest_dir is not None:
        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copytree(cycle.out_dir, dest_dir, dirs_exist_ok=True)
        print(f"  ✓ Results copied to: {dest_dir}\n")


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


def _extract_results(cycle, config_file, n_stages, RPM):
    """
    Extract a fully detailed flat result dict from a converged cycle.

    System level
    ------------
    eta_system, eta_cycle, W_net_kW, W_turbine_kW, W_pump_kW,
    W_aux_kW, Q_in_kW, Q_available_kW, heat_utilization,
    m_dot_wf_kg_s, m_dot_well_kg_s, m_dot_cooling_kg_s,
    brine_exit_T_C, heat_sink_exit_T_C,
    heater_pinch_K, cooler_pinch_K,
    expander_p_in_bar, expander_p_out_bar,
    expander_T_in_C, expander_T_out_C,
    expander_h_in_kJ_kg, expander_h_out_kJ_kg,
    compressor_p_in_bar, compressor_T_in_C,
    Dh_is_total_kJ_kg, eta_turbine_overall,
    SP_total, Vr_total, Ns_total,
    SP_clamped_any, Vr_over_any, Ns_oor_any,
    splitting_method

    Per stage  (s1_*, s2_*, … up to s5_*)
    -------------------------------------
    s{i}_p_in_bar, s{i}_p_out_bar,
    s{i}_T_in_C, s{i}_T_out_is_C, s{i}_T_out_real_C,
    s{i}_h_in_kJ_kg, s{i}_h_out_is_kJ_kg, s{i}_h_out_real_kJ_kg,
    s{i}_s_in_kJ_kgK,
    s{i}_Dh_is_kJ_kg, s{i}_work_kJ_kg,
    s{i}_SP, s{i}_SP_eval, s{i}_SP_clamped,
    s{i}_Vr, s{i}_Vr_over_limit,
    s{i}_Ns, s{i}_Ns_out_of_range,
    s{i}_eta
    """
    data = cycle.problem.cycle_data
    components = data["components"]
    energy = data["energy_analysis"]
    config_name = Path(config_file).stem

    result = {
        "config": config_name,
        "n_stages": n_stages,
        "RPM": RPM,
        "converged": True,
        "ns_invalid": False,
    }

    # ── System-level energy metrics ───────────────────────────────────
    result["eta_system"] = energy.get("system_efficiency", None)
    result["eta_cycle"] = energy.get("cycle_efficiency", None)
    result["W_net_kW"] = energy.get("net_system_power", 0) / 1e3
    components = data["components"]
    result["W_turbine_kW"] = components["expander"]["energy_analysis"]["power"] / 1e3
    result["W_pump_kW"] = components["compressor"]["energy_analysis"]["power"] / 1e3
    result["W_aux_kW"] = (
        components["heat_source_pump"]["energy_analysis"]["power"]
        + components["heat_sink_pump"]["energy_analysis"]["power"]
    ) / 1e3
    result["Q_in_kW"] = energy.get("heater_heat_flow", 0) / 1e3
    result["Q_available_kW"] = energy.get("heater_heat_flow_max", 0) / 1e3
    result["heat_utilization"] = (
        result["Q_in_kW"] / result["Q_available_kW"]
        if result["Q_available_kW"] > 0
        else None
    )
    result["m_dot_wf_kg_s"] = energy.get("mass_flow_working_fluid", None)
    result["m_dot_well_kg_s"] = energy.get("mass_flow_heating_fluid", None)
    result["m_dot_cooling_kg_s"] = energy.get("mass_flow_cooling_fluid", None)

    # ── Heat exchanger temperatures & pinch ───────────────────────────
    heater = components.get("heater", {})
    cooler = components.get("cooler", {})

    try:
        result["brine_exit_T_C"] = heater["hot_side"]["state_out"].T - 273.15
    except (KeyError, AttributeError, TypeError):
        result["brine_exit_T_C"] = None

    try:
        result["heat_sink_exit_T_C"] = cooler["cold_side"]["state_out"].T - 273.15
    except (KeyError, AttributeError, TypeError):
        result["heat_sink_exit_T_C"] = None

    try:
        dT_heater = heater.get("temperature_difference")
        result["heater_pinch_K"] = float(
            dT_heater.min() if hasattr(dT_heater, "min") else min(dT_heater)
        )
    except (TypeError, ValueError, AttributeError):
        result["heater_pinch_K"] = None

    try:
        dT_cooler = cooler.get("temperature_difference")
        result["cooler_pinch_K"] = float(
            dT_cooler.min() if hasattr(dT_cooler, "min") else min(dT_cooler)
        )
    except (TypeError, ValueError, AttributeError):
        result["cooler_pinch_K"] = None

    # ── Expander inlet / outlet states ────────────────────────────────
    exp = components.get("expander", {})
    data_out = exp.get("data_out", {})

    try:
        result["expander_p_in_bar"] = exp["state_in"].p / 1e5
        result["expander_T_in_C"] = exp["state_in"].T - 273.15
        result["expander_h_in_kJ_kg"] = exp["state_in"].h / 1e3
    except (KeyError, AttributeError):
        result["expander_p_in_bar"] = None
        result["expander_T_in_C"] = None
        result["expander_h_in_kJ_kg"] = None

    try:
        result["expander_p_out_bar"] = exp["state_out"].p / 1e5
        result["expander_T_out_C"] = exp["state_out"].T - 273.15
        result["expander_h_out_kJ_kg"] = exp["state_out"].h / 1e3
    except (KeyError, AttributeError):
        result["expander_p_out_bar"] = None
        result["expander_T_out_C"] = None
        result["expander_h_out_kJ_kg"] = None

    result["Dh_is_total_kJ_kg"] = exp.get("isentropic_work", 0) / 1e3
    result["eta_turbine_overall"] = exp.get("efficiency", None)

    # ── Overall turbine non-dimensional parameters ────────────────────
    result["SP_total"] = data_out.get("size_parameter", None)
    result["Vr_total"] = data_out.get("volume_ratio", None)
    result["Ns_total"] = data_out.get("specific_speed", None)
    result["SP_clamped_any"] = data_out.get("size_parameter_clamped", False)
    result["Vr_over_any"] = data_out.get("volume_ratio_out_of_range", False)
    result["Ns_oor_any"] = data_out.get("specific_speed_out_of_range", False)
    result["splitting_method"] = data_out.get("splitting_method", "single_stage")

    # ── Compressor inlet ──────────────────────────────────────────────
    comp = components.get("compressor", {})
    try:
        result["compressor_p_in_bar"] = comp["state_in"].p / 1e5
        result["compressor_T_in_C"] = comp["state_in"].T - 273.15
    except (KeyError, AttributeError):
        result["compressor_p_in_bar"] = None
        result["compressor_T_in_C"] = None

    # ── Per-stage detailed breakdown ──────────────────────────────────
    stage_data = data_out.get("stage_data", [])
    for i, s in enumerate(stage_data, 1):
        pfx = f"s{i}_"

        result[pfx + "p_in_bar"] = (
            s["p_in"] / 1e5 if s.get("p_in") is not None else None
        )
        result[pfx + "p_out_bar"] = (
            s["p_out"] / 1e5 if s.get("p_out") is not None else None
        )

        result[pfx + "T_in_C"] = (
            s["T_in"] - 273.15 if s.get("T_in") is not None else None
        )
        result[pfx + "T_out_is_C"] = (
            s["T_out_is"] - 273.15 if s.get("T_out_is") is not None else None
        )
        result[pfx + "T_out_real_C"] = (
            s["T_out_real"] - 273.15 if s.get("T_out_real") is not None else None
        )

        result[pfx + "h_in_kJ_kg"] = (
            s["h_in"] / 1e3 if s.get("h_in") is not None else None
        )
        result[pfx + "h_out_is_kJ_kg"] = (
            s["h_out_is"] / 1e3 if s.get("h_out_is") is not None else None
        )
        result[pfx + "h_out_real_kJ_kg"] = (
            s["h_out_real"] / 1e3 if s.get("h_out_real") is not None else None
        )

        result[pfx + "s_in_kJ_kgK"] = (
            s["s_in"] / 1e3 if s.get("s_in") is not None else None
        )

        result[pfx + "Dh_is_kJ_kg"] = (
            s["Dh_is"] / 1e3 if s.get("Dh_is") is not None else None
        )
        result[pfx + "work_kJ_kg"] = (
            s["work"] / 1e3 if s.get("work") is not None else None
        )

        result[pfx + "SP"] = s.get("SP")
        result[pfx + "SP_eval"] = s.get("SP_eval")
        result[pfx + "SP_clamped"] = s.get("SP_clamped", False)

        result[pfx + "Vr"] = s.get("Vr")
        result[pfx + "Vr_over_limit"] = s.get("Vr_over_limit", False)

        result[pfx + "Ns"] = s.get("Ns")
        result[pfx + "Ns_out_of_range"] = s.get("Ns_out_of_range", False)

        result[pfx + "eta"] = s.get("eta_stage")

    # ── List-form stage arrays ──
    result["Ns_out_of_range"] = data_out.get("specific_speed_out_of_range", False)
    result["stage_Ns"] = [s.get("Ns") for s in stage_data]
    result["stage_Ns_flags"] = [s.get("Ns_out_of_range", False) for s in stage_data]
    result["stage_Vr"] = [s.get("Vr") for s in stage_data]
    result["stage_eta"] = [s.get("eta_stage") for s in stage_data]
    result["stage_SP"] = [s.get("SP") for s in stage_data]
    result["stage_Dh_is_kJ_kg"] = [
        s["Dh_is"] / 1e3 if s.get("Dh_is") is not None else None for s in stage_data
    ]
    result["stage_p_in_bar"] = [
        s["p_in"] / 1e5 if s.get("p_in") is not None else None for s in stage_data
    ]
    result["stage_p_out_bar"] = [
        s["p_out"] / 1e5 if s.get("p_out") is not None else None for s in stage_data
    ]

    # ── Recuperator (if present) ──────────────────────────────────────
    if "recuperator" in components:
        result["Q_recuperator_kW"] = energy.get("recuperator_heat_flow", 0) / 1e3

    return result


def _check_ns_validity(result):
    """
    Check whether any stage had Ns outside [0.045, 0.20].

    Uses the specific_speed_out_of_range flag set by the Astolfi component,
    which is True if ANY single stage exceeded the regression bounds.
    Drills into per-stage data to report exactly which stages were invalid.

    Returns a human-readable problem string, or None if everything is valid.
    """
    problems = []

    # Detect which expanders are present
    expanders = []
    if "Ns_out_of_range" in result:
        expanders.append(("", "expander"))
    for prefix in ["HP", "LP"]:
        if f"{prefix}_Ns_out_of_range" in result:
            expanders.append((f"{prefix}_", prefix))

    for key_prefix, label in expanders:
        if not result.get(f"{key_prefix}Ns_out_of_range", False):
            continue  # all stages valid for this expander

        # Identify which specific stages were out of range
        stage_ns = result.get(f"{key_prefix}stage_Ns", [])
        stage_flags = result.get(f"{key_prefix}stage_Ns_flags", [])

        bad_stages = []
        for i, (ns, flag) in enumerate(zip(stage_ns, stage_flags)):
            if flag:
                ns_str = f"{ns:.3f}" if ns is not None else "?"
                bad_stages.append(f"stage {i+1} (Ns={ns_str})")

        if bad_stages:
            problems.append(f"{label}: {', '.join(bad_stages)}")
        else:
            # Top-level flag set but no stage detail available
            problems.append(f"{label}: Ns out of range (no stage detail available)")

    return "; ".join(problems) if problems else None


def _run_single(config_file, n_stages, RPM):
    """Run one optimization for a specific YAML (already correct for n_stages) at a given RPM.
    Only RPM is patched — the YAML structure (intermediate pressures, constraints) is preserved.
    """
    yaml_text = Path(config_file).read_text()
    yaml_text = re.sub(r"(RPM:\s*)\d+", rf"\g<1>{RPM}", yaml_text)

    config_dir = Path(config_file).parent
    tmp_path = config_dir / f"_tmp_run_n{n_stages}_rpm{RPM}.yaml"

    try:
        tmp_path.write_text(yaml_text)

        cycle = th.ThermodynamicCycleOptimization(str(tmp_path))
        cycle.run_optimization()

        # Check convergence — read the flag ThermOpt/pysolver_view actually set
        converged = False
        try:
            converged = bool(cycle.solver.success)
        except AttributeError:
            pass

        result = _extract_results(cycle, config_file, n_stages, RPM)
        result["converged"] = converged
        return result

    except Exception as e:
        print(f"    ✗ FAILED: {e}")
        traceback.print_exc()
        return {
            "config": Path(config_file).stem,
            "n_stages": n_stages,
            "RPM": RPM,
            "converged": False,
            "error": str(e),
        }
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


#  MODE 3b: PARAMETRIC STUDY — n_stages × RPM  (native thermopt API)
# ══════════════════════════════════════════════════════════════════════


def _build_ordering_constraints(n_stages, prefix, normalize=True):
    """
    Return the list of pressure ordering constraint dicts for n_stages.

    For n=1: empty list (no intermediate pressures).
    For n=2: two constraints  p_in > p1 > p_out
    For n=3: three constraints p_in > p1 > p2 > p_out
    ... and so on.

    The pattern from the YAML is:
      $components.expander.state_in.p / $variables.<prefix>_1           > 1
      $variables.<prefix>_{i} / $variables.<prefix>_{i+1}               > 1  (middle)
      $variables.<prefix>_{n-1} / $components.expander.state_out.p      > 1
    """
    constraints = []
    n = n_stages - 1  # number of intermediate pressures

    if n <= 0:
        return constraints

    for i in range(1, n + 2):  # i = 1 … n+1  (edges of the chain)
        if i == 1:
            lhs = "$components.expander.state_in.p"
            rhs = f"$variables.{prefix}_1"
        elif i == n + 1:
            lhs = f"$variables.{prefix}_{n}"
            rhs = "$components.expander.state_out.p"
        else:
            lhs = f"$variables.{prefix}_{i-1}"
            rhs = f"$variables.{prefix}_{i}"

        constraints.append(
            {
                "variable": f"{lhs} / {rhs}",
                "type": ">",
                "value": 1.0,
                "normalize": normalize,
            }
        )

    return constraints


def _is_ordering_constraint(c, prefix):
    """Return True if constraint c is an intermediate pressure ordering constraint."""
    v = c.get("variable", "")
    return (
        f"$variables.{prefix}_" in v
        or ("$components.expander.state_in.p" in v and f"$variables.{prefix}_" in v)
        or (f"$variables.{prefix}_" in v and "$components.expander.state_out.p" in v)
    )


def run_parametric_nstages_rpm(
    base_yaml,
    n_stages_list,
    rpm_list,
    output_dir,
    interp_var_prefix="expander_intermediate_pressure",
    interp_p_min_expr="0.9*$working_fluid.liquid_at_ambient_temperature.p",
    interp_p_max_expr="0.8*$working_fluid.critical_point.p",
    expander_inlet_p_var="expander_inlet_pressure",
    expander_outlet_p_var="compressor_inlet_pressure",
    rpm_config_path="problem_formulation.fixed_parameters.expander.RPM",
    n_stages_config_path="problem_formulation.fixed_parameters.expander.n_stages",
):
    """
    Nested parametric sweep over turbine stage count and shaft speed,
    using thermopt's native Python API — no temp YAML files.

    ThermodynamicCycleOptimization is instantiated ONCE from the base YAML.
    The live cycle.config dict is then modified in-place via the native API,
    and cycle.load_config() rebuilds the problem and solver each time the
    structure changes.

    Native thermopt methods used
    ----------------------------
    cycle.load_config(cycle.config)
        Rebuilds problem + solver from the current dict state.
    cycle.set_config_value(path, value, reload=True/False)
        Updates a single nested value by dot-separated path.
    cycle.set_constraint(variable, type, value)
        Adds or updates a named constraint.
    cycle.problem.x0_dict
        Converged variable values after run_optimization().

    Loop structure
    --------------
    Outer loop : n_stages
        Structural changes managed here:
        1. Add/remove intermediate pressure design variables from
           cycle.config["problem_formulation"]["design_variables"].
        2. Rebuild the ordering constraints (p_in > p1 > p2 > p_out)
           to match the new stage count — old ones removed, new ones added.
        3. cycle.load_config() to rebuild problem + solver.
        4. Apply cycle-level warm starts from the previous n_stages.

    Inner loop : RPM
        cycle.set_config_value(rpm_config_path, rpm) — triggers one reload.
        Converged x0_dict written back as warm starts before the next run.

    What does NOT change with n_stages
    -----------------------------------
    The array-level Vr and Ns constraints in the YAML
    ($components.expander.data_out.stage_Vr and stage_Ns) automatically
    apply to however many stages are present — they do not need to be
    added or removed.  Only the ordering constraints between intermediate
    pressures change.

    Warm start strategy
    -------------------
    Within RPM loop  : full x0_dict carried forward.
    Across n_stages  : cycle-level vars (non-intermediate-pressure) carried
                       forward.  New intermediate pressure vars get an
                       equal-pressure-ratio initial guess from the current
                       warm inlet/outlet pressures.
    On failure       : warm_vars unchanged — next run starts from last GOOD
                       solution.
    """

    base_yaml = Path(base_yaml)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total = len(n_stages_list) * len(rpm_list)
    run_n = 0

    print("=" * 76)
    print("  PARAMETRIC STUDY — n_stages × RPM  (native thermopt API)")
    print("=" * 76)
    print(f"  Base YAML     : {base_yaml.name}")
    print(f"  n_stages      : {n_stages_list}")
    print(f"  RPM           : {rpm_list}")
    print(f"  Total runs    : {total}")
    print(f"  Output        : {output_dir}")
    print()
    print("  Warm start strategy:")
    print("    RPM loop     : full x0_dict written back via set_config_value()")
    print(f"    n_stages loop: cycle vars carried; '{interp_var_prefix}_*' recomputed")
    print("  Constraints managed automatically:")
    print("    Ordering (p_in > p1 > ... > p_out): rebuilt each n_stages step")
    print("    Vr/Ns array constraints: unchanged (apply to whole stage array)")
    print("=" * 76 + "\n")

    # ── Instantiate once — only file read in the entire sweep ────────
    cycle = th.ThermodynamicCycleOptimization(str(base_yaml))
    dvars = cycle.config["problem_formulation"]["design_variables"]

    results = []
    warm_vars = {}  # {var_name: float} — updated after every converged run
    transition_vars = (
        {}
    )  # warm_vars saved from the first RPM run of each n_stages block
    # Used for the n_stages boundary transition — avoids carrying
    # over a degenerate high-RPM solution into the next stage count

    # ══════════════════════════════════════════════════════════════════
    # OUTER LOOP — n_stages
    # ══════════════════════════════════════════════════════════════════
    for n_stages in n_stages_list:

        print(f"\n{'─'*76}")
        print(f"  n_stages = {n_stages}")
        print(f"{'─'*76}")

        n_needed = max(0, n_stages - 1)

        # ── 1. Sync intermediate pressure design variables ─────────────
        existing = sorted(
            [k for k in dvars if k.startswith(interp_var_prefix + "_")],
            key=lambda k: int(k.rsplit("_", 1)[-1]),
        )

        # Remove variables (stepping to fewer stages)
        while len(existing) > n_needed:
            var_name = existing.pop()
            del dvars[var_name]
            print(f"    - removed design var  {var_name}")

        # Add variables (stepping to more stages)
        for i in range(len(existing) + 1, n_needed + 1):
            var_name = f"{interp_var_prefix}_{i}"
            # Bounds: copy from last existing variable, else use defaults
            if existing:
                prev = dvars[existing[-1]]
                bmin = prev["min"]
                bmax = prev["max"]
            else:
                bmin = interp_p_min_expr
                bmax = interp_p_max_expr
            # Temporary placeholder — will be overwritten by equal-ratio pass below
            dvars[var_name] = {"value": 1e5, "min": bmin, "max": bmax}
            existing.append(var_name)
            print(f"    + added design var  {var_name}")

        # Recompute ALL intermediate pressure values from equal pressure-ratio
        # spacing using transition_vars (first RPM of previous n_stages block).
        # This is done for every variable — both newly added ones AND any that
        # carried over from the previous stage count — so that the full set is
        # always internally consistent and guaranteed ordered p_in > p1 > ... > p_out.
        source = transition_vars if transition_vars else warm_vars
        p_in = source.get(expander_inlet_p_var)
        p_out = source.get(expander_outlet_p_var)
        if p_in is None or p_out is None or not (p_in > p_out > 0):
            p_in = warm_vars.get(expander_inlet_p_var, 3e6)
            p_out = warm_vars.get(expander_outlet_p_var, 0.1e5)

        for idx, var_name in enumerate(existing, 1):
            # Equal geometric spacing: p_i = p_in * (p_out/p_in)^(i/n_stages)
            p_guess = float(p_in * (p_out / p_in) ** (idx / n_stages))
            dvars[var_name]["value"] = p_guess
            print(
                f"    ↺ set {var_name} = {p_guess:.4e} Pa  " f"({p_guess/1e5:.4f} bar)"
            )

        # ── 2. Rebuild ordering constraints ───────────────────────────
        # Remove all existing ordering constraints, then add the correct
        # set for the current n_stages.
        all_constraints = cycle.config["problem_formulation"].get("constraints", [])

        # Keep everything that is NOT an ordering constraint
        non_ordering = [
            c
            for c in all_constraints
            if not _is_ordering_constraint(c, interp_var_prefix)
        ]

        # Build the correct ordering constraints for this n_stages
        new_ordering = _build_ordering_constraints(
            n_stages, interp_var_prefix, normalize=True
        )

        cycle.config["problem_formulation"]["constraints"] = non_ordering + new_ordering

        n_ord = len(new_ordering)
        if n_ord:
            print(
                f"    Ordering constraints: {n_ord} "
                f"(p_in > p1 {'> ... ' if n_stages > 2 else ''}> p_out)"
            )
        else:
            print(f"    Ordering constraints: none (n_stages=1)")

        # ── 3. Update n_stages in fixed parameters ─────────────────────
        cycle.set_config_value(n_stages_config_path, n_stages, reload=False)

        # ── 4. Apply cycle-level warm starts (skip intermediate pressures)
        #       Use transition_vars (from first RPM of previous n_stages block)
        #       so the n_stages boundary always starts from the least
        #       Ns-constrained solution, not a potentially degenerate high-RPM one.
        source_vars = transition_vars if transition_vars else warm_vars
        n_carried = 0
        for k, v in source_vars.items():
            if k.startswith(interp_var_prefix + "_"):
                continue
            if k in dvars:
                cycle.set_config_value(
                    f"problem_formulation.design_variables.{k}.value",
                    float(v),
                    reload=False,
                )
                n_carried += 1
        if n_carried:
            src_label = (
                "transition_vars (first RPM)" if transition_vars else "warm_vars"
            )
            print(
                f"    Cycle-level warm start: {n_carried} vars carried over "
                f"(source: {src_label})"
            )

        # ── 5. Rebuild problem + solver with updated dict ──────────────
        cycle.load_config(cycle.config)

        # ══════════════════════════════════════════════════════════════
        # INNER LOOP — RPM
        # ══════════════════════════════════════════════════════════════
        for rpm in rpm_list:
            run_n += 1
            print(
                f"\n  [{run_n:>2}/{total}] n_stages={n_stages}, RPM={rpm:>5} ... ",
                end="",
                flush=True,
            )

            # 6. Update RPM — triggers a reload internally
            cycle.set_config_value(rpm_config_path, rpm)

            # 7. Run optimizer
            try:
                cycle.run_optimization()

                # 8. Check convergence
                converged = False
                try:
                    converged = bool(cycle.solver.success)
                except AttributeError:
                    pass

                # 9. Converged → read x0_dict and write back as warm starts
                #    Not converged → leave warm_vars unchanged (start from
                #    last good solution next time)
                if converged:
                    warm_vars = {k: float(v) for k, v in cycle.problem.x0_dict.items()}
                    # Save transition_vars from the first RPM run of each n_stages
                    # block. This is the least Ns-constrained solution and gives the
                    # best thermodynamic starting point for the next stage count.
                    if rpm == rpm_list[0]:
                        transition_vars = dict(warm_vars)
                    # Write warm start values into the dict without reloading —
                    # the next iteration's set_config_value(rpm_config_path, rpm)
                    # will trigger the single reload we need.
                    for k, v in warm_vars.items():
                        if k in dvars:
                            cycle.set_config_value(
                                f"problem_formulation.design_variables.{k}.value",
                                v,
                                reload=False,
                            )

                # 10. Extract and store results
                result = _extract_results(cycle, base_yaml, n_stages, rpm)
                result["converged"] = converged

                if converged:
                    ns_problem = _check_ns_validity(result)
                    if ns_problem:
                        result["ns_invalid"] = True
                        result["skip_reason"] = f"Ns out of range: {ns_problem}"
                        eta = result.get("eta_system", 0) or 0
                        W = result.get("W_net_kW", 0) or 0
                        print(
                            f"⚠  η_sys={eta*100:.2f}%  W={W:.0f} kW  "
                            f"[Ns OOR: {ns_problem}]"
                        )
                    else:
                        result["ns_invalid"] = False
                        eta = result.get("eta_system", 0) or 0
                        W = result.get("W_net_kW", 0) or 0
                        print(f"✓  η_sys={eta*100:.2f}%  W={W:.0f} kW")
                else:
                    result["ns_invalid"] = False
                    print("✗  did not converge  (warm start unchanged)")

            except Exception as e:
                print(f"✗  FAILED: {e}")
                traceback.print_exc()
                result = {
                    "config": base_yaml.stem,
                    "n_stages": n_stages,
                    "RPM": rpm,
                    "converged": False,
                    "ns_invalid": False,
                    "error": str(e),
                }

            results.append(result)

    # ── Save combined results ─────────────────────────────────────────
    df = (
        pd.DataFrame(results)
        .sort_values(["n_stages", "RPM"], ascending=True, na_position="last")
        .reset_index(drop=True)
    )

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    csv_path = output_dir / f"parametric_nstages_rpm_{timestamp}.csv"
    xlsx_path = output_dir / f"parametric_nstages_rpm_{timestamp}.xlsx"

    df.to_csv(csv_path, index=False)
    df.to_excel(xlsx_path, index=False, sheet_name="Results")

    # ── Print summary ─────────────────────────────────────────────────
    print("\n\n" + "=" * 76)
    print("  PARAMETRIC STUDY — RESULTS SUMMARY")
    print("=" * 76)

    display_cols = ["n_stages", "RPM", "converged", "ns_invalid"]
    for col in ["eta_system", "eta_turbine", "W_net_kW", "Vr", "Ns", "skip_reason"]:
        if col in df.columns:
            display_cols.append(col)

    print(df[display_cols].to_string(index=False))

    n_ok = int(df["converged"].sum())
    n_bad = total - n_ok
    print(f"\n  Converged : {n_ok}/{total}   |   Failed : {n_bad}")
    print(f"\n  Results saved to:")
    print(f"    CSV  : {csv_path}")
    print(f"    Excel: {xlsx_path}")
    print("=" * 76 + "\n")

    return df


# ══════════════════════════════════════════════════════════════════════
#  MODE 4: CONDENSATION PRESSURE vs EFFICIENCY — THERMOPT FLUID SWEEP
# ══════════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════════
#  HELPERS: FLUID-SPECIFIC WARM START FOR PCOND SWEEP
# ══════════════════════════════════════════════════════════════════════


def _compute_fluid_warmstart(
    fluid_name,
    T_cond_K=308.15,
    T_evap_K=443.15,
    superheating_K=10.0,
    subcooling_K=2.0,
    W_net_target=45e6,
    eta_turbine=0.90,
    eta_pump=0.85,
    transcritical=False,
    k=1.20,
):
    """
    Compute physically sensible initial guess values for a given fluid
    using CoolProp, so the pcond_sweep optimizer starts from a feasible
    point regardless of which fluid is being run.

    The key variable is expander_mass_flow_rate — it is calibrated for
    Toluene in the template YAML (287 kg/s) and will be completely wrong
    for other fluids, causing the 45 MW power constraint to start with
    a huge infeasibility that the optimizer cannot recover from.

    Returns a tuple (ws, summary_str) where:
      ws          : dict of {design_variable_name: float} ready to patch into
                    the YAML value: fields.
      summary_str : a single-line string for the caller to print — no leading
                    newline, so it does not break end="" console output.
    Returns (None, error_str) if CoolProp fails, in which case the caller
    uses the YAML defaults unchanged.

    Notes on mass flow estimation
    ------------------------------
    The mass flow is estimated from W_net_target / dh_net where dh_net is the
    net specific work per unit mass of working fluid:

        dh_net = eta_turbine * dh_is_turbine  -  dh_pump / eta_pump

    The pump specific work is estimated from the incompressible formula:
        dh_pump = v_liq * (p_evap - p_cond)

    where v_liq is the specific volume of the subcooled liquid at the
    compressor inlet. This correction is significant for light refrigerants
    (10–20% of gross turbine power) and negligible for heavy fluids like
    Toluene or siloxanes. Omitting it causes systematic underestimation of
    mass flow for refrigerants, which starts the 45 MW equality constraint
    with a large infeasibility the optimizer may not recover from.

    Other notes
    -----------
    - Transcritical expander inlet uses T_crit + 50 K (not 30 K) to stay
      clear of the high-property-gradient region near the critical point
      that can cause CoolProp instability.
    - recuperator_effectiveness is included for recuperated cycle YAMLs.
      It is silently skipped by _patch_warmstart for simple YAMLs.

    Parameters
    ----------
    T_cond_K      : condensing temperature [K]  — default 35°C (sink + 13K margin)
    T_evap_K      : evaporation temperature [K] — default 170°C (well below brine pinch)
    superheating_K: superheat above sat. vapour at p_evap
    subcooling_K  : subcooling below sat. liquid at p_cond
    W_net_target  : target net power [W] used to estimate mass flow
    eta_turbine   : isentropic efficiency of the expander
    eta_pump      : isentropic efficiency of the working-fluid pump (compressor)
    """
    try:
        import CoolProp.CoolProp as CP

        T_crit = CP.PropsSI("TCRIT", "", 0, "", 0, fluid_name)
        p_crit = CP.PropsSI("PCRIT", "", 0, "", 0, fluid_name)

        # ── Condensing pressure ───────────────────────────────────────
        # Clamp to 95% of critical if fluid is supercritical at T_cond
        T_cond_use = min(T_cond_K, T_crit * 0.95)
        p_cond = CP.PropsSI("P", "T", T_cond_use, "Q", 0, fluid_name)
        p_cond = min(p_cond, 0.95 * p_crit)

        # ── Evaporation pressure ──────────────────────────────────────
        if transcritical:
            # Transcritical: p_evap = k * p_crit (supercritical high side)
            p_evap = k * p_crit
        else:
            # Subcritical: saturation pressure at T_evap_K, clamped below critical
            T_evap_use = min(T_evap_K, T_crit * 0.90)
            p_evap = CP.PropsSI("P", "T", T_evap_use, "Q", 1, fluid_name)
            p_evap = min(p_evap, 0.80 * p_crit)

        # Ensure a meaningful pressure ratio
        if p_evap <= p_cond * 2.0:
            p_evap = p_cond * 4.0

        # ── Compressor inlet: subcooled liquid ────────────────────────
        T_sat_cond = CP.PropsSI("T", "P", p_cond, "Q", 0, fluid_name)
        T_comp_in = T_sat_cond - subcooling_K
        h_comp_in = CP.PropsSI("H", "T", T_comp_in, "P", p_cond, fluid_name)

        # ── Expander inlet ────────────────────────────────────────────
        if transcritical:
            # Supercritical: no saturation curve — use T_crit + 50 K to stay
            # well clear of the high-property-gradient region near the critical
            # point, which can cause CoolProp instability at T_crit + 30 K.
            T_exp_in = T_crit + 50.0
            h_exp_in = CP.PropsSI("H", "T", T_exp_in, "P", p_evap, fluid_name)
        else:
            # Subcritical: superheated vapour above saturation temperature
            T_sat_evap = CP.PropsSI("T", "P", p_evap, "Q", 1, fluid_name)
            T_exp_in = T_sat_evap + superheating_K
            h_exp_in = CP.PropsSI("H", "T", T_exp_in, "P", p_evap, fluid_name)

        # ── Turbine specific work ─────────────────────────────────────
        s_exp_in = CP.PropsSI("S", "T", T_exp_in, "P", p_evap, fluid_name)
        h_exp_out_is = CP.PropsSI("H", "P", p_cond, "S", s_exp_in, fluid_name)
        dh_is = h_exp_in - h_exp_out_is  # isentropic turbine work [J/kg]
        dh_turbine = eta_turbine * dh_is  # actual turbine work [J/kg]

        # ── Pump specific work (incompressible approximation) ─────────
        # v_liq * (p_evap - p_cond) is the isentropic pump work.
        # Dividing by eta_pump gives the actual shaft work consumed.
        # This correction matters most for light refrigerants where pump
        # work is 10–20% of gross turbine output.
        v_liq = 1.0 / CP.PropsSI("D", "T", T_comp_in, "P", p_cond, fluid_name)
        dh_pump = v_liq * (p_evap - p_cond) / eta_pump  # actual pump work [J/kg]

        # ── Net specific work and mass flow ───────────────────────────
        dh_net = dh_turbine - dh_pump
        if dh_net > 0:
            m_dot = W_net_target / dh_net
            m_dot = max(25.0, min(m_dot, 1400.0))
        else:
            m_dot = 300.0  # fallback: pressure ratio too low for net positive work

        # ── Build warm-start dict ─────────────────────────────────────
        ws = {
            "compressor_inlet_pressure": p_cond,
            "compressor_inlet_enthalpy": h_comp_in,
            "expander_inlet_pressure": p_evap,
            "expander_inlet_enthalpy": h_exp_in,
            "expander_mass_flow_rate": m_dot,
            # Recuperated cycle only — silently skipped for simple YAMLs.
            # 0.60 is a conservative mid-range guess within [0, 0.95].
            "recuperator_effectiveness": 0.60,
        }

        pump_frac = dh_pump / dh_turbine * 100 if dh_turbine > 0 else 0.0
        summary = (
            f"warm-start ({'TC' if transcritical else 'SC'}, k={k:.2f}): "
            f"p_cond={p_cond/1e5:.3f} bar, p_evap={p_evap/1e5:.2f} bar, "
            f"m_dot={m_dot:.1f} kg/s  "
            f"(dh_is={dh_is/1e3:.1f} kJ/kg, pump={pump_frac:.1f}% of turbine)"
        )
        return ws, summary

    except Exception as e:
        return None, f"warm-start failed ({e}) — using YAML defaults"


def _patch_warmstart(text, ws):
    """
    Replace the value: fields of the given design variables in the YAML
    text with plain float values from ws dict.

    Works line-by-line: finds the variable block by name, then replaces
    only the first "value:" line inside that block with the new number.
    This approach is more robust than regex for YAML with expression-based
    bounds like "0.172758*$working_fluid.critical_point.p".

    Returns (patched_text, matched, skipped) where:
      matched : list of variable names that were successfully patched.
      skipped : list of variable names from ws that were NOT found in the
                YAML. The caller decides whether a skip is expected (e.g.
                recuperator_effectiveness absent in a simple YAML) or a
                real problem such as a typo in a variable name.
    """
    lines = text.splitlines(keepends=True)
    result = []
    matched = []
    i = 0

    while i < len(lines):
        line = lines[i]

        # Check if this line is the start of a design variable we want to patch
        matched_var = None
        for var in ws:
            stripped = line.lstrip()
            indent = len(line) - len(stripped)
            # Must be indented (inside a YAML mapping, not a top-level key)
            # and the stripped line must start with "var_name:"
            if indent >= 2 and stripped.startswith(var + ":"):
                matched_var = var
                break

        if matched_var is not None:
            result.append(line)
            i += 1
            var_indent = len(line) - len(line.lstrip())
            patched = False
            while i < len(lines):
                sub_line = lines[i]
                sub_stripped = sub_line.lstrip()
                sub_indent = len(sub_line) - len(sub_line.lstrip())

                # Stop if we have left this variable block (same or lower indent,
                # non-empty line that is not a child key)
                if (
                    sub_indent <= var_indent
                    and sub_stripped
                    and not sub_stripped.startswith("#")
                ):
                    break

                if not patched and sub_stripped.startswith("value:"):
                    leading = sub_line[:sub_indent]
                    result.append(f"{leading}value: {ws[matched_var]:.6g}\n")
                    patched = True
                else:
                    result.append(sub_line)
                i += 1

            if patched:
                matched.append(matched_var)
        else:
            result.append(line)
            i += 1

    skipped = [v for v in ws if v not in matched]
    return "".join(result), matched, skipped


def _run_one_pcond_attempt(
    fluid_name,
    category,
    base_text,
    transcritical,
    p_crit_fluid,
    turbine_model,
    n_stages,
    RPM,
    template_yaml,
):
    """
    Run a single optimization attempt for one fluid from one warm-start style
    (subcritical or transcritical).

    base_text : YAML text already patched with the correct fluid name.
    Returns a fully populated row dict.  row["converged"] is False if the run
    failed or did not converge — the caller decides what to do with it.
    The tmp file is always cleaned up before returning.
    """
    suffix = "tc" if transcritical else "sc"
    tmp_yaml = None
    row = {
        "fluid": fluid_name,
        "category": category,
        "cycle_type": None,
        "converged": False,
        "ns_invalid": False,
        "skip_reason": None,
    }
    try:
        # ── Warm start ────────────────────────────────────────────────
        ws, ws_summary = _compute_fluid_warmstart(
            fluid_name, transcritical=transcritical
        )
        # Note: ws_summary is NOT printed here — in parallel mode this function runs
        # in a subprocess and printing would produce garbled interleaved console output.
        # The warmstart outcome is captured in warmstart_winner/eta_sc/eta_tc in the row.

        text = base_text
        if ws is not None:
            text, _matched, _skipped = _patch_warmstart(text, ws)

        if turbine_model == "astolfi":
            text = re.sub(r"(n_stages:\s*)\d+", rf"\g<1>{n_stages}", text)
            text = re.sub(r"(RPM:\s*)\d+", rf"\g<1>{RPM}", text)

        tmp_yaml = Path(template_yaml).parent / f"_tmp_pcond_{fluid_name}_{suffix}.yaml"
        tmp_yaml.write_text(text)

        # ── Optimize ──────────────────────────────────────────────────
        cycle = th.ThermodynamicCycleOptimization(str(tmp_yaml))
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            cycle.run_optimization()

        converged = False
        for attr in [
            lambda: cycle.solver.success,
            lambda: cycle.solver.result.success,
            lambda: cycle.solver.solver_result.success,
        ]:
            try:
                converged = attr()
                break
            except AttributeError:
                continue

        if not converged:
            row["skip_reason"] = "optimizer did not converge"
            return row

        # ── Extract results ───────────────────────────────────────────
        data = cycle.problem.cycle_data
        energy = data["energy_analysis"]
        components = data["components"]
        exp = components["expander"]
        data_out = exp.get("data_out", {})

        # Post-convergence sub/TC classification
        try:
            p_exp_in = exp["state_in"].p
            is_transcritical = bool(p_exp_in > p_crit_fluid)
            row["cycle_type"] = "transcritical" if is_transcritical else "subcritical"
        except Exception:
            row["cycle_type"] = "unknown"

        row.update(
            {
                "converged": True,
                "eta_system": energy.get("system_efficiency"),
                "eta_cycle": energy.get("cycle_efficiency"),
                "p_cond_bar": exp["state_out"].p / 1e5,
                "p_evap_bar": exp["state_in"].p / 1e5,
                "p_crit_bar": p_crit_fluid / 1e5,
                "W_net_kW": energy.get("net_system_power", 0) / 1e3,
                "eta_turbine": exp.get("efficiency"),
                "m_wf_kg_s": energy.get("mass_flow_working_fluid"),
                "m_well_kg_s": energy.get("mass_flow_heating_fluid"),
            }
        )

        if turbine_model == "astolfi":
            row.update(
                {
                    "SP": data_out.get("size_parameter"),
                    "SP_clamped": data_out.get("size_parameter_clamped", False),
                    "Vr": data_out.get("volume_ratio"),
                    "Ns": data_out.get("specific_speed"),
                }
            )
            if data_out.get("specific_speed_out_of_range", False):
                row["ns_invalid"] = True
                row["skip_reason"] = "Ns out of range"

        return row

    except Exception as e:
        row["skip_reason"] = str(e)
        traceback.print_exc()
        return row
    finally:
        if tmp_yaml is not None and tmp_yaml.exists():
            tmp_yaml.unlink()


def _print_pcond_row(row, n_done, n_total, turbine_model):
    """Print a one-line progress summary for one completed fluid."""
    fluid = row["fluid"]
    n_left = n_total - n_done
    left_str = f"  ({n_left} left)" if n_left > 0 else "  (done)"
    prefix = f"  [{n_done:>3}/{n_total}]{left_str}  {fluid:<20}  "

    if not row["converged"]:
        reason = row.get("skip_reason") or "did not converge"
        print(f"{prefix}✗  {reason}", flush=True)
        return

    ct = row.get("cycle_type", "?")
    eta = row["eta_system"]
    p_cond = row.get("p_cond_bar")
    p_evap = row.get("p_evap_bar")
    eta_t = row.get("eta_turbine")
    winner = row.get("warmstart_winner", "")

    if turbine_model == "astolfi" and row.get("ns_invalid"):
        print(
            f"{prefix}⚠  Ns OOR [{ct}]  "
            f"η_sys={eta*100:.2f}%  p_cond={p_cond:.3f} bar  [{winner}]",
            flush=True,
        )
    else:
        eta_t_str = f"{eta_t:.3f}" if eta_t is not None else "?"
        print(
            f"{prefix}✓  [{ct}]  "
            f"η_sys={eta*100:.2f}%  "
            f"p_cond={p_cond:.3f} bar  "
            f"p_evap={p_evap:.2f} bar  "
            f"η_turb={eta_t_str}  [{winner}]",
            flush=True,
        )


def _pcond_fluid_worker(task):
    """
    Top-level worker for parallel pcond_sweep.

    Must be a module-level function (not a closure) so multiprocessing
    can pickle it.  One call = one fluid = one returned row dict.

    Task tuple
    ----------
    (fluid_name, category, template_yaml_str,
     turbine_model, n_stages, RPM,
     warmstart_tcrit_K, dual_warmstart)
    """
    import CoolProp.CoolProp as _CP

    (
        fluid_name,
        category,
        template_yaml_str,
        turbine_model,
        n_stages,
        RPM,
        warmstart_tcrit_K,
        dual_warmstart,
    ) = task

    row = {
        "fluid": fluid_name,
        "category": category,
        "cycle_type": None,
        "converged": False,
        "ns_invalid": False,
        "skip_reason": None,
        "warmstart_winner": None,
        "warmstart_eta_sc": None,
        "warmstart_eta_tc": None,
        "sc_error": None,  # skip_reason from SC attempt if it failed
        "tc_error": None,  # skip_reason from TC attempt if it failed
    }

    # ── CoolProp lookup ───────────────────────────────────────────────
    try:
        T_crit_fluid = _CP.PropsSI("TCRIT", "", 0, "", 0, fluid_name)
        p_crit_fluid = _CP.PropsSI("PCRIT", "", 0, "", 0, fluid_name)
    except Exception as cp_err:
        row["skip_reason"] = f"CoolProp T_crit lookup failed: {cp_err}"
        return row

    # ── Patch fluid name into YAML text ──────────────────────────────
    try:
        base_text = re.sub(
            r"(working_fluid:.*?\n\s+name:\s*)[^\n]+",
            lambda m: m.group(1) + fluid_name,
            Path(template_yaml_str).read_text(),
            count=1,
            flags=re.DOTALL,
        )
    except Exception as e:
        row["skip_reason"] = f"YAML read/patch failed: {e}"
        return row

    attempt_kwargs = dict(
        fluid_name=fluid_name,
        category=category,
        base_text=base_text,
        p_crit_fluid=p_crit_fluid,
        turbine_model=turbine_model,
        n_stages=n_stages,
        RPM=RPM,
        template_yaml=template_yaml_str,
    )

    # ── Run optimizer (dual or single warm start) ─────────────────────
    if dual_warmstart:
        row_sc = _run_one_pcond_attempt(**attempt_kwargs, transcritical=False)
        row_tc = _run_one_pcond_attempt(**attempt_kwargs, transcritical=True)

        sc_ok = row_sc["converged"]
        tc_ok = row_tc["converged"]

        if sc_ok and tc_ok:
            eta_sc = row_sc.get("eta_system") or 0.0
            eta_tc = row_tc.get("eta_system") or 0.0
            row = row_sc if eta_sc >= eta_tc else row_tc
            winner = "SC" if eta_sc >= eta_tc else "TC"
            row["warmstart_winner"] = winner
            row["warmstart_eta_sc"] = eta_sc
            row["warmstart_eta_tc"] = eta_tc
            row["sc_error"] = None
            row["tc_error"] = None
        elif sc_ok:
            row = row_sc
            row["warmstart_winner"] = "SC (TC failed)"
            row["warmstart_eta_sc"] = row_sc.get("eta_system")
            row["warmstart_eta_tc"] = None
            row["sc_error"] = None
            row["tc_error"] = row_tc.get("skip_reason")
        elif tc_ok:
            row = row_tc
            row["warmstart_winner"] = "TC (SC failed)"
            row["warmstart_eta_sc"] = None
            row["warmstart_eta_tc"] = row_tc.get("eta_system")
            row["sc_error"] = row_sc.get("skip_reason")
            row["tc_error"] = None
        else:
            row["skip_reason"] = "both SC and TC warm starts failed to converge"
            row["warmstart_winner"] = "none"
            row["sc_error"] = row_sc.get("skip_reason")
            row["tc_error"] = row_tc.get("skip_reason")
    else:
        ws_tc = T_crit_fluid < warmstart_tcrit_K
        row = _run_one_pcond_attempt(**attempt_kwargs, transcritical=ws_tc)
        row["warmstart_winner"] = "TC_heuristic" if ws_tc else "SC_heuristic"
        row["warmstart_eta_sc"] = None
        row["warmstart_eta_tc"] = None

    return row


def run_pcond_sweep(
    template_yaml,
    fluids,
    output_dir,
    cycle_label="simple",
    turbine_model="isentropic",
    n_stages=4,
    RPM=1000,
    warmstart_tcrit_K=475.30,
    dual_warmstart=True,
    n_workers=4,
    debug_fluids=None,
):
    """
    Sweep over candidate working fluids using a single YAML template whose
    design-variable bounds are wide enough to cover both subcritical and
    transcritical operation.

    The optimizer decides the operating regime for each fluid; the result is
    classified post-convergence by comparing expander inlet pressure to the
    fluid's critical pressure.

    Parameters
    ----------
    template_yaml : Path or str
        Single YAML template for all fluids (bounds must span sub + TC range).
    fluids : dict {fluid_name: category} or None
        If None, every pure fluid in the CoolProp database is used (excluding
        those in PCOND_EXCLUDE_FLUIDS).  Unknown fluids get category "other".
    dual_warmstart : bool
        If True (default), each fluid is optimized twice — once from a
        subcritical warm start and once from a transcritical warm start.
        The higher-efficiency converged result is kept.  This prevents the
        gradient-based optimizer from being locked into one regime by the
        warm-start heuristic, at the cost of roughly double the run time.
        If False, the single T_crit heuristic (warmstart_tcrit_K) is used.
    warmstart_tcrit_K : float
        Only used when dual_warmstart=False.  Fluids with T_crit below this
        threshold get a transcritical-style warm-start guess; others get a
        subcritical guess.
    n_workers : int
        Number of parallel worker processes.  Each worker handles one
        fluid (both SC and TC attempts) independently.  Set to 1 to
        run sequentially, which is useful for debugging.
    debug_fluids : list of str or None
        When non-empty, only the listed fluids are run.  Overrides
        the full fluid list.  Use with n_workers=1 for easy debugging.
    """
    import CoolProp.CoolProp as _CP

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Build fluid list ──────────────────────────────────────────────
    if fluids is None:
        # Enumerate all CoolProp pure fluids, skip exclusions and mixtures
        all_cp = _CP.FluidsList()
        fluids = {}
        for fname in all_cp:
            if fname in PCOND_EXCLUDE_FLUIDS:
                continue
            # Skip CoolProp pseudo-pure / mixture entries (contain dots or &)
            if "." in fname or "&" in fname:
                continue
            # Use known category if available, otherwise "other"
            category = PCOND_FLUIDS.get(fname, "other")
            if isinstance(category, str):
                fluids[fname] = category
            else:
                fluids[fname] = "other"

    # ── Debug filter — override fluid list if PCOND_DEBUG_FLUIDS is set ———
    if debug_fluids:
        fluids = {
            f: (fluids.get(f) or PCOND_FLUIDS.get(f, "other"))
            for f in debug_fluids
            if f  # skip any empty strings
        }
        print(f"  [DEBUG] Running {len(fluids)} fluid(s) only: {list(fluids.keys())}")

    print("=" * 70)
    print("  CONDENSATION PRESSURE vs EFFICIENCY — ThermOpt Fluid Sweep")
    print("=" * 70)
    print(f"  Cycle         : {cycle_label}")
    print(
        f"  Template YAML : {Path(template_yaml).name}  (single template, sub + TC bounds)"
    )
    if turbine_model == "astolfi":
        print(f"  Turbine model : Macchi-Astolfi  (n_stages={n_stages}, RPM={RPM})")
    else:
        print(f"  Turbine model : isentropic efficiency (from YAML)")
    print(f"  Total fluids  : {len(fluids)}")
    print(
        f"  Warm start    : {'dual (SC + TC, keep best)' if dual_warmstart else 'single (T_crit heuristic)'}"
    )
    print(
        f"  Workers       : {n_workers}{'  (sequential — set PCOND_N_WORKERS > 1 to parallelise)' if n_workers <= 1 else ' parallel'}"
    )
    print(f"  Sub/TC classification : post-convergence (expander inlet p vs p_crit)")
    print("=" * 70 + "\n")

    results = []

    # ── Build task list ───────────────────────────────────────────────
    template_yaml_str = str(template_yaml)
    tasks = [
        (
            fluid_name,
            category,
            template_yaml_str,
            turbine_model,
            n_stages,
            RPM,
            warmstart_tcrit_K,
            dual_warmstart,
        )
        for fluid_name, category in fluids.items()
    ]
    n_total = len(tasks)
    n_done = 0

    # ── Dispatch ─────────────────────────────────────────────────────
    import multiprocessing as _mp

    actual_workers = min(n_workers, n_total)
    if actual_workers <= 1:
        # Sequential fallback — same logic, easier to debug
        for task in tasks:
            row = _pcond_fluid_worker(task)
            n_done += 1
            _print_pcond_row(row, n_done, n_total, turbine_model)
            results.append(row)
    else:
        with _mp.Pool(processes=actual_workers) as pool:
            for row in pool.imap_unordered(_pcond_fluid_worker, tasks):
                n_done += 1
                _print_pcond_row(row, n_done, n_total, turbine_model)
                results.append(row)

    # ── Save results ──────────────────────────────────────────────────
    df = (
        pd.DataFrame(results)
        .sort_values("eta_system", ascending=False, na_position="last")
        .reset_index(drop=True)
    )

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    turb_tag = (
        "isentropic"
        if turbine_model == "isentropic"
        else f"astolfi_n{n_stages}_rpm{RPM}"
    )
    tag = f"{cycle_label}_{turb_tag}"
    xlsx_path = output_dir / f"pcond_vs_efficiency_{tag}_{timestamp}.xlsx"
    df.to_excel(xlsx_path, index=False)
    print(f"\n  Saved data  : {xlsx_path}")

    # ── Plots ─────────────────────────────────────────────────────────
    valid = [r for r in results if r["converged"] and not r.get("ns_invalid", False)]
    flagged = [r for r in results if r["converged"] and r.get("ns_invalid", False)]

    if not valid and not flagged:
        print("  No converged results to plot.")
        return df

    p1 = _plot_pcond_efficiency(valid, flagged, output_dir, tag, subset="all")
    sub_v = [r for r in valid if r.get("cycle_type") == "subcritical"]
    sub_f = [r for r in flagged if r.get("cycle_type") == "subcritical"]
    p2 = _plot_pcond_efficiency(
        sub_v, sub_f, output_dir, f"{tag}_subcritical", subset="subcritical"
    )
    tc_v = [r for r in valid if r.get("cycle_type") == "transcritical"]
    tc_f = [r for r in flagged if r.get("cycle_type") == "transcritical"]
    p3 = _plot_pcond_efficiency(
        tc_v, tc_f, output_dir, f"{tag}_transcritical", subset="transcritical"
    )

    print(f"  Saved plots : {p1.name}  |  {p2.name}  |  {p3.name}")

    n_ok = len(valid)
    n_ns = len(flagged)
    n_bad = sum(1 for r in results if not r["converged"])
    n_sub = sum(1 for r in valid if r.get("cycle_type") == "subcritical")
    n_tc = sum(1 for r in valid if r.get("cycle_type") == "transcritical")
    print(
        f"\n  Summary: {n_ok} converged ({n_sub} subcritical, {n_tc} transcritical) "
        f"| {n_ns} Ns-invalid | {n_bad} failed"
    )
    print("=" * 70 + "\n")

    return df


def _plot_pcond_efficiency(valid, flagged, output_dir, tag, subset="all"):
    """
    Thesis-quality scatter plot: condensing pressure vs system efficiency.

    subset : "all"          — combined plot, subcritical by family + transcritical teal
             "subcritical"  — subcritical fluids only, coloured by family
             "transcritical"— transcritical fluids only, coloured by family
    """
    import matplotlib.ticker as ticker

    if not valid and not flagged:
        return Path(output_dir) / f"pcond_vs_efficiency_{tag}.png"  # nothing to plot

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 11,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.linewidth": 0.8,
        }
    )

    fig, ax = plt.subplots(figsize=(13, 7))

    all_results = valid + flagged

    if subset == "transcritical":
        # All transcritical fluids coloured by family with teal base
        tc_tol = [r for r in valid if r["fluid"] == "cis-2-Butene"]
        tc_rest = [r for r in valid if r["fluid"] != "cis-2-Butene"]
        for cat, style in PCOND_CATEGORY_STYLE.items():
            cat_r = [r for r in tc_rest if r["category"] == cat]
            if not cat_r:
                continue
            ax.scatter(
                [r["p_cond_bar"] for r in cat_r],
                [r["eta_system"] * 100 for r in cat_r],
                color="#17a589",
                marker=style["marker"],
                s=90,
                edgecolors="k",
                linewidth=0.4,
                label=cat.capitalize(),
                zorder=3,
            )
        if tc_tol:
            r = tc_tol[0]
            ax.scatter(
                r["p_cond_bar"],
                r["eta_system"] * 100,
                color="#17a589",
                marker="o",
                s=180,
                edgecolors="k",
                linewidth=1.5,
                zorder=5,
                label="cis-2-Butene (reference TC)",
            )

    else:
        # Subcritical fluids coloured by family
        sub_valid = [r for r in valid if r.get("cycle_type") != "transcritical"]
        toluene = [r for r in sub_valid if r["fluid"] == "Toluene"]
        sub_rest = [r for r in sub_valid if r["fluid"] != "Toluene"]

        for cat, style in PCOND_CATEGORY_STYLE.items():
            cat_r = [r for r in sub_rest if r["category"] == cat]
            if not cat_r:
                continue
            label = (
                f"Subcritical — {cat.capitalize()}"
                if subset == "all"
                else cat.capitalize()
            )
            ax.scatter(
                [r["p_cond_bar"] for r in cat_r],
                [r["eta_system"] * 100 for r in cat_r],
                color=style["color"],
                marker=style["marker"],
                s=90,
                edgecolors="k",
                linewidth=0.4,
                label=label,
                zorder=3,
            )

        if toluene:
            r = toluene[0]
            ax.scatter(
                r["p_cond_bar"],
                r["eta_system"] * 100,
                color=PCOND_CATEGORY_STYLE["aromatic"]["color"],
                marker=PCOND_CATEGORY_STYLE["aromatic"]["marker"],
                s=180,
                edgecolors="k",
                linewidth=1.5,
                zorder=5,
                label="Toluene (reference)",
            )

        if subset == "all":
            # Also draw transcritical in teal
            tc_valid = [r for r in valid if r.get("cycle_type") == "transcritical"]
            tc_plotted = False
            for cat, style in PCOND_CATEGORY_STYLE.items():
                cat_r = [r for r in tc_valid if r["category"] == cat]
                if not cat_r:
                    continue
                label = "Transcritical" if not tc_plotted else "_nolegend_"
                tc_plotted = True
                ax.scatter(
                    [r["p_cond_bar"] for r in cat_r],
                    [r["eta_system"] * 100 for r in cat_r],
                    color="#17a589",
                    marker=style["marker"],
                    s=90,
                    edgecolors="k",
                    linewidth=0.4,
                    label=label,
                    zorder=3,
                )

    # Ns-invalid hollow markers (Astolfi mode only)
    if flagged:
        ax.scatter(
            [r["p_cond_bar"] for r in flagged],
            [r["eta_system"] * 100 for r in flagged],
            facecolors="none",
            edgecolors="#aaaaaa",
            marker="o",
            s=70,
            linewidth=0.8,
            label=r"$N_s$ out of range",
            zorder=2,
        )

    ax.set_xscale("log")

    # Fluid name labels
    for r in all_results:
        x, y = r["p_cond_bar"], r["eta_system"] * 100
        dx, dy = PCOND_LABEL_OFFSETS.get(r["fluid"], (7, 5))
        use_arrow = abs(dx) > 15 or abs(dy) > 12
        weight = "bold" if r["fluid"] in ("Toluene", "cis-2-Butene") else "normal"
        ax.annotate(
            r["fluid"],
            (x, y),
            textcoords="offset points",
            xytext=(dx, dy),
            fontsize=8,
            fontweight=weight,
            ha="left" if dx >= 0 else "right",
            color="#222222",
            arrowprops=(
                dict(arrowstyle="-", color="#aaaaaa", lw=0.5, shrinkA=0, shrinkB=2)
                if use_arrow
                else None
            ),
        )

    # 1 atm reference line
    ax.axvline(
        x=1.01325, color="#888888", linestyle="--", linewidth=0.8, alpha=0.8, zorder=1
    )
    y_top = ax.get_ylim()[1]
    ax.text(
        1.01325 * 1.04,
        y_top * 0.98,
        "1 atm",
        fontsize=8,
        color="#888888",
        va="top",
        ha="left",
        style="italic",
    )

    ax.set_xlabel("Condensing Pressure [bar]")
    ax.set_ylabel(r"System Efficiency $\eta_{sys}$ [%]")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:g}"))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.4, color="grey")
    ax.set_axisbelow(True)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        frameon=True,
        framealpha=0.95,
        edgecolor="#cccccc",
        borderpad=0.8,
    )

    fig.tight_layout()
    fig.subplots_adjust(right=0.82)

    plot_path = Path(output_dir) / f"pcond_vs_efficiency_{tag}.png"
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    plt.rcParams.update(plt.rcParamsDefault)
    return plot_path


# ══════════════════════════════════════════════════════════════════════
#  MODE 6: k-SENSITIVITY — REDUCED PRESSURE SWEEP (TRANSCRITICAL)
# ══════════════════════════════════════════════════════════════════════


def run_k_sensitivity(
    transcritical_yaml,
    tc_fluids,
    k_values,
    output_dir,
    cycle_label="simple",
    turbine_model="isentropic",
    n_stages=4,
    RPM=1000,
):
    """
    For each transcritical fluid, sweep over reduced-pressure ratios
    k = P_max / P_crit and record the cycle efficiency. This tells you
    how sensitive the transcritical ORC performance is to the choice of
    maximum operating pressure.

    For each (fluid, k) combination:
      - p_evap is patched to k * p_crit in the YAML
      - All other settings (heat source, sink, constraints) stay fixed
    """
    import CoolProp.CoolProp as _CP

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  k-SENSITIVITY — TRANSCRITICAL REDUCED PRESSURE SWEEP")
    print("=" * 70)
    print(f"  Cycle         : {cycle_label}")
    print(f"  Template YAML : {Path(transcritical_yaml).name}")
    print(f"  k values      : {k_values}")
    print(f"  Fluids        : {len(tc_fluids)}")
    print(f"  Total runs    : {len(tc_fluids) * len(k_values)}")
    print("=" * 70 + "\n")

    results = []
    total = len(tc_fluids) * len(k_values)
    run_n = 0

    for fluid_name, category in tc_fluids.items():
        try:
            T_crit = _CP.PropsSI("TCRIT", "", 0, "", 0, fluid_name)
            p_crit = _CP.PropsSI("PCRIT", "", 0, "", 0, fluid_name)
        except Exception as e:
            print(f"  ✗ {fluid_name}: CoolProp failed — {e}")
            continue

        for k in k_values:
            run_n += 1
            p_evap_target = k * p_crit
            print(
                f"  [{run_n:>3}/{total}] {fluid_name:<18} k={k:.2f}  "
                f"(p_evap={p_evap_target/1e5:.2f} bar) ... ",
                end="",
                flush=True,
            )

            row = {
                "fluid": fluid_name,
                "category": category,
                "k": k,
                "p_evap_target_bar": p_evap_target / 1e5,
                "T_crit_C": round(T_crit - 273.15, 1),
                "converged": False,
                "skip_reason": None,
            }
            tmp_yaml = None
            try:
                text = Path(transcritical_yaml).read_text()
                # Swap fluid name
                text = re.sub(
                    r"(working_fluid:.*?\n\s+name:\s*)[^\n]+",
                    lambda m: m.group(1) + fluid_name,
                    text,
                    count=1,
                    flags=re.DOTALL,
                )
                # Apply fluid-specific warm start with this k value
                ws, ws_summary = _compute_fluid_warmstart(
                    fluid_name, transcritical=True, k=k
                )
                print(f"    {ws_summary}")
                if ws is not None:
                    text, _matched, _skipped = _patch_warmstart(text, ws)

                if turbine_model == "astolfi":
                    text = re.sub(r"(n_stages:\s*)\d+", rf"\g<1>{n_stages}", text)
                    text = re.sub(r"(RPM:\s*)\d+", rf"\g<1>{RPM}", text)

                tmp_yaml = (
                    Path(transcritical_yaml).parent
                    / f"_tmp_ksens_{fluid_name}_k{k:.2f}.yaml"
                )
                tmp_yaml.write_text(text)

                cycle = th.ThermodynamicCycleOptimization(str(tmp_yaml))
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore")
                    cycle.run_optimization()

                converged = False
                for attr in [
                    lambda: cycle.solver.success,
                    lambda: cycle.solver.result.success,
                    lambda: cycle.solver.solver_result.success,
                ]:
                    try:
                        converged = attr()
                        break
                    except AttributeError:
                        continue

                if not converged:
                    row["skip_reason"] = "did not converge"
                    print("✗  did not converge")
                    results.append(row)
                    continue

                data = cycle.problem.cycle_data
                energy = data["energy_analysis"]
                exp = data["components"]["expander"]

                row.update(
                    {
                        "converged": True,
                        "eta_system": energy.get("system_efficiency"),
                        "eta_cycle": energy.get("cycle_efficiency"),
                        "p_cond_bar": exp["state_out"].p / 1e5,
                        "p_evap_bar": exp["state_in"].p / 1e5,
                        "W_net_kW": energy.get("net_system_power", 0) / 1e3,
                        "eta_turbine": exp.get("efficiency"),
                        "m_wf_kg_s": energy.get("mass_flow_working_fluid"),
                    }
                )

                print(
                    f"✓  η_sys={row['eta_system']*100:.2f}%  "
                    f"p_cond={row['p_cond_bar']:.3f} bar  "
                    f"p_evap={row['p_evap_bar']:.2f} bar"
                )

            except Exception as e:
                row["skip_reason"] = str(e)
                print(f"✗  {e}")
                traceback.print_exc()
            finally:
                if tmp_yaml is not None and tmp_yaml.exists():
                    tmp_yaml.unlink()

            results.append(row)

    # ── Save results ──────────────────────────────────────────────────
    df = (
        pd.DataFrame(results)
        .sort_values(["k", "eta_system"], ascending=[True, False], na_position="last")
        .reset_index(drop=True)
    )

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    tag = f"{cycle_label}_k_sensitivity"
    xlsx_path = output_dir / f"{tag}_{timestamp}.xlsx"
    df.to_excel(xlsx_path, index=False)
    print(f"\n  Saved data : {xlsx_path}")

    # ── Plot ──────────────────────────────────────────────────────────
    valid = [r for r in results if r["converged"]]
    if not valid:
        print("  No converged results to plot.")
        return df

    plot_path = _plot_k_sensitivity(valid, k_values, output_dir, tag)
    print(f"  Saved plot : {plot_path}")

    n_ok = sum(1 for r in results if r["converged"])
    n_bad = len(results) - n_ok
    print(
        f"\n  Summary: {n_ok} converged | {n_bad} failed  "
        f"({len(tc_fluids)} fluids × {len(k_values)} k values)"
    )
    print("=" * 70 + "\n")
    return df


def _plot_k_sensitivity(results, k_values, output_dir, tag):
    """
    Scatter plot: condensing pressure vs cycle efficiency, one colour per k value.
    Matches the style of the ORC fluid screening sensitivity analysis.
    """
    import matplotlib.ticker as ticker
    import matplotlib.cm as cm
    import numpy as np

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 11,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.linewidth": 0.8,
        }
    )

    fig, ax = plt.subplots(figsize=(13, 7))

    # Colour map: one colour per k value (viridis, dark to light)
    colours = cm.viridis(np.linspace(0.1, 0.9, len(k_values)))
    k_colour = {k: colours[i] for i, k in enumerate(k_values)}

    # Track which fluids have been labelled (for first-k label placement)
    labelled_fluids = set()

    for k in k_values:
        k_results = [r for r in results if r["k"] == k and r["converged"]]
        if not k_results:
            continue
        ax.scatter(
            [r["p_cond_bar"] for r in k_results],
            [r["eta_system"] * 100 for r in k_results],
            color=k_colour[k],
            marker="o",
            s=70,
            edgecolors="k",
            linewidth=0.3,
            label=f"k = {k:.2f}",
            zorder=3,
        )

    # Label each fluid once (at the first k value where it appears)
    for r in sorted(results, key=lambda x: x["k"]):
        if not r["converged"]:
            continue
        if r["fluid"] not in labelled_fluids:
            labelled_fluids.add(r["fluid"])
            x, y = r["p_cond_bar"], r["eta_system"] * 100
            dx, dy = PCOND_LABEL_OFFSETS.get(r["fluid"], (7, 5))
            use_arrow = abs(dx) > 15 or abs(dy) > 12
            ax.annotate(
                r["fluid"],
                (x, y),
                textcoords="offset points",
                xytext=(dx, dy),
                fontsize=8,
                ha="left" if dx >= 0 else "right",
                color="#222222",
                arrowprops=(
                    dict(arrowstyle="-", color="#aaaaaa", lw=0.5, shrinkA=0, shrinkB=2)
                    if use_arrow
                    else None
                ),
            )

    ax.set_xscale("log")
    ax.axvline(
        x=1.01325, color="#888888", linestyle="--", linewidth=0.8, alpha=0.8, zorder=1
    )
    ax.text(
        1.01325 * 1.04,
        ax.get_ylim()[1] * 0.98,
        "1 atm",
        fontsize=8,
        color="#888888",
        va="top",
        ha="left",
        style="italic",
    )

    ax.set_xlabel(r"Condensing Pressure $p_{\mathrm{cond}}$ [bar]")
    ax.set_ylabel(r"System Efficiency $\eta_{sys}$ [%]")
    ax.set_title(r"Sensitivity to Reduced Pressure $k = P_{\max}/P_c$", pad=8)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:g}"))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.4, color="grey")
    ax.set_axisbelow(True)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        frameon=True,
        framealpha=0.95,
        edgecolor="#cccccc",
        borderpad=0.8,
    )

    fig.tight_layout()
    fig.subplots_adjust(right=0.82)

    plot_path = Path(output_dir) / f"{tag}.png"
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    plt.rcParams.update(plt.rcParamsDefault)
    return plot_path


# Set to True to print the narrow bounds for every design variable — useful
# for verifying that ±50% is working correctly before a full multistart run.
# Set back to False once you are satisfied.
MULTISTART_DEBUG_BOUNDS = False


def _evaluate_bound_expr(expr, fluid_name, T_max=475.30):
    """
    Evaluate a thermopt bound expression for a specific fluid using CoolProp.

    Supports the property tokens used in thermopt YAML bound expressions:
        $working_fluid.liquid_at_ambient_temperature.p  — sat. pressure at 20 C
        $working_fluid.liquid_at_ambient_temperature.h  — sat. liquid enthalpy at 20 C
        $working_fluid.critical_point.p                 — critical pressure
        $working_fluid.critical_point.h                 — critical enthalpy
        $working_fluid.gas_at_maximum_temperature.h     — dilute gas enthalpy at T_max
        $working_fluid.triple_point_liquid.p            — triple-point pressure

    T_max defaults to 475.30 K (Toluene basecase) but is read from the YAML
    special_points.maximum_temperature when available.

    Returns the evaluated float, or None if evaluation fails.
    """
    try:
        import CoolProp.CoolProp as CP

        T_ambient = 293.15  # K (20 C)

        # Saturated liquid state at ambient temperature
        p_sat_ambient = CP.PropsSI("P", "T", T_ambient, "Q", 0, fluid_name)
        h_liq_ambient = CP.PropsSI("H", "T", T_ambient, "Q", 0, fluid_name)

        # Critical point
        p_crit = CP.PropsSI("pcrit", fluid_name)
        h_crit = CP.PropsSI("H", "P", p_crit, "Q", 0, fluid_name)

        # Dilute gas at maximum temperature (very low pressure — ideal-gas limit)
        p_triple = CP.PropsSI("ptriple", fluid_name)
        p_dilute = 1.01 * p_triple
        h_gas_max = CP.PropsSI("H", "P", p_dilute, "T", T_max, fluid_name)

        # Build substitution dict — longest tokens first to avoid partial matches
        tokens = {
            "$working_fluid.liquid_at_ambient_temperature.p": p_sat_ambient,
            "$working_fluid.liquid_at_ambient_temperature.h": h_liq_ambient,
            "$working_fluid.gas_at_maximum_temperature.h": h_gas_max,
            "$working_fluid.critical_point.p": p_crit,
            "$working_fluid.critical_point.h": h_crit,
            "$working_fluid.triple_point_liquid.p": p_triple,
        }

        evaluated = str(expr)
        for token, value in tokens.items():
            evaluated = evaluated.replace(token, repr(value))

        return float(eval(evaluated))  # noqa: S307 — controlled substitution only

    except Exception as _e:
        if MULTISTART_DEBUG_BOUNDS:
            print(
                f"    [_evaluate_bound_expr] ERROR fluid={fluid_name}: {_e}  expr='{str(expr)[:60]}'"
            )
        return None


def _narrow_bounds(vd, pct=0.50, var_name="", fluid_name=None, T_max=475.30):
    """
    Compute narrow LHS sampling bounds centred on the warm start value.

    Returns (lo, hi) = (value - pct*|value|, value + pct*|value|), capped
    at the YAML min/max bounds so we never leave the feasible region.

    If vmin/vmax are expression strings (e.g. "1*$working_fluid.critical_point.p")
    they are evaluated via CoolProp so that the narrow window can be capped
    correctly.

    IMPORTANT: abs(v0) is used for the half-range, NOT v0 directly.
    For negative warm-start values (e.g. compressor_inlet_enthalpy
    approx -149 000 J/kg), multiplying directly gives:
        v0 * (1 - pct) = -74 500  ->  LARGER than v0
        v0 * (1 + pct) = -223 500 ->  SMALLER than v0
    which inverts the bounds and collapses the sampling window.
    Using abs(v0) * pct always produces a symmetric window centred on
    v0 regardless of its sign.

    Raises ValueError — with the variable name and exact reason — for any
    of the following conditions (no silent fallback to full YAML bounds):
        • value: is not a plain number
        • min:/max: is an expression and fluid_name was not provided
        • min:/max: expression evaluation failed (CoolProp error)
        • The ±pct window collapses (hi <= lo), e.g. when v0 = 0

    Set MULTISTART_DEBUG_BOUNDS = True to print bounds for every variable.
    """
    vmin = vd["min"]
    vmax = vd["max"]
    v0 = vd["value"]

    # v0 must be a plain number — a non-numeric value: means the warm start
    # was never set to a converged result.
    if not isinstance(v0, (int, float)):
        raise ValueError(
            f"variable '{var_name}': value: is not a plain number "
            f"(got {str(v0)[:60]!r}). "
            f"The warm start must be a converged numeric value, not an expression."
        )

    # Resolve vmin — evaluate expression string if needed
    if not isinstance(vmin, (int, float)):
        if not fluid_name:
            raise ValueError(
                f"variable '{var_name}': min: is an expression "
                f"({str(vmin)[:60]!r}) but no fluid_name was provided — "
                f"cannot evaluate the bound."
            )
        vmin_resolved = _evaluate_bound_expr(vmin, fluid_name, T_max)
        if vmin_resolved is None:
            raise ValueError(
                f"variable '{var_name}': failed to evaluate min: expression "
                f"({str(vmin)[:60]!r}) for fluid '{fluid_name}' — "
                f"CoolProp returned None (check fluid name and expression tokens)."
            )
        vmin = vmin_resolved

    # Resolve vmax — evaluate expression string if needed
    if not isinstance(vmax, (int, float)):
        if not fluid_name:
            raise ValueError(
                f"variable '{var_name}': max: is an expression "
                f"({str(vmax)[:60]!r}) but no fluid_name was provided — "
                f"cannot evaluate the bound."
            )
        vmax_resolved = _evaluate_bound_expr(vmax, fluid_name, T_max)
        if vmax_resolved is None:
            raise ValueError(
                f"variable '{var_name}': failed to evaluate max: expression "
                f"({str(vmax)[:60]!r}) for fluid '{fluid_name}' — "
                f"CoolProp returned None (check fluid name and expression tokens)."
            )
        vmax = vmax_resolved

    half_range = abs(float(v0)) * pct
    lo = max(float(vmin), float(v0) - half_range)
    hi = min(float(vmax), float(v0) + half_range)

    if hi <= lo:
        raise ValueError(
            f"variable '{var_name}': ±{pct*100:.0f}% window collapsed "
            f"(v0={v0:.6g}, lo={lo:.6g}, hi={hi:.6g}, "
            f"yaml=[{vmin:.6g}, {vmax:.6g}]). "
            f"Likely cause: v0 is zero, or v0 sits exactly at a YAML boundary."
        )

    if MULTISTART_DEBUG_BOUNDS:
        print(
            f"  NARROW  {var_name:<45}  v0={v0:>14.4f}  "
            f"yaml=[{vmin:.4f}, {vmax:.4f}]  lhs=[{lo:.4f}, {hi:.4f}]"
        )
    return (lo, hi)


def _multistart_sample_worker(args):
    """
    Module-level worker for multistart_batch — runs ONE LHS sample for ONE
    (YAML, fluid) entry.  Fully picklable; safe for mp.Pool with the spawn
    start method used on Windows.

    Args tuple (all picklable primitives — no Path objects, no lambdas):
        config_file_str : str   — absolute path to the base YAML
        fluid_override  : str|None
        var_names       : list[str]   — design-variable names in declaration order
        sample          : list[float] — LHS sample in [0, 1]^n_vars
        bounds          : list[(min, max)]  — raw min/max from the YAML dict
                          (may be float or a thermopt expression string)
        run_index       : int   — 0-based sample index within this entry
        run_label       : str   — e.g. "Toluene_simple_basecase_cis-2-Butene"
        output_dir_str  : str   — absolute path to this entry's output folder

    Returns a row dict.  Two keys are internal (stripped before saving to CSV):
        _x0_dict       : {str: float} of converged variable values, or None
        _lhs_yaml_text : str containing the full YAML text of the LHS starting
                         point — returned so the main process can write
                         best_lhs_start_*.yaml without needing a temp file to
                         survive past the worker's finally block.
    """
    import os
    import io as _io
    import time
    import traceback as _tb
    import thermopt as th
    from pathlib import Path
    from ruamel.yaml import YAML

    (
        config_file_str,
        fluid_override,
        var_names,
        sample,
        bounds,
        run_index,
        run_label,
        output_dir_str,
    ) = args

    config_file = Path(config_file_str)
    output_dir = Path(output_dir_str)
    output_dir.mkdir(parents=True, exist_ok=True)

    ryaml = YAML()
    ryaml.preserve_quotes = True

    # PID in filename guarantees no collision between concurrent workers that
    # share the same config_file parent directory.
    tmp_path = (
        config_file.parent / f"_tmp_ms_{run_label}_{run_index:04d}_{os.getpid()}.yaml"
    )

    row = {
        # Grouping / identification
        "run_label": run_label,
        "run": run_index + 1,
        "config_file_str": config_file_str,
        "fluid_override": fluid_override,
        # Result columns (match run_multistart schema exactly)
        "config_name": config_file.stem,
        "fluid_name": fluid_override,  # updated below from YAML
        "converged": False,
        "eta_system": None,
        "eta_cycle": None,
        "eta_exergy": None,
        "heat_utilization": None,
        "W_net_kW": None,
        "is_transcritical": None,
        "error": None,
        "t_elapsed_s": None,
        "n_iterations": None,
        "n_func_evals": None,
        "infeasibility": None,
        "exit_message": None,
        # Internal — stripped before CSV/Excel save
        "_x0_dict": None,
        "_lhs_yaml_text": None,
    }

    try:
        with open(config_file) as f:
            cfg = ryaml.load(f)

        if fluid_override is not None:
            cfg["problem_formulation"]["fixed_parameters"]["working_fluid"][
                "name"
            ] = fluid_override

        # Actual fluid name (may differ from fluid_override if None)
        row["fluid_name"] = cfg["problem_formulation"]["fixed_parameters"][
            "working_fluid"
        ]["name"]

        # ── Write LHS starting point into design-variable value: fields ──
        # bounds[j] are the NARROW bounds (value ± 50%, capped at YAML bounds)
        # already computed in the main process — not the full YAML min/max.
        dv = cfg["problem_formulation"]["design_variables"]
        for j, var_name in enumerate(var_names):
            t = float(sample[j])
            vd = dv[var_name]
            lo, hi = bounds[j]
            # bounds are always plain floats — the main process raises if not.
            # Guard here catches any inconsistency and reports it clearly.
            if not isinstance(lo, (int, float)) or not isinstance(hi, (int, float)):
                raise ValueError(
                    f"variable '{var_name}': bounds are not numeric "
                    f"(lo={lo!r}, hi={hi!r}). "
                    f"This should have been caught in the main process."
                )
            vd["value"] = float(lo) + t * (float(hi) - float(lo))
            row[f"t_{var_name}"] = round(t, 6)

        # Serialise to text for two purposes:
        #   1. Write the actual tmp file for thermopt.
        #   2. Stash the text so the main process can save best_lhs_start_*.yaml
        #      without needing the tmp file to survive past this worker's finally.
        buf = _io.StringIO()
        ryaml.dump(cfg, buf)
        lhs_yaml_text = buf.getvalue()
        row["_lhs_yaml_text"] = lhs_yaml_text

        tmp_path.write_text(lhs_yaml_text)

        # ── Run optimiser ─────────────────────────────────────────────
        t_start = time.perf_counter()
        cycle = th.ThermodynamicCycleOptimization(str(tmp_path))
        cycle.run_optimization()
        row["t_elapsed_s"] = round(time.perf_counter() - t_start, 1)

        # ── Solver diagnostics ────────────────────────────────────────
        try:
            row["n_iterations"] = int(cycle.solver.nit)
        except AttributeError:
            pass
        try:
            row["n_func_evals"] = int(cycle.solver.nfev)
        except AttributeError:
            pass
        try:
            row["infeasibility"] = float(cycle.solver.infeasibility)
        except AttributeError:
            pass
        try:
            row["exit_message"] = str(cycle.solver.message)
        except AttributeError:
            pass

        # ── Convergence ───────────────────────────────────────────────
        converged = False
        try:
            converged = bool(cycle.solver.success)
        except AttributeError:
            pass

        if not converged:
            row["error"] = "optimizer did not converge"
            return row

        # ── Extract key metrics ───────────────────────────────────────
        ea = cycle.problem.cycle_data["energy_analysis"]
        eta = ea.get("system_efficiency", 0) or 0
        W_net = ea.get("net_system_power", 0) / 1e3

        row["converged"] = True
        row["eta_system"] = round(eta, 8)
        row["eta_cycle"] = round(ea.get("cycle_efficiency", 0) or 0, 8)
        row["W_net_kW"] = round(W_net, 2)
        if "heat_utilization" in ea:
            row["heat_utilization"] = round(ea.get("heat_utilization", 0) or 0, 6)
        elif ea.get("heater_heat_flow_max"):
            row["heat_utilization"] = round(
                ea.get("heater_heat_flow", ea.get("total_heat_input", 0))
                / ea["heater_heat_flow_max"],
                6,
            )
        else:
            row["heat_utilization"] = None
        try:
            exergy = cycle.problem.cycle_data.get("exergy_analysis", {})
            row["eta_exergy"] = round(float(exergy.get("eta_exergy", 0) or 0), 8)
        except Exception:
            pass

        # ── Thermodynamic state outputs ───────────────────────────────
        if "brine_exit_temperature" in ea:
            row["brine_exit_T_K"] = round(ea["brine_exit_temperature"], 2)
        elif "hp_brine_exit_temperature" in ea:
            row["hp_brine_exit_T_K"] = round(ea["hp_brine_exit_temperature"], 2)
            row["lp_brine_exit_T_K"] = round(ea["lp_brine_exit_temperature"], 2)

        if "mass_flow_heating_fluid" in ea:
            row["m_brine_kgs"] = round(ea["mass_flow_heating_fluid"], 2)
        elif "mass_flow_brine_hp" in ea:
            row["m_brine_hp_kgs"] = round(ea["mass_flow_brine_hp"], 2)
            row["m_brine_lp_kgs"] = round(ea["mass_flow_brine_lp"], 2)
        row["m_working_fluid_kgs"] = round(ea.get("mass_flow_working_fluid", 0), 2)

        components = cycle.problem.cycle_data.get("components", {})
        for exp_name in ["lp_expander", "expander"]:
            if exp_name in components:
                exp = components[exp_name]
                try:
                    row["expander_inlet_T_C"] = round(exp["state_in"].T - 273.15, 2)
                    row["expander_inlet_p_bar"] = round(exp["state_in"].p / 1e5, 3)
                except (KeyError, TypeError):
                    pass
                try:
                    p_in = exp["state_in"].p
                    p_crit = cycle.problem.fluid.critical_point.p
                    row["is_transcritical"] = bool(p_in > p_crit)
                except Exception:
                    pass
                break

        for cool_name in ["cooler", "condenser"]:
            if cool_name in components:
                cool = components[cool_name]
                try:
                    row["condenser_exit_T_C"] = round(cool["state_out"].T - 273.15, 2)
                except (KeyError, TypeError):
                    pass
                break

        # ── Capture converged values as plain dict NOW while cycle alive ─
        # Must happen before the finally block deletes the tmp file and before
        # the cycle object goes out of scope.
        try:
            x0 = cycle.problem.x0_dict
            row["_x0_dict"] = {k: float(v) for k, v in x0.items()}
            for var_name in var_names:
                if var_name in x0:
                    row[f"x_{var_name}"] = round(float(x0[var_name]), 6)
        except AttributeError:
            pass

    except Exception:
        row["error"] = _tb.format_exc()

    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass

    return row


def _save_multistart_entry_outputs(
    config_file, fluid_override, run_label, output_dir, rows, ryaml, timestamp
):
    """
    Called in the main process after all worker results for one (YAML, fluid)
    entry have been collected.  Saves:
      - multistart_results_<label>_<timestamp>.csv / .xlsx  — full result table
      - best_lhs_start_<label>_<timestamp>.yaml             — LHS start point
      - optimized_<label>_<timestamp>.yaml                  — converged solution

    Parameters
    ----------
    rows : list[dict]
        Row dicts returned by _multistart_sample_worker, sorted by run index.
        Internal keys (_x0_dict, _lhs_yaml_text, etc.) are present but stripped
        before saving to CSV/Excel.

    Returns (csv_path, xlsx_path, lhs_yaml_path, optimized_yaml_path, n_ok, best_eta)
    where paths are None if that file was not written.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── CSV / Excel — strip internal _ keys ───────────────────────────
    internal_keys = {
        "_x0_dict",
        "_lhs_yaml_text",
        "config_file_str",
        "fluid_override",
        "run_label",
    }
    public_rows = [{k: v for k, v in r.items() if k not in internal_keys} for r in rows]
    df = pd.DataFrame(public_rows)
    csv_path = output_dir / f"multistart_results_{run_label}_{timestamp}.csv"
    xlsx_path = output_dir / f"multistart_results_{run_label}_{timestamp}.xlsx"
    df.to_csv(csv_path, index=False)
    df.to_excel(xlsx_path, index=False, sheet_name="Multistart")

    # ── Find best converged result ─────────────────────────────────────
    converged_rows = [r for r in rows if r.get("converged")]
    best = (
        max(converged_rows, key=lambda r: r.get("eta_system") or -1.0)
        if converged_rows
        else None
    )
    n_ok = len(converged_rows)
    best_eta = best["eta_system"] if best else None

    # ── Save best LHS starting-point YAML ─────────────────────────────
    lhs_yaml_path = None
    if best and best.get("_lhs_yaml_text"):
        lhs_yaml_path = output_dir / f"best_lhs_start_{run_label}_{timestamp}.yaml"
        lhs_yaml_path.write_text(best["_lhs_yaml_text"], encoding="utf-8")

    # ── Save optimized YAML (converged values as warm start) ──────────
    optimized_yaml_path = None
    if best and best.get("_x0_dict"):
        optimized_yaml_path = output_dir / f"optimized_{run_label}_{timestamp}.yaml"
        _save_optimized_yaml(
            base_config_file=config_file,
            x0_dict=best["_x0_dict"],
            out_path=optimized_yaml_path,
            ryaml=ryaml,
            fluid_override=fluid_override,
        )

    return csv_path, xlsx_path, lhs_yaml_path, optimized_yaml_path, n_ok, best_eta


def _save_optimized_yaml(
    base_config_file, x0_dict, out_path, ryaml, fluid_override=None
):
    """
    Save converged variable values back into the YAML as value: fields.

    Parameters
    ----------
    base_config_file : Path
        The original base YAML — read fresh from disk so we always start from
        clean, comment-preserved source rather than a post-optimization state.
    x0_dict : dict
        Plain {variable_name: float} mapping of converged variable values,
        extracted from cycle.problem.x0_dict immediately after convergence
        (before the tmp file is deleted) and stored as a plain dict so this
        function has no dependency on a live ThermOpt cycle object.
    out_path : Path
        Destination for the saved YAML.
    ryaml : ruamel.yaml.YAML
        Shared ruamel instance (preserve_quotes=True).
    fluid_override : str or None
        If provided, the working fluid name is rewritten in the saved YAML so
        it is self-consistent and runnable in "optimize" mode without edits.

    Strategy
    --------
    1. Load the base YAML with ruamel (preserves formatting/comments).
    2. Apply fluid_override to the in-memory dict if provided.
    3. Dump back to a StringIO buffer — gives us clean text with the correct
       fluid name already embedded.
    4. Run _patch_warmstart on that text to overwrite each design-variable
       value: field with the actual converged float.
    """
    import io

    # Step 1 & 2: load base YAML and optionally rewrite fluid name
    with open(base_config_file) as f:
        cfg = ryaml.load(f)
    if fluid_override is not None:
        cfg["problem_formulation"]["fixed_parameters"]["working_fluid"][
            "name"
        ] = fluid_override

    # Step 3: round-trip through ruamel to text
    buf = io.StringIO()
    ryaml.dump(cfg, buf)
    text = buf.getvalue()

    # Step 4: patch converged values into value: fields
    patched, matched, skipped = _patch_warmstart(text, x0_dict)

    if skipped:
        # Variables in x0_dict that were not found in the YAML — almost always
        # harmless (thermopt may add internal bookkeeping keys), but log them so
        # the user can spot a real mismatch (e.g. a renamed design variable).
        print(
            f"    [_save_optimized_yaml] {len(skipped)} x0 key(s) not found in YAML "
            f"(probably internal thermopt keys): {skipped[:5]}"
            + (" …" if len(skipped) > 5 else "")
        )

    with open(out_path, "w") as f:
        f.write(patched)


def run_multistart(config_file, n_samples, output_dir, fluid_override=None):
    """
    Latin Hypercube Sampling multistart optimisation.

    Generates n_samples starting points distributed evenly across the full
    design variable space using LHS (one sample per interval per variable),
    runs the optimizer from each point, logs and skips failures, and saves:
      - A CSV + Excel summary of every run (eta, W_net, convergence flag).
      - best_lhs_start_*.yaml  — the LHS starting point that led to the best
        result. Re-running this in "optimize" mode will reproduce the result.
      - optimized_*.yaml       — the converged solution with value: fields set
        to the actual optimized variable values. Use this as a warm start for
        future runs; the optimizer begins right at the optimum.

    Dependencies: scipy (>=1.7 for scipy.stats.qmc), ruamel.yaml
    """
    from scipy.stats.qmc import LatinHypercube
    from ruamel.yaml import YAML
    import shutil

    config_file = Path(config_file)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Parse YAML — extract design variable names and bounds ──────────
    ryaml = YAML()
    ryaml.preserve_quotes = True
    with open(config_file) as f:
        base_config = ryaml.load(f)

    # Apply fluid override for transcritical runs
    if fluid_override is not None:
        base_config["problem_formulation"]["fixed_parameters"]["working_fluid"][
            "name"
        ] = fluid_override

    dvars = base_config["problem_formulation"]["design_variables"]
    var_names = list(dvars.keys())
    n_vars = len(var_names)

    # Extract fluid name and T_max for expression-bound evaluation
    _fluid_name = base_config["problem_formulation"]["fixed_parameters"][
        "working_fluid"
    ]["name"]
    _T_max = (
        base_config["problem_formulation"]["fixed_parameters"]
        .get("special_points", {})
        .get("maximum_temperature", 475.30)
    )
    if not isinstance(_T_max, (int, float)):
        _T_max = 475.30  # fallback if it's an expression

    print("=" * 70)
    print("  MULTISTART OPTIMISATION — Latin Hypercube Sampling")
    print("=" * 70)
    run_label = config_file.stem + (f"_{fluid_override}" if fluid_override else "")
    print(
        f"  Config    : {config_file.name}"
        + (f"  [fluid: {fluid_override}]" if fluid_override else "")
    )
    print(f"  Variables : {n_vars}")
    print(f"  Samples   : {n_samples}")
    print(f"  Output    : {output_dir}")
    print("=" * 70)
    print()
    print("  Design variable bounds:")
    for name in var_names:
        vd = dvars[name]
        print(f"    {name:<48}  min={vd['min']}  max={vd['max']}")
    print()

    # ── Pre-compute narrow bounds once — fail fast before any samples run ──
    # _narrow_bounds raises ValueError (variable name + reason) if any
    # variable's window cannot be computed.  Catching here means we abort
    # immediately with a clear message rather than repeating the same error
    # N times inside the sample loop.
    try:
        bounds = [
            _narrow_bounds(dvars[v], var_name=v, fluid_name=_fluid_name, T_max=_T_max)
            for v in var_names
        ]
    except ValueError as _be:
        print(f"\n  ✗  BOUND ERROR — aborting multistart for {run_label}:")
        print(f"     {_be}")
        return pd.DataFrame()

    # ── Generate LHS samples in [0, 1]^n_vars ─────────────────────────
    # seed=42 makes the sample set reproducible across runs.
    sampler = LatinHypercube(d=n_vars, seed=42)
    lhs_samples = sampler.random(n=n_samples)  # shape: (n_samples, n_vars)

    # ── Optimise from each LHS starting point ─────────────────────────
    results = []
    best_eta = -1.0
    best_run_idx = None
    best_x0_dict = None  # plain dict of converged values for the best run — captured
    # immediately after convergence, before the tmp file is deleted

    import time

    for i, sample in enumerate(lhs_samples):
        run_tag = f"[{i+1:>3}/{n_samples}]"
        print(f"  {run_tag} ", end="", flush=True)

        tmp_path = config_file.parent / f"_tmp_multistart_{run_label}_{i:04d}.yaml"
        row = {
            "run": i + 1,
            "config_name": config_file.stem,
            "fluid_name": (
                fluid_override
                if fluid_override
                else base_config["problem_formulation"]["fixed_parameters"][
                    "working_fluid"
                ]["name"]
            ),
            "converged": False,
            "eta_system": None,
            "eta_cycle": None,
            "eta_exergy": None,
            "heat_utilization": None,
            "W_net_kW": None,
            "is_transcritical": None,
            "error": None,
            "t_elapsed_s": None,
            "n_iterations": None,
            "n_func_evals": None,
            "infeasibility": None,
            "exit_message": None,
        }

        try:
            # Re-load the config each iteration to get a fresh ruamel tree.
            with open(config_file) as f:
                cfg = ryaml.load(f)
            if fluid_override is not None:
                cfg["problem_formulation"]["fixed_parameters"]["working_fluid"][
                    "name"
                ] = fluid_override

            dv = cfg["problem_formulation"]["design_variables"]

            for j, var_name in enumerate(var_names):
                t = float(sample[j])  # LHS sample in [0, 1]
                lo, hi = bounds[j]  # pre-computed narrow bounds
                dv[var_name]["value"] = float(lo) + t * (float(hi) - float(lo))

                # Store the t-value in the result row for reference
                row[f"t_{var_name}"] = round(t, 6)

            with open(tmp_path, "w") as f:
                ryaml.dump(cfg, f)

            # ── Run optimiser ──────────────────────────────────────────
            t_start = time.perf_counter()
            cycle = th.ThermodynamicCycleOptimization(str(tmp_path))
            cycle.run_optimization()
            t_elapsed = time.perf_counter() - t_start
            row["t_elapsed_s"] = round(t_elapsed, 1)

            # ── Solver diagnostics ─────────────────────────────────────
            try:
                row["n_iterations"] = int(cycle.solver.nit)
            except AttributeError:
                pass
            try:
                row["n_func_evals"] = int(cycle.solver.nfev)
            except AttributeError:
                pass
            try:
                row["infeasibility"] = float(cycle.solver.infeasibility)
            except AttributeError:
                pass
            try:
                row["exit_message"] = str(cycle.solver.message)
            except AttributeError:
                pass

            # Convergence check (mirrors _run_single / run_pcond_sweep)
            converged = False
            try:
                converged = bool(cycle.solver.success)
            except AttributeError:
                pass

            if not converged:
                print(f"✗  did not converge  ({t_elapsed:.0f}s)")
                row["error"] = "optimizer did not converge"
                results.append(row)
                continue

            # ── Extract key metrics ────────────────────────────────────
            ea = cycle.problem.cycle_data["energy_analysis"]
            eta = ea.get("system_efficiency", 0) or 0
            W_net = ea.get("net_system_power", 0) / 1e3

            row["converged"] = True
            row["eta_system"] = round(eta, 8)
            row["eta_cycle"] = round(ea.get("cycle_efficiency", 0) or 0, 8)
            row["W_net_kW"] = round(W_net, 2)
            if "heat_utilization" in ea:
                row["heat_utilization"] = round(ea.get("heat_utilization", 0) or 0, 6)
            elif ea.get("heater_heat_flow_max"):
                row["heat_utilization"] = round(
                    ea.get("heater_heat_flow", ea.get("total_heat_input", 0))
                    / ea["heater_heat_flow_max"],
                    6,
                )
            else:
                row["heat_utilization"] = None
            # Exergy efficiency
            try:
                exergy = cycle.problem.cycle_data.get("exergy_analysis", {})
                row["eta_exergy"] = round(float(exergy.get("eta_exergy", 0) or 0), 8)
            except Exception:
                pass

            # ── Thermodynamic outputs ──────────────────────────────────
            # Brine exit temperature(s)
            if "brine_exit_temperature" in ea:
                row["brine_exit_T_K"] = round(ea["brine_exit_temperature"], 2)
            elif "hp_brine_exit_temperature" in ea:
                row["hp_brine_exit_T_K"] = round(ea["hp_brine_exit_temperature"], 2)
                row["lp_brine_exit_T_K"] = round(ea["lp_brine_exit_temperature"], 2)

            # Mass flows
            if "mass_flow_heating_fluid" in ea:
                row["m_brine_kgs"] = round(ea["mass_flow_heating_fluid"], 2)
            elif "mass_flow_brine_hp" in ea:
                row["m_brine_hp_kgs"] = round(ea["mass_flow_brine_hp"], 2)
                row["m_brine_lp_kgs"] = round(ea["mass_flow_brine_lp"], 2)
            row["m_working_fluid_kgs"] = round(ea.get("mass_flow_working_fluid", 0), 2)

            # Expander inlet conditions (LP expander = main expander in dual pressure)
            components = cycle.problem.cycle_data.get("components", {})
            for exp_name in ["lp_expander", "expander"]:
                if exp_name in components:
                    exp = components[exp_name]
                    try:
                        row["expander_inlet_T_C"] = round(exp["state_in"].T - 273.15, 2)
                        row["expander_inlet_p_bar"] = round(exp["state_in"].p / 1e5, 3)
                    except (KeyError, TypeError):
                        pass
                    # Check if expander inlet is transcritical
                    try:
                        p_in = exp["state_in"].p
                        p_crit = cycle.problem.fluid.critical_point.p
                        row["is_transcritical"] = bool(p_in > p_crit)
                    except Exception:
                        pass
                    break

            # Condenser exit temperature
            for cool_name in ["cooler", "condenser"]:
                if cool_name in components:
                    cool = components[cool_name]
                    try:
                        row["condenser_exit_T_C"] = round(
                            cool["state_out"].T - 273.15, 2
                        )
                    except (KeyError, TypeError):
                        pass
                    break

            # ── Converged variable values ──────────────────────────────
            try:
                x0_dict = cycle.problem.x0_dict
                for var_name in var_names:
                    if var_name in x0_dict:
                        row[f"x_{var_name}"] = round(float(x0_dict[var_name]), 6)
            except AttributeError:
                pass

            print(
                f"✓  η_sys = {eta * 100:.4f}%   W_net = {W_net:.1f} kW  ({t_elapsed:.0f}s)",
                end="",
            )

            # ── Track best ─────────────────────────────────────────────
            if eta > best_eta:
                best_eta = eta
                best_run_idx = i + 1
                # Capture converged values NOW, while cycle is still alive and
                # tmp_path still exists.  We store a plain dict so there is no
                # dependency on the cycle object or the temp file surviving past
                # this iteration's finally block.
                try:
                    best_x0_dict = {
                        k: float(v) for k, v in cycle.problem.x0_dict.items()
                    }
                except AttributeError:
                    best_x0_dict = None
                # Keep a copy of the temp YAML (the LHS starting point) for the
                # best run.  Overwritten each time a better run is found.
                _candidate = output_dir / "_best_candidate.yaml"
                shutil.copy(tmp_path, _candidate)
                print("  ← best so far", end="")

            print()

        except Exception as e:
            tb = traceback.format_exc()
            print(f"✗  {e}")
            row["error"] = tb

        finally:
            if tmp_path.exists():
                tmp_path.unlink()

        results.append(row)

    # ── Save summary CSV + Excel ───────────────────────────────────────
    df = pd.DataFrame(results)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    csv_path = output_dir / f"multistart_results_{run_label}_{timestamp}.csv"
    xlsx_path = output_dir / f"multistart_results_{run_label}_{timestamp}.xlsx"
    df.to_csv(csv_path, index=False)
    df.to_excel(xlsx_path, index=False, sheet_name="Multistart")

    # ── Save LHS warm-start YAML (starting point that led to best result) ─
    # Useful for exact reproduction: running this in "optimize" mode will
    # re-run the optimizer from the same starting point.
    lhs_yaml_path = None
    candidate_path = output_dir / "_best_candidate.yaml"
    if candidate_path.exists():
        lhs_yaml_path = output_dir / f"best_lhs_start_{run_label}_{timestamp}.yaml"
        shutil.move(str(candidate_path), str(lhs_yaml_path))

    # ── Save optimized YAML (converged solution as value: fields) ─────
    # This is the proper warm start: value: fields are set to the actual
    # converged variable values, not the LHS starting point.
    optimized_yaml_path = None
    if best_x0_dict is not None:
        optimized_yaml_path = output_dir / f"optimized_{run_label}_{timestamp}.yaml"
        _save_optimized_yaml(
            base_config_file=config_file,
            x0_dict=best_x0_dict,
            out_path=optimized_yaml_path,
            ryaml=ryaml,
            fluid_override=fluid_override,
        )

    # ── Print summary ──────────────────────────────────────────────────
    n_ok = int(df["converged"].sum())
    n_bad = n_samples - n_ok

    print()
    print("=" * 70)
    print("  MULTISTART — SUMMARY")
    print("=" * 70)
    print(f"  Total runs : {n_samples}")
    print(f"  Converged  : {n_ok}")
    print(f"  Failed     : {n_bad}")
    print()

    converged_df = df[df["converged"] == True].copy()
    if not converged_df.empty:
        converged_df_sorted = converged_df.sort_values("eta_system", ascending=False)
        print("  Top 5 converged results:")
        print(f"    {'Run':>5}   {'η_sys [%]':>12}   {'W_net [kW]':>12}")
        print(f"    {'─'*5}   {'─'*12}   {'─'*12}")
        for _, r in converged_df_sorted.head(5).iterrows():
            marker = " ← BEST" if int(r["run"]) == best_run_idx else ""
            print(
                f"    {int(r['run']):>5}   {r['eta_system']*100:>11.4f}%   "
                f"{r['W_net_kW']:>11.1f} kW{marker}"
            )
        print()
        print(f"  Best η_sys : {best_eta * 100:.4f}%  (run {best_run_idx})")
    else:
        print("  No runs converged.")

    print()
    print("  Output files:")
    print(f"    CSV            : {csv_path}")
    print(f"    Excel          : {xlsx_path}")
    if lhs_yaml_path is not None:
        print(f"    LHS start YAML : {lhs_yaml_path}")
        print(f"                     (starting point that produced the best result)")
    if optimized_yaml_path is not None:
        print(f"    Optimized YAML : {optimized_yaml_path}")
        print(
            f"                     (converged solution as value: — use this as warm start)"
        )
        print()
        print("  To run full post-processing on the best result:")
        print(
            f'    1. Set CONFIG_FILE = Path(__file__).with_name("{optimized_yaml_path.name}")'
        )
        print('    2. Set MODE = "optimize"')
        print("    3. Run this script")
    print("=" * 70 + "\n")

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

    elif MODE == "batch":
        missing = [f for f in BATCH_YAML_FILES if not f.exists()]
        if missing:
            for f in missing:
                print(f"Error: config file not found: {f}")
            sys.exit(1)

        # ── Create timestamped batch folder ───────────────────────────
        batch_timestamp = datetime.now().strftime("batch_%Y-%m-%d_%H-%M-%S")
        batch_dir = BATCH_OUTPUT_ROOT / batch_timestamp
        batch_dir.mkdir(parents=True, exist_ok=True)
        print("=" * 70)
        print(f"  BATCH MODE — {len(BATCH_YAML_FILES)} YAML files queued")
        print(f"  Output root : {BATCH_OUTPUT_ROOT}")
        print(f"  Batch folder: {batch_dir.name}")
        print("=" * 70)

        batch_results = []
        for i, yaml_file in enumerate(BATCH_YAML_FILES, 1):
            print(f"\n[{i}/{len(BATCH_YAML_FILES)}] {yaml_file.name}")
            print("-" * 70)
            # One subfolder per YAML, named after the YAML stem
            yaml_dest = batch_dir / yaml_file.stem
            try:
                run_optimize(yaml_file, batch=True, dest_dir=yaml_dest)
                batch_results.append((yaml_file.name, "✓ converged"))
            except Exception as e:
                batch_results.append((yaml_file.name, f"✗ failed: {e}"))
                print(f"  ERROR: {e}")
                print(f"  Continuing with next file...")

        print("\n" + "=" * 70)
        print("  BATCH SUMMARY")
        print("=" * 70)
        for name, status in batch_results:
            print(f"  {name:<45} {status}")
        print("=" * 70 + "\n")

        # ── Save summary txt inside the batch folder ──────────────────
        n_ok = sum(1 for _, s in batch_results if s.startswith("✓"))
        n_bad = len(batch_results) - n_ok
        summary_path = batch_dir / "batch_summary.txt"
        with open(summary_path, "w", encoding="utf-8") as _f:
            _f.write("=" * 70 + "\n")
            _f.write("  BATCH SUMMARY\n")
            _f.write(f"  Completed  : {batch_timestamp}\n")
            _f.write(f"  YAML files : {len(batch_results)}\n")
            _f.write(f"  Successful : {n_ok}\n")
            _f.write(f"  Failed     : {n_bad}\n")
            _f.write(f"  Output dir : {batch_dir}\n")
            _f.write("=" * 70 + "\n\n")
            for name, status in batch_results:
                _f.write(f"  {name:<45} {status}\n")
            _f.write("\n" + "=" * 70 + "\n")
        print(f"  Summary saved: {summary_path}\n")

    elif MODE == "sweep":
        if not CONFIG_FILE.exists():
            print(f"Error: config file not found: {CONFIG_FILE}")
            sys.exit(1)
        print(f"Config: {CONFIG_FILE.name}\n")
        run_sweep(CONFIG_FILE, SWEEP_OUTPUT_DIR)

    elif MODE == "pcond_sweep":
        run_pcond_sweep(
            template_yaml=PCOND_TEMPLATES[PCOND_CYCLE],
            fluids=None if PCOND_USE_ALL_COOLPROP else PCOND_FLUIDS,
            output_dir=PCOND_OUTPUT_DIR,
            cycle_label=PCOND_CYCLE,
            turbine_model=PCOND_TURBINE_MODEL,
            n_stages=PCOND_N_STAGES,
            RPM=PCOND_RPM,
            warmstart_tcrit_K=PCOND_WARMSTART_TCRIT_K,
            dual_warmstart=PCOND_DUAL_WARMSTART,
            n_workers=PCOND_N_WORKERS,
            debug_fluids=PCOND_DEBUG_FLUIDS or None,
        )

    elif MODE == "k_sensitivity":
        run_k_sensitivity(
            transcritical_yaml=PCOND_TRANSCRITICAL_TEMPLATES[PCOND_CYCLE],
            tc_fluids=PCOND_TC_FLUIDS,
            k_values=KSENS_K_VALUES,
            output_dir=KSENS_OUTPUT_DIR,
            cycle_label=PCOND_CYCLE,
            turbine_model=PCOND_TURBINE_MODEL,
            n_stages=PCOND_N_STAGES,
            RPM=PCOND_RPM,
        )

    elif MODE == "multistart":

        # ── Sanity check: verify _evaluate_bound_expr against Toluene CoolProp values ──
        # Expected values are raw CoolProp properties at T_ambient=293.15 K, T_max=475.30 K.
        # IMPORTANT: these are the raw property values, NOT the warm-start values in the YAML.
        #   e.g. compressor_inlet_pressure warm start = 1.761479 * p_sat ≈ 5142 Pa — NOT p_sat.
        # Set _VERIFY_EXPR_EVAL = False once you have confirmed the implementation is correct.
        _VERIFY_EXPR_EVAL = False
        if _VERIFY_EXPR_EVAL:
            print("=" * 70)
            print("  Expression evaluation sanity check for Toluene")
            print("  (raw CoolProp values at T_amb=293.15 K, T_max=475.30 K)")
            print("=" * 70)
            _checks = [
                (
                    "1*$working_fluid.liquid_at_ambient_temperature.p",
                    2919.0,
                    "Pa   p_sat_ambient (Toluene vapour pressure at 20 C)",
                ),
                (
                    "6.0*$working_fluid.liquid_at_ambient_temperature.p",
                    17513.0,
                    "Pa   6*p_sat_ambient",
                ),
                ("1*$working_fluid.critical_point.p", 4126347, "Pa   p_crit"),
                ("5*$working_fluid.critical_point.p", 20631735, "Pa   5*p_crit"),
                (
                    "1*$working_fluid.gas_at_maximum_temperature.h",
                    513933.0,
                    "J/kg h_gas_max at T_max=475.30 K",
                ),
                (
                    "1*$working_fluid.liquid_at_ambient_temperature.h",
                    -166707.0,
                    "J/kg h_liq_ambient",
                ),
            ]
            all_ok = True
            for expr, expected, label in _checks:
                result = _evaluate_bound_expr(expr, "Toluene")
                if result is None:
                    print(f"  FAIL  {label:<35}  result=None  expected={expected}")
                    all_ok = False
                else:
                    rel_err = abs(result - expected) / abs(expected) * 100
                    status = "OK  " if rel_err < 1.0 else "WARN"
                    print(
                        f"  {status}  {label:<35}  result={result:>14.2f}  expected={expected:>14.2f}  err={rel_err:.3f}%"
                    )
                    if rel_err >= 1.0:
                        all_ok = False
            print("=" * 70)
            print(
                f"  Result: {'ALL CHECKS PASSED' if all_ok else 'SOME CHECKS FAILED — review _evaluate_bound_expr'}"
            )
            print("=" * 70)
            print()

        if not MULTISTART_YAML_FILES:
            print("Error: MULTISTART_YAML_FILES is empty — add at least one YAML path.")
            sys.exit(1)
        missing = [
            f
            for f in MULTISTART_YAML_FILES
            if not (Path(f[0]) if isinstance(f, tuple) else Path(f)).exists()
        ]
        # Validate any per-entry n_samples overrides are positive integers
        for _e in MULTISTART_YAML_FILES:
            if isinstance(_e, tuple) and len(_e) >= 3 and _e[2] is not None:
                if not isinstance(_e[2], int) or _e[2] < 1:
                    print(
                        f"Error: per-entry n_samples must be a positive integer, got {_e[2]!r} in {_e}"
                    )
                    sys.exit(1)
        if missing:
            for f in missing:
                print(f"Error: config file not found: {f}")
            sys.exit(1)

        if len(MULTISTART_YAML_FILES) == 1:
            # ── Single file: run sequentially in the main process ─────────
            entry = MULTISTART_YAML_FILES[0]
            if isinstance(entry, tuple):
                yaml_file = Path(entry[0])
                fluid = entry[1]
                n_samples_entry = (
                    int(entry[2])
                    if len(entry) >= 3 and entry[2] is not None
                    else MULTISTART_N_SAMPLES
                )
            else:
                yaml_file, fluid = Path(entry), None
                n_samples_entry = MULTISTART_N_SAMPLES
            label = yaml_file.stem + (f"_{fluid}" if fluid else "")
            run_multistart(
                config_file=yaml_file,
                n_samples=n_samples_entry,
                output_dir=Path(MULTISTART_OUTPUT_DIR) / label,
                fluid_override=fluid,
            )
        else:
            # ── Multiple files: run in parallel using the batch worker ────
            import multiprocessing as mp
            from scipy.stats.qmc import LatinHypercube
            from ruamel.yaml import YAML

            ryaml_main = YAML()
            ryaml_main.preserve_quotes = True

            entry_meta = []
            for entry in MULTISTART_YAML_FILES:
                if isinstance(entry, tuple):
                    yaml_file = Path(entry[0])
                    fluid = entry[1]
                    n_samples_entry = (
                        int(entry[2])
                        if len(entry) >= 3 and entry[2] is not None
                        else MULTISTART_N_SAMPLES
                    )
                else:
                    yaml_file, fluid = Path(entry), None
                    n_samples_entry = MULTISTART_N_SAMPLES

                label = yaml_file.stem + (f"_{fluid}" if fluid else "")
                fluid_dir = Path(MULTISTART_OUTPUT_DIR) / label

                with open(yaml_file) as _f:
                    cfg = ryaml_main.load(_f)
                if fluid is not None:
                    cfg["problem_formulation"]["fixed_parameters"]["working_fluid"][
                        "name"
                    ] = fluid

                dvars = cfg["problem_formulation"]["design_variables"]
                var_names = list(dvars.keys())

                # Extract fluid name and T_max for expression-bound evaluation
                _fluid_name = cfg["problem_formulation"]["fixed_parameters"][
                    "working_fluid"
                ]["name"]
                _T_max = (
                    cfg["problem_formulation"]["fixed_parameters"]
                    .get("special_points", {})
                    .get("maximum_temperature", 475.30)
                )
                if not isinstance(_T_max, (int, float)):
                    _T_max = 475.30

                # Compute narrow bounds for every design variable.
                # _narrow_bounds raises ValueError (variable name + reason) if
                # any variable's window cannot be computed.  We catch per-variable
                # so the error message identifies exactly which variable failed.
                bounds = []
                bound_error = None
                for v in var_names:
                    try:
                        bounds.append(
                            _narrow_bounds(
                                dvars[v],
                                var_name=v,
                                fluid_name=_fluid_name,
                                T_max=_T_max,
                            )
                        )
                    except ValueError as _be:
                        bound_error = str(_be)
                        break

                sampler = LatinHypercube(d=len(var_names), seed=42)
                samples = sampler.random(n=n_samples_entry).tolist()

                entry_meta.append(
                    (
                        yaml_file,
                        fluid,
                        label,
                        fluid_dir,
                        var_names,
                        bounds,
                        samples,
                        bound_error,
                        n_samples_entry,
                    )
                )

            tasks = []
            for (
                yaml_file,
                fluid,
                label,
                fluid_dir,
                var_names,
                bounds,
                samples,
                bound_error,
                n_samples_entry,
            ) in entry_meta:
                if bound_error is not None:
                    # Bounds failed — no tasks created; rows pre-populated below
                    continue
                for run_index, sample in enumerate(samples):
                    tasks.append(
                        (
                            str(yaml_file),
                            fluid,
                            var_names,
                            sample,
                            bounds,
                            run_index,
                            label,
                            str(fluid_dir),
                        )
                    )

            n_entries = len(entry_meta)
            n_skipped = sum(1 for m in entry_meta if m[7] is not None)
            n_active = n_entries - n_skipped
            n_total = len(tasks)
            n_workers = min(max(n_total, 1), mp.cpu_count(), 6)
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            # Build a label → n_samples lookup for progress and summary printing
            _n_samp_for = {m[2]: m[8] for m in entry_meta}

            # Annotation: show per-entry counts if they differ
            _all_n = [m[8] for m in entry_meta]
            if len(set(_all_n)) == 1:
                # All the same — compact display
                _samp_line = f"  Samples per entry : {_all_n[0]}"
                if n_skipped == 0:
                    task_annotation = f"{n_entries} × {_all_n[0]}"
                else:
                    task_annotation = (
                        f"{n_active} × {_all_n[0]}"
                        f"  ({n_skipped} entr{{'y' if n_skipped == 1 else 'ies'}} skipped — bound error)"
                    )
            else:
                # Mixed — list each entry
                _samp_line = "  Samples per entry : (varies — see below)"
                task_annotation = (
                    f"{n_total} total ({n_skipped} skipped)"
                    if n_skipped
                    else f"{n_total} total"
                )

            print("=" * 70)
            print(f"  MULTISTART (parallel) — {n_entries} YAML files")
            print(_samp_line)
            if len(set(_all_n)) > 1:
                for m in entry_meta:
                    skip_tag = "  [BOUND ERROR — will skip]" if m[7] is not None else ""
                    print(f"    {m[2][:50]:<50}  {m[8]} samples{skip_tag}")
            print(f"  Total tasks       : {n_total}  ({task_annotation})")
            print(
                f"  Workers           : {n_workers} (of {mp.cpu_count()} CPUs available)"
            )
            print(f"  Output root       : {MULTISTART_OUTPUT_DIR}")
            print("=" * 70)
            print()

            # Pre-populate failed rows for any entry whose bounds computation
            # raised — these entries have no tasks and never enter the pool.
            collected = {m[2]: [] for m in entry_meta}
            for (
                yaml_file,
                fluid,
                label,
                fluid_dir,
                var_names,
                bounds,
                samples,
                bound_error,
                n_samples_entry,
            ) in entry_meta:
                if bound_error is not None:
                    print(f"  [SKIP] {label} — bound error: {bound_error}")
                    for run_index in range(n_samples_entry):
                        collected[label].append(
                            {
                                "run_label": label,
                                "run": run_index + 1,
                                "config_file_str": str(yaml_file),
                                "fluid_override": fluid,
                                "config_name": yaml_file.stem,
                                "fluid_name": fluid,
                                "converged": False,
                                "eta_system": None,
                                "eta_cycle": None,
                                "eta_exergy": None,
                                "heat_utilization": None,
                                "W_net_kW": None,
                                "is_transcritical": None,
                                "error": f"bound_error: {bound_error}",
                                "t_elapsed_s": None,
                                "n_iterations": None,
                                "n_func_evals": None,
                                "infeasibility": None,
                                "exit_message": None,
                                "_x0_dict": None,
                                "_lhs_yaml_text": None,
                            }
                        )

            n_done = 0
            if n_total == 0:
                print("  (all entries had bound errors — no tasks to run)")
            else:
                with mp.Pool(processes=n_workers) as pool:
                    for row in pool.imap_unordered(_multistart_sample_worker, tasks):
                        n_done += 1
                        lbl = row["run_label"]
                        collected[lbl].append(row)
                        eta = row.get("eta_system")
                        if row["converged"] and eta is not None:
                            status = f"✓  η_sys={eta*100:.4f}%   W_net={row.get('W_net_kW', 0):.1f} kW   ({row.get('t_elapsed_s', 0):.0f}s)"
                        else:
                            err_short = row.get("error") or "did not converge"
                            err_short = err_short.strip().split("\n")[-1][:55]
                            status = f"✗  {err_short}"
                        print(
                            f"  [{n_done:>4}/{n_total}]  {lbl[:42]:<42}  "
                            f"run {row['run']:>3}/{_n_samp_for.get(lbl, MULTISTART_N_SAMPLES)}   {status}"
                        )

            print()
            print("=" * 70)
            print("  Saving per-entry outputs …")
            print("=" * 70)

            summary_rows = []
            for (
                yaml_file,
                fluid,
                label,
                fluid_dir,
                var_names,
                bounds,
                samples,
                bound_error,
                n_samples_entry,
            ) in entry_meta:
                rows = sorted(collected[label], key=lambda r: r["run"])
                (
                    csv_path,
                    xlsx_path,
                    lhs_yaml_path,
                    optimized_yaml_path,
                    n_ok,
                    best_eta,
                ) = _save_multistart_entry_outputs(
                    config_file=yaml_file,
                    fluid_override=fluid,
                    run_label=label,
                    output_dir=fluid_dir,
                    rows=rows,
                    ryaml=ryaml_main,
                    timestamp=timestamp,
                )
                n_bad = n_samples_entry - n_ok
                print(f"\n  ── {label}")
                print(f"     Converged : {n_ok}/{n_samples_entry}   Failed: {n_bad}")
                if best_eta is not None:
                    print(f"     Best η_sys: {best_eta*100:.4f}%")
                print(f"     CSV       : {csv_path.name}")
                print(f"     Excel     : {xlsx_path.name}")
                if lhs_yaml_path:
                    print(f"     LHS start : {lhs_yaml_path.name}")
                if optimized_yaml_path:
                    print(
                        f"     Optimized : {optimized_yaml_path.name}  ← use as warm start"
                    )
                summary_rows.append(
                    (label, n_ok, n_samples_entry, best_eta, optimized_yaml_path)
                )

            print()
            print("=" * 70)
            print("  MULTISTART — FINAL SUMMARY")
            print("=" * 70)
            print(
                f"  {'Entry':<50}  {'Conv':>5}  {'Best η_sys':>11}  {'Optimized YAML'}"
            )
            print(f"  {'─'*50}  {'─'*5}  {'─'*11}  {'─'*30}")
            for lbl, n_ok, n_samp, best_eta, opt_path in summary_rows:
                eta_str = f"{best_eta*100:.4f}%" if best_eta is not None else "—"
                opt_str = opt_path.name if opt_path else "none converged"
                print(f"  {lbl:<50}  {n_ok:>2}/{n_samp:<2}  {eta_str:>11}  {opt_str}")
            print("=" * 70)
            n_entries_ok = sum(1 for _, n_ok, _, _, _ in summary_rows if n_ok > 0)
            print(f"  Entries with ≥1 converged run : {n_entries_ok}/{n_entries}")
            print(f"  Output root                   : {MULTISTART_OUTPUT_DIR}")
            print("=" * 70)

    elif MODE == "parametric_study":
        if not PARAMETRIC_BASE_YAML.exists():
            print(f"Error: base YAML not found: {PARAMETRIC_BASE_YAML}")
            sys.exit(1)
        run_parametric_nstages_rpm(
            base_yaml=PARAMETRIC_BASE_YAML,
            n_stages_list=PARAMETRIC_N_STAGES_LIST,
            rpm_list=PARAMETRIC_RPM_LIST,
            output_dir=PARAMETRIC_OUTPUT_DIR,
            interp_var_prefix=PARAMETRIC_INTERP_PREFIX,
            interp_p_min_expr=PARAMETRIC_INTERP_P_MIN_EXPR,
            interp_p_max_expr=PARAMETRIC_INTERP_P_MAX_EXPR,
            expander_inlet_p_var=PARAMETRIC_EXPANDER_INLET_P_VAR,
            expander_outlet_p_var=PARAMETRIC_EXPANDER_OUTLET_P_VAR,
        )
