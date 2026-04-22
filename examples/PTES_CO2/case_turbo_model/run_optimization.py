import os
import numpy as np
import thermopt as th

CONFIG_FILE_TURBO = "../case_PTES_CO2_turbo.yaml"
CONFIG_FILE_FIXED = "../case_PTES_CO2.yaml"
OUT_DIR_BASE = "results"
os.makedirs(OUT_DIR_BASE, exist_ok=True)
DATA_FILE = "simulation_data.pkl"
DATA_FULLPATH = os.path.join(OUT_DIR_BASE, DATA_FILE)

# -------------------------------------------------------------------------- #
# Check if saved data exists; if not, run simulations and save
# -------------------------------------------------------------------------- #

if not os.path.exists(DATA_FULLPATH):
    print("Running turbo and fixed-efficiency optimizations...")

    # Turbo model optimization
    cycle_turbo = th.ThermodynamicCycleOptimization(
        CONFIG_FILE_TURBO, out_dir=os.path.join(OUT_DIR_BASE, "turbo")
    )
    cycle_turbo.set_config_value("solver_options.callbacks.plot_cycle", True)
    # cycle_turbo.set_config_value("solver_options.max_iterations", 5)
    cycle_turbo.run_optimization()
    cycle_turbo.save_results()

    # Extract efficiencies
    eta_compressor_charge = cycle_turbo.problem.cycle_data["components"][
        "compressor_charge"
    ]["data_out"]["isentropic_efficiency"]
    eta_compressor_discharge = cycle_turbo.problem.cycle_data["components"][
        "compressor_discharge"
    ]["data_out"]["isentropic_efficiency"]
    eta_expander_charge = cycle_turbo.problem.cycle_data["components"][
        "expander_charge"
    ]["data_out"]["isentropic_efficiency"]
    eta_expander_discharge = cycle_turbo.problem.cycle_data["components"][
        "expander_discharge"
    ]["data_out"]["isentropic_efficiency"]

    # Fixed-efficiency optimization
    cycle_fixed = th.ThermodynamicCycleOptimization(
        CONFIG_FILE_FIXED, out_dir=os.path.join(OUT_DIR_BASE, "fixed")
    )
    cycle_fixed.set_config_value("solver_options.callbacks.plot_cycle", True)
    # cycle_fixed.set_config_value("solver_options.max_iterations", 5)
    cycle_fixed.set_config_value(
        "problem_formulation.fixed_parameters.compressor_charge.efficiency",
        eta_compressor_charge,
    )
    cycle_fixed.set_config_value(
        "problem_formulation.fixed_parameters.compressor_discharge.efficiency",
        eta_compressor_discharge,
    )
    cycle_fixed.set_config_value(
        "problem_formulation.fixed_parameters.expander_charge.efficiency",
        eta_expander_charge,
    )
    cycle_fixed.set_config_value(
        "problem_formulation.fixed_parameters.expander_discharge.efficiency",
        eta_expander_discharge,
    )
    cycle_fixed.run_optimization()
    cycle_fixed.save_results()

    # Save both to pickle
    cycles = {
        "turbo": cycle_turbo,
        "fixed": cycle_fixed,
    }
    th.save_to_pickle(cycles, DATA_FULLPATH, timestamp=False)
    print("Simulations completed and saved.")

else:
    print("Loading existing simulation results...")
    cycles = th.load_from_pickle(DATA_FULLPATH)


# -------------------------------------------------------------------------- #
# Report summary table from saved or loaded results
# -------------------------------------------------------------------------- #

import numpy as np

# Variables to extract and convert
STATE_VARS = {
    "p": ("pressure", "bar", lambda x: x / 1e5),
    "T": ("temperature", "degC", lambda x: x - 273.15),
}
EFFICIENCY_VAR = (
    "isentropic_efficiency",
    "Isentropic efficiency",
    "%",
    lambda x: x * 100,
)

components = [
    "compressor_charge",
    "compressor_discharge",
    "expander_charge",
    "expander_discharge",
]


# Helper function for formatting
def fmt(x):
    return f"{x:.3f}" if isinstance(x, (int, float, np.floating)) else str(x)


# Define column widths
col1_width = 30
col2_width = 10
col3_width = 20
col4_width = 20
col5_width = 20

# Header
print("-" * (col1_width + col2_width + col3_width + col4_width + col5_width))
print("Summary table for pressures, temperatures, and efficiencies:")
print("-" * (col1_width + col2_width + col3_width + col4_width + col5_width))
print(
    f"{'Variable':{col1_width}s}{'Unit':>{col2_width}s}{'Modeled efficiency':>{col3_width}s}{'Fixed efficiency':>{col4_width}s}{'Deviation [%]':>{col5_width}s}"
)
print("-" * (col1_width + col2_width + col3_width + col4_width + col5_width))

# Loop through components
for comp in components:
    turbo_state_in = cycles["turbo"].problem.cycle_data["components"][comp]["state_in"]
    turbo_state_out = cycles["turbo"].problem.cycle_data["components"][comp][
        "state_out"
    ]
    fixed_state_in = cycles["fixed"].problem.cycle_data["components"][comp]["state_in"]
    fixed_state_out = cycles["fixed"].problem.cycle_data["components"][comp][
        "state_out"
    ]

    turbo_data_out = cycles["turbo"].problem.cycle_data["components"][comp]["data_out"]
    fixed_data_out = cycles["fixed"].problem.cycle_data["components"][comp]["data_out"]

    # Print header row for this component
    print(f"{comp.replace('_', ' ').capitalize()}")

    # Pressure and temperature at inlet/outlet
    for var_key, (label, unit, convert) in STATE_VARS.items():
        for loc, turbo_val_raw, fixed_val_raw in [
            ("Inlet", turbo_state_in.get(var_key), fixed_state_in.get(var_key)),
            ("Outlet", turbo_state_out.get(var_key), fixed_state_out.get(var_key)),
        ]:
            turbo_val = convert(turbo_val_raw) if turbo_val_raw is not None else None
            fixed_val = convert(fixed_val_raw) if fixed_val_raw is not None else None
            delta = 100 * (fixed_val - turbo_val) / turbo_val if turbo_val else np.nan
            var_label = f"   {loc} {label}"
            print(
                f"{var_label:{col1_width}s}{unit:>{col2_width}s}{fmt(turbo_val):>{col3_width}s}{fmt(fixed_val):>{col4_width}s}{fmt(delta):>{col5_width}s}"
            )

    # Isentropic efficiency (percentage)
    turbo_eff_raw = turbo_data_out.get(EFFICIENCY_VAR[0])
    fixed_eff_raw = fixed_data_out.get(EFFICIENCY_VAR[0])
    turbo_eff = EFFICIENCY_VAR[3](turbo_eff_raw) if turbo_eff_raw is not None else None
    fixed_eff = EFFICIENCY_VAR[3](fixed_eff_raw) if fixed_eff_raw is not None else None
    delta_eff = 100 * (fixed_eff - turbo_eff) / turbo_eff if turbo_eff else np.nan
    label = f"   {EFFICIENCY_VAR[1]}"
    print(
        f"{label:{col1_width}s}{EFFICIENCY_VAR[2]:>{col2_width}s}{fmt(turbo_eff):>{col3_width}s}{fmt(fixed_eff):>{col4_width}s}{fmt(delta_eff):>{col5_width}s}"
    )
    print("-" * (col1_width + col2_width + col3_width + col4_width + col5_width))
