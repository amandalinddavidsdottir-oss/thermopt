import os
import numpy as np
import pandas as pd
import thermopt as th

th.print_package_info()

CONFIG_FILE = "../case_PTES_CO2_turbo.yaml"
OUT_DIR_BASE = "results"
os.makedirs(OUT_DIR_BASE, exist_ok=True)

DATA_FILE = "simulation_data.pkl"
DATA_FULLPATH = os.path.join(OUT_DIR_BASE, DATA_FILE)

# -------------------------------------------------------------------------- #
# Check if saved data exists; if not, run simulations and save
# -------------------------------------------------------------------------- #
if not os.path.exists(DATA_FULLPATH):
    print("Running simulations...")

    # Unconstrained optimization
    cycle_unconstrained = th.ThermodynamicCycleOptimization(
        CONFIG_FILE, out_dir=os.path.join(OUT_DIR_BASE, "unconstrained")
    )
    cycle_unconstrained.set_config_value("solver_options.callbacks.plot_cycle", True)
    cycle_unconstrained.run_optimization()
    cycle_unconstrained.save_results()

    # Constrained optimization
    cycle_constrained = th.ThermodynamicCycleOptimization(
        CONFIG_FILE, out_dir=os.path.join(OUT_DIR_BASE, "constrained")
    )
    cycle_constrained.set_config_value("solver_options.callbacks.plot_cycle", True)

    cycle_constrained.set_constraint(
        variable="$components.expander_charge.data_out.angular_speed - $components.compressor_charge.data_out.angular_speed",
        type="=",
        value=0.0,
        normalize=1000,
    )
    cycle_constrained.set_constraint(
        variable="$components.expander_discharge.data_out.angular_speed - $components.compressor_discharge.data_out.angular_speed",
        type="=",
        value=0.0,
        normalize=1000,
    )

    cycle_constrained.run_optimization(x0=cycle_unconstrained.solver.x_final)
    cycle_constrained.save_results()

    # Save both simulations to pickle
    cycles = {
        "unconstrained": cycle_unconstrained,
        "constrained": cycle_constrained,
    }
    th.save_to_pickle(cycles, DATA_FULLPATH, timestamp=False)
    print("Both simulations saved.")
else:
    print("Loading existing simulation data...")
    cycles = th.load_from_pickle(DATA_FULLPATH)

# -------------------------------------------------------------------------- #
# Report summary table from saved or loaded results
# -------------------------------------------------------------------------- #


# Define what to extract
VARIABLES = {
    "compressor": {
        "isentropic_efficiency": ("isentropic efficiency", "-"),
        "flow_coefficient": ("flow coefficient", "-"),
        "backsweep": ("backsweep angle", "degree"),
        "angular_speed": ("rotational speed", "RPM"),
    },
    "expander": {
        "isentropic_efficiency": ("isentropic efficiency", "-"),
        "specific_speed": ("specific speed", "-"),
        "angular_speed": ("rotational speed", "RPM"),
    },
}


# Helper: format float nicely or fallback
def fmt(val):
    return f"{val:.3f}" if isinstance(val, (float, int)) else str(val)


# Header
print("-" * 125)
print("Summary table for compressor and expander performance.")
print("-" * 125)
print(
    f"{'Variable':60s} {'Unit':>10} {'Unconstrained':>15} {'Constrained':>15} {'Change [%]':>15}"
)

# Build table rows with subheaders
for role in ["charge", "discharge"]:
    # Subheader for each cycle
    print("-" * 125)
    print(f"{role.capitalize()} cycle")
    print("-" * 125)

    for machine in ["compressor", "expander"]:
        comp_name = f"{machine}_{role}"
        for var_key, (label, unit) in VARIABLES[machine].items():
            var_label = f"{machine.capitalize()} {label}"
            row = [var_label, unit]
            values = []
            for case in ["unconstrained", "constrained"]:
                data = cycles[case].problem.cycle_data["components"][comp_name][
                    "data_out"
                ]
                value = data.get(var_key)
                if var_key == "angular_speed" and value is not None:
                    value = value * 60 / (2 * np.pi)  # rad/s → RPM
                values.append(value)
                row.append(fmt(value))
            # Compute relative deviation
            try:
                if values[0] is not None and values[1] is not None and values[0] != 0:
                    deviation = 100 * (values[1] - values[0]) / values[0]
                    row.append(fmt(deviation))
                else:
                    row.append("-")
            except Exception:
                row.append("-")
            print(f"{row[0]:60s} {row[1]:>10} {row[2]:>15} {row[3]:>15} {row[4]:>15}")

# Add roundtrip efficiency row
print("-" * 125)
label = "System roundtrip efficiency"
unit = "%"
row = [label, unit]
values = []
for case in ["unconstrained", "constrained"]:
    data = cycles[case].problem.cycle_data["energy_analysis"]
    value = 100 * data.get("roundtrip_efficiency")
    values.append(value)
    row.append(fmt(value))
try:
    if values[0] is not None and values[1] is not None and values[0] != 0:
        deviation = values[1] - values[0]
        row.append(fmt(deviation))
    else:
        row.append("-")
except Exception:
    row.append("-")
print(f"{row[0]:60s} {row[1]:>10} {row[2]:>15} {row[3]:>15} {row[4]:>15}")
print("-" * 125)

# # Header
# print("-" * 125)
# print("Summary table for compressor and expander performance.")
# print("-" * 125)
# print(f"{'Variable':60s} {'Unit':>10} {'Unconstrained':>15} {'Constrained':>15} {'Change [%]':>15}")
# print("-" * 125)

# # Build table rows
# for role in ["charge", "discharge"]:
#     for machine in ["compressor", "expander"]:
#         comp_name = f"{machine}_{role}"
#         for var_key, (label, unit) in VARIABLES[machine].items():
#             var_label = f"{machine.capitalize()} {role} {label}"
#             row = [var_label, unit]
#             values = []
#             for case in ["unconstrained", "constrained"]:
#                 data = cycles[case].problem.cycle_data["components"][comp_name]["data_out"]
#                 value = data.get(var_key)
#                 if var_key == "angular_speed" and value is not None:
#                     value = value * 60 / (2 * np.pi)  # rad/s → RPM
#                 values.append(value)
#                 row.append(fmt(value))
#             # Compute relative deviation
#             try:
#                 if values[0] is not None and values[1] is not None and values[0] != 0:
#                     deviation = 100 * (values[1] - values[0]) / values[0]
#                     row.append(fmt(deviation))
#                 else:
#                     row.append("-")
#             except Exception:
#                 row.append("-")
#             print(f"{row[0]:60s} {row[1]:>10} {row[2]:>15} {row[3]:>15} {row[4]:>15}")

# # Add roundtrip efficiency row
# label = "System roundtrip efficiency"
# unit = "%"
# row = [label, unit]
# values = []
# for case in ["unconstrained", "constrained"]:
#     data = cycles[case].problem.cycle_data["energy_analysis"]
#     value = 100 * data.get("roundtrip_efficiency")
#     values.append(value)
#     row.append(fmt(value))
# try:
#     if values[0] is not None and values[1] is not None and values[0] != 0:
#         deviation = (values[1] - values[0])
#         row.append(fmt(deviation))
#     else:
#         row.append("-")
# except Exception:
#     row.append("-")
# print(f"{row[0]:60s} {row[1]:>10} {row[2]:>15} {row[3]:>15} {row[4]:>15}")

# print("-" * 125)
