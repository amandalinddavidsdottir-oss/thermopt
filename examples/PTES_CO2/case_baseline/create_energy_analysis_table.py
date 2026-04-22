import thermopt as th
import os

# Load solver
SOLVER_PATH = "results_baseline/solver.pkl"
solver = th.load_from_pickle(SOLVER_PATH)
energy = solver.problem.cycle_data["energy_analysis"]

# Output file
OUTPUT_FILE = "table_energy_analysis.txt"

# Initialize writer
with open(OUTPUT_FILE, "w") as f:

    def write(text=""):
        print(text)
        f.write(text + "\n")

    # Conversion and formatting
    def convert(key, value):
        if "power" in key or "heat_flow" in key:
            return value / 1e6, "MW"
        elif "temperature" in key:
            return value - 273.15, "°C"
        elif "mass_flow" in key:
            return value, "kg/s"
        else:
            return value, "-"

    def fmt(x):
        return f"{x:.3f}" if isinstance(x, (float, int)) else str(x)

    # Filters
    skip = ["energy_balance", "backwork_ratio", "pressure", "temperature", "efficiency"]

    # Sort order
    categories = {
        "mass_flow": 0,
        "heat_flow": 1,
        "expander_power": 2,
        "compressor_power": 3,
        "net_cycle_power": 4,
    }

    def sort_key(key):
        for cat, rank in categories.items():
            if cat in key:
                return rank
        return 99

    # Split keys
    charge_keys = sorted(
        [k for k in energy if k.endswith("_charge") and all(s not in k for s in skip)],
        key=sort_key,
    )
    discharge_keys = sorted(
        [
            k
            for k in energy
            if k.endswith("_discharge") and all(s not in k for s in skip)
        ],
        key=sort_key,
    )

    # Header
    write("-" * 75)
    write(f"{'Variable':50s} {'Value':>12} {'Unit':>10}")

    # Charge section
    write("-" * 75)
    write("Charge cycle")
    write("-" * 75)
    for key in charge_keys:
        val, unit = convert(key, energy[key])
        label = key.replace("_", " ").capitalize()
        write(f"{label:50s} {fmt(val):>12} {unit:>10}")

    # Discharge section
    write("-" * 75)
    write("Discharge cycle")
    write("-" * 75)
    for key in discharge_keys:
        val, unit = convert(key, energy[key])
        label = key.replace("_", " ").capitalize()
        write(f"{label:50s} {fmt(val):>12} {unit:>10}")

    # Final items
    write("-" * 75)
    write("Overall system metrics")
    write("-" * 75)
    metrics = {
        "COP_heat_pump": "Heat pump coefficient of performance",
        "cycle_efficiency_discharge": "Heat engine thermal efficiency",
        "roundtrip_efficiency": "PTES roundtrip efficiency",
    }

    for key, label in metrics.items():
        val, unit = convert(key, energy[key])
        write(f"{label:50s} {fmt(val):>12} {unit:>10}")
    write("-" * 75)
