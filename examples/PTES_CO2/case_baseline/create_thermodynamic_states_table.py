import pandas as pd
import numpy as np

# File path
file_path = "results_baseline/optimal_solution.xlsx"
output_txt = "component_states_summary.txt"

# Read Excel file: use row 0 as header, skip row 1 (units)
df = pd.read_excel(file_path, sheet_name="cycle_states", header=0, skiprows=[1])

# Filter for inlet and outlet states
df = df[df["state"].str.endswith(("_in", "_out"))].copy()

# Extract component and location
df["component"] = (
    df["state"]
    .str.extract(r"(.+)_in|(.+)_out")[0]
    .combine_first(df["state"].str.extract(r"(.+)_in|(.+)_out")[1])
)
df["location"] = df["state"].str.extract(r".*_(in|out)")

# Create variable label
df["Variable"] = (
    (df["component"] + " " + df["location"]).str.replace("_", " ").str.capitalize()
)

# Convert units
df["Pressure (bar)"] = df["pressure"] / 1e5
df["Temperature (C)"] = df["temperature"] - 273.15
df["Density (kg/m3)"] = df["density"]

# Organize output
columns = ["Variable", "Pressure (bar)", "Temperature (C)", "Density (kg/m3)"]
df_out = df[columns]

# Split into charge/discharge
charge_df = df_out[df_out["Variable"].str.contains("charge")]
discharge_df = df_out[df_out["Variable"].str.contains("discharge")]


# Format helper
def fmt(x):
    return f"{x:.3f}" if isinstance(x, (float, int, np.floating)) else str(x)


# Header
lines = []
header = f"{'Variable':35s} {'Pressure (bar)':>15} {'Temperature (C)':>20} {'Density (kg/m3)':>20}"
lines.append("-" * len(header))
lines.append(header)
lines.append("-" * len(header))

# Add charge section
lines.append("Charge cycle")
lines.append("-" * len(header))
for _, row in charge_df.iterrows():
    if " charge" in row["Variable"].lower():
        label = row["Variable"].replace(" charge", "").strip().capitalize()
        line = f"{label:35s} {fmt(row['Pressure (bar)']):>15} {fmt(row['Temperature (C)']):>20} {fmt(row['Density (kg/m3)']):>20}"
        lines.append(line)

# Add discharge section
lines.append("-" * len(header))
lines.append("Discharge cycle")
lines.append("-" * len(header))
for _, row in discharge_df.iterrows():
    if " discharge" in row["Variable"].lower():
        label = row["Variable"].replace(" discharge", "").strip().capitalize()
        line = f"{label:35s} {fmt(row['Pressure (bar)']):>15} {fmt(row['Temperature (C)']):>20} {fmt(row['Density (kg/m3)']):>20}"
        lines.append(line)

lines.append("-" * len(header))

# Save and print
with open(output_txt, "w") as f:
    for line in lines:
        print(line)
        f.write(line + "\n")
