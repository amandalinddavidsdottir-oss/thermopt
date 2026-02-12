"""
ORC Brine Calculation
Calculates the brine mass flow and properties going into the ORC plant
from well data (WHP, total flow, enthalpy) using CoolProp steam tables.
"""

from CoolProp.CoolProp import PropsSI

# ---- Input data ----

P_atm = 1.01325  # atmospheric pressure in bar

# Well data from spreadsheet: (WHP in bar-g, enthalpy in kJ/kg, total flow in kg/s)
well_names = ["Well 1", "Well 2", "Well 3", "Well 4"]
WHP =        [25.1,     34.8,     22.6,     17.0]       # bar-g
h_total =    [1517.2,   1845.8,   1509.1,   1156.9]     # kJ/kg
m_total =    [29.1,     21.0,     37.3,     38.5]        # kg/s

T_reinjection = 100  # degrees C


# ---- Print input data ----

print("INPUT DATA")
print("=" * 60)
for i in range(len(well_names)):
    print(f"  {well_names[i]}:  WHP = {WHP[i]} bar-g,  h = {h_total[i]} kJ/kg,  m = {m_total[i]} kg/s")
print(f"  Reinjection temperature = {T_reinjection} °C")


# ---- Equations used ----

print("\nEQUATIONS")
print("=" * 60)
print("  x = (h_total - h_f) / (h_g - h_f)        steam quality")
print("  m_steam = m_total * x")
print("  m_brine = m_total * (1 - x)")
print("  h_brine = h_f at WHP                      saturated liquid")
print("  Throttling is isenthalpic:  h_out = h_in")
print("  h_mix = sum(m_i * h_i) / sum(m_i)         adiabatic mixing")
print("  Q = m_mix * (h_mix - h_reinj)              available heat")


# ---- Step 1: Split each well into steam and brine ----

print("\nSTEP 1: STEAM/BRINE SPLIT")
print("=" * 60)

m_brine = []
h_brine = []

for i in range(len(well_names)):
    P = WHP[i] + P_atm  # convert to bar absolute

    # look up steam table values at this pressure
    hf = PropsSI('H', 'P', P * 1e5, 'Q', 0, 'Water') / 1000   # saturated liquid enthalpy
    hg = PropsSI('H', 'P', P * 1e5, 'Q', 1, 'Water') / 1000   # saturated vapor enthalpy

    # calculate steam quality
    x = (h_total[i] - hf) / (hg - hf)

    # split the flow
    steam = m_total[i] * x
    brine = m_total[i] * (1 - x)

    m_brine.append(brine)
    h_brine.append(hf)  # brine leaves separator as saturated liquid

    print(f"  {well_names[i]}:  P = {P:.2f} bar-a,  hf = {hf:.1f},  hg = {hg:.1f},  x = {x:.3f},  m_brine = {brine:.1f} kg/s")


# ---- Step 2: Throttle all brine streams to the lowest well pressure ----

print("\nSTEP 2: THROTTLE TO SYSTEM PRESSURE")
print("=" * 60)

P_system = min(WHP) + P_atm
T_system = PropsSI('T', 'P', P_system * 1e5, 'Q', 0, 'Water') - 273.15
hf_system = PropsSI('H', 'P', P_system * 1e5, 'Q', 0, 'Water') / 1000
hg_system = PropsSI('H', 'P', P_system * 1e5, 'Q', 1, 'Water') / 1000

print(f"  System pressure = {P_system:.2f} bar-a  (T_sat = {T_system:.1f} °C)")

m_brine_throttled = []

for i in range(len(well_names)):
    # throttling is isenthalpic, so h stays the same
    # if h_brine > hf at system pressure, some of it flashes to steam
    if h_brine[i] > hf_system:
        x_flash = (h_brine[i] - hf_system) / (hg_system - hf_system)
        m_liquid = m_brine[i] * (1 - x_flash)
        print(f"  {well_names[i]}:  flash = {x_flash*100:.2f}%,  m_liquid = {m_liquid:.1f} kg/s")
    else:
        m_liquid = m_brine[i]
        print(f"  {well_names[i]}:  no flash,  m_liquid = {m_liquid:.1f} kg/s")

    m_brine_throttled.append(m_liquid)


# ---- Step 3: Mix all brine streams ----
# After throttling, all streams are saturated liquid at system pressure
# so they all have h = hf_system and T = T_sat at system pressure.
# Mixing just adds up the flows.

m_mix = sum(m_brine_throttled)
h_mix = hf_system  # all streams are at the same enthalpy
T_mix = T_system

h_reinj = PropsSI('H', 'T', T_reinjection + 273.15, 'P', P_system * 1e5, 'Water') / 1000
Q_available = m_mix * (h_mix - h_reinj) / 1000  # MW


# ---- Results ----

print("\nORC BRINE INPUT PARAMETERS")
print("=" * 60)
print(f"  Pressure:        {P_system:.2f} bar-a")
print(f"  Temperature:     {T_mix:.1f} °C")
print(f"  Enthalpy:        {h_mix:.1f} kJ/kg")
print(f"  Mass flow:       {m_mix:.1f} kg/s")
print(f"  Reinjection T:   {T_reinjection} °C")
print(f"  Available heat:  {Q_available:.2f} MW_th")
print("=" * 60)
