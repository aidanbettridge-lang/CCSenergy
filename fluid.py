import streamlit as st
import numpy as np
import plotly.graph_objects as go
from CoolProp.CoolProp import PropsSI, PhaseSI
from CoolProp import __version__ as coolprop_version

# =====================================
# HEADER
# =====================================
st.image("https://ccsenergy.com.au/wp-content/themes/custom-theme/assets/images/logo.svg", width=250)
st.title("CO₂ Phase Envelope & Property Estimator")

st.write("""
This tool estimates **phase**, **density**, and **viscosity** of **pure CO₂**  
and plots its **phase envelope (vapor-liquid boundary)** with critical and triple points labeled.

**Backend:** CoolProp (HEOS)  
""")

# =====================================
# BACKEND CHECK
# =====================================
backend = "HEOS::CO2"
st.info(f"Using backend: **{backend}** (CoolProp v{coolprop_version})")

# =====================================
# USER INPUTS - RESERVOIR CONDITIONS
# =====================================
st.header("Reservoir Conditions")

temperature_c = st.number_input("Temperature (°C)", min_value=-80.0, max_value=200.0, value=40.0, format="%.2f")
pressure_mpa = st.number_input("Pressure (MPa)", min_value=0.01, max_value=50.0, value=10.0, format="%.2f")

T = temperature_c + 273.15  # K
P = pressure_mpa * 1e6       # Pa

# =====================================
# PROPERTY CALCULATIONS
# =====================================
try:
    density = PropsSI("D", "T", T, "P", P, backend)
    viscosity = PropsSI("VISCOSITY", "T", T, "P", P, backend)
    phase = PhaseSI("T", T, "P", P, backend)
    error_message = None
except Exception as e:
    density = viscosity = phase = None
    error_message = str(e)

# =====================================
# PHASE ENVELOPE CALCULATION (pure CO₂)
# =====================================
st.header("Phase Envelope")

T_min = 200.0   # K
T_max = 320.0   # K
T_points = np.linspace(T_min, T_max, 250)
P_sat = []

for T_i in T_points:
    try:
        P_i = PropsSI("P", "T", T_i, "Q", 0, backend)
        P_sat.append(P_i / 1e6)  # Pa → MPa
    except:
        P_sat.append(np.nan)

# Critical and triple points for CO₂
T_crit = PropsSI("TCRIT", backend)
P_crit = PropsSI("PCRIT", backend)
T_triple = PropsSI("T_TRIPLE", backend)
P_triple = PropsSI("P_TRIPLE", backend)

# =====================================
# PLOT PHASE ENVELOPE WITH LABELS
# =====================================
fig = go.Figure()

# Vapor-liquid boundary
fig.add_trace(go.Scatter(
    x=T_points - 273.15,
    y=P_sat,
    mode='lines',
    name='Vapor-Liquid Boundary',
    line=dict(color='royalblue', width=2)
))

# Critical point marker
fig.add_trace(go.Scatter(
    x=[T_crit - 273.15],
    y=[P_crit / 1e6],
    mode='markers+text',
    text=["Critical Point"],
    textposition="top center",
    marker=dict(color='orange', size=10, symbol='star'),
    name='Critical Point'
))

# Triple point marker
fig.add_trace(go.Scatter(
    x=[T_triple - 273.15],
    y=[P_triple / 1e6],
    mode='markers+text',
    text=["Triple Point"],
    textposition="bottom center",
    marker=dict(color='green', size=10, symbol='diamond'),
    name='Triple Point'
))

# Reservoir condition
fig.add_trace(go.Scatter(
    x=[temperature_c],
    y=[pressure_mpa],
    mode='markers+text',
    text=["Reservoir"],
    textposition="top center",
    marker=dict(color='red', size=10, symbol='circle'),
    name='Reservoir Condition'
))

fig.update_layout(
    title="CO₂ Phase Envelope",
    xaxis_title="Temperature (°C)",
    yaxis_title="Pressure (MPa)",
    template="plotly_white",
    height=650,
    legend=dict(
        bgcolor='rgba(255,255,255,0.8)',
        bordercolor='gray',
        borderwidth=0.5
    )
)

# =====================================
# OUTPUT RESULTS
# =====================================
st.header("Estimated CO₂ Properties at Reservoir Conditions")

if error_message:
    st.error(f"Error calculating properties: {error_message}")
else:
    st.write(f"**Density:** {density:,.2f} kg/m³")
    st.write(f"**Viscosity:** {viscosity*1000:,.4f} mPa·s")
    st.write(f"**Phase State:** {phase}")

st.plotly_chart(fig, use_container_width=True)

# =====================================
# FOOTER
# =====================================
st.markdown("---")
st.markdown("© 2025 CCS Energy Pty Ltd. All rights reserved.")
