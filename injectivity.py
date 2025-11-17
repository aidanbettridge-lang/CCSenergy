import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import pi, log
from CoolProp.CoolProp import PropsSI

# =====================================
# PAGE CONFIGURATION
# =====================================
st.set_page_config(page_title="CO₂ Injection — THP Estimator", layout="wide")
st.image("https://ccsenergy.com.au/wp-content/themes/custom-theme/assets/images/logo.svg", width=250)
st.title("CO₂ Injection Well Pressure & Temperature Estimator")

# =====================================
# CONSTANTS & HELPER FUNCTIONS
# =====================================
G = 9.81  # m/s²

def haaland_friction(Re, eD):
    if Re <= 0 or np.isnan(Re): return np.nan
    if Re < 2300: return 64.0 / Re
    return 1.0 / (-1.8 * np.log10((eD / 3.7)**1.11 + 6.9 / Re))**2

def to_pa(val, unit):
    return val * 1e5 if unit == 'bar' else val * 6894.76

def from_pa(val, unit):
    return val / 1e5 if unit == 'bar' else val / 6894.76

def length_to_m(val, unit):
    return val / 1000.0 if unit == 'mm' else val * 0.0254

# =====================================
# SIDEBAR INPUTS
# =====================================
with st.sidebar:
    st.header("Input Parameters")
    pressure_unit = st.selectbox("Pressure unit", ['bar', 'psi'], index=0)
    tubing_unit = st.selectbox("Tubing ID unit", ['mm', 'inch'], index=0)

    P_res_in = st.number_input(f"Reservoir pressure ({pressure_unit})", value=120.0)
    T_res_c = st.number_input("Reservoir temperature (°C)", value=60.0)
    T_surf_c = st.number_input("Surface injection temperature (°C)", value=20.0)

    MD = st.number_input("Measured depth (MD) of top perforation (m)", value=2500.0)
    TVD = st.number_input("True vertical depth (TVD) of top perforation (m)", value=2400.0)

    tubing_id_in = st.number_input(f"Tubing inner diameter ({tubing_unit})", value=88.9)
    tubing_rough_mm = st.number_input("Tubing roughness (mm)", value=0.05)
    skin = st.number_input("Skin factor", value=0.0)

    q_tpd = st.number_input("Injection rate (t/d)", value=1000.0)
    perm_mD = st.number_input("Permeability (mD)", value=100.0)
    poro = st.number_input("Porosity (fraction)", value=0.25)
    h_res = st.number_input("Reservoir thickness (m)", value=10.0)
    rw = st.number_input("Wellbore radius (m)", value=0.0889)
    re = st.number_input("Drainage radius (m)", value=100.0)

    n_segments = st.slider("Number of segments (for plotting)", 20, 1000, 200, 10)
    sens_percent = st.slider("Mass flow sensitivity (%)", 1, 100, 50)

# =====================================
# UNIT CONVERSIONS
# =====================================
P_res_pa = to_pa(P_res_in, pressure_unit)
T_res_K = T_res_c + 273.15
T_surf_K = T_surf_c + 273.15
D_tub_m = length_to_m(tubing_id_in, tubing_unit)
eps_m = tubing_rough_mm / 1000.0
perm_m2 = perm_mD * 9.869233e-16

# =====================================
# CALCULATE CO₂ PROPERTIES
# =====================================
try:
    rho_res = PropsSI('D', 'T', T_res_K, 'P', P_res_pa, 'CO2')
    mu_res = PropsSI('V', 'T', T_res_K, 'P', P_res_pa, 'CO2')
except:
    st.error("CoolProp failed to evaluate CO₂ at reservoir conditions.")
    st.stop()

# =====================================
# MAIN CALCULATION
# =====================================
mass_flow_kg_s = q_tpd * 1000 / 86400
q_vol_m3s = mass_flow_kg_s / rho_res
A_tub = pi * (D_tub_m**2) / 4

h_conv = 20.0  # W/(m²·K)
A_surf = pi * D_tub_m
cp_CO2 = 850  # J/(kg·K), approximate

if st.button("Calculate"):

    # --- ΔP to inject across reservoir ---
    if perm_m2 > 0:
        dP_sandface = (mu_res * q_vol_m3s / (2 * pi * perm_m2 * h_res)) * (log(re / rw) + skin)
    else:
        dP_sandface = 0.0

    # --- Sandface BHP ---
    BHP_pa = P_res_pa + dP_sandface

    # --- DEPTH GRID ---
    depths = np.linspace(0, TVD, n_segments)

    # --- INIT ARRAYS ---
    T_flow_K = np.zeros_like(depths)
    T_shut_K = T_surf_K + (T_res_K - T_surf_K) * (depths / TVD)
    P_flow_pa = np.zeros_like(depths)
    rho_flow = np.zeros_like(depths)
    mu_flow = np.zeros_like(depths)

    # --- INITIAL CONDITIONS ---
    T_flow_K[0] = T_surf_K
    P_flow_pa[-1] = BHP_pa
    try:
        rho_flow[-1] = PropsSI('D', 'T', T_res_K, 'P', BHP_pa, 'CO2')
        mu_flow[-1] = PropsSI('V', 'T', T_res_K, 'P', BHP_pa, 'CO2')
    except:
        rho_flow[-1] = rho_res
        mu_flow[-1] = mu_res

    # --- FLOWING TEMPERATURE PROFILE (surface→bottom) ---
    dz = depths[1] - depths[0]
    for i in range(1, n_segments):
        dT = h_conv * A_surf * (T_res_K - T_flow_K[i-1]) / (mass_flow_kg_s * cp_CO2) * dz
        T_flow_K[i] = T_flow_K[i-1] + dT

    # --- UPWARD FLOWING PRESSURE PROFILE (sandface→surface) ---
    for i in reversed(range(1, n_segments)):
        dz = depths[i] - depths[i-1]
        v = q_vol_m3s / A_tub
        try:
            rho_local = PropsSI('D','T',T_flow_K[i],'P',P_flow_pa[i],'CO2')
            mu_local = PropsSI('V','T',T_flow_K[i],'P',P_flow_pa[i],'CO2')
        except:
            rho_local = rho_flow[i]
            mu_local = mu_flow[i]
        Re = rho_local * v * D_tub_m / (mu_local if mu_local>0 else 1e-6)
        f = haaland_friction(Re, eps_m / D_tub_m)
        dP_h = rho_local * G * dz
        dP_f = f * dz / D_tub_m * (rho_local * v**2 / 2)
        P_flow_pa[i-1] = P_flow_pa[i] - dP_h + dP_f  # <<< hydrostatic subtracted
        rho_flow[i-1] = rho_local
        mu_flow[i-1] = mu_local

    # --- Flowing Tubing Head Pressure ---
    FTHP_pa_surface = P_flow_pa[0]

    # --- SHUT-IN PRESSURE & DENSITY PROFILE ---
    P_shut_pa = P_res_pa - rho_flow * G * (TVD - depths)
    rho_shut = np.zeros_like(depths)
    for i, P in enumerate(P_shut_pa):
        try:
            rho_shut[i] = PropsSI('D', 'T', T_shut_K[i], 'P', P, 'CO2')
        except:
            rho_shut[i] = rho_flow[i]

    # --- RESULTS DISPLAY ---
    st.subheader("Results")
    st.write(f"**Sandface BHP:** {from_pa(BHP_pa, pressure_unit):.3f} {pressure_unit}")
    st.write(f"**ΔP across reservoir:** {from_pa(dP_sandface, pressure_unit):.3f} {pressure_unit}")
    st.write(f"**Flowing Tubing Head Pressure (surface THP):** {from_pa(FTHP_pa_surface, pressure_unit):.3f} {pressure_unit}")
    st.write(f"**Mass flow (kg/s):** {mass_flow_kg_s:.3f}")

    # --- PLOTS ---
    fig, ax = plt.subplots(1,3,figsize=(18,6))
    ax[0].plot(from_pa(P_flow_pa, pressure_unit), depths, label='Flowing THP', color='blue')
    ax[0].plot(from_pa(P_shut_pa, pressure_unit), depths, linestyle='--', label='Shut-in THP', color='red')
    ax[0].invert_yaxis()
    ax[0].set_xlabel(f"Pressure ({pressure_unit})")
    ax[0].set_ylabel("Depth (m)")
    ax[0].legend()
    ax[0].grid(True)
    ax[0].set_title("Pressure vs Depth")

    ax[1].plot(T_flow_K - 273.15, depths, label='Flowing Temp', color='blue')
    ax[1].plot(T_shut_K - 273.15, depths, linestyle='--', label='Shut-in Temp', color='red')
    ax[1].invert_yaxis()
    ax[1].set_xlabel("Temperature (°C)")
    ax[1].set_ylabel("Depth (m)")
    ax[1].legend()
    ax[1].grid(True)
    ax[1].set_title("Temperature vs Depth")

    ax[2].plot(rho_flow, depths, label='Flowing Density', color='green')
    ax[2].plot(rho_shut, depths, linestyle='--', label='Shut-in Density', color='orange')
    ax[2].invert_yaxis()
    ax[2].set_xlabel("Density (kg/m³)")
    ax[2].set_ylabel("Depth (m)")
    ax[2].legend()
    ax[2].grid(True)
    ax[2].set_title("Density vs Depth")

    st.pyplot(fig)

    # --- SENSITIVITY: Surface THP vs Injection Rate ---
    st.subheader("Injection THP Sensitivity to Mass Flow")
    q_min = q_tpd * (1 - sens_percent/100)
    q_max = q_tpd * (1 + sens_percent/100)
    q_range = np.linspace(q_min, q_max, 11)
    thp_sens = []

    for q_var in q_range:
        mf = q_var * 1000 / 86400
        qv = mf / rho_res
        P_profile = np.zeros_like(depths)
        P_profile[-1] = BHP_pa
        for i in reversed(range(1, n_segments)):
            dz = depths[i] - depths[i-1]
            v = qv / A_tub
            try:
                rho_local = PropsSI('D','T',T_flow_K[i],'P',P_profile[i],'CO2')
                mu_local = PropsSI('V','T',T_flow_K[i],'P',P_profile[i],'CO2')
            except:
                rho_local = rho_flow[i]
                mu_local = mu_flow[i]
            Re = rho_local * v * D_tub_m / (mu_local if mu_local>0 else 1e-6)
            f = haaland_friction(Re, eps_m / D_tub_m)
            dP_h = rho_local * G * dz
            dP_f = f * dz / D_tub_m * (rho_local * v**2 / 2)
            P_profile[i-1] = P_profile[i] - dP_h + dP_f  # hydrostatic subtracted
        thp_sens.append(from_pa(P_profile[0], pressure_unit))

    fig3, ax3 = plt.subplots(figsize=(6,4))
    ax3.plot(q_range, thp_sens, marker='o')
    ax3.set_xlabel("Injection Rate (t/d)")
    ax3.set_ylabel(f"Flowing THP ({pressure_unit})")
    ax3.set_title("Surface Tubing Head Pressure vs Injection Rate")
    ax3.grid(True)
    st.pyplot(fig3)

    # --- CSV EXPORT ---
    df = pd.DataFrame({
        "Depth_m": depths,
        "Flowing_Temp_C": T_flow_K - 273.15,
        "Shut_in_Temp_C": T_shut_K - 273.15,
        f"Flowing_Pressure_{pressure_unit}": from_pa(P_flow_pa, pressure_unit),
        f"Shut_in_Pressure_{pressure_unit}": from_pa(P_shut_pa, pressure_unit),
        "Flowing_Density_kg_m3": rho_flow,
        "Shut_in_Density_kg_m3": rho_shut
    })
    csv_buf = df.to_csv(index=False)
    st.download_button("Download CSV", data=csv_buf.encode('utf-8'),
                       file_name="CO2_injection_THP_profile.csv", mime="text/csv")

    st.success("Calculation complete ✅")

# =====================================
# FOOTER
# =====================================
st.markdown("---")
st.markdown("© 2025 CCS Energy Pty Ltd. All rights reserved. info@ccsenergy.com.au")



