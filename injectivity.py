import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import pi, log
from CoolProp.CoolProp import PropsSI

# =====================================
# HEADER
# =====================================
st.set_page_config(page_title="CO2 Injection Well - THP Estimator", layout="wide")
st.image("https://ccsenergy.com.au/wp-content/themes/custom-theme/assets/images/logo.svg", width=250)
st.title("CO₂ Injection — Tubing Head Pressure Estimator")

G = 9.81  # gravity (m/s²)

# =====================================
# FUNCTIONS
# =====================================
def haaland_friction(Re, eD):
    if Re <= 0 or np.isnan(Re):
        return np.nan
    if Re < 2300:
        return 64.0 / Re
    return 1.0 / (-1.8 * np.log10((eD / 3.7)**1.11 + 6.9 / Re))**2

def to_pa(val, unit): return val*1e5 if unit=='bar' else val*6894.76
def from_pa(val, unit): return val/1e5 if unit=='bar' else val/6894.76
def length_to_m(val, unit): return val/1000.0 if unit=='mm' else val*0.0254

# =====================================
# SIDEBAR INPUTS
# =====================================
with st.sidebar:
    pressure_unit = st.selectbox("Pressure unit", ['bar','psi'], index=0)
    tubing_unit = st.selectbox("Tubing ID unit", ['mm','inch'], index=0)

    P_res_in = st.number_input(f"Reservoir pressure ({pressure_unit})", value=120.0)
    T_res_c = st.number_input("Reservoir temperature (°C)", value=60.0)
    T_surf_c = st.number_input("Surface temperature (°C)", value=20.0)
    MD = st.number_input("Measured depth (m)", value=2500.0)
    TVD = st.number_input("True vertical depth (m)", value=2400.0)
    tubing_id_in = st.number_input(f"Tubing inner diameter ({tubing_unit})", value=88.9)
    tubing_rough_mm = st.number_input("Tubing roughness (mm)", value=0.05)
    skin = st.number_input("Skin factor", value=0.0)
    q_tpd = st.number_input("Injection rate (t/d)", value=1000.0)
    perm_mD = st.number_input("Permeability (mD)", value=100.0)
    poro = st.number_input("Porosity", value=0.25)
    h_res = st.number_input("Reservoir thickness (m)", value=10.0)
    rw = st.number_input("Wellbore radius (m)", value=0.0889)
    re = st.number_input("Drainage radius (m)", value=100.0)
    show_sensitivity = st.checkbox("Show Rate vs FTHP Sensitivity (±100%)", value=False)

# =====================================
# CONSTANTS AND PRE-CALCS
# =====================================
P_res_pa = to_pa(P_res_in, pressure_unit)
T_res_K = T_res_c + 273.15
T_surf_K = T_surf_c + 273.15
D_tub_m = length_to_m(tubing_id_in, tubing_unit)
eps_m = tubing_rough_mm / 1000.0
perm_m2 = perm_mD * 9.869233e-16
A_tub = pi * D_tub_m**2 / 4.0
mass_flow_kg_s = q_tpd * 1000 / 86400  # t/d → kg/s

# get base fluid properties at reservoir conditions
try:
    rho_res = PropsSI('D','T',T_res_K,'P',P_res_pa,'CO2')
    mu_res = PropsSI('V','T',T_res_K,'P',P_res_pa,'CO2')
except:
    st.error("CoolProp error at reservoir conditions.")
    st.stop()

# =====================================
# CALCULATION
# =====================================
if st.button("Calculate"):
    # --- Hydrostatic pressures ---
    # Simple average density approximation
    rho_static_avg = rho_res
    rho_flow_avg = rho_res
    dP_hydro_static = rho_static_avg * G * TVD
    dP_hydro_flow = rho_flow_avg * G * TVD

    # --- Sandface pressure loss ---
    q_vol = mass_flow_kg_s / rho_res
    sandface_loss = (mu_res*q_vol)/(2*pi*perm_m2*h_res)*(log(re/rw)+skin) if (perm_m2>0 and h_res>0) else 0.0

    # --- Friction pressure along tubing ---
    vel = q_vol / A_tub
    Re = rho_res * vel * D_tub_m / mu_res
    f = haaland_friction(Re, eps_m/D_tub_m)
    dp_friction = f * (MD/D_tub_m) * (rho_res*vel**2/2.0)

    # --- THP calculations ---
    SITHP_pa = P_res_pa - dP_hydro_static
    FTHP_pa = P_res_pa + sandface_loss + dp_friction - dP_hydro_flow

    # --- Display results ---
    st.subheader("Results")
    st.write(f"Shut-in THP = {from_pa(SITHP_pa, pressure_unit):.3f} {pressure_unit}")
    st.write(f"Flowing THP = {from_pa(FTHP_pa, pressure_unit):.3f} {pressure_unit}")
    st.write(f"ΔTHP = {from_pa(FTHP_pa - SITHP_pa, pressure_unit):.3f} {pressure_unit}")
    st.write(f"Sandface loss = {from_pa(sandface_loss, pressure_unit):.3f} {pressure_unit}")
    st.write(f"Friction loss = {from_pa(dp_friction, pressure_unit):.3f} {pressure_unit}")
    st.write(f"Hydrostatic (static) = {from_pa(dP_hydro_static, pressure_unit):.3f} {pressure_unit}")
    st.write(f"Hydrostatic (flowing) = {from_pa(dP_hydro_flow, pressure_unit):.3f} {pressure_unit}")

    # --- Pressure vs depth (simple linear) ---
    tvd_nodes = np.linspace(0, TVD, 20)
    pressure_shut_profile = np.linspace(P_res_pa - dP_hydro_static, P_res_pa, 20)
    pressure_flow_profile = np.linspace(FTHP_pa - (FTHP_pa-SITHP_pa), FTHP_pa, 20)

    fig, ax = plt.subplots(figsize=(5,8))
    ax.plot(from_pa(pressure_flow_profile, pressure_unit), tvd_nodes, label="Flowing profile", color='blue')
    ax.plot(from_pa(pressure_shut_profile, pressure_unit), tvd_nodes, '--', label="Shut-in profile", color='red')
    ax.set_xlabel(f"Pressure ({pressure_unit})")
    ax.set_ylabel("TVD (m)")
    ax.invert_yaxis()
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

    # --- Sensitivity plot ---
    if show_sensitivity:
        q_vals = np.linspace(0.5*q_tpd, 2.0*q_tpd, 15)
        fthp_vals = []
        for qv in q_vals:
            q_vol_tmp = qv*1000/86400/rho_res
            vel_tmp = q_vol_tmp / A_tub
            Re_tmp = rho_res*vel_tmp*D_tub_m/mu_res
            f_tmp = haaland_friction(Re_tmp, eps_m/D_tub_m)
            dp_fric_tmp = f_tmp*(MD/D_tub_m)*(rho_res*vel_tmp**2/2)
            FTHP_tmp = P_res_pa + sandface_loss + dp_fric_tmp - dP_hydro_flow
            fthp_vals.append(from_pa(FTHP_tmp, pressure_unit))
        fig2, ax2 = plt.subplots()
        ax2.plot(q_vals, fthp_vals, '-o', color='green')
        ax2.axvline(q_tpd, color='gray', linestyle='--')
        ax2.set_xlabel("Injection rate (t/d)")
        ax2.set_ylabel(f"Flowing THP ({pressure_unit})")
        ax2.grid(True)
        st.pyplot(fig2)

    # --- CSV export ---
    df = pd.DataFrame({
        "TVD_m": tvd_nodes,
        "Shut_in_Pressure_"+pressure_unit: from_pa(pressure_shut_profile, pressure_unit),
        "Flowing_Pressure_"+pressure_unit: from_pa(pressure_flow_profile, pressure_unit),
    })
    st.download_button("Download CSV", data=df.to_csv(index=False).encode("utf-8"),
                       file_name="CO2_injection_profile.csv", mime="text/csv")

# =====================================
# FOOTER
# =====================================
st.markdown("---")
st.markdown("© 2025 CCS Energy Pty Ltd. All rights reserved.")
