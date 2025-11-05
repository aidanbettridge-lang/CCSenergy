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
st.title("CO₂ Injection — Tubing Head Pressure Estimator")

# =====================================
# CONSTANTS & HELPER FUNCTIONS
# =====================================
G = 9.81  # m/s²

def haaland_friction(Re, eD):
    """Haaland equation for turbulent flow friction factor."""
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
    T_surf_c = st.number_input("Surface temperature (°C)", value=20.0)

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

if st.button("Calculate"):
    # --- Temperature & density profile ---
    depths = np.linspace(0, TVD, n_segments)
    T_profile_K = T_surf_K + (T_res_K - T_surf_K) * (depths / TVD)
    rho_profile = []
    for T in T_profile_K:
        try:
            rho_profile.append(PropsSI('D', 'T', T, 'P', P_res_pa, 'CO2'))
        except:
            rho_profile.append(rho_res)
    rho_profile = np.array(rho_profile)
    rho_avg = np.mean(rho_profile)

    # --- Pressure components ---
    dP_hydro = rho_avg * G * TVD  # hydrostatic head
    if perm_m2 > 0:
        dP_sandface = (mu_res * q_vol_m3s / (2 * pi * perm_m2 * h_res)) * (log(re / rw) + skin)
    else:
        dP_sandface = 0.0

    Re = rho_res * (q_vol_m3s / A_tub) * D_tub_m / (mu_res if mu_res > 0 else 1e-6)
    f = haaland_friction(Re, eps_m / D_tub_m)
    dP_fric = f * (MD / D_tub_m) * (rho_res * (q_vol_m3s / A_tub)**2 / 2)

    # --- Flowing and Shut-in THP ---
    SITHP_pa = P_res_pa - dP_hydro
    FTHP_pa = P_res_pa + dP_sandface + dP_fric - dP_hydro

    # =====================================
    # RESULTS DISPLAY
    # =====================================
    st.subheader("Results")
    st.write(f"**Reservoir (BHP):** {from_pa(P_res_pa, pressure_unit):.3f} {pressure_unit}")
    st.write(f"**Hydrostatic Head:** {from_pa(dP_hydro, pressure_unit):.3f} {pressure_unit}")
    st.write(f"**Sandface ΔP:** {from_pa(dP_sandface, pressure_unit):.3f} {pressure_unit}")
    st.write(f"**Frictional ΔP:** {from_pa(dP_fric, pressure_unit):.3f} {pressure_unit}")
    st.markdown("---")
    st.write(f"**Shut-in THP:** {from_pa(SITHP_pa, pressure_unit):.3f} {pressure_unit}")
    st.write(f"**Flowing THP:** {from_pa(FTHP_pa, pressure_unit):.3f} {pressure_unit}")
    st.write(f"ΔTHP = {from_pa(FTHP_pa - SITHP_pa, pressure_unit):.3f} {pressure_unit}")

    if FTHP_pa > SITHP_pa:
        st.success("✅ Flowing THP > Shut-in THP — physics consistent.")
    else:
        st.error("❌ Flowing THP < Shut-in THP — check inputs or logic.")

    # =====================================
    # PRESSURE VS DEPTH PLOT
    # =====================================
    pressure_shut_profile = P_res_pa - rho_profile * G * (TVD - depths)
    pressure_flow_profile = P_res_pa + dP_sandface + dP_fric - rho_profile * G * (TVD - depths)

    fig, ax = plt.subplots(figsize=(5, 8))
    ax.plot(from_pa(pressure_flow_profile, pressure_unit), depths, label='Flowing THP', color='blue')
    ax.plot(from_pa(pressure_shut_profile, pressure_unit), depths, linestyle='--', label='Shut-in THP', color='red')
    ax.invert_yaxis()
    ax.set_xlabel(f"Pressure ({pressure_unit})")
    ax.set_ylabel("Depth (m)")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    # =====================================
    # CSV EXPORT
    # =====================================
    df = pd.DataFrame({
        "Depth_m": depths,
        "Temperature_C": T_profile_K - 273.15,
        f"Flowing_Pressure_{pressure_unit}": from_pa(pressure_flow_profile, pressure_unit),
        f"Shut_in_Pressure_{pressure_unit}": from_pa(pressure_shut_profile, pressure_unit),
        "Density_kg_m3": rho_profile
    })

    summary = pd.DataFrame({
        "Parameter": [
            "Reservoir Pressure (BHP)",
            "Hydrostatic Head",
            "Sandface ΔP",
            "Frictional ΔP",
            "Shut-in THP",
            "Flowing THP"
        ],
        "Value": [
            from_pa(P_res_pa, pressure_unit),
            from_pa(dP_hydro, pressure_unit),
            from_pa(dP_sandface, pressure_unit),
            from_pa(dP_fric, pressure_unit),
            from_pa(SITHP_pa, pressure_unit),
            from_pa(FTHP_pa, pressure_unit)
        ],
        "Unit": [pressure_unit] * 6
    })

    csv_buf = df.to_csv(index=False) + '\n\n' + summary.to_csv(index=False)
    st.download_button("Download CSV", data=csv_buf.encode('utf-8'),
                       file_name="CO2_injection_THP_profile.csv", mime="text/csv")

    st.success("Calculation complete ✅")

# =====================================
# FOOTER
# =====================================
st.markdown("---")
st.markdown("© 2025 CCS Energy Pty Ltd. All rights reserved.")

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
    h_res = st.number_input("Reservoir thickness (m)", value=10.0)
    rw = st.number_input("Wellbore radius (m)", value=0.0889)
    re = st.number_input("Drainage radius (m)", value=100.0)
    n_segments = st.slider("Number of segments", 20, 500, 200, 10)
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

    # --- Depth nodes ---
    md_nodes = np.linspace(0, MD, n_segments)
    tvd_nodes = np.linspace(0, TVD, n_segments)
    dz_md = np.diff(md_nodes)
    dz_tvd = np.diff(tvd_nodes)

    # --- Hydrostatic pressures ---
    # Static hydrostatics (SITHP)
    SITHP_pa = P_res_pa - rho_res * G * TVD

    # --- Sandface loss ---
    q_vol = mass_flow_kg_s / rho_res
    sandface_loss = (mu_res*q_vol)/(2*pi*perm_m2*h_res)*(log(re/rw)+skin) if (perm_m2>0 and h_res>0) else 0.0

    # --- FLOWING PRESSURE (Bottom-up) ---
    pressure_flow = np.zeros(n_segments)
    pressure_flow[-1] = P_res_pa + sandface_loss  # start at reservoir + sandface

    for i in range(n_segments-2, -1, -1):
        T_loc = T_surf_K + (T_res_K - T_surf_K) * (tvd_nodes[i+1] / TVD)
        try:
            rho_loc = PropsSI('D','T',T_loc,'P',pressure_flow[i+1],'CO2')
            mu_loc = PropsSI('V','T',T_loc,'P',pressure_flow[i+1],'CO2')
        except:
            rho_loc = rho_res
            mu_loc = mu_res

        vel_loc = q_vol / A_tub
        Re_loc = rho_loc * vel_loc * D_tub_m / mu_loc
        f_loc = haaland_friction(Re_loc, eps_m/D_tub_m)
        dp_friction = f_loc * (dz_md[i]/D_tub_m) * (rho_loc*vel_loc**2 / 2)
        dp_hydro = rho_loc * G * dz_tvd[i]

        pressure_flow[i] = pressure_flow[i+1] - dp_hydro - dp_friction

    FTHP_pa = pressure_flow[0]

    # --- Display results ---
    st.subheader("Results")
    st.write(f"Shut-in THP = {from_pa(SITHP_pa, pressure_unit):.3f} {pressure_unit}")
    st.write(f"Flowing THP = {from_pa(FTHP_pa, pressure_unit):.3f} {pressure_unit}")
    st.write(f"ΔTHP = {from_pa(FTHP_pa - SITHP_pa, pressure_unit):.3f} {pressure_unit}")
    st.write(f"Sandface loss = {from_pa(sandface_loss, pressure_unit):.3f} {pressure_unit}")

    # --- Pressure vs depth plot ---
    fig, ax = plt.subplots(figsize=(5,8))
    ax.plot(from_pa(pressure_flow, pressure_unit), tvd_nodes, label="Flowing profile", color='blue')
    ax.plot(from_pa(np.linspace(P_res_pa - rho_res*G*TVD, P_res_pa, n_segments), pressure_unit),
            tvd_nodes, '--', label="Shut-in profile", color='red')
    ax.set_xlabel(f"Pressure ({pressure_unit})")
    ax.set_ylabel("TVD (m)")
    ax.invert_yaxis()
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

    # --- Sensitivity plot ---
    if show_sensitivity:
        q_vals = np.linspace(0.5*q_tpd, 2*q_tpd, 15)
        fthp_vals = []
        for qv in q_vals:
            q_vol_tmp = qv * 1000 / 86400 / rho_res
            pressure_tmp = np.zeros(n_segments)
            pressure_tmp[-1] = P_res_pa + sandface_loss
            for i in range(n_segments-2, -1, -1):
                dp_hydro = rho_res * G * dz_tvd[i]
                vel_tmp = q_vol_tmp / A_tub
                Re_tmp = rho_res * vel_tmp * D_tub_m / mu_res
                f_tmp = haaland_friction(Re_tmp, eps_m/D_tub_m)
                dp_fric_tmp = f_tmp * (dz_md[i]/D_tub_m) * (rho_res*vel_tmp**2/2)
                pressure_tmp[i] = pressure_tmp[i+1] - dp_hydro - dp_fric_tmp
            fthp_vals.append(from_pa(pressure_tmp[0], pressure_unit))
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
        "Flowing_Pressure_"+pressure_unit: from_pa(pressure_flow, pressure_unit),
        "Shut_in_Pressure_"+pressure_unit: from_pa(np.linspace(P_res_pa - rho_res*G*TVD, P_res_pa, n_segments), pressure_unit)
    })
    st.download_button("Download CSV", data=df.to_csv(index=False).encode("utf-8"),
                       file_name="CO2_injection_profile.csv", mime="text/csv")

# =====================================
# FOOTER
# =====================================
st.markdown("---")
st.markdown("© 2025 CCS Energy Pty Ltd. All rights reserved.")

