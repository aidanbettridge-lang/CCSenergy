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

G = 9.81  # gravity acceleration (m/s²)

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

def compute_shut_in_profile_simple(n_segments, MD, TVD, P_res_pa, T_surf_K, T_res_K):
    """Static (shut-in) profile"""
    md_nodes = np.linspace(0.0, MD, n_segments)
    tvd_nodes = (md_nodes / MD) * TVD
    dz_tvd = np.diff(tvd_nodes)
    pressure_shut = np.zeros(n_segments)
    rho_shut = np.zeros(n_segments)

    pressure_shut[-1] = P_res_pa
    for i in range(n_segments-2, -1, -1):
        T_deep = T_surf_K + (T_res_K - T_surf_K) * (tvd_nodes[i+1] / TVD)
        try:
            rho_deep = PropsSI('D', 'T', T_deep, 'P', pressure_shut[i+1], 'CO2')
        except:
            rho_deep = rho_res
        rho_shut[i+1] = rho_deep
        pressure_shut[i] = pressure_shut[i+1] - rho_deep * G * dz_tvd[i]

    try:
        rho_shut[0] = PropsSI('D', 'T', T_surf_K, 'P', pressure_shut[0], 'CO2')
    except:
        rho_shut[0] = rho_res

    return pressure_shut, rho_shut, tvd_nodes

def compute_flowing_profile_simple(q_tpd, n_segments, MD, TVD, P_res_pa,
                                  T_surf_K, T_res_K, D_tub_m, eps_m, perm_m2,
                                  h_res, rw, re, skin):
    """Flowing (injection) profile"""
    mass_flow_kg_s = q_tpd * 1000.0 / 86400.0
    md_nodes = np.linspace(0.0, MD, n_segments)
    tvd_nodes = (md_nodes / MD) * TVD
    dz_md = np.diff(md_nodes)
    dz_tvd = np.diff(tvd_nodes)
    pressure_flow = np.zeros(n_segments)
    rho_flow = np.zeros(n_segments)

    rho_flow[-1] = rho_res
    T_bottom = T_surf_K + (T_res_K - T_surf_K) * (tvd_nodes[-1] / TVD)
    try:
        mu_bottom = PropsSI('V', 'T', T_bottom, 'P', P_res_pa, 'CO2')
    except:
        mu_bottom = mu_res

    q_vol_bottom = mass_flow_kg_s / rho_flow[-1]
    sandface_loss = (mu_bottom * q_vol_bottom) / (2.0 * pi * perm_m2 * h_res) * (np.log(re / rw) + skin) if (perm_m2>0 and h_res>0) else 0.0

    pressure_flow[-1] = P_res_pa + sandface_loss
    total_friction = 0.0

    for i in range(n_segments-2, -1, -1):
        T_deep = T_surf_K + (T_res_K - T_surf_K) * (tvd_nodes[i+1] / TVD)
        try:
            rho_deep = PropsSI('D', 'T', T_deep, 'P', pressure_flow[i+1], 'CO2')
        except:
            rho_deep = rho_res
        rho_flow[i+1] = rho_deep

        q_vol_local = mass_flow_kg_s / rho_deep
        vel_loc = q_vol_local / (pi * (D_tub_m**2) / 4.0)

        try:
            mu_deep = PropsSI('V', 'T', T_deep, 'P', pressure_flow[i+1], 'CO2')
        except:
            mu_deep = mu_res

        Re_loc = rho_deep * vel_loc * D_tub_m / (mu_deep if mu_deep>0 else 1e-12)
        f_loc = haaland_friction(Re_loc, eps_m / D_tub_m) if Re_loc>0 else 0.0
        dp_f = f_loc * (dz_md[i] / D_tub_m) * (rho_deep * vel_loc**2 / 2.0)
        dp_h = rho_deep * G * dz_tvd[i]
        pressure_flow[i] = pressure_flow[i+1] - dp_h - dp_f
        total_friction += dp_f

    try:
        rho_flow[0] = PropsSI('D', 'T', T_surf_K, 'P', pressure_flow[0], 'CO2')
    except:
        rho_flow[0] = rho_res

    return pressure_flow, rho_flow, sandface_loss, total_friction, tvd_nodes


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

# get base fluid props
try:
    rho_res = PropsSI('D','T',T_res_K,'P',P_res_pa,'CO2')
    mu_res = PropsSI('V','T',T_res_K,'P',P_res_pa,'CO2')
except:
    st.error("CoolProp error at reservoir conditions.")
    st.stop()

# =====================================
# MAIN CALCULATION
# =====================================
if st.button("Calculate"):
    # --- SHUT-IN PROFILE ---
    pressure_shut, rho_shut, tvd_nodes = compute_shut_in_profile_simple(n_segments, MD, TVD, P_res_pa, T_surf_K, T_res_K)
    SITHP_pa = pressure_shut[0]
    rho_static_avg = np.mean(rho_shut)
    dP_hydro_static = rho_static_avg * G * TVD

    # --- FLOWING PROFILE ---
    pressure_flow, rho_flow, sandface_loss, total_fric, tvd_nodes = compute_flowing_profile_simple(
        q_tpd, n_segments, MD, TVD, P_res_pa, T_surf_K, T_res_K, D_tub_m, eps_m,
        perm_m2, h_res, rw, re, skin
    )
    FTHP_pa = pressure_flow[0]
    rho_flow_avg = np.mean(rho_flow)
    dP_hydro_flow = rho_flow_avg * G * TVD

    # --- DISPLAY RESULTS ---
    st.subheader("Results")
    st.write(f"Reservoir Pressure (BHP): {from_pa(P_res_pa, pressure_unit):.3f} {pressure_unit}")
    st.write(f"Hydrostatic (static): {from_pa(dP_hydro_static, pressure_unit):.3f} {pressure_unit}")
    st.write(f"Hydrostatic (flowing): {from_pa(dP_hydro_flow, pressure_unit):.3f} {pressure_unit}")
    st.write(f"Sandface loss: {from_pa(sandface_loss, pressure_unit):.3f} {pressure_unit}")
    st.write(f"Frictional loss: {from_pa(total_fric, pressure_unit):.3f} {pressure_unit}")
    st.write("---")
    st.write(f"Shut-in THP = {from_pa(SITHP_pa, pressure_unit):.3f} {pressure_unit}")
    st.write(f"Flowing THP = {from_pa(FTHP_pa, pressure_unit):.3f} {pressure_unit}")
    st.write(f"ΔTHP = {from_pa(FTHP_pa - SITHP_pa, pressure_unit):.3f} {pressure_unit}")

    if FTHP_pa > SITHP_pa:
        st.success("✅ Flowing THP > Shut-in THP — physics consistent.")
    else:
        st.error("❌ Flowing THP < Shut-in THP — check input parameters.")

    # --- PLOT PRESSURE VS DEPTH ---
    fig, ax = plt.subplots(figsize=(5,8))
    ax.plot(from_pa(pressure_flow, pressure_unit), tvd_nodes, label="Flowing profile", color='blue')
    ax.plot(from_pa(pressure_shut, pressure_unit), tvd_nodes, '--', label="Shut-in profile", color='red')
    ax.set_xlabel(f"Pressure ({pressure_unit})")
    ax.set_ylabel("TVD (m)")
    ax.invert_yaxis()
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    # --- SENSITIVITY PLOT ---
    if show_sensitivity:
        q_vals = np.linspace(0.5*q_tpd, 2.0*q_tpd, 15)
        fthp_vals = []
        for qv in q_vals:
            p_flow_tmp, rho_flow_tmp, sf_tmp, fric_tmp, _ = compute_flowing_profile_simple(
                qv, n_segments, MD, TVD, P_res_pa, T_surf_K, T_res_K,
                D_tub_m, eps_m, perm_m2, h_res, rw, re, skin
            )
            fthp_vals.append(from_pa(p_flow_tmp[0], pressure_unit))
        fig2, ax2 = plt.subplots()
        ax2.plot(q_vals, fthp_vals, '-o', color='green')
        ax2.axvline(q_tpd, color='gray', linestyle='--')
        ax2.set_xlabel("Injection rate (t/d)")
        ax2.set_ylabel(f"Flowing THP ({pressure_unit})")
        ax2.grid(True)
        st.pyplot(fig2)

    # --- CSV EXPORT ---
    df_profile = pd.DataFrame({
        "TVD_m": tvd_nodes,
        "Flowing_Pressure_"+pressure_unit: from_pa(pressure_flow, pressure_unit),
        "Shut_in_Pressure_"+pressure_unit: from_pa(pressure_shut, pressure_unit),
        "Flowing_Density_kgm3": rho_flow,
        "Static_Density_kgm3": rho_shut
    })

    csv_buf = df_profile.to_csv(index=False)
    st.download_button("Download CSV", data=csv_buf.encode("utf-8"),
                       file_name="CO2_injection_profile.csv", mime="text/csv")

    st.success("Calculation complete")

# =====================================
# FOOTER
# =====================================
st.markdown("---")
st.markdown("© 2025 CCS Energy Pty Ltd. All rights reserved.")
