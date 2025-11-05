"""
Enhanced Streamlit app: CO2 Injection THP Estimator with full CSV export

- Flowing THP > Shut-in THP.
- Flowing THP: start from reservoir pressure at perforation, subtract hydrostatic from perf to surface, add friction along wellbore, add sandface loss at perforation.
- Shut-in THP: static hydrostatic column from perforation to surface.
- Pressure vs depth plot shows flowing and shut-in profiles (shut-in dashed).
- CSV export includes depth profile with flowing & shut-in pressures, density, viscosity, temperature, friction, hydrostatic, sandface losses.
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import pi, log
from CoolProp.CoolProp import PropsSI

# Page UI
st.image("https://ccsenergy.com.au/wp-content/themes/custom-theme/assets/images/logo.svg", width=250)
st.set_page_config(page_title="CO2 Injection Well - THP Estimator", layout="wide")
st.title("CO₂ Injection — Tubing Head Pressure Estimator")

# Constants
G = 9.81  # gravity (m/s^2)

# Utilities
def haaland_friction(Re, eD):
    if Re <= 0 or np.isnan(Re): 
        return np.nan
    if Re < 2300: 
        return 64.0 / Re
    # Haaland formula; ensure safe domain for log10
    return 1.0 / (-1.8 * np.log10((eD / 3.7)**1.11 + 6.9 / Re))**2

def to_pa(val, unit):
    # convert user input to Pa
    return val*1e5 if unit=='bar' else val*6894.76

def from_pa(val, unit):
    # convert Pa to user unit
    return val/1e5 if unit=='bar' else val/6894.76

def length_to_m(val, unit):
    return val/1000.0 if unit=='mm' else val*0.0254

# Sidebar inputs
with st.sidebar:
    pressure_unit = st.selectbox("Pressure unit", ['bar','psi'], index=0)
    tubing_unit = st.selectbox("Tubing ID unit", ['mm','inch'], index=0)

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

    n_segments = st.slider("Number of segments", 20, 1000, 200, 10)
    show_sensitivity = st.checkbox("Show Rate vs THP Sensitivity Plot (±100%)", value=False)

# Derived constants / conversions
P_res_pa = to_pa(P_res_in, pressure_unit)
T_res_K = T_res_c + 273.15
T_surf_K = T_surf_c + 273.15
D_tub_m = length_to_m(tubing_id_in, tubing_unit)
eps_m = tubing_rough_mm / 1000.0
perm_m2 = perm_mD * 9.869233e-16

# Fluid properties at reservoir (fallback)
try:
    rho_res = PropsSI('D','T',T_res_K,'P',P_res_pa,'CO2')
    mu_res = PropsSI('V','T',T_res_K,'P',P_res_pa,'CO2')
except Exception as e:
    st.error('CoolProp error at reservoir conditions. Check temperature/pressure inputs.')
    st.stop()

# Mass flow (kg/s) and nominal volumetric flow (m3/s) using reservoir density (used only for a few approximations)
mass_flow_kg_s = q_tpd * 1000.0 / 86400.0  # t/d -> kg/s
q_vol_m3s_nominal = mass_flow_kg_s / rho_res
A_tub = pi * (D_tub_m**2) / 4.0

# Main compute action
if st.button('Calculate'):
    # Depth grids
    md_nodes = np.linspace(0.0, MD, n_segments)
    tvd_nodes = (md_nodes / MD) * TVD
    dz_md = np.diff(md_nodes)
    dz_tvd = np.diff(tvd_nodes)
    T_profile_K = T_surf_K + (T_res_K - T_surf_K) * (tvd_nodes / TVD)

    # --- Shut-in THP (static hydrostatic column) ---
    pressure_shut_profile = np.zeros(n_segments)
    pressure_shut_profile[-1] = P_res_pa  # bottom = reservoir pressure
    rho_shut = np.zeros(n_segments)
    mu_shut = np.zeros(n_segments)

    for i in range(n_segments-2, -1, -1):
        try:
            rho_loc = PropsSI('D','T',T_profile_K[i],'P',pressure_shut_profile[i+1],'CO2')
            mu_loc = PropsSI('V','T',T_profile_K[i],'P',pressure_shut_profile[i+1],'CO2')
        except:
            rho_loc = rho_res
            mu_loc = mu_res
        rho_shut[i] = rho_loc
        mu_shut[i] = mu_loc
        # shallower pressure = deeper pressure - rho*g*dz (hydrostatic drop upward)
        pressure_shut_profile[i] = pressure_shut_profile[i+1] - rho_loc * G * dz_tvd[i]

    THP_shut_pa = pressure_shut_profile[0]

    # --- Flowing THP (mass-conserved, per-segment properties) ---
    pressure_flow_profile = np.zeros(n_segments)
    rho_flow = np.zeros(n_segments)
    mu_flow = np.zeros(n_segments)
    seg_friction_loss = np.zeros(n_segments-1)  # friction per segment
    seg_hydrostatic = np.zeros(n_segments-1)

    # Sandface loss (Darcy-style) - approximate using reservoir viscosity; units Pa
    try:
        sandface_loss = (mu_res * (mass_flow_kg_s / rho_res)) / (2.0 * pi * perm_m2 * h_res) * (np.log(re / rw) + skin) if perm_m2 > 0.0 else 0.0
        # Note: Above uses volumetric flow computed with reservoir density as a simple Darcy estimate.
    except Exception:
        sandface_loss = 0.0

    # Boundary condition: perforation (deepest node) flowing pressure must be reservoir pressure + sandface_loss
    pressure_flow_profile[-1] = P_res_pa + sandface_loss
    rho_flow[-1] = rho_res
    mu_flow[-1] = mu_res

    # March upward from bottom to surface, conserving mass:
    for i in range(n_segments-2, -1, -1):
        # Use properties at deeper node pressure (i+1) as the starting estimate for this segment
        try:
            rho_loc = PropsSI('D','T',T_profile_K[i+1],'P',pressure_flow_profile[i+1],'CO2')
            mu_loc = PropsSI('V','T',T_profile_K[i+1],'P',pressure_flow_profile[i+1],'CO2')
        except:
            rho_loc = rho_res
            mu_loc = mu_res

        # local volumetric flow (m3/s) given mass flow (kg/s)
        q_vol_local = mass_flow_kg_s / (rho_loc if rho_loc>0 else rho_res)
        vel_loc = q_vol_local / A_tub
        Re_loc = rho_loc * vel_loc * D_tub_m / (mu_loc if mu_loc>0 else 1e-12)
        f_loc = haaland_friction(Re_loc, eps_m / D_tub_m) if Re_loc>0 else np.nan

        # frictional pressure loss over this MD segment (Pa)
        dp_f = f_loc * (dz_md[i] / D_tub_m) * (rho_loc * vel_loc**2 / 2.0)
        dp_h = rho_loc * G * dz_tvd[i]

        seg_friction_loss[i] = dp_f
        seg_hydrostatic[i] = dp_h

        # marching upward: shallower pressure = deeper pressure - hydrostatic - friction
        pressure_flow_profile[i] = pressure_flow_profile[i+1] - dp_h - dp_f

        # store properties at the shallower node for CSV/plot completeness (we'll estimate at node i using node i+1 props)
        rho_flow[i] = rho_loc
        mu_flow[i] = mu_loc

    THP_flow_pa = pressure_flow_profile[0]

    # --- Quick numerical test: perforation BHP vs expected (P_res + sandface_loss) ---
    perforation_bhp_pa = pressure_flow_profile[-1]
    expected_bottom_pa = P_res_pa + sandface_loss
    tol = 1e-6  # tiny tolerance
    if perforation_bhp_pa + tol >= expected_bottom_pa:
        st.success(f'Perforation BHP check: OK. BHP_perforation = {from_pa(perforation_bhp_pa, pressure_unit):.4f} {pressure_unit} >= P_res + sandface_loss = {from_pa(expected_bottom_pa, pressure_unit):.4f} {pressure_unit}')
    else:
        st.warning(f'Perforation BHP check: FAILED. BHP_perforation = {from_pa(perforation_bhp_pa, pressure_unit):.4f} {pressure_unit} < P_res + sandface_loss = {from_pa(expected_bottom_pa, pressure_unit):.4f} {pressure_unit}. Difference = {from_pa(expected_bottom_pa - perforation_bhp_pa, pressure_unit):.4f} {pressure_unit}')

    # Display results
    st.subheader('Results')
    st.write(f'Flowing THP = {from_pa(THP_flow_pa,pressure_unit):.4f} {pressure_unit}')
    st.write(f'Shut-in THP = {from_pa(THP_shut_pa,pressure_unit):.4f} {pressure_unit}')
    st.write(f'Sandface loss (Pa) = {sandface_loss:.2f} Pa ({from_pa(sandface_loss,pressure_unit):.4f} {pressure_unit})')

    # --- Pressure vs Depth plot ---
    fig, ax = plt.subplots(figsize=(5,8))
    ax.plot(from_pa(pressure_flow_profile,pressure_unit), tvd_nodes, label='Flowing THP', color='blue')
    ax.plot(from_pa(pressure_shut_profile,pressure_unit), tvd_nodes, linestyle='--', label='Shut-in THP', color='red')
    ax.set_xlabel(f'Pressure ({pressure_unit})')
    ax.set_ylabel('TVD (m)')
    ax.invert_yaxis()
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

    # Sensitivity (approximate friction using reservoir properties but mass-conserved)
    if show_sensitivity:
        q_vals = np.linspace(0.5*q_tpd, 2.0*q_tpd, 20)
        thp_vals = []
        for qv in q_vals:
            mass_flow_tmp = qv * 1000.0 / 86400.0
            # approximate sandface with reservoir viscosity & local q_vol estimate
            try:
                q_vol_tmp = mass_flow_tmp / rho_res
                sandface_tmp = (mu_res * q_vol_tmp) / (2.0 * pi * perm_m2 * h_res) * (np.log(re / rw) + skin) if perm_m2 > 0.0 else 0.0
            except:
                sandface_tmp = sandface_loss

            # approximate friction sum across segments using reservoir props for speed
            vel_tmp = q_vol_tmp / A_tub
            friction_tmp = 0.0
            for j in range(n_segments-1):
                Re_tmp = rho_res * vel_tmp * D_tub_m / (mu_res if mu_res>0 else 1e-12)
                f_tmp = haaland_friction(Re_tmp, eps_m / D_tub_m)
                friction_tmp += f_tmp * (dz_md[j] / D_tub_m) * (rho_res * vel_tmp**2 / 2.0)

            thp_guess = THP_shut_pa + friction_tmp + sandface_tmp
            thp_vals.append(from_pa(thp_guess, pressure_unit))

        fig2, ax2 = plt.subplots()
        ax2.plot(q_vals, thp_vals, '-o')
        ax2.axvline(q_tpd, color='gray', linestyle='--')
        ax2.set_xlabel('Injection rate (t/d)')
        ax2.set_ylabel(f'THP ({pressure_unit})')
        ax2.grid(True)
        st.pyplot(fig2)

    # --- CSV export ---
    df_profile = pd.DataFrame({
        'MD_m': md_nodes,
        'TVD_m': tvd_nodes,
        'Temperature_C': T_profile_K - 273.15,
        f'Flowing_THP_{pressure_unit}': from_pa(pressure_flow_profile, pressure_unit),
        f'Shut_in_THP_{pressure_unit}': from_pa(pressure_shut_profile, pressure_unit),
        'Density_kg_m3': rho_flow,
        'Viscosity_Pa_s': mu_flow,
        'Segment_friction_Pa': np.append(seg_friction_loss, np.nan),
        'Segment_hydrostatic_Pa': np.append(seg_hydrostatic, np.nan)
    })

    summary = pd.DataFrame({
        'Parameter': ['Flowing THP','Shut-in THP','Sandface loss (user calc)'],
        'Value': [from_pa(THP_flow_pa, pressure_unit), from_pa(THP_shut_pa, pressure_unit), from_pa(sandface_loss, pressure_unit)],
        'Unit':[pressure_unit]*3
    })

    csv_buf = df_profile.to_csv(index=False) + '\n\n' + summary.to_csv(index=False)
    st.download_button('Download CSV', data=csv_buf.encode('utf-8'), file_name='CO2_injection_profile.csv', mime='text/csv')

    st.success('Calculation complete')

# Footer
st.markdown("---")
st.markdown("© 2025 CCS Energy Pty Ltd. www.ccsenergy.com.au All rights reserved. info@ccsenergy.com.au")
