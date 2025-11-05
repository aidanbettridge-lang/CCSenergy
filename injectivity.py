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

    # --- Shut-in THP (static hydrostatic) ---
    pressure_shut_profile = np.zeros(n_segments)
    pressure_shut_profile[-1] = P_res_pa
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
        pressure_shut_profile[i] = pressure_shut_profile[i+1] - rho_loc * G * dz_tvd[i]
    THP_shut_pa = pressure_shut_profile[0]

    # --- Flowing THP with iterative sandface convergence ---
    # initialization
    max_iter = 50
    tol_pa = 1e-3
    # initial sandface guess using reservoir props (as before)
    try:
        q_vol_nominal = mass_flow_kg_s / rho_res
        sandface_guess = (mu_res * q_vol_nominal) / (2.0 * pi * perm_m2 * h_res) * (np.log(re / rw) + skin) if perm_m2 > 0 else 0.0
    except:
        sandface_guess = 0.0

    sandface = sandface_guess
    prev_perf_pressure = None

    for it in range(max_iter):
        # set bottom (perforation) flowing pressure = reservoir + sandface
        pressure_flow_profile = np.zeros(n_segments)
        pressure_flow_profile[-1] = P_res_pa + sandface

        rho_flow = np.zeros(n_segments)
        mu_flow = np.zeros(n_segments)
        seg_friction_loss = np.zeros(n_segments-1)
        seg_hydrostatic = np.zeros(n_segments-1)

        # set bottom node props using PropsSI at perforation conditions if possible
        try:
            rho_flow[-1] = PropsSI('D','T',T_profile_K[-1],'P',pressure_flow_profile[-1],'CO2')
            mu_flow[-1] = PropsSI('V','T',T_profile_K[-1],'P',pressure_flow_profile[-1],'CO2')
        except:
            rho_flow[-1] = rho_res
            mu_flow[-1] = mu_res

        # march upward conserving mass: q_vol_local = mass_flow / rho_loc (rho at segment deeper node)
        for i in range(n_segments-2, -1, -1):
            # use properties at deeper node (i+1) to evaluate losses over the segment
            rho_loc = rho_flow[i+1]
            mu_loc = mu_flow[i+1]
            if rho_loc == 0 or np.isnan(rho_loc):
                rho_loc = rho_res
            if mu_loc == 0 or np.isnan(mu_loc):
                mu_loc = mu_res

            q_vol_local = mass_flow_kg_s / (rho_loc if rho_loc>0 else rho_res)
            vel_loc = q_vol_local / A_tub
            Re_loc = rho_loc * vel_loc * D_tub_m / (mu_loc if mu_loc>0 else 1e-12)
            f_loc = haaland_friction(Re_loc, eps_m / D_tub_m) if Re_loc>0 else np.nan

            dp_f = f_loc * (dz_md[i] / D_tub_m) * (rho_loc * vel_loc**2 / 2.0)
            dp_h = rho_loc * G * dz_tvd[i]

            seg_friction_loss[i] = dp_f
            seg_hydrostatic[i] = dp_h

            # marching upward: shallower pressure = deeper pressure - hydrostatic - friction
            pressure_flow_profile[i] = pressure_flow_profile[i+1] - dp_h - dp_f

            # estimate properties at the shallower node (i) using pressure just computed there
            try:
                rho_flow[i] = PropsSI('D','T',T_profile_K[i],'P',pressure_flow_profile[i],'CO2')
                mu_flow[i] = PropsSI('V','T',T_profile_K[i],'P',pressure_flow_profile[i],'CO2')
            except:
                rho_flow[i] = rho_res
                mu_flow[i] = mu_res

        THP_flow_pa = pressure_flow_profile[0]

        # recompute sandface using the bottom node local properties (use bottom mu and bottom rho)
        rho_bottom = rho_flow[-1] if rho_flow[-1]>0 else rho_res
        mu_bottom = mu_flow[-1] if mu_flow[-1]>0 else mu_res
        q_vol_bottom = mass_flow_kg_s / rho_bottom
        try:
            sandface_new = (mu_bottom * q_vol_bottom) / (2.0 * pi * perm_m2 * h_res) * (np.log(re / rw) + skin) if perm_m2 > 0 else 0.0
        except:
            sandface_new = sandface

        # check convergence on perforation pressure (or sandface)
        perf_pressure = pressure_flow_profile[-1]
        if prev_perf_pressure is not None and abs(perf_pressure - prev_perf_pressure) < tol_pa:
            sandface = sandface_new
            break

        prev_perf_pressure = perf_pressure
        sandface = sandface_new
    else:
        st.warning(f'Sandface iteration did not fully converge in {max_iter} iters (last diff = {abs(perf_pressure - prev_perf_pressure):.3f} Pa)')

    # totals for reporting
    total_friction = np.nansum(seg_friction_loss)
    total_hydro_shut = np.nansum([rho_shut[i]*G*dz_tvd[i] for i in range(n_segments-1)])
    total_hydro_flow = np.nansum(seg_hydrostatic)
    hydro_diff = total_hydro_shut - total_hydro_flow

    # compute difference THP_flow - THP_shut
    thp_diff_pa = THP_flow_pa - THP_shut_pa

    # Show numeric breakdown
    st.subheader("Pressure breakdown (surface-referenced)")
    st.write(f"Flowing THP = {from_pa(THP_flow_pa, pressure_unit):.6f} {pressure_unit}")
    st.write(f"Shut-in THP = {from_pa(THP_shut_pa, pressure_unit):.6f} {pressure_unit}")
    st.write(f"THP_flow - THP_shut = {from_pa(thp_diff_pa, pressure_unit):.6f} {pressure_unit}")
    st.write("---")
    st.write(f"Sandface loss (Pa) = {sandface:.3f} Pa ({from_pa(sandface, pressure_unit):.6f} {pressure_unit})")
    st.write(f"Total frictional loss up the tubing (Pa) = {total_friction:.3f} Pa ({from_pa(total_friction, pressure_unit):.6f} {pressure_unit})")
    st.write(f"Total hydrostatic (shut-in) (Pa) = {total_hydro_shut:.3f} Pa ({from_pa(total_hydro_shut, pressure_unit):.6f} {pressure_unit})")
    st.write(f"Total hydrostatic (flowing) (Pa) = {total_hydro_flow:.3f} Pa ({from_pa(total_hydro_flow, pressure_unit):.6f} {pressure_unit})")
    st.write(f"Hydrostatic difference (shut - flow) (Pa) = {hydro_diff:.3f} Pa ({from_pa(hydro_diff, pressure_unit):.6f} {pressure_unit})")

    # Final sanity check
    if thp_diff_pa >= -1e-6:
        st.success("Flowing THP >= Shut-in THP (as expected for injection).")
    else:
        st.error("Flowing THP < Shut-in THP — still inconsistent. See breakdown above to diagnose.")
        # Suggest likely causes
        st.write("- If total frictional loss > sandface + hydrostatic difference, that explains the result.")
        st.write("- Check permeability/skin/flow rate; if very high friction (tiny ID or extremely high rate) friction may dominate.")
        st.write("- If the sandface model is inappropriate for your reservoir (non-Darcy, multi-phase, or partial penetration), include a tailored IPR or iterative well/reservoir coupling.")

    # Plot and CSV export (same as before)
    fig, ax = plt.subplots(figsize=(5,8))
    ax.plot(from_pa(pressure_flow_profile,pressure_unit), tvd_nodes, label='Flowing THP')
    ax.plot(from_pa(pressure_shut_profile,pressure_unit), tvd_nodes, linestyle='--', label='Shut-in THP')
    ax.set_xlabel(f'Pressure ({pressure_unit})')
    ax.set_ylabel('TVD (m)')
    ax.invert_yaxis()
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

    # CSV export includes segment losses
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
        'Parameter': ['Flowing THP','Shut-in THP','Sandface loss (Pa)','Total friction (Pa)','Hydrostatic diff (Pa)'],
        'Value': [THP_flow_pa, THP_shut_pa, sandface, total_friction, hydro_diff],
        'Unit': ['Pa']*5
    })

    csv_buf = df_profile.to_csv(index=False) + '\n\n' + summary.to_csv(index=False)
    st.download_button('Download CSV', data=csv_buf.encode('utf-8'), file_name='CO2_injection_profile.csv', mime='text/csv')

    st.success('Calculation complete')

# Footer
st.markdown("---")
st.markdown("© 2025 CCS Energy Pty Ltd. www.ccsenergy.com.au All rights reserved. info@ccsenergy.com.au")

