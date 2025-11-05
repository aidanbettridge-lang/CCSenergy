"""
CO2 Injection — Tubing Head Pressure Estimator (patched)
- Uses separate density profiles for static and flowing hydrostatics.
- Flowing profile computed iteratively (per-segment, mass-conserved) so rho, hydrostatic, and friction are consistent.
- Includes FTHP vs Rate sensitivity plot.
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import pi, log
from CoolProp.CoolProp import PropsSI

# ---------------------------
# Page config / UI
# ---------------------------
st.set_page_config(page_title="CO₂ Injection — THP Estimator", layout="wide")
st.image("https://ccsenergy.com.au/wp-content/themes/custom-theme/assets/images/logo.svg", width=250)
st.title("CO₂ Injection — Tubing Head Pressure Estimator")

# ---------------------------
# Constants and helpers
# ---------------------------
G = 9.81  # m/s^2

def haaland_friction(Re, eD):
    if Re <= 0 or np.isnan(Re):
        return np.nan
    if Re < 2300:
        return 64.0 / Re
    # Haaland formula
    return 1.0 / (-1.8 * np.log10((eD / 3.7)**1.11 + 6.9 / Re))**2

def to_pa(val, unit):
    return val * 1e5 if unit == 'bar' else val * 6894.76

def from_pa(val, unit):
    return val / 1e5 if unit == 'bar' else val / 6894.76

def length_to_m(val, unit):
    return val / 1000.0 if unit == 'mm' else val * 0.0254

# ---------------------------
# Sidebar inputs
# ---------------------------
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

    q_tpd = st.number_input("Base injection rate (t/d)", value=1000.0)
    perm_mD = st.number_input("Permeability (mD)", value=100.0)
    poro = st.number_input("Porosity (fraction)", value=0.25)
    h_res = st.number_input("Reservoir thickness (m)", value=10.0)
    rw = st.number_input("Wellbore radius (m)", value=0.0889)
    re = st.number_input("Drainage radius (m)", value=100.0)

    n_segments = st.slider("Number of segments (for profiles)", 20, 1000, 200, 10)

    st.markdown("---")
    st.header("Sensitivity (optional)")
    show_sensitivity = st.checkbox("Show FTHP vs Rate Sensitivity", value=True)
    qmin = st.number_input("Min rate (t/d)", value=200.0)
    qmax = st.number_input("Max rate (t/d)", value=3000.0)
    qstep = st.number_input("Rate step (t/d)", value=200.0)

# ---------------------------
# Convert units / derived
# ---------------------------
P_res_pa = to_pa(P_res_in, pressure_unit)
T_res_K = T_res_c + 273.15
T_surf_K = T_surf_c + 273.15
D_tub_m = length_to_m(tubing_id_in, tubing_unit)
eps_m = tubing_rough_mm / 1000.0
perm_m2 = perm_mD * 9.869233e-16

# ---------------------------
# CO2 base properties at reservoir (fallback)
# ---------------------------
try:
    rho_res = PropsSI('D', 'T', T_res_K, 'P', P_res_pa, 'CO2')
    mu_res = PropsSI('V', 'T', T_res_K, 'P', P_res_pa, 'CO2')
except Exception as e:
    st.error("CoolProp error evaluating reservoir CO₂ properties. Check inputs.")
    st.stop()

A_tub = pi * (D_tub_m**2) / 4.0

# ---------------------------
# Helper: compute static (shut-in) profile
# ---------------------------
def compute_shut_in_profile(n_segments):
    """
    Returns:
      pressure_shut_profile (Pa, array length n_segments, top->bottom indexed 0..N-1? We'll keep 0=surface, -1=perforation),
      rho_shut_profile (kg/m3, same length),
      THP_shut_pa (surface pressure)
    Approach: march from bottom (perforation) upward: P[i] = P[i+1] - rho_loc*g*dz
    """
    md_nodes = np.linspace(0.0, MD, n_segments)
    tvd_nodes = (md_nodes / MD) * TVD
    dz_tvd = np.diff(tvd_nodes)

    pressure_shut = np.zeros(n_segments)
    rho_shut = np.zeros(n_segments)

    # boundary condition: bottom = reservoir pressure
    pressure_shut[-1] = P_res_pa
    rho_shut[-1] = rho_res
    # march upward
    for i in range(n_segments-2, -1, -1):
        T_loc = T_surf_K + (T_res_K - T_surf_K) * (tvd_nodes[i+1] / TVD)  # use deeper node T for properties
        try:
            rho_loc = PropsSI('D', 'T', T_loc, 'P', pressure_shut[i+1], 'CO2')
        except:
            rho_loc = rho_res
        rho_shut[i+1] = rho_loc
        pressure_shut[i] = pressure_shut[i+1] - rho_loc * G * dz_tvd[i]
    # for the top node 0, ensure rho_shut[0] defined (estimate from top pressure)
    try:
        rho_shut[0] = PropsSI('D', 'T', T_surf_K, 'P', pressure_shut[0], 'CO2')
    except:
        rho_shut[0] = rho_res

    THP_shut_pa = pressure_shut[0]
    md_nodes = np.linspace(0.0, MD, n_segments)
    tvd_nodes = (md_nodes / MD) * TVD
    return pressure_shut, rho_shut, THP_shut_pa, tvd_nodes

# ---------------------------
# Helper: compute flowing profile (iterative per-segment)
# ---------------------------
def compute_flowing_profile(q_tpd, n_segments, max_iter=60, tol=1e-3):
    """
    Iteratively computes flowing pressure profile (Pa) and rho profile (kg/m3)
    Returns:
      pressure_flow_profile (Pa), rho_flow_profile (kg/m3), THP_flow_pa (surface), sandface_loss (Pa), total_friction (Pa), tvd_nodes
    Approach:
      - Start with initial guess for rho_flow_profile = rho_shut (or rho_res)
      - Set sandface_loss using mu at bottom and q_vol based on bottom rho
      - March upward computing dp_h (rho_loc * g * dz) and dp_f (Haaland using local q_vol = mass / rho_loc)
      - Update node pressures
      - Recompute densities from new pressures and iterate until THP converges
    """
    mass_flow_kg_s = q_tpd * 1000.0 / 86400.0
    md_nodes = np.linspace(0.0, MD, n_segments)
    tvd_nodes = (md_nodes / MD) * TVD
    dz_md = np.diff(md_nodes)
    dz_tvd = np.diff(tvd_nodes)

    # initial rho guess: use static profile densities (more realistic than rho_res everywhere)
    _, rho_shut_guess, _, _ = compute_shut_in_profile(n_segments)
    rho_flow = rho_shut_guess.copy()
    pressure_flow = np.full(n_segments, P_res_pa)  # initialize with reservoir everywhere
    pressure_flow[-1] = P_res_pa  # bottom start

    total_friction = 0.0
    sandface_loss = 0.0

    for it in range(max_iter):
        # compute bottom mu and rho for sandface calc
        rho_bottom = rho_flow[-1] if rho_flow[-1] > 0 else rho_res
        # estimate bottom viscosity (use temperature at bottom node)
        T_bottom = T_surf_K + (T_res_K - T_surf_K) * (tvd_nodes[-1] / TVD)
        try:
            mu_bottom = PropsSI('V', 'T', T_bottom, 'P', pressure_flow[-1], 'CO2')
        except:
            mu_bottom = mu_res

        # sandface loss using bottom mu and bottom rho to compute q_vol at sandface
        q_vol_bottom = mass_flow_kg_s / (rho_bottom if rho_bottom>0 else rho_res)
        if perm_m2 > 0 and h_res > 0:
            try:
                sandface_loss_new = (mu_bottom * q_vol_bottom) / (2.0 * pi * perm_m2 * h_res) * (log(re / rw) + skin)
            except:
                sandface_loss_new = 0.0
        else:
            sandface_loss_new = 0.0

        # boundary condition: perforation flowing pressure = P_res + sandface_loss
        pressure_flow[-1] = P_res_pa + sandface_loss_new

        # march upward from bottom to surface computing segment losses using local properties
        seg_friction = np.zeros(n_segments-1)
        seg_hydro = np.zeros(n_segments-1)
        for i in range(n_segments-2, -1, -1):
            # use deeper node properties to compute segment behaviour
            rho_loc = rho_flow[i+1] if rho_flow[i+1] > 0 else rho_res
            # local volumetric flow
            q_vol_local = mass_flow_kg_s / rho_loc
            vel_loc = q_vol_local / A_tub
            # local viscosity at deeper node pressure and temperature at deeper node
            T_loc = T_surf_K + (T_res_K - T_surf_K) * (tvd_nodes[i+1] / TVD)
            try:
                mu_loc = PropsSI('V', 'T', T_loc, 'P', pressure_flow[i+1], 'CO2')
            except:
                mu_loc = mu_res

            # friction factor and frictional loss across segment (use MD for length)
            Re_loc = rho_loc * vel_loc * D_tub_m / (mu_loc if mu_loc>0 else 1e-12)
            f_loc = haaland_friction(Re_loc, eps_m / D_tub_m) if Re_loc>0 else 0.0
            dp_f = f_loc * (dz_md[i] / D_tub_m) * (rho_loc * vel_loc**2 / 2.0)
            dp_h = rho_loc * G * dz_tvd[i]

            # shallower pressure = deeper pressure - hydrostatic - friction
            pressure_flow[i] = pressure_flow[i+1] - dp_h - dp_f

            seg_friction[i] = dp_f
            seg_hydro[i] = dp_h

        # update densities at nodes from computed pressures
        rho_new = np.zeros(n_segments)
        for j in range(n_segments):
            Tj = T_surf_K + (T_res_K - T_surf_K) * (tvd_nodes[j] / TVD)
            try:
                rho_new[j] = PropsSI('D', 'T', Tj, 'P', pressure_flow[j], 'CO2')
            except:
                rho_new[j] = rho_res

        # evaluate convergence on surface THP (top node) and bottom sandface loss
        THP_top = pressure_flow[0]
        total_friction_new = np.nansum(seg_friction)
        # check if THP changed small enough
        if it > 0 and abs(THP_top - prev_THP_top) < tol and abs(sandface_loss_new - sandface_loss) < tol:
            sandface_loss = sandface_loss_new
            rho_flow = rho_new
            total_friction = total_friction_new
            break

        # update for next iter
        prev_THP_top = THP_top
        sandface_loss = sandface_loss_new
        rho_flow = rho_new
        total_friction = total_friction_new
    else:
        st.warning(f"Flowing profile iteration did not converge in {max_iter} iterations (last THP change = {abs(THP_top - prev_THP_top):.3f} Pa)")

    THP_flow_pa = pressure_flow[0]
    return pressure_flow, rho_flow, THP_flow_pa, sandface_loss, total_friction, tvd_nodes

# ---------------------------
# Single-run function (used by sensitivity)
# ---------------------------
def calculate_all(q_tpd, n_segments):
    # shut-in
    pressure_shut, rho_shut, THP_shut_pa, tvd_nodes = compute_shut_in_profile(n_segments)
    # flowing
    pressure_flow, rho_flow, THP_flow_pa, sandface_loss, total_friction, tvd_nodes = compute_flowing_profile(q_tpd, n_segments)
    return {
        'pressure_shut': pressure_shut,
        'rho_shut': rho_shut,
        'THP_shut_pa': THP_shut_pa,
        'pressure_flow': pressure_flow,
        'rho_flow': rho_flow,
        'THP_flow_pa': THP_flow_pa,
        'sandface_loss': sandface_loss,
        'total_friction': total_friction,
        'tvd_nodes': tvd_nodes
    }

# ---------------------------
# Run calculations when user clicks
# ---------------------------
if st.button("Calculate"):
    results = calculate_all(q_tpd, n_segments)

    THP_shut_pa = results['THP_shut_pa']
    THP_flow_pa = results['THP_flow_pa']
    sandface_loss = results['sandface_loss']
    total_friction = results['total_friction']
    pressure_shut = results['pressure_shut']
    pressure_flow = results['pressure_flow']
    rho_shut = results['rho_shut']
    rho_flow = results['rho_flow']
    tvd_nodes = results['tvd_nodes']

    # summary numbers
    st.subheader("Results")
    st.write(f"Reservoir Pressure (BHP) = {from_pa(P_res_pa, pressure_unit):.4f} {pressure_unit}")
    st.write(f"Shut-in THP = {from_pa(THP_shut_pa, pressure_unit):.4f} {pressure_unit}")
    st.write(f"Flowing THP = {from_pa(THP_flow_pa, pressure_unit):.4f} {pressure_unit}")
    st.write(f"ΔTHP (Flowing - Shut-in) = {from_pa(THP_flow_pa - THP_shut_pa, pressure_unit):.4f} {pressure_unit}")
    st.write(f"Sandface loss = {from_pa(sandface_loss, pressure_unit):.4f} {pressure_unit}")
    st.write(f"Total tubing friction (sum segments) = {from_pa(total_friction, pressure_unit):.4f} {pressure_unit}")

    if THP_flow_pa >= THP_shut_pa - 1e-6:
        st.success("Flowing THP >= Shut-in THP (expected for injection).")
    else:
        st.error("Flowing THP < Shut-in THP (unexpected) — check inputs and outputs above.")

    # ---------------------------
    # Pressure vs depth plot (shut and flow)
    # ---------------------------
    fig, ax = plt.subplots(figsize=(5, 8))
    ax.plot(from_pa(pressure_flow, pressure_unit), tvd_nodes, label='Flowing', color='blue')
    ax.plot(from_pa(pressure_shut, pressure_unit), tvd_nodes, linestyle='--', label='Shut-in', color='red')
    ax.set_xlabel(f'Pressure ({pressure_unit})')
    ax.set_ylabel('TVD (m)')
    ax.invert_yaxis()
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

    # ---------------------------
    # CSV export with profiles and per-seg losses
    # ---------------------------
    # compute per-segment friction and hydrostatic for final flow profile (for CSV)
    dz_md = np.diff(np.linspace(0.0, MD, n_segments))
    dz_tvd = np.diff(tvd_nodes)
    seg_friction_loss = np.zeros(n_segments-1)
    seg_hydrostatic = np.zeros(n_segments-1)
    mass_flow_kg_s = q_tpd * 1000.0 / 86400.0
    for i in range(n_segments-1):
        rho_loc = rho_flow[i+1] if rho_flow[i+1]>0 else rho_res
        q_vol_local = mass_flow_kg_s / rho_loc
        vel_loc = q_vol_local / A_tub
        T_loc = T_surf_K + (T_res_K - T_surf_K) * (tvd_nodes[i+1] / TVD)
        try:
            mu_loc = PropsSI('V', 'T', T_loc, 'P', pressure_flow[i+1], 'CO2')
        except:
            mu_loc = mu_res
        Re_loc = rho_loc * vel_loc * D_tub_m / (mu_loc if mu_loc>0 else 1e-12)
        f_loc = haaland_friction(Re_loc, eps_m / D_tub_m) if Re_loc>0 else 0.0
        seg_friction_loss[i] = f_loc * (dz_md[i] / D_tub_m) * (rho_loc * vel_loc**2 / 2.0)
        seg_hydrostatic[i] = rho_loc * G * dz_tvd[i]

    df_profile = pd.DataFrame({
        'TVD_m': tvd_nodes,
        'Pressure_Flowing_'+pressure_unit: from_pa(pressure_flow, pressure_unit),
        'Pressure_Shut_in_'+pressure_unit: from_pa(pressure_shut, pressure_unit),
        'Density_flowing_kg_m3': rho_flow,
        'Density_shut_kg_m3': rho_shut
    })

    # Append segment arrays as extra columns offset by one so lengths match
    df_profile['Seg_friction_Pa'] = np.append(seg_friction_loss, np.nan)
    df_profile['Seg_hydrostatic_Pa'] = np.append(seg_hydrostatic, np.nan)

    summary = pd.DataFrame({
        'Parameter': ['Reservoir pressure (Pa)','THP_shut (Pa)','THP_flow (Pa)','Sandface loss (Pa)','Total friction (Pa)'],
        'Value': [P_res_pa, THP_shut_pa, THP_flow_pa, sandface_loss, total_friction],
        'Unit': ['Pa']*5
    })

    csv_buf = df_profile.to_csv(index=False) + '\n\n' + summary.to_csv(index=False)
    st.download_button('Download CSV', data=csv_buf.encode('utf-8'), file_name='CO2_THP_profiles.csv', mime='text/csv')

    st.success("Calculation complete")

    # ---------------------------
    # Sensitivity
    # ---------------------------
    if show_sensitivity:
        st.subheader("FTHP Sensitivity to Injection Rate")
        rates = np.arange(qmin, qmax + qstep, qstep)
        fthp_vals = []
        sithp_vals = []
        for q in rates:
            r = calculate_all(q, n_segments)
            fthp_vals.append(from_pa(r['THP_flow_pa'], pressure_unit))
            sithp_vals.append(from_pa(r['THP_shut_pa'], pressure_unit))

        fig2, ax2 = plt.subplots()
        ax2.plot(rates, fthp_vals, '-o', label='FTHP')
        ax2.plot(rates, sithp_vals, '--', label='SITHP')
        ax2.set_xlabel('Injection rate (t/d)')
        ax2.set_ylabel(f'Tubing head pressure ({pressure_unit})')
        ax2.grid(True)
        ax2.legend()
        st.pyplot(fig2)

        df_sens = pd.DataFrame({
            'Rate_tpd': rates,
            'FTHP_'+pressure_unit: fthp_vals,
            'SITHP_'+pressure_unit: sithp_vals
        })
        st.download_button('Download sensitivity CSV', data=df_sens.to_csv(index=False).encode('utf-8'),
                           file_name='FTHP_sensitivity.csv', mime='text/csv')

# ---------------------------
# Footer
# ---------------------------
st.markdown("---")
st.markdown("© 2025 CCS Energy Pty Ltd. All rights reserved. info@ccsenergy.com.au")
