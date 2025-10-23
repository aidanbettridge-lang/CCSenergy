import streamlit as st
from CoolProp.CoolProp import PropsSI

# Header with company logo
st.image("https://ccsenergy.com.au/wp-content/themes/custom-theme/assets/images/logo.svg", width=250)
st.title("CO₂ Storage Capacity Estimator")

# Description
st.write("""
This tool estimates the CO₂ storage capacity of a subsurface reservoir
based on basic reservoir parameters. It also estimates CO₂ density under reservoir conditions.
""")

st.header("Reservoir Parameters")

# Input fields
area_km2 = st.number_input("Reservoir Area (km²)", min_value=0.0, format="%.2f")
thickness_m = st.number_input("Reservoir Thickness (m)", min_value=0.0, format="%.2f")
porosity_percent = st.number_input("Porosity (%)", min_value=0.0, max_value=100.0, format="%.2f")
saturation_percent = st.number_input("Max CO₂ Saturation (%)", min_value=0.0, max_value=100.0, format="%.2f")

st.header("Reservoir Conditions")

# New input fields for reservoir conditions
temperature_c = st.number_input("Reservoir Temperature (°C)", min_value=-50.0, max_value=200.0,
                                value=40.0, format="%.2f")
pressure_mpa = st.number_input("Reservoir Pressure (MPa)", min_value=0.1, max_value=50.0,
                                value=10.0, format="%.2f")

# Automatically calculate CO2 density when conditions change
density_calculated = None
try:
    T = temperature_c + 273.15  # Kelvin
    P = pressure_mpa * 1e6      # Pascal
    density_calculated = PropsSI("D", "T", T, "P", P, "CO2")
    density_display = f"{density_calculated:,.2f} kg/m³"
except Exception as e:
    density_display = f"Error calculating density: {e}"

st.write(f"**Estimated CO₂ Density at these conditions:** {density_display}")

st.header("Storage Capacity Calculation")

# Option to input or use calculated density
use_calc_density = st.checkbox("Use the calculated CO₂ density above", value=True)

if use_calc_density and (density_calculated is not None):
    co2_density = density_calculated
else:
    co2_density = st.number_input("CO₂ Density (kg/m³)", min_value=0.0, format="%.2f")

if st.button("Estimate Storage Capacity"):
    # Convert area to m²
    area_m2 = area_km2 * 1e6

    # Convert percentages to fractions
    porosity = porosity_percent / 100
    saturation = saturation_percent / 100

    # Calculate pore volume
    pore_volume_m3 = area_m2 * thickness_m * porosity * saturation

    # Calculate mass of CO₂ stored
    co2_mass_kg = pore_volume_m3 * co2_density
    co2_mass_tonnes = co2_mass_kg / 1000

    # Display results
    st.success(f"Estimated CO₂ Storage Capacity: {co2_mass_tonnes:,.2f} tonnes")
    st.write(f"Using CO₂ Density: {co2_density:,.2f} kg/m³")

# Footer
st.markdown("---")
st.markdown("© 2025 CCS Energy Pty Ltd. All rights reserved")
