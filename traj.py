import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# =====================================
# HEADER
# =====================================
st.image("https://ccsenergy.com.au/wp-content/themes/custom-theme/assets/images/logo.svg", width=250)
st.set_page_config(page_title="Well Trajectory Viewer", layout="wide")

st.title("Well Trajectory Viewer")
st.write(
    "Upload a well survey (MD, Inclination, Azimuth), view 2D/3D trajectories, "
    "and interpolate between MD and TVD."
)

# -------------------------
# Helper: Minimum Curvature
# -------------------------
def compute_minimum_curvature(df, md_col="MD", inc_col="Inclination", azi_col="Azimuth"):
    """
    Computes North, East, TVD, and Horizontal displacement from MD, Inc, Azi
    using the minimum curvature method.
    Assumes:
      - MD in metres
      - Inc, Azi in degrees
      - TVD at first point = 0.0 m
    """
    df = df.copy()
    
    # Ensure sorted by MD
    df = df.sort_values(md_col).reset_index(drop=True)

    md = df[md_col].values.astype(float)
    inc_deg = df[inc_col].values.astype(float)
    azi_deg = df[azi_col].values.astype(float)

    # Convert to radians
    inc = np.deg2rad(inc_deg)
    azi = np.deg2rad(azi_deg)

    n = np.zeros_like(md)  # North
    e = np.zeros_like(md)  # East
    v = np.zeros_like(md)  # Vertical (TVD, positive downwards)

    for i in range(1, len(md)):
        dmd = md[i] - md[i - 1]
        if dmd <= 0:
            continue  # skip bad data

        inc1, inc2 = inc[i - 1], inc[i]
        azi1, azi2 = azi[i - 1], azi[i]

        cos_dogleg = (
            np.sin(inc1) * np.sin(inc2) * np.cos(azi2 - azi1)
            + np.cos(inc1) * np.cos(inc2)
        )
        # Numerical safety
        cos_dogleg = np.clip(cos_dogleg, -1.0, 1.0)
        dogleg = np.arccos(cos_dogleg)

        if dogleg == 0:
            rf = 1.0
        else:
            rf = (2 / dogleg) * np.tan(dogleg / 2)

        dN = 0.5 * dmd * rf * (np.sin(inc1) * np.cos(azi1) + np.sin(inc2) * np.cos(azi2))
        dE = 0.5 * dmd * rf * (np.sin(inc1) * np.sin(azi1) + np.sin(inc2) * np.sin(azi2))
        dV = 0.5 * dmd * rf * (np.cos(inc1) + np.cos(inc2))  # TVD increment (downwards)

        n[i] = n[i - 1] + dN
        e[i] = e[i - 1] + dE
        v[i] = v[i - 1] + dV

    hd = np.sqrt(n**2 + e**2)  # Horizontal displacement

    df["North"] = n
    df["East"] = e
    df["TVD"] = v
    df["HD"] = hd

    return df

# -------------------------
# File Upload
# -------------------------
st.sidebar.header("Upload Well Survey")
uploaded_file = st.sidebar.file_uploader(
    "Upload CSV with columns: MD, Inclination, Azimuth",
    type=["csv"],
)

example_text = """
Expected CSV columns (case-sensitive by default):
- MD (measured depth, m)
- Inclination (degrees)
- Azimuth (degrees)
"""
st.sidebar.info(example_text)

if uploaded_file is not None:
    try:
        df_raw = pd.read_csv(uploaded_file)

        # Try to be a bit flexible on column names
        col_map = {}
        for col in df_raw.columns:
            lc = col.strip().lower()
            if lc in ["md", "measured depth", "measured_depth"]:
                col_map[col] = "MD"
            elif lc in ["inclination", "inc"]:
                col_map[col] = "Inclination"
            elif lc in ["azimuth", "azi", "az"]:
                col_map[col] = "Azimuth"

        df_raw = df_raw.rename(columns=col_map)

        required_cols = ["MD", "Inclination", "Azimuth"]
        if not all(c in df_raw.columns for c in required_cols):
            st.error(
                f"Missing required columns. Found: {list(df_raw.columns)}. "
                f"Need at least: {required_cols}"
            )
        else:
            st.subheader("Uploaded Survey (first 10 rows)")
            st.dataframe(df_raw.head(10))

            # Compute trajectory
            df_traj = compute_minimum_curvature(df_raw)

            st.subheader("Computed Trajectory (first 10 rows)")
            st.dataframe(
                df_traj[["MD", "Inclination", "Azimuth", "North", "East", "TVD", "HD"]].head(10)
            )

            # -------------------------
            # Plots (stacked vertically)
            # -------------------------
            st.subheader("Trajectory Plots")

            # 2D view
            st.markdown("**2D View: TVD vs Horizontal Displacement**")
            fig_2d = px.line(
                df_traj,
                x="HD",
                y="TVD",
                labels={"HD": "Horizontal Displacement (m)", "TVD": "TVD (m)"},
            )
            # Depth increasing downwards -> reverse y-axis
            fig_2d.update_yaxes(autorange="reversed")
            fig_2d.update_layout(height=500)
            st.plotly_chart(fig_2d, use_container_width=True)

            # 3D view (below 2D)
            st.markdown("**3D View: East–North–TVD**")
            fig_3d = go.Figure(
                data=[
                    go.Scatter3d(
                        x=df_traj["East"],
                        y=df_traj["North"],
                        z=df_traj["TVD"],
                        mode="lines+markers",
                        marker=dict(size=3),
                        line=dict(width=4),
                    )
                ]
            )
            fig_3d.update_layout(
                scene=dict(
                    xaxis_title="East (m)",
                    yaxis_title="North (m)",
                    zaxis_title="TVD (m)",
                    zaxis=dict(autorange="reversed"),  # depth downwards
                ),
                height=500,
            )
            st.plotly_chart(fig_3d, use_container_width=True)

            # -------------------------
            # Interpolation Tool
            # -------------------------
            st.subheader("MD / TVD Interpolation")

            df_interp = df_traj[["MD", "TVD"]].dropna().copy()

            # Ensure monotonic MD
            df_interp = df_interp.sort_values("MD")
            md_vals = df_interp["MD"].values
            tvd_vals = df_interp["TVD"].values

            # Quick check: TVD mostly increasing
            if not np.all(np.diff(tvd_vals) >= -0.01):
                st.warning(
                    "TVD is not strictly increasing with MD. "
                    "Interpolation may not be unique in some ranges."
                )

            mode = st.radio(
                "Interpolation mode",
                ["Given MD → TVD", "Given TVD → MD"],
                horizontal=True,
            )

            if mode == "Given MD → TVD":
                md_input = st.number_input(
                    "Enter MD (m)",
                    min_value=float(md_vals.min()),
                    max_value=float(md_vals.max()),
                    value=float(md_vals.min()),
                )
                if st.button("Calculate TVD"):
                    tvd_est = np.interp(md_input, md_vals, tvd_vals)
                    st.success(
                        f"Estimated TVD at MD = {md_input:.2f} m is **{tvd_est:.2f} m**"
                    )

            else:  # Given TVD -> MD
                tvd_min, tvd_max = float(tvd_vals.min()), float(tvd_vals.max())
                tvd_input = st.number_input(
                    "Enter TVD (m)",
                    min_value=tvd_min,
                    max_value=tvd_max,
                    value=tvd_min,
                )
                if st.button("Calculate MD"):
                    # Need TVD sorted for np.interp
                    sort_idx = np.argsort(tvd_vals)
                    tvd_sorted = tvd_vals[sort_idx]
                    md_sorted = md_vals[sort_idx]
                    md_est = np.interp(tvd_input, tvd_sorted, md_sorted)
                    st.success(
                        f"Estimated MD at TVD = {tvd_input:.2f} m is **{md_est:.2f} m**"
                    )

    except Exception as e:
        st.error(f"Error reading or processing file: {e}")

else:
    st.info("Upload a CSV file to begin.")

# =====================================
# FOOTER
# =====================================
st.markdown("---")
st.markdown("© 2025 CCS Energy Pty Ltd. All rights reserved. info@ccsenergy.com.au")