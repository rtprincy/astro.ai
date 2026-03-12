import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from astropy.timeseries import LombScargle
from lk_stat_package import lk_stat
import base64
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Astro Time-Series Analyser",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded"
)
# Custom CSS for dark theme and styling
# st.markdown("""
# <style>
#     .main {
#         background-color: #0e1117;
#         color: #ffffff;
#     }
#     .sidebar .sidebar-content {
#         background-color: #1a1c23;
#         color: #ffffff;
#     }
#     .stButton>button {
#         background-color: #4CAF50;
#         color: white;
#         border-radius: 8px;
#         border: none;
#         padding: 10px 20px;
#         font-size: 16px;
#     }
#     .stButton>button:hover {
#         background-color: #45a049;
#     }
#     .stTextInput>div>div>input {
#         background-color: #2d3748;
#         color: #ffffff;
#         border-radius: 5px;
#     }
#     .stSelectbox>div>div>select {
#         background-color: #2d3748;
#         color: #ffffff;
#         border-radius: 5px;
#     }
#     .stHeader {
#         color: #4CAF50;
#     }
#     .stSubheader {
#         color: #81c784;
#     }
#     .stWrite {
#         color: #e8f5e8;
#     }
#     .css-1d391kg {
#         background-color: #0e1117;
#     }
# </style>
# """, unsafe_allow_html=True)

# Header with title and description
st.markdown("""
<meta property="og:image" content="https://github.com/rtprincy/astro.ai/blob/main/streamlit_page.png>
<meta property="og:title" content="Time-series analysis tools for astronomy">
<meta property="og:description" content="Time-series frequency search and visualisation">
<div style="text-align: center; padding: 20px;">
    <img src="https://github.com/rtprincy/rtprincy.github.io/blob/main/assets/background_astroml.png?raw=true" 
         style="width: 100%; max-width: 800px; border-radius: 10px; margin-bottom: 20px;" 
         alt="Time-Series Analysis">
    <h1 style="color: #4CAF50;">🌌 Time-Series Analysis Tools</h1>
    <p style="font-size: 18px; color:  black;">
        Analyse time-series data with advanced periodogram techniques.
        Upload your CSV data and discover periodic signals in your observations.
    </p>
</div>
""", unsafe_allow_html=True)


# Configure matplotlib for dark theme
plt.style.use('default')
plt.rcParams['figure.dpi']=300
plt.rcParams['xtick.minor.visible']=False
plt.rcParams['ytick.minor.visible']=False
plt.rcParams['axes.labelsize']=25
plt.rcParams['xtick.labelsize']=22
plt.rcParams['ytick.labelsize']=22


def freq_grid(times,oversampling_factor=10,f0=None,fn=None):
    times=np.sort(times)
    df = 1.0 / (times.max() - times.min())
    if (f0 is None) | (int(f0)==0):
        f0 = df
    if fn is None:
        fn = 0.5 / np.median(np.diff(times)) 
        
    return np.arange(f0, fn, df / oversampling_factor)

# use sidebar for controls; outputs on main area
uploaded_file = None
df = None
compute_button = False
manual_button = False

with st.sidebar:
    st.markdown("### 📊 Input Parameters")
    st.markdown("Upload your astronomical time-series data and configure analysis settings.")

    # let user choose whether to upload or use demo data
    data_source = st.radio("📁 Data Source", ["Upload CSV file", "Use local demo light curve"], index=0)

    if data_source == "Upload CSV file":
        uploaded_file = st.file_uploader("📁 Upload CSV file", type="csv", help="Select a CSV file containing your time-series data.")
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
    else:
        # load the bundled demo file
        try:
            df = pd.read_csv("demo_lc.csv")
            st.info("ℹ️ Loaded demo light curve from `demo_lc.csv`.")
        except Exception as e:
            st.error(f"❌ Failed to load demo file: {e}")
            df = None

    if df is not None:
        # default selections: first three columns
        x_default = 0
        y_default = 1 if len(df.columns) > 1 else 0
        yerr_default = 3 if len(df.columns) > 2 else 0  # index within error_options (0 -> None)

        x_column = st.selectbox("⏰ Select Time Column (X)", df.columns, index=x_default, help="Choose the column containing time data.")
        y_column = st.selectbox("📊 Select Magnitude/Flux Column (Y)", df.columns, index=y_default, help="Choose the column containing magnitude or flux data.")
        error_options = ["None"] + list(df.columns)
        yerr_column = st.selectbox("📏 Select Error Column (Yerr)", error_options, index=yerr_default, help="Optional: Choose the column containing error data.")
        filter_options = ["None"] + list(df.columns)
        filter_column = st.selectbox("🔍 Select Filter Column (optional)", filter_options, help="Optional: Choose a column to filter data by.")
        filter_value = st.text_input("Filter Value (e.g., 'q')", value="q", help="Enter the value to filter by.") if filter_column != "None" else None
        col1, col2 = st.columns(2)
        with col1:
            f0 = st.text_input("📈 Minimum Frequency (c/d)", value="0", help="Minimum frequency in cycles per day.")
            fn = st.text_input("📉 Maximum Frequency (c/d)", value="5", help="Maximum frequency in cycles per day.")
        with col2:
            oversampling_factor = st.text_input("🔄 Oversampling Factor (≤20)", value="5", help="Factor for frequency grid resolution.")
        try:
            fn = float(fn)
            f0 = float(f0)
            oversampling_factor = int(oversampling_factor)
            if oversampling_factor > 20:
                st.error("❌ Oversampling factor must be ≤ 20.")
                compute_disabled = True
            else:
                compute_disabled = False
        except ValueError:
            st.error("❌ Please enter valid numeric values for frequency and oversampling.")
            compute_disabled = True

        compute_button = st.button("🚀 Compute Periodogram and Find Best Period", disabled=compute_disabled)
        st.markdown("---")
        manual_input = st.text_input("🔧 Manual period in days (optional)", help="Enter a custom period for phase folding analysis.")
        manual_button = st.button("✅ Apply Manual Period")
        # Note: plotting happens automatically below the dataframe preview; manual period is applied when OK is clicked

# main output area
if df is not None:
    # indicate source
    if 'data_source' in locals() and data_source == "Use local demo light curve":
        st.info("ℹ️ Analysis running on demo data (demo_lc.csv)")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("### 📋 Data Preview")
        st.dataframe(df.head(), use_container_width=True)
    with col2:
        st.markdown("### 📈 Data Summary")
        st.write(f"**Rows:** {len(df)}")
        st.write(f"**Columns:** {len(df.columns)}")
        st.write(f"**Selected X:** {x_column}")
        st.write(f"**Selected Y:** {y_column}")

    if compute_button:
        x = df[x_column].values
        y = df[y_column].values
        yerr = df[yerr_column].values if yerr_column != "None" and yerr_column in df.columns else None
        if filter_column != "None" and filter_column in df.columns:
            filter_mask = df[filter_column].astype(str) == filter_value
            x = x[filter_mask]
            y = y[filter_mask]
            if yerr is not None:
                yerr = yerr[filter_mask]
            st.info(f"ℹ️ Applied filter '{filter_column} = {filter_value}': using {len(x)} data points.")
        ls = LombScargle(x, y, dy=yerr, normalization="psd")
        frequency = freq_grid(x, oversampling_factor=oversampling_factor, f0=f0, fn=fn)
        period = 1 / frequency
        power = ls.power(frequency)
        theta = lk_stat(period, y, yerr, x)

        psi = 2 * power / theta
        psi_norm = psi / psi.max()
        best_idx = np.argmax(psi_norm)
        best_freq = frequency[best_idx]
        best_period = 1 / best_freq
        st.session_state.x=x
        st.session_state.y=y
        st.session_state.yerr=yerr
        st.session_state.best_period = best_period
        st.session_state.psi_norm = psi_norm
        st.session_state.frequency = frequency
        st.session_state.best_freq = best_freq
        st.success(f"✅ Best period found: {best_period:.4f} days ({best_period*24:.2f} hours / {best_period*24*60:.1f} minutes)")

    # handle manual period submission
    if manual_button:
        try:
            val = float(manual_input)
            if val > 0:
                st.session_state.manual_period = val
                st.success(f"✅ Manual period set to {val:.4f} days")
            else:
                st.warning("⚠️ Please enter a positive number for manual period.")
        except ValueError:
            st.error("❌ Manual period must be a valid number.")


    # plot periodogram if computed
    if 'psi_norm' in st.session_state:
        x = st.session_state.x
        y = st.session_state.y
        yerr = st.session_state.yerr
        st.markdown("### 🌟 Original time-series")
        fig, ax = plt.subplots(figsize=(10, 6))
        if yerr is not None:
            ax.errorbar(
                x=x-min(x),  # shift time to start at zero for better visualization
                y=y,
                yerr=yerr,
                fmt="o",
                markersize=4,
                color='C0',
                ecolor='C0',
                capsize=2,
                label="Data points"
            )
        else:
            ax.plot(x, y, "o", markersize=4, color='C0', label="Data points")
        ax.set_xlabel("Time - %.5f"%(min(x)))
        ax.set_ylabel("Magnitude/Flux")
        ax.invert_yaxis()
        ax.set_title("Original time-series", fontsize=20)
        ax.legend(fontsize=20)
        # ax.grid(True, alpha=0.3)
        st.pyplot(fig)

        st.markdown("### 📊 Hybrid Psi Periodogram")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(st.session_state.frequency, st.session_state.psi_norm, color='k', linewidth=2)
        ax.set_xlabel("Frequency (cycles/day)")
        ax.set_ylabel("Normalised Psi")
        ax.plot(st.session_state.best_freq, 1, "o", markersize=4, color="red",label=f"Best Frequency: {st.session_state.best_freq:.4f} c/d")
        ax.set_title("Periodogram Analysis", fontsize=20)
        ax.legend(fontsize=20)
        # ax.grid(True, alpha=0.3)
        st.pyplot(fig)

    # phase-folding uses manual period if provided, otherwise uses stored best period
    
    # collect periods
    period_best = st.session_state.get("best_period", None)
    # manual period comes from session state if user clicked OK
    period_manual = st.session_state.get('manual_period', None)

    # now plot best & manual periods
    if period_best and x is not None and y is not None:
        st.info(f"ℹ️ Using best period from frequency search: {period_best:.4f} days")
        phase = (x / period_best) % 1
        sorted_indices = np.argsort(phase)
        phase = phase[sorted_indices]
        y_best = y[sorted_indices]
        yerr_best = yerr[sorted_indices] if yerr is not None else None
        st.markdown("### 🔍 Best Period Analysis")
        fig, ax = plt.subplots(figsize=(10, 6))
        if yerr_best is not None:
            ax.errorbar(
                x=phase,
                y=y_best,
                yerr=yerr_best,
                markersize=3,
                fmt="o",
                color="k",
                ecolor='k',
                capsize=2
            
            )
            ax.errorbar(x=phase + 1, y=y_best, yerr=yerr_best, markersize=3, fmt="o", color="k", ecolor='k', capsize=2)
        else:
            ax.plot(phase, y_best, "o", markersize=3, color="k")
            ax.plot(phase + 1, y_best, "o", markersize=3, color="k")
        ax.set_xlabel("Phase")
        ax.set_ylabel("Magnitude/Flux")
        ax.invert_yaxis()
        ax.set_title(f"Phase-folded time-series (Best Period: {period_best:.4f} days)", fontsize=20)
        # ax.grid(True, alpha=0.3)
        st.pyplot(fig)

    if period_manual and x is not None and y is not None:
        st.info(f"ℹ️ Using manually entered period: {period_manual:.4f} days")
        phase = (x / period_manual) % 1
        sorted_indices = np.argsort(phase)
        phase = phase[sorted_indices]
        y_man = y[sorted_indices]
        yerr_man = yerr[sorted_indices] if yerr is not None else None
        st.markdown("### 🎯 Manual Period Analysis")
        fig, ax = plt.subplots(figsize=(10, 6))
        if yerr_man is not None:
            ax.errorbar(
                x=phase,
                y=y_man,
                yerr=yerr_man,
                markersize=3,
                fmt="o",
                color="k",
                ecolor='k',
                capsize=2
            )
            ax.errorbar(x=phase + 1, y=y_man, yerr=yerr_man, markersize=3, fmt="o", color="k", ecolor='k', capsize=2)
        else:
            ax.plot(phase, y_man, "o", markersize=3, color="k")
            ax.plot(phase + 1, y_man, "o", markersize=3, color="k")
        ax.set_xlabel("Phase")
        ax.set_ylabel("Magnitude/Flux")
        ax.invert_yaxis()
        ax.set_title(f"Phase-folded timeseries (Manual Period: {period_manual:.4f} days)", fontsize=20)
        # ax.grid(True, alpha=0.3)
        st.pyplot(fig)