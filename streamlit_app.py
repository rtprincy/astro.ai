import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from astropy.timeseries import LombScargle
from lk_stat_package import lk_stat

plt.rcParams['figure.dpi']=250
plt.rcParams['lines.color']='k'
plt.rcParams['axes.edgecolor']='k'
plt.rcParams['xtick.minor.visible']=False
plt.rcParams['ytick.minor.visible']=False
plt.rcParams['axes.labelsize']=22
plt.rcParams['xtick.labelsize']=20
plt.rcParams['ytick.labelsize']=20

st.title("Time-series analysis")

def freq_grid(times,oversampling_factor=10,f0=None,fn=None):
    times=np.sort(times)
    df = 1.0 / (times.max() - times.min())
    if f0 is None:
        f0 = df
    if fn is None:
        fn = 0.5 / np.median(np.diff(times)) 
    return np.arange(f0, fn, df / oversampling_factor)

# use sidebar for controls; outputs on main area
uploaded_file = None

df = None
with st.sidebar:
    st.header("Input Parameters")
    uploaded_file = st.file_uploader("Upload the time-series data (only csv file is supported)", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        x_column = st.selectbox("Select Time Column (X)", df.columns)
        y_column = st.selectbox("Select Magnitude/Flux Column (Y)", df.columns)
        error_options = ["None"] + list(df.columns)
        yerr_column = st.selectbox("Select Magnitude/Flux error Column (Yerr)", error_options)
        filter_options = ["None"] + list(df.columns)
        filter_column = st.selectbox("Select Filter Column (optional)", filter_options)
        filter_value = st.text_input("Filter Value (e.g., 'q')", value="q") if filter_column != "None" else None
        f0 = st.text_input("Minimum Frequency", value="0")
        fn = st.text_input("Maximum Frequency", value="50")
        oversampling_factor = st.text_input("Oversampling Factor (< 20)", value="5")
        try:
            fn = float(fn)
            f0 = float(f0)
            oversampling_factor = int(oversampling_factor)
            if oversampling_factor > 20:
                st.error("Oversampling factor must be less than or equal to 20.")
                compute_disabled = True
            else:
                compute_disabled = False
        except ValueError:
            st.error("Please enter valid numeric values for all inputs.")
            compute_disabled = True

        manual_period = st.number_input(
            "Enter period in days for manual phase folding (optional). Otherwise, the best period will be used once computed:",
            min_value=0.0,
            step=0.01,
            value=0.0,
        )
        compute_button = st.button("Compute Periodogram and Find Best Period", disabled=compute_disabled)
        # Note: plotting happens automatically below the dataframe preview; no separate button needed

# main output area
if uploaded_file:
    st.write("**Data Preview:**")
    st.write(df.head())
    x = df[x_column].values
    y = df[y_column].values
    yerr = df[yerr_column].values if yerr_column != "None" and yerr_column in df.columns else None
    st.subheader("Original Lightcurve")
    plt.figure(figsize=(8, 4))
    if yerr is not None:
        plt.errorbar(
            x=x-min(x),  # shift time to start at zero for better visualization
            y=y,
            yerr=yerr,
            fmt="o",
            markersize=5,
            label="Original Lightcurve",
        )
    else:
        plt.plot(x, y, "o", markersize=5, label="Original Lightcurve")
    plt.xlabel("Time (MJD - %.5f)"%(min(x)), fontsize=18)
    plt.ylabel("Magnitude/Flux", fontsize=18)
    plt.gca().invert_yaxis()
    plt.title("Original Lightcurve")
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    st.pyplot(plt)

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
            st.write(f"Applied filter '{filter_column} = {filter_value}': using {len(x)} data points.")
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
        st.session_state.best_period = best_period
        st.session_state.psi_norm = psi_norm
        st.session_state.frequency = frequency
        st.session_state.best_freq = best_freq
        st.write(f"Best period found: {best_period:.4f} days")

    # after displaying the preview we always show the lightcurve plots (with filtering applied)
    x = df[x_column].values
    y = df[y_column].values
    yerr = df[yerr_column].values if yerr_column != "None" else None
    if filter_column != "None" and filter_column in df.columns:
        filter_mask = df[filter_column].astype(str) == filter_value
        x = x[filter_mask]
        y = y[filter_mask]
        if yerr is not None:
            yerr = yerr[filter_mask]
        st.write(
            f"Applied filter '{filter_column} == {filter_value}': using {len(x)} data points for plotting."
        )

    # plot periodogram if computed
    if 'psi_norm' in st.session_state:
        st.subheader("Hybrid Psi Periodogram")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(st.session_state.frequency, st.session_state.psi_norm)
        ax.set_xlabel("Frequency")
        ax.set_ylabel("Normalized Psi")
        ax.axvline(
            st.session_state.best_freq,
            color="r",
            linestyle="--",
            label=f"Best Frequency: {st.session_state.best_freq:.4f} c/d",
        )
        ax.legend()
        st.pyplot(fig)

    # phase-folding uses manual period if provided, otherwise uses stored best period
    if manual_period > 0:
        period_days = manual_period
        st.write(f"Using manually entered period: {period_days:.4f} days")
    elif "best_period" in st.session_state:
        period_days = st.session_state.best_period
        st.write(f"Using best period from frequency search: {period_days:.4f} days")
    else:
        period_days = None

    if period_days and period_days > 0:
        phase = (x / period_days) % 1
        sorted_indices = np.argsort(phase)
        phase = phase[sorted_indices]
        y = y[sorted_indices]
        yerr = yerr[sorted_indices] if yerr is not None else None
        st.subheader("Phase-Folded Lightcurve")
        plt.figure(figsize=(8, 4))
        if yerr is not None:
            plt.errorbar(
                x=phase,
                y=y,
                yerr=yerr,
                markersize=5,
                fmt="o",
                color="k",
                label="Phase-Folded Lightcurve",
            )
            plt.errorbar(x=phase + 1, y=y, yerr=yerr, markersize=5, fmt="o", color="k")
        else:
            plt.plot(phase, y, "ok", markersize=5, label="Phase-Folded Lightcurve")
            plt.plot(phase + 1, y, "ok", markersize=5)
        plt.xlabel("Phase", fontsize=18)
        plt.ylabel("Magnitude/Flux", fontsize=18)
        plt.gca().invert_yaxis()
        plt.title(f"Phase-Folded Lightcurve (Period = {period_days} days)")
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        st.pyplot(plt)