import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from astropy.timeseries import LombScargle
from lk_stat_package import lk_stat

st.title("Time-series analysis")
def freq_grid(times,oversample_factor=10,f0=None,fn=None):
    times=np.sort(times)
    df = 1.0 / (times.max() - times.min())
    if f0 is None:
        f0 = df
    if fn is None:
        fn = 0.5 / np.median(np.diff(times)) 
    return np.arange(f0, fn, df / oversample_factor)

# File uploader
uploaded_file = st.file_uploader("Upload the time-series data (only csv file is supported)", type="csv")
if uploaded_file:
    # Read uploaded CSV
    df = pd.read_csv(uploaded_file)
    st.write("Data Preview:", df.head())

    # Column selection
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
        # Validate oversampling_factor
        if oversampling_factor > 20:
            st.error("Oversampling factor must be less than or equal to 20.")
            compute_disabled = True
        else:
            compute_disabled = False
    except ValueError:
        st.error("Please enter valid numeric values for all inputs.")
        compute_disabled = True
    
    # Frequency search section

    st.subheader("Frequency search using a hybrid periodogram (Lomb-Scargle + Lafler-Kinman)")
    if st.button("Compute Periodogram and Find Best Period", disabled=compute_disabled):
        # Extract columns
        x = df[x_column].values
        y = df[y_column].values
        yerr = df[yerr_column].values if yerr_column != "None" and yerr_column in df.columns else None
        
        # Apply filter if specified
        if filter_column != "None" and filter_column in df.columns:
            filter_mask = df[filter_column].astype(str) == filter_value
            x = x[filter_mask]
            y = y[filter_mask]
            if yerr is not None:
                yerr = yerr[filter_mask]
            st.write(f"Applied filter '{filter_column} = {filter_value}': using {len(x)} data points.")
        
        # Compute Lomb-Scargle periodogram
        ls = LombScargle(x, y, dy=yerr,normalization="psd")
        
        # Define frequency range
        frequency = freq_grid(x,oversample_factor=oversampling_factor, f0=f0, fn=fn)
        period = 1 / frequency
        power = ls.power(frequency)
        theta = lk_stat(period, y, yerr, x)

        psi = 2*power/theta
        psi_norm=psi/psi.max()
        # Find the best frequency
        best_idx = np.argmax(psi_norm)
        best_freq = frequency[best_idx]
        best_period = 1 / best_freq
        
        # Store best_period in session state
        st.session_state.best_period = best_period
        
        st.write(f"Best period found: {best_period:.4f} days")
        
        # Plot the periodogram
        st.subheader("Hybrid Psi Periodogram")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(frequency, psi_norm)
        ax.set_xlabel("Frequency")
        ax.set_ylabel("Normalized Psi")
        ax.axvline(best_freq, color='r', linestyle='--', label=f'Best Frequency: {best_freq:.4f} c/d')
        ax.legend()
        st.pyplot(fig)

    # Input for period in days (optional manual override)
    manual_period = st.number_input("Enter period in days for manual phase folding (optional). Otherwise, click Plot to phase fold with the best period found:", min_value=0.0, step=0.01, value=0.0)
    
    if st.button("Plot"):
        # Extract columns
        x = df[x_column].values
        y = df[y_column].values
        yerr = df[yerr_column].values if yerr_column != "None" else None
        
        # Apply filter if specified
        if filter_column != "None" and filter_column in df.columns:
            filter_mask = df[filter_column].astype(str) == filter_value
            x = x[filter_mask]
            y = y[filter_mask]
            if yerr is not None:
                yerr = yerr[filter_mask]
            st.write(f"Applied filter '{filter_column} == {filter_value}': using {len(x)} data points for plotting.")
        
        # Plot original lightcurve
        st.subheader("Original Lightcurve")
        plt.figure(figsize=(8, 4))
        if yerr is not None:
            plt.errorbar(x=x, y=y,yerr=yerr, fmt='o', markersize=5, label="Original Lightcurve")
        else:
            plt.plot(x, y, 'o', markersize=5, label="Original Lightcurve")
        plt.xlabel("Time",fontsize=18)
        plt.ylabel("Magnitude/Flux",fontsize=18)
        plt.gca().invert_yaxis()  # Typical for magnitude plots
        plt.title("Original Lightcurve")
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        
        # plt.legend()
        st.pyplot(plt)

        # Determine period to use for phase folding
        if manual_period > 0:
            period_days = manual_period
            st.write(f"Using manually entered period: {period_days:.4f} days")
        elif 'best_period' in st.session_state:
            period_days = st.session_state.best_period
            st.write(f"Using best period from frequency search: {period_days:.4f} days")
        else:
            st.warning("No period available. Please run frequency search first or enter a manual period.")
            period_days = None

        # Check if a valid period is available
        if period_days and period_days > 0:
            # Phase folding
            phase = (x / period_days) % 1  # Compute phase
            sorted_indices = np.argsort(phase)  # Sort by phase
            phase = phase[sorted_indices]
            y = y[sorted_indices]
            yerr = yerr[sorted_indices] if yerr is not None else None

            # Plot phase-folded lightcurve
            st.subheader("Phase-Folded Lightcurve")
            plt.figure(figsize=(8, 4))
            if yerr is not None:
                plt.errorbar(x=phase,y=y,yerr=yerr,markersize=5,fmt='o',color='k', label="Phase-Folded Lightcurve")
                plt.errorbar(x=phase+1,y=y,yerr=yerr,markersize=5,fmt='o',color='k')
            else:
                plt.plot(phase, y, 'ok', markersize=5, label="Phase-Folded Lightcurve")
                plt.plot(phase+1, y, 'ok', markersize=5)
            plt.xlabel("Phase",fontsize=18)
            plt.ylabel("Magnitude/Flux",fontsize=18)
            plt.gca().invert_yaxis()  # Typical for magnitude plots
            plt.title(f"Phase-Folded Lightcurve (Period = {period_days} days)")
            plt.xticks(fontsize=18)
            plt.yticks(fontsize=18)
            # plt.legend()
            st.pyplot(plt)
        else:
            st.warning("Please enter a valid period in days to phase-fold the lightcurve.")
