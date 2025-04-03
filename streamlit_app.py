import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

st.title("Phase Fold and visualise lightcurves")

# File uploader
uploaded_file = st.file_uploader("Upload a CSV file containing the lightcurve", type="csv")
if uploaded_file:
    # Read uploaded CSV
    df = pd.read_csv(uploaded_file)
    st.write("Data Preview:", df.head())

    # Column selection
    x_column = st.selectbox("Select Time Column (X)", df.columns)
    y_column = st.selectbox("Select Magnitude/Flux Column (Y)", df.columns)
    yerr_column=st.selectbox("Select Magnitude/Flux error Column (Yerr)",df.columns)

    # Input for period in days
    period_days = st.number_input("Enter the period in days for phase folding:", min_value=0.0)

    if st.button("Plot"):
        # Extract columns
        x = df[x_column].values
        y = df[y_column].values
        yerr=df[yerr_column].values
        
        # Plot original lightcurve
        st.subheader("Original Lightcurve")
        plt.figure(figsize=(8, 4))
        plt.errorbar(x=x, y=y,yerr=yerr, fmt='o', markersize=5, label="Original Lightcurve")
        plt.xlabel("Time",fontsize=18)
        plt.ylabel("Magnitude/Flux",fontsize=18)
        plt.gca().invert_yaxis()  # Typical for magnitude plots
        plt.title("Original Lightcurve")
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        
        # plt.legend()
        st.pyplot(plt)

        # Check if a valid period is entered
        if period_days > 0:
            # Phase folding
            phase = (x / period_days) % 1  # Compute phase
            sorted_indices = np.argsort(phase)  # Sort by phase
            phase = phase[sorted_indices]
            y = y[sorted_indices]

            # Plot phase-folded lightcurve
            st.subheader("Phase-Folded Lightcurve")
            plt.figure(figsize=(8, 4))
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
