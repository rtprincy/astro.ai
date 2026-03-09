# Astro.ai - Lightcurve Analysis Tool

A Streamlit app for visualizing and analyzing astronomical lightcurves, including phase folding and frequency search using the Lomb-Scargle periodogram.

## Features

- Upload CSV files containing lightcurve data
- Visualize original lightcurves
- **Data filtering**: Optional filter column to select specific data points (e.g., quality flags)
- Perform frequency search using Lomb-Scargle periodogram to find periodic signals
- **Automatic phase folding**: Uses the best period found in frequency search for phase folding
- Manual period override option for phase folding

### How to run it on your own machine

1. Install the requirements

   ```
   $ pip install -r requirements.txt
   ```

2. Run the app

   ```
   $ streamlit run streamlit_app.py
   ```
