import streamlit as st
import pandas as pd
import numpy as np
from scipy.interpolate import griddata
import plotly.graph_objects as go

from db_utils import load_iv_surface_snapshot, get_distinct_tickers, get_timestamps_for_ticker

st.title("Historical IV Surface Viewer")

tickers = get_distinct_tickers()
if not tickers:
    st.info("No snapshots available yet.")
    st.stop()

selected_ticker = st.selectbox("Select Ticker:", tickers)
timestamps = get_timestamps_for_ticker(selected_ticker)
selected_timestamp = st.selectbox("Select Snapshot Time:", timestamps)

if st.button("Load IV Surface"):
    df = load_iv_surface_snapshot(selected_ticker, selected_timestamp)
    if df.empty:
        st.warning("No data found.")
    else:
        X = df["timeToExpiration"].values
        Y = df["moneyness"].values
        Z = df["impliedVolatility"].values

        grid_x = np.linspace(X.min(), X.max(), 50)
        grid_y = np.linspace(Y.min(), Y.max(), 50)
        T_grid, Y_grid = np.meshgrid(grid_x, grid_y)
        Z_grid = griddata((X, Y), Z, (T_grid, Y_grid), method='linear')
        Z_grid = np.nan_to_num(Z_grid, nan=np.nanmean(Z))

        fig = go.Figure(data=[go.Surface(
            x=T_grid,
            y=Y_grid,
            z=Z_grid,
            colorscale='Viridis',
            colorbar_title='IV (%)'
        )])

        fig.update_layout(
            title=f"{selected_ticker} IV Surface ({selected_timestamp})",
            scene=dict(
                xaxis_title="Time to Expiration (Years)",
                yaxis_title="Moneyness",
                zaxis_title="Implied Volatility (%)"
            ),
            width=900,
            height=750
        )

        st.plotly_chart(fig)
