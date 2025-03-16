import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm
from scipy.interpolate import griddata
import plotly.graph_objects as go

# --- Black-Scholes and IV functions ---
def bs_call_price(S, K, T, r, sigma, q=0):
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def implied_volatility(price, S, K, T, r, q=0):
    if T <= 0 or price <= 0:
        return np.nan
    def objective_function(sigma):
        return bs_call_price(S, K, T, r, sigma, q) - price
    try:
        return brentq(objective_function, 1e-6, 5)
    except (ValueError, RuntimeError):
        return np.nan

# --- Streamlit UI ---
st.title("3D Implied Volatility Surface for SPY Options")

# Sidebar parameters
r = st.sidebar.number_input('Risk-Free Rate', value=0.015, step=0.001, format="%.3f")
moneyness_toggle = st.sidebar.radio("Y-axis:", ("Strike Price ($)", "Moneyness"))

# Fetch SPY data
st.write("Fetching SPY options data...")
spy = yf.Ticker("SPY")
spot_price = spy.history(period="1d")["Close"].iloc[-1]
st.write(f"Current SPY Spot Price: **${spot_price:.2f}**")

# Fetch options chain & compute IVs
iv_data = []
expirations = spy.options
progress = st.progress(0)

for idx, expiry in enumerate(expirations[:10]):  # Limit to first 10 for speed
    opt_chain = spy.option_chain(expiry)
    calls = opt_chain.calls
    calls = calls[(calls["bid"] > 0) & (calls["ask"] > 0)].copy()
    
    # Mid price & time to expiry
    calls["mid"] = (calls["bid"] + calls["ask"]) / 2
    T = (pd.to_datetime(expiry) - pd.Timestamp.today()).days / 365
    calls["timeToExpiration"] = T
    calls["moneyness"] = calls["strike"] / spot_price
    
    # Compute IV
    calls["impliedVolatility"] = calls.apply(
        lambda row: implied_volatility(row["mid"], spot_price, row["strike"], T, r), axis=1)
    
    calls.dropna(subset=["impliedVolatility"], inplace=True)
    iv_data.append(calls[["strike", "timeToExpiration", "impliedVolatility", "moneyness"]])
    
    progress.progress((idx + 1) / 10)

iv_df = pd.concat(iv_data).reset_index(drop=True)
progress.empty()

# Sidebar filters
if moneyness_toggle == "Strike Price ($)":
    min_strike = st.sidebar.slider("Min Strike", float(iv_df["strike"].min()), float(iv_df["strike"].max()), float(iv_df["strike"].min()))
    max_strike = st.sidebar.slider("Max Strike", float(iv_df["strike"].min()), float(iv_df["strike"].max()), float(iv_df["strike"].max()))
    filtered_df = iv_df[(iv_df["strike"] >= min_strike) & (iv_df["strike"] <= max_strike)]
    Y_vals = filtered_df["strike"]
    y_label = "Strike Price ($)"
else:
    min_money = st.sidebar.slider("Min Moneyness", float(iv_df["moneyness"].min()), float(iv_df["moneyness"].max()), float(iv_df["moneyness"].min()))
    max_money = st.sidebar.slider("Max Moneyness", float(iv_df["moneyness"].min()), float(iv_df["moneyness"].max()), float(iv_df["moneyness"].max()))
    filtered_df = iv_df[(iv_df["moneyness"] >= min_money) & (iv_df["moneyness"] <= max_money)]
    Y_vals = filtered_df["moneyness"]
    y_label = "Moneyness"

# --- Visualization Toggle ---
plot_type = st.radio("Plot Type:", ("3D Scatter", "3D Surface (Interpolated)"))

X = filtered_df["timeToExpiration"].values
Y = Y_vals.values
Z = filtered_df["impliedVolatility"].values * 100  # %

if plot_type == "3D Scatter":
    fig = go.Figure(data=[go.Scatter3d(
        x=X, y=Y, z=Z,
        mode="markers",
        marker=dict(
            size=4,
            color=Z,
            colorscale="Viridis",
            colorbar_title="IV (%)"
        )
    )])
else:
    # Interpolation
    grid_x = np.linspace(X.min(), X.max(), 50)
    grid_y = np.linspace(Y.min(), Y.max(), 50)
    T_grid, K_grid = np.meshgrid(grid_x, grid_y)
    
    Z_grid = griddata((X, Y), Z, (T_grid, K_grid), method='linear')
    Z_grid = np.nan_to_num(Z_grid, nan=np.nanmean(Z))  # Fill missing values
    
    fig = go.Figure(data=[go.Surface(
        x=T_grid,
        y=K_grid,
        z=Z_grid,
        colorscale='Viridis',
        colorbar_title='IV (%)'
    )])

# --- Layout ---
fig.update_layout(
    title="3D Implied Volatility Surface",
    scene=dict(
        xaxis_title="Time to Expiration (Years)",
        yaxis_title=y_label,
        zaxis_title="Implied Volatility (%)"
    ),
    width=900,
    height=800
)

st.plotly_chart(fig)

# --- Display basic stats ---
st.write(f"Number of Options Used: **{len(filtered_df)}**")
st.write(f"Average IV: **{filtered_df['impliedVolatility'].mean() * 100:.2f}%**")
