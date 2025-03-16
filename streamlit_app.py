import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm
from scipy.interpolate import griddata
import plotly.graph_objects as go
from datetime import timedelta

# --- Black-Scholes and IV functions ---
def bs_call_price(S, K, T, r, sigma, q=0):
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def implied_volatility(price, S, K, T, r, q=0):
    if T <= 0 or price <= 0:
        return np.nan
    def objective(sigma):
        return bs_call_price(S, K, T, r, sigma, q) - price
    try:
        return brentq(objective, 1e-6, 5)
    except (ValueError, RuntimeError):
        return np.nan

# --- Streamlit UI ---
st.set_page_config(page_title="IV Surface", layout="wide")
st.title("3D Implied Volatility Surface for SPY Options")

# --- Sidebar parameters ---
st.sidebar.header("Model Parameters")
r = st.sidebar.number_input('Risk-Free Rate (%)', value=1.5, step=0.1, format="%.2f") / 100
dividend_yield = 0  # Assuming no dividend for SPY

st.sidebar.header("Strike Price Filter (% of Spot Price)")
min_strike_pct = st.sidebar.number_input('Minimum (%)', min_value=50.0, max_value=199.0, value=80.0, step=1.0)
max_strike_pct = st.sidebar.number_input('Maximum (%)', min_value=51.0, max_value=200.0, value=120.0, step=1.0)

if min_strike_pct >= max_strike_pct:
    st.sidebar.error("Minimum % must be less than Maximum %")
    st.stop()

# --- Fetch SPY data ---
st.sidebar.write("Fetching SPY options data...")
spy = yf.Ticker("SPY")
spot_price = spy.history(period="1d")["Close"].iloc[-1]
st.sidebar.info(f"SPY Spot Price: **${spot_price:.2f}**")

# --- Fetch options chain & compute IVs ---
iv_data = []
expirations = spy.options
progress = st.sidebar.progress(0)
valid_expiries = []

for idx, expiry in enumerate(expirations):
    expiry_date = pd.to_datetime(expiry)
    if (expiry_date - pd.Timestamp.today()).days <= 7:
        continue  # Skip near-term expiry
    
    valid_expiries.append(expiry)
    opt_chain = spy.option_chain(expiry)
    calls = opt_chain.calls
    calls = calls[(calls["bid"] > 0) & (calls["ask"] > 0)].copy()
    
    # Mid price & time to expiry
    T = (expiry_date - pd.Timestamp.today()).days / 365
    calls["mid"] = (calls["bid"] + calls["ask"]) / 2
    calls["timeToExpiration"] = T
    calls["moneyness"] = calls["strike"] / spot_price
    
    # Filter strikes based on percentage
    min_strike = spot_price * (min_strike_pct / 100)
    max_strike = spot_price * (max_strike_pct / 100)
    calls = calls[(calls["strike"] >= min_strike) & (calls["strike"] <= max_strike)]
    
    # Compute IV
    calls["impliedVolatility"] = calls.apply(
        lambda row: implied_volatility(row["mid"], spot_price, row["strike"], T, r, dividend_yield), axis=1)
    
    calls.dropna(subset=["impliedVolatility"], inplace=True)
    iv_data.append(calls[["strike", "timeToExpiration", "impliedVolatility", "moneyness"]])
    
    progress.progress((idx + 1) / len(expirations))

progress.empty()

if not iv_data:
    st.error("No valid option data found. Try adjusting strike percentage range.")
    st.stop()

iv_df = pd.concat(iv_data).reset_index(drop=True)

# --- Visualization ---
X = iv_df["timeToExpiration"].values
Y = iv_df["moneyness"].values
Z = iv_df["impliedVolatility"].values * 100  # %

# Interpolation
grid_x = np.linspace(X.min(), X.max(), 50)
grid_y = np.linspace(Y.min(), Y.max(), 50)
T_grid, K_grid = np.meshgrid(grid_x, grid_y)

Z_grid = griddata((X, Y), Z, (T_grid, K_grid), method='linear')
Z_grid = np.nan_to_num(Z_grid, nan=np.nanmean(Z))

fig = go.Figure(data=[go.Surface(
    x=T_grid,
    y=K_grid,
    z=Z_grid,
    colorscale='Viridis',
    colorbar_title='IV (%)'
)])

fig.update_layout(
    title="3D Implied Volatility Surface (Moneyness)",
    scene=dict(
        xaxis_title="Time to Expiration (Years)",
        yaxis_title="Moneyness (Strike / Spot)",
        zaxis_title="Implied Volatility (%)"
    ),
    width=900,
    height=750
)

st.plotly_chart(fig)

# --- Sidebar Stats ---
st.sidebar.write(f"Number of Options Used: **{len(iv_df)}**")
st.sidebar.write(f"Average IV: **{iv_df['impliedVolatility'].mean() * 100:.2f}%**")
