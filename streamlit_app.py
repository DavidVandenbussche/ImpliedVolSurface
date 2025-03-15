import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.stats import norm

# Black-Scholes Call Price Function
def bs_call_price(S, K, T, r, sigma, q=0):
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

# Brent's Method to Compute IV
def implied_volatility(price, S, K, T, r, q=0):
    if T <= 0 or price <= 0:
        return np.nan
    def objective_function(sigma):
        return bs_call_price(S, K, T, r, sigma, q) - price
    try:
        return brentq(objective_function, 1e-6, 5)
    except (ValueError, RuntimeError):
        return np.nan

# Streamlit UI
st.title("Implied Volatility Smile for SPY Options")

# Fetch SPY options data
spy = yf.Ticker("SPY")
spot_price = spy.history(period="1d")["Close"].iloc[-1]

# Select expiration date
expirations = spy.options
chosen_expiry = st.selectbox("Select Expiration Date", expirations)

# Fetch options chain
option_chain = spy.option_chain(chosen_expiry)
calls = option_chain.calls

# Filter out illiquid options
calls = calls[(calls["bid"] > 0) & (calls["ask"] > 0)].copy()
calls["mid"] = (calls["bid"] + calls["ask"]) / 2

# Compute time to expiration in years
T = (pd.to_datetime(chosen_expiry) - pd.Timestamp.today()).days / 365
calls["time_to_expiration"] = T

# Compute IV
calls["implied_vol"] = calls.apply(lambda row: implied_volatility(row["mid"], spot_price, row["strike"], T, r=0.015), axis=1)
calls.dropna(subset=["implied_vol"], inplace=True)
calls.sort_values("strike", inplace=True)

# Plot IV Smile
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(calls["strike"], calls["implied_vol"] * 100, marker="o", linestyle="-", label=f"IV Curve ({chosen_expiry})")
ax.set_xlabel("Strike Price")
ax.set_ylabel("Implied Volatility (%)")
ax.set_title(f"Implied Volatility Smile for {chosen_expiry}")
ax.legend()
ax.grid()

# Show the plot
st.pyplot(fig)
