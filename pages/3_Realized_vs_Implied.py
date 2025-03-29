import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from iv_surface_calculator import compute_iv_surface

st.set_page_config(page_title="RV vs IV", layout="wide")
st.title("Realized vs Implied Volatility")

st.sidebar.header("Ticker & Parameters")
ticker_symbol = st.sidebar.text_input("Ticker", value="SPY").upper()
rolling_days = st.sidebar.selectbox("Rolling Window for Realized Volatility (days)", [21, 30, 60, 90])

if not ticker_symbol:
    st.warning("Enter a valid ticker.")
    st.stop()

ticker = yf.Ticker(ticker_symbol)

try:
    hist = ticker.history(period="6mo")
    if hist.empty:
        st.error("No historical price data found.")
        st.stop()
    close_prices = hist["Close"]
except Exception as e:
    st.error(f"Error fetching data: {e}")
    st.stop()

log_returns = np.log(close_prices / close_prices.shift(1))
realized_vol = log_returns.rolling(rolling_days).std() * np.sqrt(252) * 100

# Latest realized vol
latest_rv = realized_vol.dropna().iloc[-1]

# --- Implied Vol ---
try:
    spot_price = close_prices.iloc[-1]
    expirations = ticker.options
    calls_df = []

    for exp in expirations:
        expiry_date = pd.Timestamp(exp)
        if expiry_date <= pd.Timestamp.today() + pd.Timedelta(days=7):
            continue

        try:
            chain = ticker.option_chain(exp)
            calls = chain.calls
        except:
            continue

        calls = calls[(calls["bid"] > 0) & (calls["ask"] > 0)]
        calls["mid"] = (calls["bid"] + calls["ask"]) / 2
        calls["expirationDate"] = expiry_date
        calls_df.append(calls)

    if not calls_df:
        st.warning("No valid options found.")
        st.stop()

    options_df = pd.concat(calls_df)
    options_df['daysToExpiration'] = (options_df['expirationDate'] - pd.Timestamp.today().normalize()).dt.days
    options_df['timeToExpiration'] = options_df['daysToExpiration'] / 365

    # Only keep options with maturity ~rolling_days
    target_days = rolling_days
    options_df = options_df[np.abs(options_df['daysToExpiration'] - target_days) <= 5]

    # Add strike filter around ATM
    options_df = options_df[(options_df['strike'] >= spot_price * 0.95) & (options_df['strike'] <= spot_price * 1.05)]

    if options_df.empty:
        st.warning("No ATM options with matching expiration found.")
        st.stop()

    r = 0.015
    dividend_yield = 0.0
    iv_df = compute_iv_surface(options_df, spot_price, r, dividend_yield)
    latest_iv = iv_df["impliedVolatility"].mean()

except Exception as e:
    st.error(f"Error while computing IV: {e}")
    st.stop()

# --- Display Results ---
st.subheader(f"{ticker_symbol}: Realized vs Implied Volatility ({rolling_days} days)")

st.metric(label="Realized Volatility (%)", value=f"{latest_rv:.2f}")
st.metric(label="Implied Volatility (%)", value=f"{latest_iv:.2f}")
vrp = latest_iv - latest_rv
st.metric(label="Volatility Risk Premium (IV - RV)", value=f"{vrp:.2f}", delta=f"{vrp:.2f}")

# --- Plot ---
fig = go.Figure()
fig.add_trace(go.Scatter(x=realized_vol.index, y=realized_vol, mode='lines', name='Realized Vol (%)'))
fig.update_layout(title=f"{ticker_symbol} Realized Volatility ({rolling_days}-day window)", yaxis_title="Volatility (%)")
st.plotly_chart(fig)
