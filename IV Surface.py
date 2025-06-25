import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.interpolate import griddata
import plotly.graph_objects as go
from datetime import timedelta

from iv_surface_calculator import compute_iv_surface  # Import clean IV logic

st.set_page_config(page_title="IV Surface")
st.title("3D Implied Volatility Surface for Options")

# --- Sidebar parameters ---

st.sidebar.header("Model Parameters")
r = st.sidebar.number_input('Risk-Free Rate (%)', value=1.5, step=0.1, format="%.2f") / 100
st.sidebar.header("Dividend Yield")
dividend_yield = st.sidebar.number_input('Dividend Yield (%)', value=0.0, step=0.1, format="%.2f") / 100

st.sidebar.header("Ticker Selection")
default_tickers = ['SPY', 'AAPL', 'TSLA', 'MSFT']
selected_default = st.sidebar.selectbox("Choose a Ticker", default_tickers)
custom_ticker = st.sidebar.text_input("Or enter custom ticker:", value="").upper()

# Priority to custom input
if custom_ticker:
    ticker_symbol = custom_ticker
else:
    ticker_symbol = selected_default

st.sidebar.header("Y-axis Parameter")
y_axis_option = st.sidebar.radio("Select Y-axis:", ("Moneyness", "Strike Price ($)"))

st.sidebar.header("Strike Price Filter (% of Spot Price)")
min_strike_pct = st.sidebar.number_input('Minimum (%)', min_value=50.0, max_value=199.0, value=70.0, step=1.0)
max_strike_pct = st.sidebar.number_input('Maximum (%)', min_value=51.0, max_value=200.0, value=130.0, step=1.0)

if min_strike_pct >= max_strike_pct:
    st.sidebar.error("Minimum % must be less than Maximum %")
    st.stop()

# --- Fetch ticker data ---
ticker = yf.Ticker(ticker_symbol)

spot_history = ticker.history(period="5d")
if spot_history.empty:
    st.error(f"Failed to fetch spot price for {ticker_symbol}.")
    st.stop()
spot_price = spot_history["Close"].iloc[-1]

# --- Fetch options chain & build data ---
option_data = []
today = pd.Timestamp.today().normalize()

try:
    expirations = ticker.options
except Exception as e:
    st.error(f"Error fetching {ticker_symbol} options chain: {e}")
    st.stop()

# Filter expirations > 7 days
exp_dates = [pd.Timestamp(exp) for exp in expirations if pd.Timestamp(exp) > today + timedelta(days=7)]

if not exp_dates:
    st.error("No valid expiration dates found (more than 7 days ahead).")
    st.stop()

# --- Collect all calls first ---
for expiry_date in exp_dates:
    try:
        opt_chain = ticker.option_chain(expiry_date.strftime('%Y-%m-%d'))
        calls = opt_chain.calls
    except Exception as e:
        continue  # Skip problematic chains

    calls = calls[(calls["bid"] > 0) & (calls["ask"] > 0)]
    for _, row in calls.iterrows():
        option_data.append({
            'expirationDate': expiry_date,
            'strike': row['strike'],
            'bid': row['bid'],
            'ask': row['ask'],
            'mid': (row['bid'] + row['ask']) / 2
        })

if not option_data:
    st.error("No valid option data found after filtering bids/asks.")
    st.stop()

# --- Create DataFrame ---
options_df = pd.DataFrame(option_data)

# --- Time to Expiration & Filters ---
options_df['daysToExpiration'] = (options_df['expirationDate'] - today).dt.days
options_df['timeToExpiration'] = options_df['daysToExpiration'] / 365

# Filter strikes based on spot price
options_df = options_df[
    (options_df['strike'] >= spot_price * (min_strike_pct / 100)) &
    (options_df['strike'] <= spot_price * (max_strike_pct / 100))
]

if options_df.empty:
    st.error("No options left after applying strike price filter. Adjust range.")
    st.stop()

# --- Compute Implied Volatility ---
with st.spinner("Calculating Implied Volatility..."):
    options_df = compute_iv_surface(options_df, spot_price, r, dividend_yield)

if options_df.empty:
    st.error("No valid IVs computed. Try adjusting filter parameters.")
    st.stop()

# --- Prepare data for plotting ---
X = options_df["timeToExpiration"].values
Y_vals = options_df["moneyness"].values if y_axis_option == "Moneyness" else options_df["strike"].values
Z = options_df["impliedVolatility"].values

y_label = "Moneyness (Strike / Spot)" if y_axis_option == "Moneyness" else "Strike Price ($)"

# --- Grid interpolation ---
grid_x = np.linspace(X.min(), X.max(), 50)
grid_y = np.linspace(Y_vals.min(), Y_vals.max(), 50)
T_grid, Y_grid = np.meshgrid(grid_x, grid_y)

Z_grid = griddata((X, Y_vals), Z, (T_grid, Y_grid), method='linear')
Z_grid = np.nan_to_num(Z_grid, nan=np.nanmean(Z))

# --- Plotly Surface Plot ---
fig = go.Figure(data=[go.Surface(
    x=T_grid,
    y=Y_grid,
    z=Z_grid,
    colorscale='Viridis',
    colorbar_title='IV (%)'
)])

fig.update_layout(
    title=f"3D Implied Volatility Surface ({y_label})",
    scene=dict(
        xaxis_title="Time to Expiration (Years)",
        yaxis_title=y_label,
        zaxis_title="Implied Volatility (%)"
    ),
    width=900,
    height=750,
    margin=dict(l=65, r=50, b=65, t=90)
)

st.plotly_chart(fig)

# --- Sidebar Stats ---
st.sidebar.write(f"Number of Options Used: **{len(options_df)}**")
st.sidebar.write(f"Average IV: **{options_df['impliedVolatility'].mean():.2f}%**")

# st.write("---")
# st.markdown(
#     """
#     **Created by David Vandenbussche | [LinkedIn](https://www.linkedin.com/in/vandenbusschedavid/)**  
#     🎓 National Unversity of Singapore 
#     """
# )
