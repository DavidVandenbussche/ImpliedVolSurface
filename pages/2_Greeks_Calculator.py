import streamlit as st
import pandas as pd
from utils.black_scholes_model import BlackScholes

st.set_page_config(page_title="Greeks Calculator", layout="wide")
st.title("📊 Option Greeks Calculator")

st.sidebar.header("Input Parameters")
spot_price = st.sidebar.number_input("Stock Price", min_value=1.0, max_value=50000.0, value=4000.0, step=1.0)
strike_price = st.sidebar.number_input("Strike Price", min_value=1.0, max_value=50000.0, value=4000.0, step=1.0)
time_to_expiry = st.sidebar.number_input("Time to Expiry (Years)", min_value=0.01, max_value=5.0, value=1.0, step=0.01)
volatility = st.sidebar.number_input("Volatility (%)", min_value=1.0, max_value=300.0, value=20.0, step=0.1) / 100
risk_free_rate = st.sidebar.number_input("Risk-Free Rate (%)", min_value=0.0, max_value=20.0, value=2.0, step=0.1) / 100
option_type = st.sidebar.radio("Option Type", options=['Call', 'Put'])

# Error handling
if spot_price <= 0 or strike_price <= 0 or time_to_expiry <= 0 or volatility <= 0:
    st.error("Inputs must be positive numbers.")
    st.stop()

# Instantiate model
bs_model = BlackScholes(r=risk_free_rate, s=spot_price, k=strike_price, t=time_to_expiry, sigma=volatility)

# Display Option Price
st.write("### Option Price")
option_price = bs_model.option(option_type)
st.metric(label=f"{option_type} Option Price", value=f"${option_price:.2f}")

# Display Greeks
st.write("### Greeks")
greeks = bs_model.greeks(option_type)
greeks_df = pd.DataFrame.from_dict(greeks, orient='index', columns=['Value'])
st.table(greeks_df)

# Visualizations
st.write("### Greeks Visualizations")
cols = st.columns(3)
for idx, greek in enumerate(greeks.keys()):
    fig = bs_model.greek_visualisation(option_type, greek)
    with cols[idx % 2]:
        st.plotly_chart(fig, use_container_width=True)

st.write("---")
st.markdown(
    """
    **Created by David Vandenbussche | [LinkedIn](https://www.linkedin.com/in/vandenbusschedavid/)**  
    🎓 National University of Singapore 
    """
)
