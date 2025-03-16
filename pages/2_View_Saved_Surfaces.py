import streamlit as st
import pandas as pd
from db_utils import load_iv_surface

st.title("View Saved Implied Volatility Surfaces")

st.write("Retrieve and visualize previously saved IV surfaces:")

# Load and display from SQLite
load_iv_surface()

st.markdown(
    """
    **Created by David Vandenbussche | [LinkedIn](https://www.linkedin.com/in/vandenbusschedavid/)**  
    🎓 National Unversity of Singapore 
    """
)
