import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq

# --- Black-Scholes Call Price ---
def bs_call_price(S, K, T, r, sigma, q=0):
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

# --- Implied Volatility Solver ---
def implied_volatility(price, S, K, T, r, q=0):
    if T <= 0 or price <= 0:
        return np.nan
    def objective(sigma):
        return bs_call_price(S, K, T, r, sigma, q) - price
    try:
        return brentq(objective, 1e-6, 5)
    except (ValueError, RuntimeError):
        return np.nan

# --- Compute IV Surface for Options DataFrame ---
def compute_iv_surface(options_df, spot_price, r, dividend_yield):
    options_df['impliedVolatility'] = options_df.apply(
        lambda row: implied_volatility(
            price=row['mid'],
            S=spot_price,
            K=row['strike'],
            T=row['timeToExpiration'],
            r=r,
            q=dividend_yield
        ), axis=1
    )
    options_df.dropna(subset=['impliedVolatility'], inplace=True)
    options_df['impliedVolatility'] *= 100  # percentage
    options_df['moneyness'] = options_df['strike'] / spot_price
    return options_df
