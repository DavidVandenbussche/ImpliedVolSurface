import yfinance as yf
import pandas as pd
import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm
from datetime import timedelta


# --- Black-Scholes Formula ---
def bs_call_price(S, K, T, r, sigma, q=0):
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


# --- Implied Volatility Calculation ---
def implied_volatility(price, S, K, T, r, q=0):
    if T <= 0 or price <= 0:
        return np.nan

    def objective(sigma):
        return bs_call_price(S, K, T, r, sigma, q) - price

    try:
        return brentq(objective, 1e-6, 5)
    except (ValueError, RuntimeError):
        return np.nan


# --- Core Function: Calculate IV Surface ---
def calculate_iv_surface(ticker_symbol, r, dividend_yield, min_strike_pct, max_strike_pct):
    ticker = yf.Ticker(ticker_symbol)
    today = pd.Timestamp.today().normalize()

    # --- Spot Price ---
    spot_history = ticker.history(period="5d")
    if spot_history.empty:
        raise ValueError(f"Failed to fetch spot price for {ticker_symbol}.")
    spot_price = spot_history["Close"].iloc[-1]

    # --- Expirations ---
    try:
        expirations = ticker.options
    except Exception:
        raise ValueError(f"No options data for {ticker_symbol}.")

    exp_dates = [pd.Timestamp(exp) for exp in expirations if pd.Timestamp(exp) > today + timedelta(days=7)]
    if not exp_dates:
        raise ValueError("No valid expiration dates found (more than 7 days ahead).")

    # --- Option Data Collection ---
    option_data = []
    for expiry_date in exp_dates:
        try:
            opt_chain = ticker.option_chain(expiry_date.strftime('%Y-%m-%d'))
            calls = opt_chain.calls
        except Exception:
            continue  # Skip problematic chains

        calls = calls[(calls["bid"] > 0) & (calls["ask"] > 0)]

        for _, row in calls.iterrows():
            option_data.append({
                'ticker': ticker_symbol,
                'calculation_date': today,
                'expiration_date': expiry_date,
                'strike': row['strike'],
                'bid': row['bid'],
                'ask': row['ask'],
                'mid': (row['bid'] + row['ask']) / 2,
                'risk_free_rate': r,
                'dividend_yield': dividend_yield,
                'min_strike_pct': min_strike_pct,
                'max_strike_pct': max_strike_pct
            })

    if not option_data:
        raise ValueError("No valid options found after filtering bids/asks.")

    options_df = pd.DataFrame(option_data)

    # --- Time to Expiration & Filters ---
    options_df['daysToExpiration'] = (options_df['expiration_date'] - today).dt.days
    options_df['timeToExpiration'] = options_df['daysToExpiration'] / 365

    # Filter strikes based on spot price
    options_df = options_df[
        (options_df['strike'] >= spot_price * (min_strike_pct / 100)) &
        (options_df['strike'] <= spot_price * (max_strike_pct / 100))
    ]

    if options_df.empty:
        raise ValueError("No options left after applying strike price filter.")

    # --- Calculate Implied Volatility ---
    options_df['implied_volatility'] = options_df.apply(
        lambda row: implied_volatility(
            price=row['mid'],
            S=spot_price,
            K=row['strike'],
            T=row['timeToExpiration'],
            r=r,
            q=dividend_yield
        ), axis=1
    )

    options_df.dropna(subset=['implied_volatility'], inplace=True)
    options_df['implied_volatility'] *= 100  # Convert to %

    # Add moneyness
    options_df['moneyness'] = options_df['strike'] / spot_price

    if options_df.empty:
        raise ValueError("No valid IVs computed.")

    return options_df
