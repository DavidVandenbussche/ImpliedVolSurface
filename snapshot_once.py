import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

from iv_surface_calculator import compute_iv_surface
from db_utils import save_iv_surface_snapshot

def fetch_and_save_snapshots():
    tickers_to_snapshot = ["SPY", "AAPL", "TSLA", "MSFT"]

    for ticker_symbol in tickers_to_snapshot:
        ticker = yf.Ticker(ticker_symbol)
        spot_history = ticker.history(period="5d")
        if spot_history.empty:
            continue  # Skip if no spot price

        spot_price = spot_history["Close"].iloc[-1]

        option_data = []
        today_date = pd.Timestamp.today().normalize()
        expirations = ticker.options
        exp_dates = [pd.Timestamp(exp) for exp in expirations if pd.Timestamp(exp) > today_date + timedelta(days=7)]

        for expiry_date in exp_dates:
            try:
                opt_chain = ticker.option_chain(expiry_date.strftime('%Y-%m-%d'))
                calls = opt_chain.calls
            except:
                continue
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
            continue  # Skip if no valid option data

        options_df = pd.DataFrame(option_data)
        options_df['daysToExpiration'] = (options_df['expirationDate'] - today_date).dt.days
        options_df['timeToExpiration'] = options_df['daysToExpiration'] / 365

        risk_free_rate = 0.015
        dividend_yield = 0.0

        options_df = compute_iv_surface(options_df, spot_price, risk_free_rate, dividend_yield)

        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        save_iv_surface_snapshot(options_df, ticker_symbol, timestamp)

if __name__ == "__main__":
    fetch_and_save_snapshots()
