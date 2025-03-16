import sqlite3
import pandas as pd

DB_PATH = "iv_surfaces.db"
TABLE_NAME = "iv_data"

def save_iv_surface_snapshot(df, ticker, timestamp):
    df['ticker'] = ticker
    df['timestamp'] = timestamp
    conn = sqlite3.connect(DB_PATH)
    df.to_sql(TABLE_NAME, conn, if_exists='append', index=False)
    conn.close()

def load_iv_surface_snapshot(ticker, timestamp):
    conn = sqlite3.connect(DB_PATH)
    query = f"""
        SELECT * FROM {TABLE_NAME}
        WHERE ticker = ? AND timestamp = ?
    """
    df = pd.read_sql_query(query, conn, params=(ticker, timestamp))
    conn.close()
    return df

def get_distinct_tickers():
    conn = sqlite3.connect(DB_PATH)
    query = f"SELECT DISTINCT ticker FROM {TABLE_NAME}"
    tickers = pd.read_sql_query(query, conn)['ticker'].tolist()
    conn.close()
    return tickers

def get_timestamps_for_ticker(ticker):
    conn = sqlite3.connect(DB_PATH)
    query = f"SELECT DISTINCT timestamp FROM {TABLE_NAME} WHERE ticker = ?"
    timestamps = pd.read_sql_query(query, conn, params=(ticker,))['timestamp'].tolist()
    conn.close()
    return timestamps
