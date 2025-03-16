import sqlite3
import pandas as pd
import streamlit as st

DB_PATH = "iv_surfaces.db"
TABLE_NAME = "iv_data"

# --- Save IV Surface Snapshot ---
def save_iv_surface_snapshot(df, ticker, timestamp, db_path=DB_PATH, table_name=TABLE_NAME):
    df['ticker'] = ticker
    df['timestamp'] = timestamp
    conn = sqlite3.connect(db_path)
    df.to_sql(table_name, conn, if_exists='append', index=False)
    conn.close()
    st.success(f"Snapshot for {ticker} saved at {timestamp}.")

# --- Load IV Surface by Ticker and Timestamp ---
def load_iv_surface_snapshot(ticker, timestamp, db_path=DB_PATH, table_name=TABLE_NAME):
    conn = sqlite3.connect(db_path)
    query = f"""
        SELECT * FROM {table_name}
        WHERE ticker = ? AND timestamp = ?
    """
    df = pd.read_sql_query(query, conn, params=(ticker, timestamp))
    conn.close()
    return df

# --- Get Available Tickers ---
def get_distinct_tickers(db_path=DB_PATH, table_name=TABLE_NAME):
    conn = sqlite3.connect(db_path)
    query = f"SELECT DISTINCT ticker FROM {table_name}"
    tickers = pd.read_sql_query(query, conn)['ticker'].tolist()
    conn.close()
    return tickers

# --- Get Available Timestamps for Ticker ---
def get_timestamps_for_ticker(ticker, db_path=DB_PATH, table_name=TABLE_NAME):
    conn = sqlite3.connect(db_path)
    query = f"SELECT DISTINCT timestamp FROM {table_name} WHERE ticker = ?"
    timestamps = pd.read_sql_query(query, conn, params=(ticker,))['timestamp'].tolist()
    conn.close()
    return timestamps
