import sqlite3
import pandas as pd
import streamlit as st

# --- Function to save DataFrame to SQLite ---
def save_iv_surface(df, db_path="iv_surfaces.db", table_name="iv_data"):
    conn = sqlite3.connect(db_path)
    df.to_sql(table_name, conn, if_exists='append', index=False)
    conn.close()
    st.success("IV Surface saved to database.")

# --- Function to read saved DataFrame ---
def load_iv_surface(db_path="iv_surfaces.db", table_name="iv_data"):
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
        st.write("Loaded Saved IV Surfaces:")
        st.dataframe(df.head(20))  # Display first few rows
    except Exception as e:
        st.error(f"Error loading data: {e}")
    finally:
        conn.close()
