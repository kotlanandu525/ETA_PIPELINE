# load.py
import os
import math
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
from supabase import create_client,Client
from time import sleep
 
load_dotenv()
 
BASE_DIR = Path(__file__).resolve().parents[0]
STAGED_DIR = BASE_DIR / "data" / "staged"
 
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
 
if not SUPABASE_URL or not SUPABASE_KEY:
    raise SystemExit("Please set SUPABASE_URL and SUPABASE_KEY in your .env")
 
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
 
TABLE_NAME = "weather_data"
 
CREATE_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS public.{TABLE_NAME} (
    id BIGSERIAL PRIMARY KEY,
    time TIMESTAMP,
    date DATE,
    hour INTEGER,
    temperature_c DOUBLE PRECISION,
    relative_humidity DOUBLE PRECISION,
    wind_speed_kmh DOUBLE PRECISION,
    temp_category TEXT,
    feels_like_c DOUBLE PRECISION
);
"""
 
def create_table_if_not_exists():
    """
    Try to create table via SQL RPC if available. If RPC not available,
    print SQL so user can run it in Supabase SQL UI.
    """
    try:
        # Many Supabase clients do not expose execute_sql via rpc - this may fail.
        # Attempt, but handle gracefully.
        print("🔧 Attempting to create table in Supabase (if permitted)...")
        supabase.rpc("execute_sql", {"query": CREATE_TABLE_SQL}).execute()
        print("✅ create_table_if_not_exists: RPC executed (or table exists).")
    except Exception as e:
        print(f"⚠️  Could not create table via RPC: {e}")
        print("ℹ️  Please run the following SQL in your Supabase SQL editor if table doesn't exist:")
        print(CREATE_TABLE_SQL)
 
def _read_staged_csv(staged_path: str) -> pd.DataFrame:
    df = pd.read_csv(staged_path)
 
    # Convert timestamps to string for Supabase insert
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], errors="coerce").astype(str)
 
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").astype(str)
 
    return df
 
 
def load_to_supabase(staged_csv_path: str, batch_size: int = 100):
    if not Path(staged_csv_path).exists():
        raise FileNotFoundError(f"Staged CSV not found at {staged_csv_path}")
 
    df = _read_staged_csv(staged_csv_path)
    total = len(df)
    print(f"📦 Loading {total} rows into Supabase table '{TABLE_NAME}' in batches of {batch_size} ...")
 
    # convert NaN to None for JSON serialization
    df = df.where(pd.notnull(df), None)
    records = df.to_dict(orient="records")
 
    for i in range(0, total, batch_size):
        batch = records[i:i + batch_size]
        try:
            res = supabase.table(TABLE_NAME).insert(batch).execute()
            # supabase-py: res has .error attribute or .status_code depending on version
            # We print a short success message. If an error, print it.
            if hasattr(res, "error") and res.error:
                print(f"⚠️  Batch {i//batch_size + 1} error: {res.error}")
            else:
                end = min(i + batch_size, total)
                print(f"✅ Inserted rows {i+1}-{end} of {total}")
        except Exception as e:
            print(f"⚠️  Exception while inserting batch {i//batch_size + 1}: {e}")
            # optional: exponential backoff retry
            print("Retrying after 3s ...")
            sleep(3)
            try:
                supabase.table(TABLE_NAME).insert(batch).execute()
                print("✅ Retry success")
            except Exception as e2:
                print(f"❌ Retry failed: {e2}")
                # continue to next batch
                continue
 
    print("🎯 Load complete.")
 
if __name__ == "__main__":
    staged_files = sorted([str(p) for p in STAGED_DIR.glob("weather_staged_*.csv")])
    if not staged_files:
        raise SystemExit("No staged CSV found. Run transform.py first.")
    create_table_if_not_exists()
    load_to_supabase(staged_files[-1], batch_size=100)
 
 