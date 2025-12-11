# load.py
"""
Load staged air quality CSV into Supabase table `air_quality_data`.

Features:
- Create table (tries RPC; prints SQL if not possible)
- Batch insert (batch_size=200)
- Convert NaN -> None
- Convert datetime to ISO strings
- Retry failed batches (2 retries)
- Print a summary of inserted rows and failures

Usage:
    from load import load_csv_to_supabase
    load_csv_to_supabase("data/staged/air_quality_transformed_20251211T...csv")

Or run as CLI:
    python load.py path/to/staged.csv
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

BATCH_SIZE = int(os.getenv("LOAD_BATCH_SIZE", "200"))
RETRY_COUNT = int(os.getenv("LOAD_RETRY_COUNT", "2"))  # number of retries after initial failure
RETRY_BACKOFF = float(os.getenv("LOAD_RETRY_BACKOFF", "2.0"))  # seconds multiplier between retries

TABLE_NAME = "air_quality_data"

CREATE_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS public.{TABLE_NAME} (
    id BIGSERIAL PRIMARY KEY,
    city TEXT,
    time TIMESTAMP,
    pm10 FLOAT,
    pm2_5 FLOAT,
    carbon_monoxide FLOAT,
    nitrogen_dioxide FLOAT,
    sulphur_dioxide FLOAT,
    ozone FLOAT,
    uv_index FLOAT,
    aqi_category TEXT,
    severity_score FLOAT,
    risk_flag TEXT,
    hour INTEGER
);
"""

# Try import supabase client
try:
    from supabase import create_client  # type: ignore
    _HAS_SUPABASE = True
except Exception:
    _HAS_SUPABASE = False


def _get_supabase_client():
    if not _HAS_SUPABASE:
        return None
    if not SUPABASE_URL or not SUPABASE_KEY:
        return None
    try:
        return create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception:
        return None


def create_table_if_not_exists():
    """
    Attempt to create table via RPC; if not possible, print SQL for manual execution.
    """
    client = _get_supabase_client()
    if client is None:
        print("⚠️ Supabase client not available. Please run this SQL in Supabase SQL editor to create the table:\n")
        print(CREATE_TABLE_SQL)
        return

    try:
        print("🔧 Attempting to create table in Supabase via RPC (if available)...")
        # Many supabase setups won't allow raw SQL via RPC. Attempting nonetheless.
        client.rpc("execute_sql", {"query": CREATE_TABLE_SQL}).execute()
        print("✅ create_table_if_not_exists: RPC executed or table exists.")
    except Exception as e:
        print(f"⚠️ Could not create table via RPC: {e}")
        print("\nRun this SQL manually in Supabase SQL editor:\n")
        print(CREATE_TABLE_SQL)


def _prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert dataframe columns to appropriate types and format:
    - Ensure 'time' column is ISO strings (if exists)
    - Convert NaNs -> None
    - Keep only the columns expected by the table (extra columns are ignored)
    """
    expected_cols = [
        "city",
        "time",
        "pm10",
        "pm2_5",
        "carbon_monoxide",
        "nitrogen_dioxide",
        "sulphur_dioxide",
        "ozone",
        "uv_index",
        "aqi_category",
        "severity",       # note: transform may call this 'severity' — we'll map to severity_score
        "severity_score", # accept either
        "risk_class",     # transform may call this 'risk_class' -> map to risk_flag
        "risk_flag",
        "hour",
    ]

    # normalize column names: allow alias names
    df = df.copy()

    # map common alternate column names
    if "severity" in df.columns and "severity_score" not in df.columns:
        df = df.rename(columns={"severity": "severity_score"})
    if "risk_class" in df.columns and "risk_flag" not in df.columns:
        df = df.rename(columns={"risk_class": "risk_flag"})
    if "aqi_category" not in df.columns and "aqi" in df.columns:
        df = df.rename(columns={"aqi": "aqi_category"})

    # Keep only expected columns (if missing, add them with None)
    keep_cols = []
    for c in ["city", "time", "pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide",
              "sulphur_dioxide", "ozone", "uv_index", "aqi_category", "severity_score",
              "risk_flag", "hour"]:
        keep_cols.append(c)
        if c not in df.columns:
            df[c] = None

    df = df[keep_cols]

    # Convert time -> ISO strings (if pandas datetime-like)
    if "time" in df.columns:
        # If already datetime dtype, convert to ISO; else try to coerce
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        # Represent as ISO 8601 strings with timezone info if available; supabase accepts ISO strings
        df["time"] = df["time"].apply(lambda x: x.isoformat() if not pd.isna(x) else None)

    # Convert numeric columns to numeric and NaN -> None
    numeric_cols = ["pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide",
                    "sulphur_dioxide", "ozone", "uv_index", "severity_score", "hour"]
    for nc in numeric_cols:
        if nc in df.columns:
            df[nc] = pd.to_numeric(df[nc], errors="coerce")

    # Convert NaN (numpy.nan) to None for JSON serialization
    df = df.where(pd.notnull(df), None)

    return df


def load_df_to_supabase(df: pd.DataFrame, batch_size: int = BATCH_SIZE, retry_count: int = RETRY_COUNT) -> dict:
    """
    Load a pandas DataFrame into Supabase table in batches.
    Returns a summary dict: { inserted: int, failed_batches: int, failures: [details] }
    """
    client = _get_supabase_client()
    prepared = _prepare_dataframe(df)
    records = prepared.to_dict(orient="records")
    total = len(records)
    if total == 0:
        print("⚠️ No records to load.")
        return {"inserted": 0, "failed_batches": 0, "failures": []}

    if client is None:
        # If Supabase not configured, print sample and return
        print("⚠️ Supabase client not configured or package missing. Printing sample rows (first 10):")
        for r in records[:10]:
            print(r)
        return {"inserted": 0, "failed_batches": 0, "failures": []}

    inserted_count = 0
    failed_batches = 0
    failures = []

    # Insert in batches
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch = records[start:end]
        attempt = 0
        success = False
        last_err = None

        while attempt <= retry_count:
            attempt += 1
            try:
                res = client.table(TABLE_NAME).insert(batch).execute()
                # supabase client versions differ: either res has .error or res.status_code etc.
                if hasattr(res, "error") and res.error:
                    last_err = res.error
                    raise Exception(f"Supabase error: {res.error}")
                # success
                inserted_count += len(batch)
                print(f"✅ Inserted rows {start+1}-{end} (batch size {len(batch)})")
                success = True
                break
            except Exception as e:
                last_err = e
                if attempt <= retry_count:
                    backoff = RETRY_BACKOFF * (2 ** (attempt - 1))
                    print(f"⚠️ Batch {start+1}-{end} failed attempt {attempt}/{retry_count}. Retrying in {backoff:.1f}s. Error: {e}")
                    time.sleep(backoff)
                else:
                    print(f"❌ Batch {start+1}-{end} failed after {attempt-1} retries. Error: {e}")
        if not success:
            failed_batches += 1
            failures.append({"batch_start": start+1, "batch_end": end, "error": str(last_err)})

    summary = {"inserted": inserted_count, "failed_batches": failed_batches, "failures": failures}
    print("\n=== LOAD SUMMARY ===")
    print(f"Total records: {total}")
    print(f"Inserted: {inserted_count}")
    print(f"Failed batches: {failed_batches}")
    if failures:
        print("Failures details:")
        for f in failures:
            print(f)
    return summary


def load_csv_to_supabase(staged_csv_path: str, batch_size: int = BATCH_SIZE, retry_count: int = RETRY_COUNT) -> dict:
    """
    Convenience wrapper: read CSV into DataFrame and call load_df_to_supabase.
    """
    p = Path(staged_csv_path)
    if not p.exists():
        raise FileNotFoundError(f"Staged CSV not found: {staged_csv_path}")

    df = pd.read_csv(p)
    return load_df_to_supabase(df, batch_size=batch_size, retry_count=retry_count)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python load.py path/to/staged.csv")
        sys.exit(1)

    staged = sys.argv[1]
    create_table_if_not_exists()
    result = load_csv_to_supabase(staged)
    print("Done. Result:", result)
