from dotenv import load_dotenv
import os
import pandas as pd
from supabase import create_client
from pathlib import Path
import matplotlib.pyplot as plt
 
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
 
BASE_DIR = Path(__file__).resolve().parents[0]
PROCESSED_DIR = BASE_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
 
TABLE_NAME = "weather_data"
 
if not SUPABASE_URL or not SUPABASE_KEY:
    raise SystemExit("Please set SUPABASE_URL and SUPABASE_KEY in your .env")
 
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
 
 
def _extract_data_from_response(res):
    """
    Try multiple strategies to extract a list-of-dicts from a supabase response object.
    Returns a Python list (possibly empty).
    """
    # 1) .data attribute (preferred)
    data = getattr(res, "data", None)
    if isinstance(data, list):
        return data
 
    # 2) dict-like with 'data' key
    try:
        if isinstance(res, dict) and "data" in res and isinstance(res["data"], list):
            return res["data"]
    except Exception:
        pass
 
    # 3) If res is a list/tuple, try to find the element that is a list of dicts
    if isinstance(res, (list, tuple)):
        for item in res:
            if isinstance(item, list) and all(isinstance(x, dict) for x in item):
                return item
        if len(res) > 0 and isinstance(res[0], dict):
            return list(res)
 
    # 4) If res has .json() method -> try that
    json_like = getattr(res, "json", None)
    if callable(json_like):
        try:
            j = res.json()
            if isinstance(j, dict) and "data" in j and isinstance(j["data"], list):
                return j["data"]
        except Exception:
            pass
 
    # Nothing matched — return empty list
    return []
 
 
def fetch_table(limit: int | None = None) -> pd.DataFrame:
    """
    Fetch table from Supabase and return a cleaned DataFrame.
    `limit` is optional to restrict rows fetched (useful for debugging).
    """
    print(f"🔍 Fetching data from Supabase table '{TABLE_NAME}' ...")
    query = supabase.table(TABLE_NAME).select("*")
    if limit:
        query = query.limit(limit)
    res = query.execute()
 
    data = _extract_data_from_response(res)
 
    # Debugging helper (uncomment if needed)
    # print("DEBUG raw response type:", type(res))
    # print("DEBUG extracted data sample:", data[:2])
 
    df = pd.DataFrame(data)
 
    if df.empty:
        print("⚠️  No rows extracted into DataFrame. Check Supabase table in web UI.")
        return df
 
    # Normalize and coerce types
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
 
    for col in ["temperature_c", "relative_humidity", "wind_speed_kmh", "feels_like_c"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
 
    if "hour" in df.columns:
        df["hour"] = pd.to_numeric(df["hour"], errors="coerce").astype("Int64")
 
    return df
 
 
def analyze_and_save(df: pd.DataFrame):
    if df.empty:
        print("No data to analyze.")
        return
 
    print("ℹ️  Data info:")
    print(df.info())
 
    # Basic metrics
    summary = {
        "rows": len(df),
        "time_min": str(df["time"].min()) if "time" in df.columns else None,
        "time_max": str(df["time"].max()) if "time" in df.columns else None,
        "temp_mean": float(df["temperature_c"].mean()) if "temperature_c" in df.columns else None,
        "humidity_mean": float(df["relative_humidity"].mean()) if "relative_humidity" in df.columns else None,
        "wind_mean": float(df["wind_speed_kmh"].mean()) if "wind_speed_kmh" in df.columns else None,
    }
    print("🔎 Summary metrics:")
    for k, v in summary.items():
        print(f"  - {k}: {v}")
 
    # Save high-level summary CSV
    summary_df = pd.DataFrame([summary])
    summary_csv = PROCESSED_DIR / "analysis_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"✅ Saved analysis summary to {summary_csv}")
 
    # Hourly average temperature (if possible)
    if {"date", "hour", "temperature_c"}.issubset(df.columns):
        hourly = df.groupby(["date", "hour"], as_index=False)["temperature_c"].mean()
        hourly_csv = PROCESSED_DIR / "hourly_avg_temp.csv"
        hourly.to_csv(hourly_csv, index=False)
        print(f"✅ Saved hourly average temperature to {hourly_csv}")
 
    # Plots (saved to processed dir)
    try:
        if "temperature_c" in df.columns:
            plt.figure(figsize=(8, 4))
            df["temperature_c"].plot(kind="hist", bins=30)
            plt.title("Temperature distribution")
            plt.xlabel("Temperature (°C)")
            plt.tight_layout()
            plt.savefig(PROCESSED_DIR / "temperature_hist.png")
            plt.close()
            print(f"✅ Saved temperature histogram to {PROCESSED_DIR / 'temperature_hist.png'}")
 
        if {"date", "temperature_c"}.issubset(df.columns):
            daily = df.groupby("date", as_index=False)["temperature_c"].mean()
            plt.figure(figsize=(10, 4))
            plt.plot(daily["date"], daily["temperature_c"], marker="o")
            plt.title("Daily Average Temperature")
            plt.ylabel("Temperature (°C)")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(PROCESSED_DIR / "daily_avg_temp.png")
            plt.close()
            print(f"✅ Saved daily average temperature plot to {PROCESSED_DIR / 'daily_avg_temp.png'}")
    except Exception as e:
        print(f"⚠️  Plotting failed: {e}")
 
 
def run_analysis(limit: int | None = None):
    df = fetch_table(limit=limit)
    analyze_and_save(df)
 
 
if __name__ == "__main__":
    # For quick debugging you can pass a limit: run_analysis(limit=10)
    run_analysis()