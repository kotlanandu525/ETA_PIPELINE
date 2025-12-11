# transform.py
import json
from pathlib import Path
import pandas as pd
from datetime import datetime
from typing import List
 
BASE_DIR = Path(__file__).resolve().parents[0]
RAW_DIR = BASE_DIR / "data" / "raw"
STAGED_DIR = BASE_DIR / "data" / "staged"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
 
STAGED_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
 
def _flatten_weather_json(json_path: str) -> pd.DataFrame:
    """
    Convert Open-Meteo hourly JSON payload to a flat DataFrame:
    columns: time, temperature_2m, relativehumidity_2m, windspeed_10m
    """
    with open(json_path, "r") as f:
        payload = json.load(f)
 
    hourly = payload.get("hourly", {})
    # Expect arrays of equal length for 'time' and each metric
    times = hourly.get("time", [])
    temps = hourly.get("temperature_2m", [])
    rh = hourly.get("relativehumidity_2m", [])
    wind = hourly.get("windspeed_10m", [])
 
    rows = []
    for i, t in enumerate(times):
        rows.append({
            "time": t,
            "temperature_2m": temps[i] if i < len(temps) else None,
            "relativehumidity_2m": rh[i] if i < len(rh) else None,
            "windspeed_10m": wind[i] if i < len(wind) else None
        })
 
    df = pd.DataFrame(rows)
    return df
 
def transform_data(raw_json_paths: List[str]) -> str:
    """
    Read one or more raw JSONs, flatten, clean, feature-engineer, and save staged CSV.
    Returns staged CSV path.
    """
    dfs = []
    for p in raw_json_paths:
        print(f"🔁 Transforming {p} ...")
        df = _flatten_weather_json(p)
        dfs.append(df)
 
    if not dfs:
        raise ValueError("No raw JSON files provided to transform")
 
    df = pd.concat(dfs, ignore_index=True)
 
    # --- Basic cleaning ---
    # Ensure time column is datetime
    df["time"] = pd.to_datetime(df["time"])
    # Rename columns to consistent snake_case
    df = df.rename(columns={
        "temperature_2m": "temperature_c",
        "relativehumidity_2m": "relative_humidity",
        "windspeed_10m": "wind_speed_kmh"
    })
 
    # Convert units if needed (Open-Meteo returns m/s for wind by default? It might return km/h depending on params.)
    # We'll assume values are provided consistently; leave as-is but coerce to numeric.
    for col in ["temperature_c", "relative_humidity", "wind_speed_kmh"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
 
    # --- Feature engineering ---
    df["date"] = df["time"].dt.date
    df["hour"] = df["time"].dt.hour
    # temp category
    df["temp_category"] = pd.cut(df["temperature_c"],
                                 bins=[-100, 0, 10, 20, 30, 100],
                                 labels=["very_cold", "cold", "mild", "warm", "hot"])
    # feels_like proxy: temp adjusted by humidity (simple proxy)
    df["feels_like_c"] = df["temperature_c"] - (df["relative_humidity"] / 100) * 2
 
    # --- Drop rows with all NaNs for sensors ---
    df = df.dropna(subset=["temperature_c", "relative_humidity", "wind_speed_kmh"], how="all")
 
    staged_path = STAGED_DIR / f"weather_staged_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(staged_path, index=False)
    print(f"✅ Transformed data saved at: {staged_path}")
    return str(staged_path)
 
if __name__ == "__main__":
    # Convenience: transform the latest raw file
    raw_files = sorted([str(p) for p in RAW_DIR.glob("weather_*.json")])
    if not raw_files:
        raise SystemExit("No raw weather JSON files found. Run extract.py first.")
    transform_data([raw_files[-1]])
 
 