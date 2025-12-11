# transform.py
"""
Transform step for AtmosTrack Open-Meteo air quality raw JSON files.

Reads raw JSON files in data/raw/*_raw_*.json (Open-Meteo air-quality),
flattens hourly arrays into a row-per-hour tabular CSV, computes features,
and saves a staged CSV to data/staged/.

Required output columns:
  city, time, pm10, pm2_5, carbon_monoxide, nitrogen_dioxide, sulphur_dioxide, ozone, uv_index

Derived columns:
  aqi_category (from pm2_5), severity (weighted sum), risk_class, hour

Usage:
  python transform.py
"""
from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[0]
RAW_DIR = BASE_DIR / "data" / "raw"
STAGED_DIR = BASE_DIR / "data" / "staged"
STAGED_DIR.mkdir(parents=True, exist_ok=True)

# Expected pollutant keys in hourly block
POLLUTANTS = [
    "pm10",
    "pm2_5",
    "carbon_monoxide",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "ozone",
    "uv_index",
]

def _list_raw_files() -> List[Path]:
    return sorted(RAW_DIR.glob("*_raw_*.json"))

def _extract_hourly_from_payload(payload: Dict[str, Any]) -> Optional[Dict[str, List[Any]]]:
    """
    Extract the hourly dict (mapping from variable name to list) from various possible payload shapes.
    Open-Meteo typical shape: { ..., "hourly": {"time": [...], "pm10":[...], ...} }
    """
    if not isinstance(payload, dict):
        return None
    # common key
    if "hourly" in payload and isinstance(payload["hourly"], dict):
        return payload["hourly"]

    # sometimes nested under 'data' or other keys
    for k in ("data", "results", "result"):
        v = payload.get(k)
        if isinstance(v, dict) and "hourly" in v and isinstance(v["hourly"], dict):
            return v["hourly"]

    # fallback: look for a dict value that contains 'time' and at least one pollutant array
    for v in payload.values():
        if isinstance(v, dict) and "time" in v and isinstance(v["time"], list):
            # check presence of at least one pollutant key
            if any(p in v for p in POLLUTANTS):
                return v
    return None

def _rows_from_file(path: Path) -> pd.DataFrame:
    """
    Parse one raw JSON file and return a DataFrame with columns:
      city, time, <pollutants...>, raw_filename
    """
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    # try to determine city name from payload top-level keys or filename
    city = None
    # open-meteo returned city sometimes in 'city' or 'location' or not at all
    if isinstance(payload, dict):
        city = payload.get("city") or payload.get("location") or payload.get("name")

    # fallback to filename prefix
    if not city:
        # filename pattern: <city>_raw_<ts>.json
        fname = path.stem  # e.g. delhi_raw_20251211T...
        parts = fname.split("_raw_")
        if parts and parts[0]:
            city = parts[0].replace("_", " ").title()
        else:
            city = "unknown"

    hourly = _extract_hourly_from_payload(payload)
    if hourly is None:
        # nothing to parse
        return pd.DataFrame(columns=["city", "time"] + POLLUTANTS + ["raw_filename"])

    times = hourly.get("time") or hourly.get("times") or []
    n = len(times)

    # prepare rows
    rows = []
    for i, t in enumerate(times):
        row = {"city": city, "time": t, "raw_filename": path.name}
        for p in POLLUTANTS:
            # Open-Meteo uses pm2_5 name as 'pm2_5' usually; try direct key and a few alternatives
            val = None
            # prefer exact key
            if p in hourly and isinstance(hourly[p], list) and i < len(hourly[p]):
                val = hourly[p][i]
            else:
                # try common variants
                alt_keys = []
                if p == "pm2_5":
                    alt_keys = ["pm2_5", "pm2.5", "pm25", "pm2_5_value"]
                elif p == "carbon_monoxide":
                    alt_keys = ["carbon_monoxide", "co"]
                elif p == "nitrogen_dioxide":
                    alt_keys = ["nitrogen_dioxide", "no2"]
                elif p == "sulphur_dioxide":
                    alt_keys = ["sulphur_dioxide", "so2"]
                elif p == "ozone":
                    alt_keys = ["ozone", "o3"]
                elif p == "pm10":
                    alt_keys = ["pm10", "pm10_value"]
                elif p == "uv_index":
                    alt_keys = ["uv_index", "uv_index_value"]

                for ak in alt_keys:
                    if ak in hourly and isinstance(hourly[ak], list) and i < len(hourly[ak]):
                        val = hourly[ak][i]
                        break
            row[p] = val
        rows.append(row)

    df = pd.DataFrame(rows)
    return df

def _aqi_category_from_pm25(pm25: Optional[float]) -> Optional[str]:
    if pm25 is None or pd.isna(pm25):
        return None
    try:
        v = float(pm25)
    except Exception:
        return None
    if v <= 50:
        return "Good"
    if 51 <= v <= 100:
        return "Moderate"
    if 101 <= v <= 200:
        return "Unhealthy"
    if 201 <= v <= 300:
        return "Very Unhealthy"
    return "Hazardous"

def _compute_severity(row: pd.Series) -> Optional[float]:
    """
    severity = (pm2_5 * 5) + (pm10 * 3) +
               (nitrogen_dioxide * 4) + (sulphur_dioxide * 4) +
               (carbon_monoxide * 2) + (ozone * 3)
    """
    # Use 0 for missing in computation (alternatively could use NaN and propagate)
    def _safe(val):
        try:
            if val is None:
                return 0.0
            return float(val)
        except Exception:
            return 0.0

    pm2_5 = _safe(row.get("pm2_5"))
    pm10 = _safe(row.get("pm10"))
    no2 = _safe(row.get("nitrogen_dioxide"))
    so2 = _safe(row.get("sulphur_dioxide"))
    co = _safe(row.get("carbon_monoxide"))
    o3 = _safe(row.get("ozone"))

    return (pm2_5 * 5.0) + (pm10 * 3.0) + (no2 * 4.0) + (so2 * 4.0) + (co * 2.0) + (o3 * 3.0)

def _risk_from_severity(sev: Optional[float]) -> Optional[str]:
    if sev is None:
        return None
    try:
        v = float(sev)
    except Exception:
        return None
    if v > 400:
        return "High Risk"
    if v > 200:
        return "Moderate Risk"
    return "Low Risk"

def transform_all() -> str:
    raw_files = _list_raw_files()
    if not raw_files:
        raise SystemExit("No raw files found in data/raw/. Run extract.py first.")

    dfs = []
    for p in raw_files:
        try:
            df = _rows_from_file(p)
            if not df.empty:
                dfs.append(df)
        except Exception as e:
            print(f"⚠️ Skipping {p.name} due to error: {e}")

    if not dfs:
        raise SystemExit("No rows extracted from raw files.")

    all_df = pd.concat(dfs, ignore_index=True)

    # Convert time -> datetime
    all_df["time"] = pd.to_datetime(all_df["time"], errors="coerce")

    # Ensure pollutant numeric columns and rename pm2_5 if alternate keys used
    for col in POLLUTANTS:
        # Some files may have pm2.5 as pm2_5 or pm25 etc; previous extraction populates pm2_5 column specifically
        all_df[col] = pd.to_numeric(all_df[col], errors="coerce")

    # Drop rows where all pollutant columns are NaN
    pollutant_cols = POLLUTANTS.copy()
    all_df = all_df.dropna(subset=pollutant_cols, how="all").reset_index(drop=True)

    # Derived features
    all_df["aqi_category"] = all_df["pm2_5"].apply(_aqi_category_from_pm25)
    # severity as numeric
    all_df["severity"] = all_df.apply(_compute_severity, axis=1)
    all_df["risk_class"] = all_df["severity"].apply(_risk_from_severity)
    # hour of day
    all_df["hour"] = all_df["time"].dt.hour

    # Reorder columns: city, time, pollutants..., derived...
    out_cols = ["city", "time"] + POLLUTANTS + ["aqi_category", "severity", "risk_class", "hour", "raw_filename"]
    # ensure present
    out_cols = [c for c in out_cols if c in all_df.columns]

    staged_path = STAGED_DIR / f"air_quality_transformed_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.csv"
    all_df.to_csv(staged_path, columns=out_cols, index=False)
    print(f"✅ Transformed data saved to: {staged_path}  (rows: {len(all_df)})")
    return str(staged_path)

if __name__ == "__main__":
    print("Starting transform step: flattening raw Open-Meteo air-quality JSON -> staged CSV")
    staged = transform_all()
    print("Done. Staged file:", staged)
