# etl_analysis.py
"""
ETL Analysis for Air Quality pipeline.

- Reads data from Supabase table `air_quality_data` (if SUPABASE_URL / SUPABASE_KEY present),
  otherwise falls back to the latest staged CSV in data/staged/.
- Computes KPI metrics and writes CSVs to data/processed/:
    - summary_metrics.csv
    - city_risk_distribution.csv
    - pollution_trends.csv
- Produces PNG visualizations in data/processed/:
    - pm25_histogram.png
    - risk_bar_per_city.png
    - pm25_hourly_line.png
    - severity_vs_pm25_scatter.png
"""
from __future__ import annotations

import os
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv

load_dotenv()

# Optional supabase client
try:
    from supabase import create_client  # type: ignore
    _HAS_SUPABASE = True
except Exception:
    _HAS_SUPABASE = False

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
TABLE_NAME = os.getenv("SUPABASE_TABLE", "air_quality_data")

BASE_DIR = Path(__file__).resolve().parents[0]
STAGED_DIR = BASE_DIR / "data" / "staged"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def _get_supabase_client():
    if not _HAS_SUPABASE:
        return None
    if not SUPABASE_URL or not SUPABASE_KEY:
        return None
    try:
        return create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception:
        return None


def _fetch_from_supabase() -> Optional[pd.DataFrame]:
    client = _get_supabase_client()
    if client is None:
        return None
    try:
        # select all rows
        res = client.table(TABLE_NAME).select("*").execute()
        # result shape varies by client version; try res.data or res
        data = None
        if hasattr(res, "data"):
            data = res.data
        elif isinstance(res, dict) and "data" in res:
            data = res["data"]
        else:
            # some versions return an object with .json()
            try:
                data = res.json()
            except Exception:
                data = None

        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data)
        return df
    except Exception as e:
        print("⚠️ Failed to fetch from Supabase:", e)
        return None


def _latest_staged_csv() -> Optional[Path]:
    files = sorted(STAGED_DIR.glob("*.csv"))
    return files[-1] if files else None


def _load_data() -> pd.DataFrame:
    """
    Load data from Supabase or fallback to latest staged CSV.
    Ensures proper dtypes and returns DataFrame.
    """
    df = _fetch_from_supabase()
    if df is not None:
        if df.empty:
            print("⚠️ Supabase returned empty dataset; falling back to staged CSV.")
        else:
            print("✅ Loaded data from Supabase.")
    if df is None or df.empty:
        staged = _latest_staged_csv()
        if not staged:
            raise SystemExit("No data available from Supabase and no staged CSV found in data/staged/.")
        print(f"ℹ️ Loading staged CSV fallback: {staged}")
        df = pd.read_csv(staged, parse_dates=["time"], infer_datetime_format=True)
    else:
        # Try to coerce 'time' column to datetime if present
        if "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"], errors="coerce", infer_datetime_format=True)

    # Normalize expected columns and aliases
    alias_map = {
        "severity": "severity_score",
        "severity_score": "severity_score",
        "risk_class": "risk_flag",
        "risk_flag": "risk_flag",
        "aqi_category": "aqi_category",
        "pm2_5": "pm2_5",
        "pm10": "pm10",
    }
    # Nothing strict here, just ensure important columns exist
    for col in ["city", "time", "pm2_5", "pm10", "ozone", "severity_score", "risk_flag", "carbon_monoxide",
                "nitrogen_dioxide", "sulphur_dioxide", "uv_index"]:
        if col not in df.columns:
            df[col] = None

    # convert numeric columns
    numeric_cols = ["pm2_5", "pm10", "ozone", "carbon_monoxide", "nitrogen_dioxide", "sulphur_dioxide", "uv_index", "severity_score", "hour"]
    for nc in numeric_cols:
        if nc in df.columns:
            df[nc] = pd.to_numeric(df[nc], errors="coerce")

    # ensure 'risk_flag' and 'aqi_category' are string types
    if "risk_flag" in df.columns:
        df["risk_flag"] = df["risk_flag"].astype("string")

    # drop rows without city or time
    df = df.dropna(subset=["city", "time"], how="any").reset_index(drop=True)
    return df


def compute_kpis(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes requested KPIs:
    - City with highest average PM2.5
    - City with highest average severity_score
    - Percentage of High/Moderate/Low risk hours (global)
    - Hour of day with worst AQI (based on average pm2_5)
    Returns metrics dict and also prints summary.
    """
    metrics = {}

    # City with highest average PM2.5
    pm25_by_city = df.groupby("city")["pm2_5"].mean().dropna()
    if not pm25_by_city.empty:
        city_highest_pm25 = pm25_by_city.idxmax()
        metrics["city_highest_avg_pm2_5"] = {"city": city_highest_pm25, "avg_pm2_5": pm25_by_city.max()}
    else:
        metrics["city_highest_avg_pm2_5"] = None

    # City with highest average severity_score
    sev_by_city = df.groupby("city")["severity_score"].mean().dropna()
    if not sev_by_city.empty:
        city_highest_sev = sev_by_city.idxmax()
        metrics["city_highest_avg_severity"] = {"city": city_highest_sev, "avg_severity": sev_by_city.max()}
    else:
        metrics["city_highest_avg_severity"] = None

    # Percentages of risk flags (High/Moderate/Low) globally
    if "risk_flag" in df.columns:
        risk_counts = df["risk_flag"].value_counts(dropna=True)
        total_risk = risk_counts.sum() if not risk_counts.empty else 0
        risk_percentages = {}
        for key in ["High Risk", "Moderate Risk", "Low Risk"]:
            cnt = int(risk_counts.get(key, 0))
            pct = (cnt / total_risk * 100) if total_risk > 0 else 0.0
            risk_percentages[key] = {"count": cnt, "percentage": round(pct, 2)}
        metrics["risk_distribution_global"] = risk_percentages
    else:
        metrics["risk_distribution_global"] = None

    # Hour of day with worst AQI (based on average pm2_5)
    if "time" in df.columns and "pm2_5" in df.columns:
        df["hour_of_day"] = df["time"].dt.hour
        hour_avg = df.groupby("hour_of_day")["pm2_5"].mean().dropna()
        if not hour_avg.empty:
            worst_hour = int(hour_avg.idxmax())
            metrics["worst_hour_by_avg_pm2_5"] = {"hour": worst_hour, "avg_pm2_5": hour_avg.max()}
        else:
            metrics["worst_hour_by_avg_pm2_5"] = None
    else:
        metrics["worst_hour_by_avg_pm2_5"] = None

    # Print a short summary
    print("\n=== KPI SUMMARY ===")
    print("City with highest avg PM2.5:", metrics["city_highest_avg_pm2_5"])
    print("City with highest avg severity:", metrics["city_highest_avg_severity"])
    print("Global risk distribution (counts & %):", metrics["risk_distribution_global"])
    print("Hour with worst avg PM2.5:", metrics["worst_hour_by_avg_pm2_5"])

    return metrics


def build_city_risk_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds a table of counts and percentages of risk_flag per city.
    Returns DataFrame with columns: city, risk_flag, count, pct
    """
    # ensure risk_flag exists
    if "risk_flag" not in df.columns:
        return pd.DataFrame(columns=["city", "risk_flag", "count", "pct"])

    counts = df.groupby(["city", "risk_flag"]).size().reset_index(name="count")
    total_by_city = df.groupby("city").size().reset_index(name="total")
    total_by_city = total_by_city.rename(columns={"size": "total"}) if "size" not in total_by_city.columns else total_by_city
    # merge totals
    counts = counts.merge(df.groupby("city").size().rename("total").reset_index(), on="city", how="left")
    counts["pct"] = counts["count"] / counts["total"] * 100.0
    counts["pct"] = counts["pct"].round(2)
    return counts[["city", "risk_flag", "count", "pct"]]


def build_pollution_trends(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each city, returns time -> pm2_5, pm10, ozone
    Returns DataFrame with columns: city, time, pm2_5, pm10, ozone
    """
    cols = ["city", "time", "pm2_5", "pm10", "ozone"]
    existing = [c for c in cols if c in df.columns]
    trends = df[existing].copy()
    # sort for readability
    trends = trends.sort_values(["city", "time"]).reset_index(drop=True)
    return trends


def save_csvs(summary_metrics: Dict[str, Any], risk_df: pd.DataFrame, trends_df: pd.DataFrame) -> Dict[str, Path]:
    """
    Save required CSVs to data/processed/ and return dict of paths.
    summary_metrics: dict (we'll convert to a one-row DataFrame with JSON columns)
    """
    proc_paths = {}

    # summary_metrics.csv: represent metrics as flattened rows
    summary_path = PROCESSED_DIR / "summary_metrics.csv"
    # Create a simple one-row DataFrame with metrics JSON
    summary_row = {
        "city_highest_avg_pm2_5": None,
        "city_highest_avg_severity": None,
        "risk_distribution_global": None,
        "worst_hour_by_avg_pm2_5": None,
    }
    if summary_metrics.get("city_highest_avg_pm2_5"):
        summary_row["city_highest_avg_pm2_5"] = str(summary_metrics["city_highest_avg_pm2_5"])
    if summary_metrics.get("city_highest_avg_severity"):
        summary_row["city_highest_avg_severity"] = str(summary_metrics["city_highest_avg_severity"])
    summary_row["risk_distribution_global"] = str(summary_metrics.get("risk_distribution_global"))
    summary_row["worst_hour_by_avg_pm2_5"] = str(summary_metrics.get("worst_hour_by_avg_pm2_5"))
    pd.DataFrame([summary_row]).to_csv(summary_path, index=False)
    proc_paths["summary_metrics"] = summary_path

    # city_risk_distribution.csv
    risk_path = PROCESSED_DIR / "city_risk_distribution.csv"
    risk_df.to_csv(risk_path, index=False)
    proc_paths["city_risk_distribution"] = risk_path

    # pollution_trends.csv
    trends_path = PROCESSED_DIR / "pollution_trends.csv"
    trends_df.to_csv(trends_path, index=False)
    proc_paths["pollution_trends"] = trends_path

    print("\nSaved CSVs:")
    for k, p in proc_paths.items():
        print(f" - {k}: {p}")
    return proc_paths


def make_plots(df: pd.DataFrame) -> Dict[str, Path]:
    """
    Create and save plots:
    - Histogram of PM2.5
    - Bar chart of risk flags per city
    - Line chart of hourly PM2.5 trends (hour 0-23)
    - Scatter: severity_score vs pm2_5
    Returns dict of saved PNG paths.
    NOTE: Uses matplotlib and saves each plot separately (no color arguments).
    """
    paths = {}

    # Histogram of PM2.5
    pm25 = df["pm2_5"].dropna()
    plt.figure()
    plt.hist(pm25, bins=30)
    plt.title("Histogram of PM2.5")
    plt.xlabel("PM2.5")
    plt.ylabel("Count")
    p_hist = PROCESSED_DIR / "pm25_histogram.png"
    plt.tight_layout()
    plt.savefig(p_hist)
    plt.close()
    paths["pm25_histogram"] = p_hist

   
    # Line chart hourly PM2.5 trends
    if "time" in df.columns and "pm2_5" in df.columns:
        df["hour_of_day"] = df["time"].dt.hour
        hourly = df.groupby("hour_of_day")["pm2_5"].mean().reset_index()
        plt.figure()
        plt.plot(hourly["hour_of_day"], hourly["pm2_5"], marker="o")
        plt.title("Average PM2.5 by Hour of Day")
        plt.xlabel("Hour of day")
        plt.ylabel("Average PM2.5")
        p_line = PROCESSED_DIR / "pm25_hourly_line.png"
        plt.tight_layout()
        plt.savefig(p_line)
        plt.close()
        paths["pm25_hourly_line"] = p_line

    # Scatter: severity_score vs pm2_5
    if "severity_score" in df.columns and "pm2_5" in df.columns:
        scatter_df = df.dropna(subset=["severity_score", "pm2_5"])
        plt.figure()
        plt.scatter(scatter_df["pm2_5"], scatter_df["severity_score"])
        plt.title("Severity Score vs PM2.5")
        plt.xlabel("PM2.5")
        plt.ylabel("Severity Score")
        p_scatter = PROCESSED_DIR / "severity_vs_pm25_scatter.png"
        plt.tight_layout()
        plt.savefig(p_scatter)
        plt.close()
        paths["severity_vs_pm25_scatter"] = p_scatter

    print("\nSaved plots:")
    for k, p in paths.items():
        print(f" - {k}: {p}")
    return paths


def run_analysis_and_export():
    df = _load_data()
    metrics = compute_kpis(df)
    risk_df = build_city_risk_distribution(df)
    trends_df = build_pollution_trends(df)

    csv_paths = save_csvs(metrics, risk_df, trends_df)
    plot_paths = make_plots(df)

    return {"csvs": csv_paths, "plots": plot_paths, "metrics": metrics}


if __name__ == "__main__":
    print("Starting ETL analysis...")
    result = run_analysis_and_export()
    print("\nETL analysis complete. Outputs:")
    print(result)