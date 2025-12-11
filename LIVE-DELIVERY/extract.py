# extract.py
"""
Extract step for AtmosTrack (Open-Meteo Air Quality).
Fetches hourly pollutant data for 5 cities and saves raw JSON files.

Saves to:
  data/raw/<city>_raw_<YYYYmmddTHHMMSSZ>.json

Features:
- Retry logic (MAX_RETRIES, exponential backoff)
- Graceful failure handling
- Logging to console and logs/extract.log
- Returns list of saved file paths (and prints a summary)
"""
from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import requests
from dotenv import load_dotenv

load_dotenv()

# ---- Configuration (override via .env) ----
BASE_DIR = Path(__file__).resolve().parents[0]
RAW_DIR = Path(os.getenv("RAW_DIR", BASE_DIR / "data" / "raw"))
LOG_DIR = Path(os.getenv("LOG_DIR", BASE_DIR / "logs"))
RAW_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
BACKOFF_FACTOR = float(os.getenv("BACKOFF_FACTOR", "1.0"))  # seconds base for exponential backoff
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "20"))  # seconds per request
SLEEP_BETWEEN_CALLS = float(os.getenv("SLEEP_BETWEEN_CALLS", "0.5"))  # polite pause between city calls

LOG_FILE = LOG_DIR / "extract.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, encoding="utf-8"), logging.StreamHandler()],
)
logger = logging.getLogger("extract")

# ---- Cities (latitude, longitude) ----
CITIES = {
    "Delhi": (28.7041, 77.1025),
    "Mumbai": (19.0760, 72.8777),
    "Bengaluru": (12.9716, 77.5946),
    "Hyderabad": (17.3850, 78.4867),
    "Kolkata": (22.5726, 88.3639),
}

# ---- API base and parameters ----
API_BASE = os.getenv("OPEN_METEO_AQ_BASE", "https://air-quality-api.open-meteo.com/v1/air-quality")
HOURLY_PARAMS = "pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,ozone,sulphur_dioxide,uv_index"


def _utc_ts() -> str:
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


def _safe_city_filename(city: str) -> str:
    return city.strip().lower().replace(" ", "_")


def _timestamped_path_for(city: str) -> Path:
    fname = f"{_safe_city_filename(city)}_raw_{_utc_ts()}.json"
    return RAW_DIR / fname


def _save_json(payload: object, city: str) -> str:
    path = _timestamped_path_for(city)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
    except Exception as e:
        # fallback to plain text
        alt = path.with_suffix(".txt")
        with open(alt, "w", encoding="utf-8") as f:
            f.write(repr(payload))
        logger.warning("Failed to JSON-dump payload for %s: %s — wrote text to %s", city, e, alt)
        return str(alt.resolve())
    return str(path.resolve())


def _fetch_city(city: str, lat: float, lon: float, max_retries: int = MAX_RETRIES) -> Dict[str, Optional[str]]:
    """
    Fetch city data with retries. Returns a dict:
      { "city": city, "success": True/False, "path": <saved_path> or None, "error": <message> or None }
    """
    url = API_BASE
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": HOURLY_PARAMS,
    }

    attempt = 0
    last_err = None

    # Use a session — optionally ignore env proxies for debugging by setting session.trust_env = False
    session = requests.Session()
    # session.trust_env = False  # uncomment if you need to ignore HTTP(S)_PROXY env vars during debugging

    while attempt < max_retries:
        attempt += 1
        try:
            logger.info("Requesting %s for %s (attempt %d/%d) params=%s", url, city, attempt, max_retries, params)
            resp = session.get(url, params=params, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            try:
                data = resp.json()
            except ValueError:
                data = {"raw_text": resp.text}

            saved_path = _save_json(data, city)
            logger.info("✅ [%s] saved raw response -> %s", city, saved_path)
            return {"city": city, "success": True, "path": saved_path, "error": None}
        except requests.RequestException as e:
            last_err = str(e)
            # Determine status if available
            status = None
            try:
                status = e.response.status_code  # type: ignore[attr-defined]
            except Exception:
                status = None

            # If client error (4xx) except 429, don't retry
            if status and 400 <= status < 500 and status != 429:
                logger.error("❌ [%s] non-retriable client error %s: %s", city, status, e)
                return {"city": city, "success": False, "path": None, "error": f"{status} {e}"}

            backoff = BACKOFF_FACTOR * (2 ** (attempt - 1))
            logger.warning("⚠️ [%s] attempt %d failed: %s — backing off %.1fs", city, attempt, e, backoff)
            time.sleep(backoff)
        except Exception as e:
            last_err = str(e)
            logger.exception("⚠️ [%s] unexpected error on attempt %d: %s", city, attempt, e)
            backoff = BACKOFF_FACTOR * (2 ** (attempt - 1))
            time.sleep(backoff)

    logger.error("❌ [%s] failed after %d attempts. Last error: %s", city, max_retries, last_err)
    return {"city": city, "success": False, "path": None, "error": last_err}


def fetch_all_cities(cities: Optional[Dict[str, tuple]] = None) -> List[Dict[str, Optional[str]]]:
    """
    Fetch data for all cities (use CITIES by default).
    Returns list of result dicts for each city.
    """
    if cities is None:
        cities = CITIES

    results: List[Dict[str, Optional[str]]] = []
    for city, (lat, lon) in cities.items():
        res = _fetch_city(city, lat, lon)
        results.append(res)
        time.sleep(SLEEP_BETWEEN_CALLS)
    return results


if __name__ == "__main__":
    logger.info("Starting AtmosTrack Open-Meteo Air Quality extract")
    logger.info("Cities: %s", ", ".join(CITIES.keys()))
    summary = fetch_all_cities()
    logger.info("Extraction finished — summary:")
    for item in summary:
        if item.get("success"):
            logger.info(" - %s: saved -> %s", item["city"], item["path"])
        else:
            logger.error(" - %s: FAILED -> %s", item["city"], item.get("error"))
    print("\nExtraction summary:")
    for item in summary:
        city = item.get("city")
        status = "OK" if item.get("success") else "FAILED"
        info = item.get("path") or item.get("error")
        print(f" - {city}: {status} -> {info}")
