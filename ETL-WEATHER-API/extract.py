# extract.py
import json
from datetime import datetime
from pathlib import Path
import requests
from dotenv import load_dotenv
import os
 
load_dotenv()
 
BASE_DIR = Path(__file__).resolve().parents[0]
RAW_DIR = BASE_DIR / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)
 
LAT = os.getenv("LAT", "17.3850")
LON = os.getenv("LON", "78.4867")
FORECAST_DAYS = int(os.getenv("FORECAST_DAYS", "1"))
 
def extract_weather_data(lat: str = LAT, lon: str = LON, days: int = FORECAST_DAYS):
    """
    Call Open-Meteo API and store raw JSON to data/raw/.
    Returns path to saved file.
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,relativehumidity_2m,windspeed_10m",
        "forecast_days": days,
        "timezone": "auto"
    }
 
    print(f"⏳ Requesting weather data for lat={lat}, lon={lon}, days={days} ...")
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
 
    filename = RAW_DIR / f"weather_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    filename.write_text(json.dumps(data, indent=2))
    print(f"✅ Extracted weather data and saved to: {filename}")
    return str(filename)
 
if __name__ == "__main__":
    extract_weather_data()
 
 