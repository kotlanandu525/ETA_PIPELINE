# supabase_test.py
from supabase import create_client
import os
from dotenv import load_dotenv
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise SystemExit("Set SUPABASE_URL and SUPABASE_KEY in .env or environment.")

client = create_client(SUPABASE_URL, SUPABASE_KEY)
print("Client created:", bool(client))

# Try a safe test insert (uncomment to run)
test_row = {
    "city": "TEST_CITY",
    "time": "2025-01-01T00:00:00",
    "pm10": 1.0, "pm2_5": 1.0, "carbon_monoxide": 1.0,
    "nitrogen_dioxide": 1.0, "sulphur_dioxide": 1.0, "ozone": 1.0,
    "uv_index": 0.0, "aqi_category": "Good", "severity_score": 1.0, "risk_flag": "Low Risk", "hour": 0
}
try:
    res = client.table("air_quality_data").insert(test_row).execute()
    print("Insert result (data):", getattr(res, "data", res))
except Exception as e:
    print("Insert failed:", e)
