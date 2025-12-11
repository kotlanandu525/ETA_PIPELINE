# run_pipeline.py
import time
from extract import extract_weather_data
from transform import transform_data
import load_data as load
from etl_analysis import run_analysis

def run_full_pipeline():
    # 1) Extract
    raw_file = extract_weather_data()
    time.sleep(1)

    # 2) Transform
    staged_csv = transform_data([raw_file])

    # 3) Load
    load.create_table_if_not_exists()
    load.load_to_supabase(staged_csv, batch_size=100)

    # 4) Analysis
    run_analysis()

if __name__ == "__main__":
    run_full_pipeline()
