import os
import seaborn as sns # type: ignore
import pandas as pd # type: ignore
 
def extract_data():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # go up one level
    data_dir = os.path.join(base_dir, "data", "raw")
    os.makedirs(data_dir, exist_ok=True)
 
    df = sns.load_dataset("titanic")
    raw_path = os.path.join(data_dir, "titanic_raw.csv")
    df.to_csv(raw_path, index=False)
 
    print(f"✅ Data extracted and saved at: {raw_path}")
    return raw_path
 
if __name__ == "__main__":
    extract_data()