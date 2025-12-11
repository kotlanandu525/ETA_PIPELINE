# ===========================
# load.py
# ===========================
# Purpose: Load transformed Titanic dataset into Supabase using Supabase client
 
import os
import pandas as pd
from supabase import create_client, Client 
from dotenv import load_dotenv 
 
# Initialize Supabase client
def get_supabase_client():
    """Initialize and return Supabase client."""
    load_dotenv()
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
   
    if not url or not key:
        raise ValueError("❌ Missing SUPABASE_URL or SUPABASE_KEY in .env")
       
    return create_client(url, key)
 
# # ------------------------------------------------------
# # Step 1: Create table if not exists
# # ------------------------------------------------------
def create_table_if_not_exists():
    """
    Ensures the titanic_data table exists in Supabase.
    """
    try:
        supabase = get_supabase_client()
       
        # Try to create the table using raw SQL
        create_table_sql = """
       CREATE TABLE IF NOT EXISTS public.titanic_data (
    id BIGSERIAL PRIMARY KEY,
 
    survived INTEGER,
    pclass INTEGER,
    sex TEXT,
    age DOUBLE PRECISION,
    sibsp INTEGER,
    parch INTEGER,
    fare DOUBLE PRECISION,
    embarked TEXT,
 
    class TEXT,
    who TEXT,
    deck TEXT,
    embark_town TEXT,
 
    alone BOOLEAN,
    family_size INTEGER,
    is_alone BOOLEAN,
    title TEXT
);
 
        """
       
        try:
            # Execute raw SQL to create table
            supabase.rpc('execute_sql', {'query': create_table_sql}).execute()
            print("✅ Table 'titanic_data' created or already exists")
        except Exception as e:
            print(f"ℹ️  Note: {e}")
            print("ℹ️  Table will be created on first insert")
 
    except Exception as e:
        print(f"⚠️  Error checking/creating table: {e}")
        print("ℹ️  Trying to continue with data insertion...")
 
# ------------------------------------------------------
# Step 2: Load CSV data into Supabase table
# ------------------------------------------------------
def load_to_supabase(staged_path: str, table_name: str = "titanic_data"):
    """
    Load a transformed CSV into a Supabase table.
 
    Args:
        staged_path (str): Path to the transformed CSV file.
        table_name (str): Supabase table name. Default is 'titanic_data'.
    """
    # Convert to absolute path
    if not os.path.isabs(staged_path):
        staged_path = os.path.abspath(os.path.join(os.path.dirname(__file__), staged_path))
   
    print(f"🔍 Looking for data file at: {staged_path}")
   
    if not os.path.exists(staged_path):
        print(f"❌ Error: File not found at {staged_path}")
        print("ℹ️  Please run transform.py first to generate the transformed data")
        return
 
    try:
        # Initialize Supabase client
        supabase = get_supabase_client()
       
        # Read the CSV in chunks
        batch_size = 50  # Reduced batch size for better reliability
        df = pd.read_csv(staged_path)
        total_rows = len(df)
       
        print(f"📊 Loading {total_rows} rows into '{table_name}'...")
       
        # Process in batches
        for i in range(0, total_rows, batch_size):
            batch = df.iloc[i:i + batch_size].copy()
            # Convert NaN to None for proper NULL handling
            batch = batch.where(pd.notnull(batch), None)
            records = batch.to_dict('records')
           
            try:
                response = supabase.table(table_name).insert(records).execute()
                if hasattr(response, 'error') and response.error:
                    print(f"⚠️  Error in batch {i//batch_size + 1}: {response.error}")
                else:
                    end = min(i + batch_size, total_rows)
                    print(f"✅ Inserted rows {i+1}-{end} of {total_rows}")
            except Exception as e:
                print(f"⚠️  Error in batch {i//batch_size + 1}: {str(e)}")
                continue
 
        print(f"🎯 Finished loading data into '{table_name}'.")
 
    except Exception as e:
        print(f"❌ Error loading data: {e}")
 
# ------------------------------------------------------
# Step 3: Run as standalone script
# ------------------------------------------------------
if __name__ == "__main__":
    # Path relative to the script location
    staged_csv_path = os.path.join("..", "data", "staged", "titanic_transformed.csv")
    create_table_if_not_exists()  # Ensure table exists
    load_to_supabase(staged_csv_path)
 