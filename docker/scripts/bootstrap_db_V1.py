# ./docker/scripts/bootstrap_db.py
"""
Bootstrap – Database Initialization & Data Import

This script initializes the PostgreSQL database for the flight delay prediction
system. It creates the required schemas and tables, then imports the raw flight
data from CSV files (downloaded from Kaggle via `src.data.load_from_local`).

The import is idempotent: if the `raw.flights` table already contains data,
the script skips the import and exits gracefully.

------------------------------------------------------------------------------
Workflow
------------------------------------------------------------------------------
1. Connect to PostgreSQL (fastapi_db)
2. Create schemas: `api`, `raw`
3. Create `api.predictions` table for storing prediction results
4. Check if `raw.flights` already exists and contains rows
   → If yes: skip import
   → If no: proceed with import
5. Load data from CSV files via `load_from_local()` generator
6. Write data in chunks (chunksize=50000) to `raw.flights`
7. Create index on `FlightDate` for faster queries
8. Delete the CSV files after import to free disk space

------------------------------------------------------------------------------
Dependencies
------------------------------------------------------------------------------
- PostgreSQL database (fastapi_db)
- KaggleHub for dataset download (handled by `src.data`)
- Environment variables: POSTGRES_USER, POSTGRES_PASSWORD

------------------------------------------------------------------------------
Usage
------------------------------------------------------------------------------
    python docker/scripts/bootstrap_db.py

The script is typically invoked via Docker Compose:
    docker compose --profile init up bootstrap

------------------------------------------------------------------------------
Notes
------------------------------------------------------------------------------
- The script is idempotent – safe to run multiple times.
- CSV files are deleted after successful import to conserve space.
- The index on `FlightDate` is created only if data was imported.
"""

import sys, os
import subprocess 
# The project root is the parent directory of the script (docker/scripts/ → repo/)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from sqlalchemy import create_engine, text
import os
import glob

# --------------------------------------------------------------------------
# CONFIGURATION
# --------------------------------------------------------------------------
DB_USER = os.environ.get("POSTGRES_USER", "testuser")
DB_PASS = os.environ.get("POSTGRES_PASSWORD", "testuser")
DB_HOST = "postgres"
DB_PORT = "5432"
DB_NAME = "fastapi_db"
DATA_DIR = "/app/flight_data"  # oder /app/flight_data – anpassen!
DB_URI = "postgresql://testuser:testuser@postgres:5432/fastapi_db"

def bootstrap():
    engine = create_engine(f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}")
# --------------------------------------------------------------------------
# CREATE API SCHEMA AND PREDICTIONS TABLE
# --------------------------------------------------------------------------
    with engine.connect() as conn:
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS api;"))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS api.predictions (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMPTZ DEFAULT NOW(),
                flight_uid TEXT,
                input_features JSONB,
                prediction_reg DOUBLE PRECISION,
                prediction_class INTEGER,
                model_version_reg TEXT,
                model_version_class TEXT,
                ground_truth JSONB DEFAULT NULL,
                prediction_class_proba DOUBLE PRECISION DEFAULT NULL
            );
        """))
        conn.commit()

# --------------------------------------------------------------------------
# CREATE RAW SCHEMA
# --------------------------------------------------------------------------
    with engine.connect() as conn:
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS raw;"))
        conn.commit()

# --------------------------------------------------------------------------
# CHECK EXISTING DATA
# --------------------------------------------------------------------------
    table_exists = False
    with engine.connect() as conn:
        result = conn.execute(text(
            "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_schema = 'raw' AND table_name = 'flights')"
        ))
        table_exists = result.scalar()
    if table_exists:
        with engine.connect() as conn:
            count = conn.execute(text("SELECT COUNT(*) FROM raw.flights")).scalar()
            if count > 0:
                print(f"Tabelle raw.flights enthält bereits {count} Zeilen – Import übersprungen.")
                return

# --------------------------------------------------------------------------
# DATA IMPORT
# --------------------------------------------------------------------------
    # If not, load and import data
    print("Importiere Daten aus CSV-Dateien ...")
    from src.data import load_from_local
    from src.data import load_from_kaggle
    generator = load_from_local()

    len_all = 0
    first = True
    for df in generator:
        len_all += len(df)
        print(f"Writing  Data Frame to PostgreSQL.")
        if first:
            df.to_sql("flights", engine, schema="raw", if_exists="replace", index=False, chunksize=50000)
            print(f"{len_all} Rows have been written to PostgreSQL.")
            first = False
        else:
            df.to_sql("flights", engine, schema="raw", if_exists="append", index=False, chunksize=50000)
            print(f"{len_all} Rows have been written to PostgreSQL.")


# --------------------------------------------------------------------------
# CREATE INDEX AND CLEAN UP CSV FILES
# --------------------------------------------------------------------------
    if len_all > 0:
        with engine.connect() as conn:
            conn.execute(text('CREATE INDEX IF NOT EXISTS idx_flight_date ON raw.flights ("FlightDate");'))
            conn.commit()
    else:
        print("No data imported – skipping index creation.")

    csv_files = glob.glob(os.path.join("./flight_data", "*.csv"))
    for f in csv_files:
        try:
            os.remove(f)
            print(f"Deleted: {f}")
        except Exception as e:
            print(f"Could not delete {f}: {e}")

# --------------------------------------------------------------------------
# GRAFANA TOKEN GENERATION
# --------------------------------------------------------------------------
    if len_all > 0:
        print("Generating Grafana API token...")
        token_script = "/app/docker/scripts/generate_grafana_token.sh"
        if os.path.exists(token_script):
            try:
                import subprocess
                result = subprocess.run(
                    ["bash", token_script],
                    capture_output=True,
                    text=True,
                    timeout=120
                )
                if result.returncode == 0:
                    print("✅ Grafana token generated successfully.")
                else:
                    print(f"Token generation failed (exit {result.returncode}):")
                    print(result.stderr)
            except Exception as e:
                print(f"Token generation error: {e}")
        else:
            print(f"Token script not found: {token_script}")                  
        
    print(f"Import finished. {len_all} rows have been added to raw.flights.")

if __name__ == "__main__":
    bootstrap()