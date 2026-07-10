# ./docker/scripts/bootstrap_db.py
import sys, os
# Das Projekt-Root ist das übergeordnete Verzeichnis des Skripts (docker/scripts/ → repo/)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


# docker/scripts/bootstrap_db.py
from sqlalchemy import create_engine, text
import os
import glob

DB_USER = os.environ.get("POSTGRES_USER", "testuser")
DB_PASS = os.environ.get("POSTGRES_PASSWORD", "testuser")
DB_HOST = "postgres"
DB_PORT = "5432"
DB_NAME = "fastapi_db"

def bootstrap():
    engine = create_engine(f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}")
    # Create Schema for prediction logging
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

    # Schema raw anlegen, falls nicht vorhanden
    with engine.connect() as conn:
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS raw;"))
        conn.commit()

    # Prüfen, ob Tabelle raw.flights existiert und Zeilen enthält
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

    # Falls nicht, Daten laden und importieren
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
        
    print(f"Import finished. {len_all} rows have been added to raw.flights.")

if __name__ == "__main__":
    bootstrap()