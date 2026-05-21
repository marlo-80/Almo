import sys, os
# Das Projekt-Root ist das übergeordnete Verzeichnis des Skripts (docker/scripts/ → repo/)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


# docker/scripts/bootstrap_db.py
from sqlalchemy import create_engine, text
import os

DB_USER = os.environ.get("POSTGRES_USER", "vikmar")
DB_PASS = os.environ.get("POSTGRES_PASSWORD", "vikmar")
DB_HOST = "postgres"
DB_PORT = "5432"
DB_NAME = "fastapi_db"

def bootstrap():
    engine = create_engine(f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

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
    df = load_from_local()
    
    print(f"Schreibe {len(df)} Zeilen nach PostgreSQL …")
    df.to_sql("flights", engine, schema="raw", if_exists="replace", index=False, chunksize=5000)

    with engine.connect() as conn:
        conn.execute(text('CREATE INDEX IF NOT EXISTS idx_flight_date ON raw.flights ("FlightDate");'))
        conn.commit()
    print(f"Import abgeschlossen. {len(df)} Zeilen in raw.flights eingefügt.")

if __name__ == "__main__":
    bootstrap()