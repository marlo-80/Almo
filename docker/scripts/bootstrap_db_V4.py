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
import io
import math

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
    print("Prüfe lokale CSV-Dateien und lade ggf. von Kaggle...")
    from src.data import load_from_local
    generator = load_from_local()

    # Holt das erste Element, um die Struktur via Pandas zu erzeugen
    try:
        first_df = next(generator)
        with engine.connect() as conn:
            first_df.head(0).to_sql("flights", conn, schema="raw", if_exists="replace", index=False)
            conn.commit()
        print("Tabellenstruktur 'raw.flights' wurde erfolgreich vorbereitet.")
    except StopIteration:
        print("Keine CSV-Dateien zum Importieren gefunden.")

    print("Tabellenstruktur erfolgreich vorbereitet. Daten-Download abgeschlossen.")

if __name__ == "__main__":
    bootstrap()
