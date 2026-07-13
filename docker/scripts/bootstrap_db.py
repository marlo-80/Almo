#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Bootstrap – Database Initialization & Data Import

This script initializes the PostgreSQL database for the flight delay prediction
system. It creates the required schemas and tables, then imports the raw flight
data from CSV files using a high-performance COPY-based approach.

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
5. Load data from CSV files via staging table + COPY
6. Transform and insert into final `raw.flights` table
7. Create index on `FlightDate` for faster queries
8. Delete the CSV files after import to free disk space
9. Generate Grafana API token for the demo

------------------------------------------------------------------------------
Dependencies
------------------------------------------------------------------------------
- PostgreSQL database (fastapi_db)
- Kaggle-CLI for dataset download (handled by `ensure_data_downloaded`)
- Environment variables: POSTGRES_USER, POSTGRES_PASSWORD
- Python packages: pandas, psycopg2, sqlalchemy

------------------------------------------------------------------------------
Usage
------------------------------------------------------------------------------
    python docker/scripts/bootstrap_db.py

The script is typically invoked via Docker Compose:
    docker compose --profile init up bootstrap
"""

import sys
import os
import subprocess
import glob
import re
import time
from io import StringIO

import pandas as pd
import psycopg2
from sqlalchemy import create_engine, text

# --------------------------------------------------------------------------
# CONFIGURATION
# --------------------------------------------------------------------------
DB_USER = os.environ.get("POSTGRES_USER", "testuser")
DB_PASS = os.environ.get("POSTGRES_PASSWORD", "testuser")
DB_HOST = "postgres"
DB_PORT = "5432"
DB_NAME = "fastapi_db"
DB_URI = f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
DATA_DIR = "/app/flight_data"


# --------------------------------------------------------------------------
# DATA DOWNLOAD (Kaggle)
# --------------------------------------------------------------------------
def ensure_data_downloaded():
    """Lädt und entpackt die Kaggle-Daten, falls sie nicht existieren."""
    os.makedirs(DATA_DIR, exist_ok=True)

    # Prüfen, ob bereits CSVs vorhanden sind
    csv_files = [f for f in os.listdir(DATA_DIR) if f.startswith("Combined_Flights_") and f.endswith(".csv")]
    if csv_files:
        print(f"✅ CSVs bereits vorhanden ({len(csv_files)} Dateien). Überspringe Download.")
        return

    # Kaggle-CLI installieren (falls nicht vorhanden)
    try:
        subprocess.run(["kaggle", "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("📦 Installiere Kaggle-CLI...")
        subprocess.run([sys.executable, "-m", "pip", "install", "kaggle"], check=True)

    # Download
    print("📥 Lade Dataset von Kaggle herunter...")
    subprocess.run([
        "kaggle", "datasets", "download", "-d",
        "robikscube/flight-delay-dataset-20182022", "-p", DATA_DIR
    ], check=True)

    # Entpacken
    zip_file = None
    for f in os.listdir(DATA_DIR):
        if f.endswith(".zip"):
            zip_file = os.path.join(DATA_DIR, f)
            break
    if not zip_file:
        raise RuntimeError("ZIP-Datei nicht gefunden.")

    print(f"📂 Entpacke {zip_file}...")
    #subprocess.run(["unzip", "-o", zip_file, "-d", DATA_DIR], check=True)
    subprocess.run(["unzip", "-o", zip_file, "*.csv", "-x", "*/*", "-d", DATA_DIR], check=True)
    os.remove(zip_file)
    print("✅ Daten bereit.")


# --------------------------------------------------------------------------
# DATA IMPORT (COPY-based, high performance)
# --------------------------------------------------------------------------
def import_csv_to_db() -> int:
    """
    Importiert alle CSV-Dateien performant über eine Staging-Tabelle.

    Returns:
        int: Anzahl der importierten Zeilen.
    """
    ensure_data_downloaded()

    conn = psycopg2.connect(DB_URI)
    conn.autocommit = True
    cur = conn.cursor()

    # Schemas vorbereiten
    cur.execute("CREATE SCHEMA IF NOT EXISTS raw;")
    cur.execute("DROP TABLE IF EXISTS raw.flights CASCADE;")
    cur.execute("DROP TABLE IF EXISTS raw.flights_staging CASCADE;")

    # 1. Staging-Tabelle erstellen (nimmt alle Text-Formate auf)
    cur.execute("""
        CREATE TABLE raw.flights_staging (
            "Year" TEXT, "Quarter" TEXT, "Month" TEXT, "DayofMonth" TEXT, "DayOfWeek" TEXT, "FlightDate" TEXT,
            "Origin" TEXT, "OriginCityName" TEXT, "OriginState" TEXT, "OriginAirportID" TEXT,
            "Dest" TEXT, "DestCityName" TEXT, "DestState" TEXT, "DestAirportID" TEXT,
            "Distance" TEXT, "DistanceGroup" TEXT,
            "Marketing_Airline_Network" TEXT, "Operating_Airline" TEXT,
            "Flight_Number_Marketing_Airline" TEXT, "Flight_Number_Operating_Airline" TEXT, "Tail_Number" TEXT,
            "CRSDepTime" TEXT, "CRSArrTime" TEXT, "CRSElapsedTime" TEXT,
            "DepTimeBlk" TEXT,
            "ArrDelay" TEXT, "ArrDelayMinutes" TEXT, "ArrDel15" TEXT, "ArrivalDelayGroups" TEXT,
            "DepDelay" TEXT, "DepDelayMinutes" TEXT,
            "Cancelled" TEXT, "Diverted" TEXT
        );
    """)
    print("✅ Staging-Tabelle erstellt.")

    target_columns = [
        "Year", "Quarter", "Month", "DayofMonth", "DayOfWeek", "FlightDate",
        "Origin", "OriginCityName", "OriginState", "OriginAirportID",
        "Dest", "DestCityName", "DestState", "DestAirportID",
        "Distance", "DistanceGroup",
        "Marketing_Airline_Network", "Operating_Airline",
        "Flight_Number_Marketing_Airline", "Flight_Number_Operating_Airline", "Tail_Number",
        "CRSDepTime", "CRSArrTime", "CRSElapsedTime", "DepTimeBlk",
        "ArrDelay", "ArrDelayMinutes", "ArrDel15", "ArrivalDelayGroups",
        "DepDelay", "DepDelayMinutes",
        "Cancelled", "Diverted"
    ]

    csv_files = sorted([
        f for f in os.listdir(DATA_DIR)
        if re.match(r"Combined_Flights_\d{4}\.csv", f)
    ])
    if not csv_files:
        print("⚠️ Keine passenden CSV-Dateien gefunden.")
        cur.close()
        conn.close()
        return 0

    # 2. Daten in die Staging-Tabelle streamen
    total_rows = 0
    for file in csv_files:
        print(f"📄 Streame {file} in Staging...")
        file_path = os.path.join(DATA_DIR, file)

        chunk_iter = pd.read_csv(
            file_path,
            chunksize=150000,
            usecols=target_columns,
            engine="c",
            dtype_backend="pyarrow"
        )

        for chunk in chunk_iter:
            output = StringIO()
            chunk.to_csv(output, index=False, header=False, na_rep="")
            output.seek(0)

            current_columns_str = ", ".join([f'"{col}"' for col in chunk.columns])
            sql = f"""
                COPY raw.flights_staging ({current_columns_str})
                FROM STDIN
                WITH (FORMAT CSV, HEADER FALSE, NULL '');
            """
            cur.copy_expert(sql, output)
            total_rows += len(chunk)

    print(f"✅ {total_rows} Zeilen erfolgreich im Staging zwischengespeichert.")

    # 3. Finale Tabelle mit dbt-kompatiblen Datentypen
    cur.execute("""
        CREATE TABLE raw.flights (
            "Year" INT, "Quarter" INT, "Month" INT, "DayofMonth" INT, "DayOfWeek" INT, "FlightDate" DATE,
            "Origin" VARCHAR(10), "OriginCityName" VARCHAR(100), "OriginState" VARCHAR(2), "OriginAirportID" INT,
            "Dest" VARCHAR(10), "DestCityName" VARCHAR(100), "DestState" VARCHAR(2), "DestAirportID" INT,
            "Distance" INT, "DistanceGroup" INT,
            "Marketing_Airline_Network" VARCHAR(10), "Operating_Airline" VARCHAR(10),
            "Flight_Number_Marketing_Airline" INT, "Flight_Number_Operating_Airline" INT, "Tail_Number" VARCHAR(20),
            "CRSDepTime" INT, "CRSArrTime" INT, "CRSElapsedTime" INT,
            "DepTimeBlk" VARCHAR(10),
            "ArrDelay" FLOAT, "ArrDelayMinutes" FLOAT, "ArrDel15" FLOAT, "ArrivalDelayGroups" FLOAT,
            "DepDelay" FLOAT, "DepDelayMinutes" FLOAT,
            "Cancelled" BOOLEAN, "Diverted" BOOLEAN
        );
    """)
    print("✅ Finale Tabelle raw.flights mit dbt-kompatiblen Datentypen erstellt.")

    # 4. Transformation und Übertragung
    print("🔄 Transformiere und kopiere Daten in finale Tabelle...")
    cur.execute("""
        INSERT INTO raw.flights (
            "Year", "Quarter", "Month", "DayofMonth", "DayOfWeek", "FlightDate",
            "Origin", "OriginCityName", "OriginState", "OriginAirportID",
            "Dest", "DestCityName", "DestState", "DestAirportID",
            "Distance", "DistanceGroup",
            "Marketing_Airline_Network", "Operating_Airline",
            "Flight_Number_Marketing_Airline", "Flight_Number_Operating_Airline", "Tail_Number",
            "CRSDepTime", "CRSArrTime", "CRSElapsedTime", "DepTimeBlk",
            "ArrDelay", "ArrDelayMinutes", "ArrDel15", "ArrivalDelayGroups",
            "DepDelay", "DepDelayMinutes",
            "Cancelled", "Diverted"
        )
        SELECT
            NULLIF("Year", '')::NUMERIC::INT, NULLIF("Quarter", '')::NUMERIC::INT,
            NULLIF("Month", '')::NUMERIC::INT, NULLIF("DayofMonth", '')::NUMERIC::INT,
            NULLIF("DayOfWeek", '')::NUMERIC::INT, NULLIF("FlightDate", '')::DATE,
            "Origin", "OriginCityName", "OriginState",
            NULLIF("OriginAirportID", '')::NUMERIC::INT,
            "Dest", "DestCityName", "DestState",
            NULLIF("DestAirportID", '')::NUMERIC::INT,
            NULLIF("Distance", '')::NUMERIC::INT, NULLIF("DistanceGroup", '')::NUMERIC::INT,
            "Marketing_Airline_Network", "Operating_Airline",
            NULLIF("Flight_Number_Marketing_Airline", '')::NUMERIC::INT,
            NULLIF("Flight_Number_Operating_Airline", '')::NUMERIC::INT,
            "Tail_Number",
            NULLIF("CRSDepTime", '')::NUMERIC::INT, NULLIF("CRSArrTime", '')::NUMERIC::INT,
            NULLIF("CRSElapsedTime", '')::NUMERIC::INT, "DepTimeBlk",
            NULLIF("ArrDelay", '')::FLOAT, NULLIF("ArrDelayMinutes", '')::FLOAT,
            NULLIF("ArrDel15", '')::FLOAT, NULLIF("ArrivalDelayGroups", '')::FLOAT,
            NULLIF("DepDelay", '')::FLOAT, NULLIF("DepDelayMinutes", '')::FLOAT,
            CASE WHEN "Cancelled" = 'True' THEN TRUE WHEN "Cancelled" = 'False' THEN FALSE ELSE NULL END,
            CASE WHEN "Diverted" = 'True' THEN TRUE WHEN "Diverted" = 'False' THEN FALSE ELSE NULL END
        FROM raw.flights_staging;
    """)

    # Staging aufräumen
    cur.execute("DROP TABLE IF EXISTS raw.flights_staging;")
    cur.close()
    conn.close()

    print(f"🎉 FERTIG! {total_rows} Zeilen in raw.flights importiert.")
    return total_rows


# --------------------------------------------------------------------------
# GRAFANA TOKEN GENERATION
# --------------------------------------------------------------------------
def grafana_token_generation():
    """Generiert den Grafana API-Token (einmalig)."""
    print("Generating Grafana API token...")
    token_script = "/app/docker/scripts/grafana_token_generation.sh"
    if os.path.exists(token_script):
        try:
            result = subprocess.run(
                ["bash", token_script],
                capture_output=True,
                text=True,
                timeout=120
            )
            if result.returncode == 0:
                print("✅ Grafana token generated successfully.")
            else:
                print(f"⚠️ Token generation failed (exit {result.returncode}):")
                print(result.stderr)
        except Exception as e:
            print(f"⚠️ Token generation error: {e}")
    else:
        print(f"⚠️ Token script not found: {token_script}")


# --------------------------------------------------------------------------
# MAIN BOOTSTRAP
# --------------------------------------------------------------------------
def bootstrap():
    """Hauptfunktion: Initialisiert die Datenbank und importiert die Daten."""
    engine = create_engine(DB_URI)

    # 1. Schema api und Tabelle api.predictions erstellen
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

    # 2. Prüfen, ob raw.flights bereits Daten enthält
    with engine.connect() as conn:
        result = conn.execute(text(
            "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_schema = 'raw' AND table_name = 'flights')"
        ))
        table_exists = result.scalar()

    if table_exists:
        with engine.connect() as conn:
            count = conn.execute(text("SELECT COUNT(*) FROM raw.flights")).scalar()
            if count > 0:
                print(f"✅ Tabelle raw.flights enthält bereits {count} Zeilen – Import übersprungen.")
                return

    # 3. Daten importieren
    total_rows = import_csv_to_db()

    # 4. Index erstellen (falls Daten importiert wurden)
    if total_rows > 0:
        with engine.connect() as conn:
            conn.execute(text('CREATE INDEX IF NOT EXISTS idx_flight_date ON raw.flights ("FlightDate");'))
            conn.commit()
        print("✅ Index auf FlightDate erstellt.")

        # 5. Grafana-Token generieren
        grafana_token_generation()

    # 6. CSV-Dateien löschen
    csv_files = glob.glob(os.path.join("./flight_data", "*.csv"))
    for f in csv_files:
        try:
            os.remove(f)
            print(f"🗑️  Gelöscht: {f}")
        except Exception as e:
            print(f"⚠️ Konnte {f} nicht löschen: {e}")

    print(f"✅ Bootstrap abgeschlossen. {total_rows} Zeilen in raw.flights.")


if __name__ == "__main__":
    bootstrap()