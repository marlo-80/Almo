#!/usr/bin/env python3
"""
Importiert die Flugdaten aus CSV-Dateien in PostgreSQL.
- Lädt bei Bedarf von Kaggle herunter (mit unzip)
- Importiert mit COPY (sehr schnell)
- Idempotent: überspringt, wenn Daten bereits vorhanden sind
"""
import os
import subprocess
import sys
import pandas as pd
from sqlalchemy import create_engine, text
import psycopg2
from io import StringIO
import re

DATA_DIR = "/app/flight_data"  # oder /app/flight_data – anpassen!
DB_URI = "postgresql://testuser:testuser@postgres:5432/fastapi_db"

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
    subprocess.run(["unzip", "-o", zip_file, "-d", DATA_DIR], check=True)
    os.remove(zip_file)
    print("✅ Daten bereit.")

import os
import re
from io import StringIO
import pandas as pd
import psycopg2

# Falls diese Variablen nicht global definiert sind, hier anpassen/einkommentieren:
# DB_URI = "postgresql://testuser:testuser@postgres:5432/fastapi_db"
# DATA_DIR = "./flight_data"


def import_csv_to_db():
    """Importiert alle CSV-Dateien performant über eine Staging-Tabelle

    und transformiert die Datentypen am Ende exakt passend für dbt.
    """
    conn = psycopg2.connect(DB_URI)
    conn.autocommit = True
    cur = conn.cursor()

    # Schemas vorbereiten
    cur.execute("CREATE SCHEMA IF NOT EXISTS raw;")
    cur.execute("DROP TABLE IF EXISTS raw.flights CASCADE;")
    cur.execute("DROP TABLE IF EXISTS raw.flights_staging CASCADE;")

    # 1. Staging-Tabelle erstellen (Nimmt alle Text-Formate klaglos auf)
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
        "Year",
        "Quarter",
        "Month",
        "DayofMonth",
        "DayOfWeek",
        "FlightDate",
        "Origin",
        "OriginCityName",
        "OriginState",
        "OriginAirportID",
        "Dest",
        "DestCityName",
        "DestState",
        "DestAirportID",
        "Distance",
        "DistanceGroup",
        "Marketing_Airline_Network",
        "Operating_Airline",
        "Flight_Number_Marketing_Airline",
        "Flight_Number_Operating_Airline",
        "Tail_Number",
        "CRSDepTime",
        "CRSArrTime",
        "CRSElapsedTime",
        "DepTimeBlk",
        "ArrDelay",
        "ArrDelayMinutes",
        "ArrDel15",
        "ArrivalDelayGroups",
        "DepDelay",
        "DepDelayMinutes",
        "Cancelled",
        "Diverted",
    ]

    csv_files = sorted(
        [
            f
            for f in os.listdir(DATA_DIR)
            if re.match(r"Combined_Flights_\d{4}\.csv", f)
        ]
    )
    if not csv_files:
        print("⚠️ Keine passenden CSV-Dateien gefunden.")
        cur.close()
        conn.close()
        return

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
            dtype_backend="pyarrow",
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

    print(
        f"✅ {total_rows} Zeilen erfolgreich im Staging zwischengespeichert."
    )

    # 3. Finale Tabelle erstellen (Jetzt mit echtem BOOLEAN für dbt!)
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
    print(
        "✅ Finale Tabelle raw.flights mit dbt-kompatiblen Datentypen erstellt."
    )

    # 4. Übertragung & Transformation per SQL (Übersetzt 'True'/'False' in echte Booleans)
    print("🔄 Transformiere und kopiere Daten in finale Tabelle...")
    cur.execute("""
        INSERT INTO raw.flights (
            "Year", "Quarter", "Month", "DayofMonth", "DayOfWeek", "FlightDate",
            "Origin", "OriginCityName", "OriginState", "OriginAirportID", "Dest", "DestCityName", "DestState", "DestAirportID",
            "Distance", "DistanceGroup", "Marketing_Airline_Network", "Operating_Airline",
            "Flight_Number_Marketing_Airline", "Flight_Number_Operating_Airline", "Tail_Number",
            "CRSDepTime", "CRSArrTime", "CRSElapsedTime", "DepTimeBlk",
            "ArrDelay", "ArrDelayMinutes", "ArrDel15", "ArrivalDelayGroups", "DepDelay", "DepDelayMinutes",
            "Cancelled", "Diverted"
        )
        SELECT 
            NULLIF("Year", '')::NUMERIC::INT, NULLIF("Quarter", '')::NUMERIC::INT, NULLIF("Month", '')::NUMERIC::INT, NULLIF("DayofMonth", '')::NUMERIC::INT, NULLIF("DayOfWeek", '')::NUMERIC::INT, NULLIF("FlightDate", '')::DATE,
            "Origin", "OriginCityName", "OriginState", NULLIF("OriginAirportID", '')::NUMERIC::INT, "Dest", "DestCityName", "DestState", NULLIF("DestAirportID", '')::NUMERIC::INT,
            NULLIF("Distance", '')::NUMERIC::INT, NULLIF("DistanceGroup", '')::NUMERIC::INT, "Marketing_Airline_Network", "Operating_Airline",
            NULLIF("Flight_Number_Marketing_Airline", '')::NUMERIC::INT, NULLIF("Flight_Number_Operating_Airline", '')::NUMERIC::INT, "Tail_Number",
            NULLIF("CRSDepTime", '')::NUMERIC::INT, NULLIF("CRSArrTime", '')::NUMERIC::INT, NULLIF("CRSElapsedTime", '')::NUMERIC::INT, "DepTimeBlk",
            NULLIF("ArrDelay", '')::FLOAT, NULLIF("ArrDelayMinutes", '')::FLOAT, NULLIF("ArrDel15", '')::FLOAT, NULLIF("ArrivalDelayGroups", '')::FLOAT, NULLIF("DepDelay", '')::FLOAT, NULLIF("DepDelayMinutes", '')::FLOAT,
            CASE WHEN "Cancelled" = 'True' THEN TRUE WHEN "Cancelled" = 'False' THEN FALSE ELSE NULL END,
            CASE WHEN "Diverted" = 'True' THEN TRUE WHEN "Diverted" = 'False' THEN FALSE ELSE NULL END
        FROM raw.flights_staging;
    """)

    # Staging aufräumen
    cur.execute("DROP TABLE IF EXISTS raw.flights_staging;")

    cur.close()
    conn.close()
    print(
        f"🎉 FERTIG! Alle {total_rows} Zeilen sind dbt-kompatibel formatiert in raw.flights."
    )


if __name__ == "__main__":
    print("=== Daten-Import gestartet ===")
    import_csv_to_db()

