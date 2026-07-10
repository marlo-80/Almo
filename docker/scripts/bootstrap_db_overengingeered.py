# ./docker/scripts/bootstrap_db.py
import sys, os
import io
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

    # Ersetze die alte Schleife in bootstrap_db.py durch:
    len_all = 0
    first = True
    
    # Rohe DBAPI-Verbindung von SQLAlchemy abgreifen, um SQLAlchemy zu umgehen
    with engine.connect() as conn:
        dbapi_conn = conn.connection.dbapi_connection if hasattr(conn.connection, 'dbapi_connection') else conn.connection
        
        for df in generator:
            len_all += len(df)
            print(f"Writing Data Frame chunk to PostgreSQL...")
            
            if first:
                # Nur beim ersten Mal darf Pandas die Tabelle via SQLAlchemy erstellen
                df.head(0).to_sql("flights", engine, schema="raw", if_exists="replace", index=False)
                first = False
            
            # Ab hier: Direkter, nativer Treiberzugriff ohne SQLAlchemy-Metadaten-Overhead
            with dbapi_conn.cursor() as cur:
                # DataFrame superschnell im RAM als CSV-String formatieren
                s_buf = io.StringIO()
                df.to_csv(s_buf, header=False, index=False, sep='\t', na_rep='\\N')
                s_buf.seek(0)
                
                # Nutzt den echten, ungedrosselten C-Befehl des Treibers
                cur.copy_from(s_buf, "raw.flights", sep='\t', null='\\N', columns=[f'"{c}"' for c in df.columns])
                
        dbapi_conn.commit()  # Erst ganz am Ende einmal alles auf die NVMe schreiben



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