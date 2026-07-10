# docker/simulator/simulate_traffic.py
import random, time, os
import httpx
import mlflow
from mlflow.types import DataType
from sqlalchemy import create_engine, text
import pandas as pd
import numpy as np

API_URL = os.environ.get("API_URL", "http://api:8000/predict")
SLEEP_SEC = float(os.environ.get("SLEEP_SEC", 2.0))
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
DB_URI = "postgresql://testuser:testuser@postgres:5432/fastapi_db"
DB_SOURCE_TABLE = os.environ.get("DB_SOURCE_TABLE", "dbt_staging.intra_covid_1M")

# Zwei Modelle aus zentraler Konfiguration
from flows.config import API_MODELS

MODEL_REG_NAME  = os.environ.get("MODEL_NAME_REG", API_MODELS["regression"]["model_name"])
MODEL_REG_ALIAS = os.environ.get("MODEL_ALIAS_REG", API_MODELS["regression"]["alias"])
MODEL_CLS_NAME  = os.environ.get("MODEL_NAME_CLS", API_MODELS["classification"]["model_name"])
MODEL_CLS_ALIAS = os.environ.get("MODEL_ALIAS_CLS", API_MODELS["classification"]["alias"])


def enforce_schema(df: pd.DataFrame, signature) -> pd.DataFrame:
    """Konvertiert alle Spalten eines DataFrame exakt in die von der MLflow-Signatur verlangten Typen."""
    df = df.copy()
    for col in signature.inputs.inputs:
        name = col.name
        if name not in df.columns:
            continue
        dtype = col.type
        if dtype in (DataType.double, DataType.float):
            df[name] = pd.to_numeric(df[name], errors='coerce').astype('float64')
        elif dtype == DataType.string:
            df[name] = df[name].astype(str)
        elif dtype in (DataType.long, DataType.integer):
            df[name] = pd.to_numeric(df[name], errors='coerce').astype('int64')
    return df


def main():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    # Beide Modelle laden
    reg_uri = f"models:/{MODEL_REG_NAME}@{MODEL_REG_ALIAS}"
    cls_uri = f"models:/{MODEL_CLS_NAME}@{MODEL_CLS_ALIAS}"
    print(f"Lade Regressor : {reg_uri}")
    print(f"Lade Classifier: {cls_uri}")
    reg_model = mlflow.pyfunc.load_model(reg_uri)
    cls_model = mlflow.pyfunc.load_model(cls_uri)

    # Features aus beiden Signaturen vereinigen
    reg_cols = [col.name for col in reg_model.metadata.signature.inputs.inputs]
    cls_cols = [col.name for col in cls_model.metadata.signature.inputs.inputs]
    all_feature_cols = sorted(set(reg_cols + cls_cols))
    print(f"Vereinigte Features: {all_feature_cols}")

    engine = create_engine(DB_URI)
    # Alle benötigten Spalten + flight_uid + Ground-Truth laden
    db_cols = ", ".join(set(all_feature_cols) | {"flight_uid", "arr_delay_minutes", "arr_del15"})
    query = f"SELECT {db_cols} FROM {DB_SOURCE_TABLE}"
    with engine.connect() as conn:
        rows = conn.execute(text(query)).fetchall()
    samples = [dict(row._mapping) for row in rows]
    print(f"Samples geladen: {len(samples)}")

    while True:
        sample = random.choice(samples)
        # DataFrames für beide Modelle bauen und Typen erzwingen
        df_reg = pd.DataFrame([sample])
        df_reg = enforce_schema(df_reg[reg_cols], reg_model.metadata.signature)
        df_cls = pd.DataFrame([sample])
        df_cls = enforce_schema(df_cls[cls_cols], cls_model.metadata.signature)

        # Payload mit allen Features aus der Vereinigungsmenge
        payload = {}
        # Wir nehmen die Werte aus df_reg oder df_cls, beide sind nach Schema korrekt
        for col in all_feature_cols:
            if col in df_reg.columns:
                val = df_reg[col].iloc[0]
            elif col in df_cls.columns:
                val = df_cls[col].iloc[0]
            else:
                val = None
            # Typ wandeln in native Python-Typen für JSON
            if isinstance(val, (np.integer,)):
                val = int(val)
            elif isinstance(val, (np.floating,)):
                val = float(val)
            elif isinstance(val, np.bool_):
                val = bool(val)
            elif isinstance(val, np.ndarray):
                val = val.tolist()
            payload[col] = val

        payload["flight_uid"] = sample.get("flight_uid")
        payload["ground_truth"] = {
            "arr_delay_minutes": float(sample.get("arr_delay_minutes", 0.0)),
            "arr_del15": int(sample.get("arr_del15", 0))
        }

        try:
            r = httpx.post(API_URL, json=payload, timeout=10)
            print(f"Status {r.status_code}, Predict: {r.json()}")
        except Exception as e:
            print(f"Fehler: {e}")
        time.sleep(SLEEP_SEC)


if __name__ == "__main__":
    main()