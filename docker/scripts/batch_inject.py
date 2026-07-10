# docker/scripts/batch_inject.py
import sys
import pandas as pd
import numpy as np
import mlflow
from mlflow.types import DataType
from sqlalchemy import create_engine, text
import json

# --- Parameter -------------------------------------------------------------
if len(sys.argv) < 4 or len(sys.argv) > 5:
    print("Usage: batch_inject.py <start_date> <end_date> <approx_rows> [source_table]")
    sys.exit(1)

start_date   = sys.argv[1]
end_date     = sys.argv[2]
approx_rows  = int(sys.argv[3])
SOURCE_TABLE = sys.argv[4] if len(sys.argv) == 5 else "dbt_staging.intra_covid_100k"

print(f"Lade exakt {approx_rows} Zeilen aus {SOURCE_TABLE} (Zeitraum {start_date} – {end_date}) ...")

# --- Konfiguration ---------------------------------------------------------
MLFLOW_URI = "http://mlflow:5000"
DB_URI = "postgresql://testuser:testuser@postgres:5432/fastapi_db"

# --- Modelle laden ---------------------------------------------------------
mlflow.set_tracking_uri(MLFLOW_URI)
reg = mlflow.pyfunc.load_model("models:/regressor@champion")
cls = mlflow.pyfunc.load_model("models:/classifier@champion")

# --- Daten laden -----------------------------------------------------------
engine = create_engine(DB_URI)
with engine.connect() as conn:
    conn.execute(text("SELECT setseed(0.123456789)"))

query = f"""
    SELECT *
    FROM {SOURCE_TABLE}
    WHERE flight_date >= '{start_date}'
      AND flight_date <  '{end_date}'
    ORDER BY random()
    LIMIT {approx_rows}
"""
df = pd.read_sql(query, engine)
if len(df) == 0:
    print(f"Keine Zeilen für den Zeitraum {start_date} – {end_date} gefunden. Überspringe Batch.")
    sys.exit(0)

print(f"Geladene Zeilen: {len(df)}")

# --- Ground‑Truth sichern ---------------------------------------------------
true_reg   = df["arr_delay_minutes"]
true_class = df["arr_del15"]
uids = df["flight_uid"].copy()

# --- Feature‑Spalten dynamisch aus den Signaturen holen --------------------
def get_feature_columns(signature):
    return [col.name for col in signature.inputs.inputs]

feature_cols_reg = get_feature_columns(reg.metadata.signature)
feature_cols_cls = get_feature_columns(cls.metadata.signature)

print(f"Features Regressor : {feature_cols_reg}")
print(f"Features Classifier: {feature_cols_cls}")

# Alle jemals benötigten Features für das Logging vereinigen
union_features = sorted(set(feature_cols_reg) | set(feature_cols_cls))

# DataFrames bauen
features_all = df[union_features]          # für das Logging
features_reg = df[feature_cols_reg]        # nur Regressor‑Features
features_cls = df[feature_cols_cls]        # nur Classifier‑Features

# --- Hilfsfunktion: Schema-Enforcement -------------------------------------
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

features_reg = enforce_schema(features_reg, reg.metadata.signature)
features_cls = enforce_schema(features_cls, cls.metadata.signature)

# --- Batch‑Predictions -----------------------------------------------------
print("Führe Batch‑Predictions durch ...")
reg_preds = reg.predict(features_reg)
cls_preds = cls.predict(features_cls)

# --- In Datenbank schreiben ------------------------------------------------
print("Schreibe in api.predictions ...")
with engine.connect() as conn:
    for i, row_all in features_all.iterrows():
        uid = uids[i] if i in uids.index else None
        gt_json = json.dumps({
            "arr_delay_minutes": float(true_reg[i]),
            "arr_del15": int(true_class[i])
        })
        conn.execute(
            text("""
                INSERT INTO api.predictions
                    (flight_uid, input_features, prediction_reg, prediction_class,
                     model_version_reg, model_version_class, ground_truth)
                VALUES (:uid, :feat, :reg, :cls, 'regressor@champion', 'classifier@champion', :gt)
            """),
            {
                "uid": uid,
                "feat": json.dumps(row_all.to_dict()),
                "reg": float(reg_preds[i]),
                "cls": int(cls_preds[i]),
                "gt": gt_json,
            }
        )
    conn.commit()

print(f"{len(features_all)} Predictions in api.predictions geschrieben.")