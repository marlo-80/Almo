"""
Batch Injection – Bulk Prediction Generator

This script loads flight data from a PostgreSQL source table, applies the
champion models (regressor and classifier) registered in MLflow, and writes
the predictions to the `api.predictions` table.

The injected data serves as the foundation for drift detection (Evidently)
and continuous model monitoring in Grafana.

------------------------------------------------------------------------------
Usage
------------------------------------------------------------------------------
python batch_inject.py <start_date> <end_date> <approx_rows> [source_table]

Arguments:
    start_date      : Start of the time range (YYYY-MM-DD)
    end_date        : End of the time range (YYYY-MM-DD)
    approx_rows     : Number of rows to load (LIMIT)
    source_table    : (optional) Source table name.
                      Default: dbt_staging.intra_covid_100k

Example:
    python batch_inject.py 2020-01-01 2020-02-01 5000 dbt_staging.intra_covid_100k

------------------------------------------------------------------------------
Workflow
------------------------------------------------------------------------------
1. Connect to PostgreSQL database (fastapi_db)
2. Load MLflow models: 'regressor@champion' and 'classifier@champion'
3. Sample random rows from the source table (using setseed for reproducibility)
4. Extract feature columns from MLflow model signatures
5. Apply models to the loaded data
6. Store results in api.predictions:
   - flight_uid
   - input_features (JSONB)
   - prediction_reg (DOUBLE PRECISION)
   - prediction_class (INTEGER)
   - model_version_reg / model_version_class
   - ground_truth (JSONB)

------------------------------------------------------------------------------
Dependencies
------------------------------------------------------------------------------
- MLflow tracking server at http://mlflow:5000
- PostgreSQL database (fastapi_db)
- Registered models: regressor@champion, classifier@champion

------------------------------------------------------------------------------
Notes
------------------------------------------------------------------------------
- The script is deterministic: the random seed is fixed.
- If no rows exist for the given time range, the script exits gracefully.
- The `api.predictions` table must exist prior to execution.
"""

import sys
import pandas as pd
import numpy as np
import mlflow
from mlflow.types import DataType
from sqlalchemy import create_engine, text
import json


# --------------------------------------------------------------------------
# COMMAND-LINE ARGUMENTS
# --------------------------------------------------------------------------
if len(sys.argv) < 4 or len(sys.argv) > 5:
    print("Usage: batch_inject.py <start_date> <end_date> <approx_rows> [source_table]")
    sys.exit(1)

start_date   = sys.argv[1]
end_date     = sys.argv[2]
approx_rows  = int(sys.argv[3])
SOURCE_TABLE = sys.argv[4] if len(sys.argv) == 5 else "dbt_staging.intra_covid_100k"

print(f"Lade exakt {approx_rows} Zeilen aus {SOURCE_TABLE} (Zeitraum {start_date} – {end_date}) ...")

# --------------------------------------------------------------------------
# CONFIGURATION
# --------------------------------------------------------------------------
MLFLOW_URI = "http://mlflow:5000"
DB_URI = "postgresql://testuser:testuser@postgres:5432/fastapi_db"

# --------------------------------------------------------------------------
# LOAD MODELS
# --------------------------------------------------------------------------
mlflow.set_tracking_uri(MLFLOW_URI)
reg = mlflow.pyfunc.load_model("models:/regressor@champion")
cls = mlflow.pyfunc.load_model("models:/classifier@champion")

# --------------------------------------------------------------------------
# LOAD DATA
# --------------------------------------------------------------------------
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

# --------------------------------------------------------------------------
# GROUND TRUTH AND FEATURE EXTRACTION
# --------------------------------------------------------------------------
true_reg   = df["arr_delay_minutes"]
true_class = df["arr_del15"]
uids = df["flight_uid"].copy()

def get_feature_columns(signature):
    return [col.name for col in signature.inputs.inputs]

feature_cols_reg = get_feature_columns(reg.metadata.signature)
feature_cols_cls = get_feature_columns(cls.metadata.signature)

print(f"Features Regressor : {feature_cols_reg}")
print(f"Features Classifier: {feature_cols_cls}")

# All features needed for logging (union of both sets)
union_features = sorted(set(feature_cols_reg) | set(feature_cols_cls))

features_all = df[union_features]          # for logging
features_reg = df[feature_cols_reg]        # regressor features only
features_cls = df[feature_cols_cls]        # classifier features only

# --------------------------------------------------------------------------
# HELPER FUNCTION: SCHEMA ENFORCEMENT
# --------------------------------------------------------------------------
def enforce_schema(df: pd.DataFrame, signature) -> pd.DataFrame:
    """Convert all columns of a DataFrame exactly to the types required by the MLflow signature."""
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

# --------------------------------------------------------------------------
# BATCH PREDICTIONS
# --------------------------------------------------------------------------
print("Führe Batch‑Predictions durch ...")
reg_preds = reg.predict(features_reg)
cls_preds = cls.predict(features_cls)

# --------------------------------------------------------------------------
# WRITE PREDICTIONS TO DATABASE
# --------------------------------------------------------------------------
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