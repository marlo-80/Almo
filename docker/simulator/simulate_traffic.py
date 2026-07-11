# docker/simulator/simulate_traffic.py
"""
Traffic Simulator – Continuous API Load Generator

This script simulates live traffic to the flight delay prediction API by
repeatedly selecting random flight samples from a PostgreSQL source table,
formatting them according to the registered MLflow model signatures, and
sending prediction requests to the FastAPI `/predict` endpoint.

The simulator runs indefinitely, making it useful for:
- Testing API performance under load
- Generating real-time prediction data for Grafana dashboards
- Feeding the drift detection pipeline with streaming data

------------------------------------------------------------------------------
Workflow
------------------------------------------------------------------------------
1. Load the champion regressor and classifier models from MLflow
2. Extract feature columns from both model signatures
3. Query the source table for random flight samples (with ground truth)
4. In an infinite loop:
   a. Pick a random sample
   b. Build separate DataFrames for regressor and classifier features
   c. Enforce schema matching MLflow signatures
   d. Assemble a unified payload with all features
   e. Send a POST request to the prediction API
   f. Print the response status and prediction
   g. Wait for the configured sleep interval

------------------------------------------------------------------------------
Configuration (Environment Variables)
------------------------------------------------------------------------------
API_URL               : Prediction endpoint (default: http://api:8000/predict)
SLEEP_SEC             : Delay between requests in seconds (default: 2.0)
MLFLOW_TRACKING_URI   : MLflow tracking server (default: http://mlflow:5000)
DB_SOURCE_TABLE       : Source table for flight samples
                        (default: dbt_staging.intra_covid_1M)
MODEL_NAME_REG        : Regressor model name (default: from API_MODELS)
MODEL_ALIAS_REG       : Regressor alias (default: champion)
MODEL_NAME_CLS        : Classifier model name (default: from API_MODELS)
MODEL_ALIAS_CLS       : Classifier alias (default: champion)

------------------------------------------------------------------------------
Dependencies
------------------------------------------------------------------------------
- MLflow tracking server with registered models (regressor@champion,
  classifier@champion)
- PostgreSQL database (fastapi_db) with flight data
- FastAPI service running at the configured URL

------------------------------------------------------------------------------
Usage
------------------------------------------------------------------------------
    python docker/simulator/simulate_traffic.py

In production, the simulator is typically started via Docker Compose:
    docker compose up -d simulator

------------------------------------------------------------------------------
Notes
------------------------------------------------------------------------------
- The script requires that both models are registered in MLflow.
- The source table must contain the required feature columns and ground truth
  (`arr_delay_minutes`, `arr_del15`).
- The loop runs until the process is terminated (Ctrl+C or SIGTERM).
- All numeric values are converted to native Python types for JSON serialization.
"""

import random, time, os
import httpx
import mlflow
from mlflow.types import DataType
from sqlalchemy import create_engine, text
import pandas as pd
import numpy as np

# Two models from central configuration
from flows.config import API_MODELS

# --------------------------------------------------------------------------
# CONFIGURATION
# --------------------------------------------------------------------------
API_URL = os.environ.get("API_URL", "http://api:8000/predict")
SLEEP_SEC = float(os.environ.get("SLEEP_SEC", 2.0))
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
DB_URI = "postgresql://testuser:testuser@postgres:5432/fastapi_db"
DB_SOURCE_TABLE = os.environ.get("DB_SOURCE_TABLE", "dbt_staging.intra_covid_1M")

MODEL_REG_NAME  = os.environ.get("MODEL_NAME_REG", API_MODELS["regression"]["model_name"])
MODEL_REG_ALIAS = os.environ.get("MODEL_ALIAS_REG", API_MODELS["regression"]["alias"])
MODEL_CLS_NAME  = os.environ.get("MODEL_NAME_CLS", API_MODELS["classification"]["model_name"])
MODEL_CLS_ALIAS = os.environ.get("MODEL_ALIAS_CLS", API_MODELS["classification"]["alias"])

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

# --------------------------------------------------------------------------
# MAIN SIMULATION LOOP
# --------------------------------------------------------------------------
def main():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    # --------------------------------------------------------------------------
    # LOAD MODELS
    # --------------------------------------------------------------------------
    reg_uri = f"models:/{MODEL_REG_NAME}@{MODEL_REG_ALIAS}"
    cls_uri = f"models:/{MODEL_CLS_NAME}@{MODEL_CLS_ALIAS}"
    print(f"Loading Regressor : {reg_uri}")
    print(f"Loading Classifier: {cls_uri}")
    reg_model = mlflow.pyfunc.load_model(reg_uri)
    cls_model = mlflow.pyfunc.load_model(cls_uri)

    # Merge features from both signatures
    reg_cols = [col.name for col in reg_model.metadata.signature.inputs.inputs]
    cls_cols = [col.name for col in cls_model.metadata.signature.inputs.inputs]
    all_feature_cols = sorted(set(reg_cols + cls_cols))
    print(f"Unified Features: {all_feature_cols}")

    # --------------------------------------------------------------------------
    # LOAD SAMPLES
    # --------------------------------------------------------------------------
    engine = create_engine(DB_URI)
    # Load all required columns + flight_uid + ground truth
    db_cols = ", ".join(set(all_feature_cols) | {"flight_uid", "arr_delay_minutes", "arr_del15"})
    query = f"SELECT {db_cols} FROM {DB_SOURCE_TABLE}"
    with engine.connect() as conn:
        rows = conn.execute(text(query)).fetchall()
    samples = [dict(row._mapping) for row in rows]
    print(f"Samples loaded: {len(samples)}")

    # --------------------------------------------------------------------------
    # SIMULATION LOOP
    # --------------------------------------------------------------------------
    while True:
        sample = random.choice(samples)
        # Build DataFrames for both models and enforce types
        df_reg = pd.DataFrame([sample])
        df_reg = enforce_schema(df_reg[reg_cols], reg_model.metadata.signature)
        df_cls = pd.DataFrame([sample])
        df_cls = enforce_schema(df_cls[cls_cols], cls_model.metadata.signature)

        # Payload with all features from the union set
        payload = {}
        # Take values from df_reg or df_cls, both are schema-correct
        for col in all_feature_cols:
            if col in df_reg.columns:
                val = df_reg[col].iloc[0]
            elif col in df_cls.columns:
                val = df_cls[col].iloc[0]
            else:
                val = None
            # Convert types to native Python types for JSON serialization
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
            print(f"Error: {e}")
        time.sleep(SLEEP_SEC)


if __name__ == "__main__":
    main()