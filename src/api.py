# src/api.py
"""
FastAPI Service – Flight Delay Prediction API with Prometheus Monitoring

This module implements the main REST API for flight delay predictions. It
loads the champion regression and classification models from MLflow at
startup, serves predictions via the `/predict` endpoint, and exposes
administrative endpoints for updating metrics, triggering retraining, and
reloading models.

The API is instrumented with Prometheus metrics (via `prometheus_fastapi_instrumentator`
and custom gauges/counters) and integrated with Grafana for real-time
monitoring of drift scores, model performance, and system health.

------------------------------------------------------------------------------
Key Features
------------------------------------------------------------------------------
- Model Loading: At startup (lifespan), loads `regressor@champion` and
  `classifier@champion` from MLflow. If models are missing, it logs warnings
  but continues running (graceful degradation).
- Prediction: `/predict` accepts flight features (JSON), applies both models,
  stores the input, predictions, and ground truth (if provided) in the
  `api.predictions` table, and returns regression and classification results.
- Admin Endpoints:
  - `/admin/reload-model` → reloads champion models from MLflow.
  - `/admin/drift-metrics` → updates drift scores and performance metrics.
  - `/admin/champion-metrics` → updates champion baseline metrics.
  - `/admin/data-stats` → refreshes training/prediction row counts.
  - `/admin/top-airlines` → updates per-airline delay rates.
  - `/admin/baseline` → sets dynamic drift baseline (for demos).
  - `/admin/retrain` → appends current predictions to `dbt_staging.retrain`,
    clears `api.predictions`, and triggers a retraining flow (via subprocess).
  - `/admin/retrain-status` → sets retrain status gauge.
  - `/admin/drift-alarm` → sets drift alarm gauge.
  - `/admin/init-champion-metrics` → loads champion metrics from MLflow and
    initializes drift metrics to champion values.
- Health Check: `/health` returns whether regression/classification models are loaded.

------------------------------------------------------------------------------
Prometheus Metrics (Exposed via /metrics)
------------------------------------------------------------------------------
The service exposes a rich set of custom metrics for monitoring:
- Drift metrics: drift score, actual/predicted delay rates, classification F1,
  ROC‑AUC, accuracy, precision, recall, specificity, confidence mean.
- Regression metrics: MAE, RMSE, R², residual skewness, rolling std.
- Champion baselines: RMSE, MAE, R², F1, ROC‑AUC, etc. for comparison.
- Operational: prediction count, request duration, DB write duration,
  model load duration, model age (hours).
- System: training rows, prediction rows, top delay airport, top airlines.

------------------------------------------------------------------------------
Environment Variables
------------------------------------------------------------------------------
- MLFLOW_TRACKING_URI : MLflow server URL (default: http://mlflow:5000)
- DB_URI              : PostgreSQL connection string
                       (default: postgresql://vikmar:vikmar@postgres:5432/fastapi_db)

Note: The default DB_URI uses `vikmar` – it is recommended to override via
      environment variables (e.g., in `docker-compose.yml`).

------------------------------------------------------------------------------
Dependencies
------------------------------------------------------------------------------
- FastAPI, Uvicorn
- MLflow (model loading, client)
- SQLAlchemy (PostgreSQL)
- Prometheus Client (custom metrics)
- pandas, numpy, json, time

------------------------------------------------------------------------------
Usage
------------------------------------------------------------------------------
The service is typically started via Docker Compose:
    docker compose up -d api

Or manually:
    uvicorn src.api:app --host 0.0.0.0 --port 8000

------------------------------------------------------------------------------
Notes
------------------------------------------------------------------------------
- Models are loaded once at startup and can be reloaded via `/admin/reload-model`.
- Prediction results are stored asynchronously (with ground truth when provided).
- The `/admin/retrain` endpoint uses subprocess to launch training flows as
  background jobs – ensure that the flow scripts are accessible and that the
  environment is correctly set (PYTHONPATH, etc.).
- The API is designed to be robust: missing models or database tables do not
  prevent startup; they are logged as warnings and handled gracefully at runtime.
"""

# --------------------------------------------------------------------------
# IMPORTS
# --------------------------------------------------------------------------
import pandas as pd
import os
import json
import time
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from contextlib import asynccontextmanager
from prometheus_fastapi_instrumentator import Instrumentator
import mlflow
from mlflow.tracking import MlflowClient
from sqlalchemy import create_engine, text

from flows.config import API_MODELS
from prometheus_client import Gauge, Counter, Histogram

# --------------------------------------------------------------------------
# ENVIRONMENT VARIABLES AND DATABASE ENGINE
# --------------------------------------------------------------------------
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
DB_URI = os.environ.get("DB_URI", "postgresql://vikmar:vikmar@postgres:5432/fastapi_db")
engine = create_engine(DB_URI)

# --------------------------------------------------------------------------
# LIFESPAN FUNCTION (MODEL LOADING AND INITIALIZATION)
# --------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Loading regressor and classifier from MLFlow registry."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    # ----- Measure model load time -----
    start_load = time.time()

    for task, cfg in API_MODELS.items():
        model_name = cfg["model_name"]
        alias = cfg["alias"]
        model_uri = f"models:/{model_name}@{alias}"
        try:
            if task == "classification":
                pipeline = mlflow.sklearn.load_model(model_uri)
                mv = client.get_model_version_by_alias(model_name, alias)
                version_str = f"{model_name}_v{mv.version}@{alias}"
                setattr(app.state, f"{task}_pipeline", pipeline)
                setattr(app.state, f"{task}_version", version_str)
                print(f"Model '{task}' loaded: {version_str}")
            else:
                pipeline = mlflow.pyfunc.load_model(model_uri)
            mv = client.get_model_version_by_alias(model_name, alias)
            version_str = f"{model_name}_v{mv.version}@{alias}"
            setattr(app.state, f"{task}_pipeline", pipeline)
            setattr(app.state, f"{task}_version", version_str)
            print(f"Model '{task}' loaded: {version_str}")
        except Exception as e:
            print(f"WARNING: Model '{task}' not loaded – {e}")
            setattr(app.state, f"{task}_pipeline", None)
            setattr(app.state, f"{task}_version", "not_loaded")

    load_duration = time.time() - start_load
    MODEL_LOAD_DURATION_SECONDS.set(load_duration)

    # ----- Load champion baselines from MLflow -----
    try:
        for model_name in ['regressor', 'classifier']:
            mv = client.get_model_version_by_alias(model_name, 'champion')
            run = client.get_run(mv.run_id)
            metrics = run.data.metrics
            if model_name == 'regressor':
                CHAMPION_REGRESSOR_RMSE.set(metrics.get('rmse', 0.0))
                CHAMPION_REGRESSOR_MAE.set(metrics.get('mae', 0.0))
                CHAMPION_REGRESSOR_R2.set(metrics.get('r2', 0.0))
                CHAMPION_REGRESSOR_RESIDUAL_SKEWNESS.set(metrics.get('residual_skewness', 0.0))
            else:
                CHAMPION_CLASSIFIER_F1.set(metrics.get('f1', 0.0))
                CHAMPION_CLASSIFIER_ROC_AUC.set(metrics.get('roc_auc', 0.0))
                CHAMPION_CLASSIFIER_ACCURACY.set(metrics.get('accuracy', 0.0))
                CHAMPION_CLASSIFIER_PRECISION.set(metrics.get('precision', 0.0))
                CHAMPION_CLASSIFIER_RECALL.set(metrics.get('recall', 0.0))
                CHAMPION_CLASSIFIER_SPECIFICITY.set(metrics.get('specificity', 0.0))
                CHAMPION_CLASSIFIER_CONFIDENCE_MEAN.set(metrics.get('confidence_mean', 0.0))
        print("Champion baselines loaded from MLflow.")
    except Exception as e:
        print(f"WARNING: Could not load champion baselines – {e}")

    # ----- Initialize row counts -----
    with engine.connect() as conn:
        try:
            TRAIN_ROWS.set(conn.execute(text("SELECT COUNT(*) FROM dbt_staging.retrain")).scalar())
        except Exception:
            TRAIN_ROWS.set(0)
            print("WARNING: Table dbt_staging.retrain does not exist.")
        try:
            PREDICTION_ROWS.set(conn.execute(text("SELECT COUNT(*) FROM api.predictions")).scalar())
        except Exception as e:
            PREDICTION_ROWS.set(0)
            print(f"WARNING: Could not read prediction rows: {e}")


    # ----- Calculate model age (both models) -----
    from datetime import datetime, timezone
    for model_name, gauge in [('regressor', MODEL_AGE_HOURS_REGRESSOR),
                               ('classifier', MODEL_AGE_HOURS_CLASSIFIER)]:
        try:
            mv = client.get_model_version_by_alias(model_name, 'champion')
            run = client.get_run(mv.run_id)
            start_time = run.info.start_time / 1000.0   # ms → s
            age_seconds = datetime.now(timezone.utc).timestamp() - start_time
            gauge.set(age_seconds / 3600.0)
        except Exception:
            gauge.set(0.0)

    # ----- Provide champion model info in Prometheus -----
    CHAMPION_MODEL_INFO.clear()
    if app.state.regression_pipeline is not None:
        CHAMPION_MODEL_INFO.labels(
            model="regressor",
            version=app.state.regression_version
        ).set(1)
    if app.state.classification_pipeline is not None:
        CHAMPION_MODEL_INFO.labels(
            model="classifier",
            version=app.state.classification_version
        ).set(1)

    # ----- Drift baseline (constant) -----
    # DRIFT_BASELINE.set(0.05)

    yield

    # Cleanup
    for task in API_MODELS:
        delattr(app.state, f"{task}_pipeline")


# --------------------------------------------------------------------------
# FASTAPI APPLICATION AND INSTRUMENTATION
# --------------------------------------------------------------------------
app = FastAPI(
    title="Flight Delay Prediction API",
    description="Liefert Regressions- und Klassifikationsvorhersage",
    version="2.0",
    lifespan=lifespan,
)
Instrumentator().instrument(app).expose(app)

# --------------------------------------------------------------------------
# PROMETHEUS METRICS
# --------------------------------------------------------------------------
# -- Drift metrics --
DRIFT_SCORE = Gauge("data_drift_score", "Overall data drift score (0=no drift, 1=full drift)")
DRIFT_ACTUAL_RATE = Gauge("prediction_drift_actual_rate", "Actual fraction of delayed flights in current batch")
DRIFT_PREDICTED_RATE = Gauge("prediction_drift_predicted_rate", "Predicted fraction of delayed flights in current batch")
DRIFT_RATE_DELTA = Gauge("prediction_drift_rate_delta", "Predicted Delay Rate minus Actual Delay Rate")
DRIFT_CLASS_F1 = Gauge("prediction_drift_class_f1", "F1 score of the classifier in the current batch")
DRIFT_CLASS_ROC_AUC = Gauge("prediction_drift_class_roc_auc", "ROC-AUC of the classifier in the current batch")
DRIFT_CLASS_ACCURACY = Gauge("prediction_drift_class_accuracy", "Accuracy of the classifier in the current batch")
DRIFT_CLASS_PRECISION = Gauge("prediction_drift_class_precision", "Precision of the classifier in the current batch")
DRIFT_CLASS_RECALL = Gauge("prediction_drift_class_recall", "Recall of the classifier in the current batch")
DRIFT_CLASS_SPECIFICITY = Gauge("prediction_drift_class_specificity", "Specificity (True Negative Rate) of the classifier in the current batch")
DRIFT_MAE = Gauge("prediction_drift_mae", "Mean Absolute Error between regression prediction and actual")
DRIFT_REGRESSOR_RMSE = Gauge("prediction_drift_rmse", "RMSE of the regressor in the current batch")
DRIFT_REGRESSOR_R2 = Gauge("prediction_drift_r2", "R² of the regressor in the current batch")
DRIFT_CLASS_CONFIDENCE_MEAN = Gauge("prediction_drift_class_confidence_mean", "Mean predicted probability (class 1) in the current batch")
DRIFT_RESIDUAL_SKEWNESS = Gauge("prediction_drift_residual_skewness", "Skewness of residuals (true - prediction) in the current batch")
DRIFT_PREDICTION_STDDEV_ROLLING = Gauge("prediction_stddev_rolling", "Rolling standard deviation of the last 100 regression predictions")

# -- Champion baselines (regressor) --
CHAMPION_REGRESSOR_RMSE = Gauge("champion_regressor_rmse", "RMSE of the current champion regressor")
CHAMPION_REGRESSOR_MAE  = Gauge("champion_regressor_mae",  "MAE of the current champion regressor")
CHAMPION_REGRESSOR_R2   = Gauge("champion_regressor_r2",   "R² of the current champion regressor")
CHAMPION_REGRESSOR_RESIDUAL_SKEWNESS = Gauge("champion_regressor_residual_skewness", "Residual skewness of the current champion regressor")

# -- Champion baselines (classifier) --
CHAMPION_CLASSIFIER_F1      = Gauge("champion_classifier_f1",      "F1 score of the current champion classifier")
CHAMPION_CLASSIFIER_ROC_AUC = Gauge("champion_classifier_roc_auc", "ROC-AUC of the current champion classifier")
CHAMPION_CLASSIFIER_ACCURACY = Gauge("champion_classifier_accuracy", "Accuracy of the current champion classifier")
CHAMPION_CLASSIFIER_PRECISION = Gauge("champion_classifier_precision", "Precision of the current champion classifier")
CHAMPION_CLASSIFIER_RECALL    = Gauge("champion_classifier_recall",    "Recall of the current champion classifier")
CHAMPION_CLASSIFIER_SPECIFICITY = Gauge("champion_classifier_specificity", "Specificity of the current champion classifier")
CHAMPION_CLASSIFIER_CONFIDENCE_MEAN = Gauge("champion_classifier_confidence_mean", "Mean confidence (class 1) of the current champion classifier")

# -- Other metrics --
TRAIN_ROWS = Gauge("train_rows", "Number of rows in the training dataset")
PREDICTION_ROWS = Gauge("prediction_rows", "Number of rows in api.predictions")
TOP_DELAY_AIRPORT = Gauge("top_delay_airport_id", "Origin airport ID with the most delays in the current batch")
MODEL_AGE_HOURS_REGRESSOR = Gauge("model_age_hours_regressor", "Age of the current champion regressor in hours")
MODEL_AGE_HOURS_CLASSIFIER = Gauge("model_age_hours_classifier", "Age of the current champion classifier in hours")
TOP_AIRLINE_DELAY_RATE = Gauge("top_airline_delay_rate", "Predicted delay rate per airline", ["rank", "airline"])
PREDICTION_COUNT = Counter("predictions_total", "Total number of prediction requests served")
DRIFT_BASELINE = Gauge("drift_baseline", "Baseline drift score (expected noise level)")

# -- Operational metrics --
PREDICTION_DURATION_SECONDS = Histogram("prediction_duration_seconds", "Model prediction time (excl. DB write)")
DB_WRITE_DURATION_SECONDS = Histogram("db_write_duration_seconds", "Duration of INSERT into api.predictions")
MODEL_LOAD_DURATION_SECONDS = Gauge("model_load_duration_seconds", "Time to load models from MLflow at startup")

# -- Metrics for demo / retraining --
RETRAIN_STATUS = Gauge("retrain_status", "1 if new champion was promoted after drift retraining")
DRIFT_BASELINE_DYNAMIC = Gauge("drift_baseline_dynamic", "Monthly adjusted drift baseline")
CHAMPION_MODEL_INFO = Gauge(
    "champion_model_info",
    "Current champion model version",
    ["model", "version"]
)


DRIFT_ALARM_ACTIVE = Gauge("drift_alarm_active", "1 while drift alarm is active, 0 otherwise")

# --------------------------------------------------------------------------
# REQUEST/RESPONSE MODELS
# --------------------------------------------------------------------------
class PredictionOutput(BaseModel):
    regression_prediction: float
    classification_prediction: int
    classification_proba: float | None = None


# --------------------------------------------------------------------------
# PREDICTION ENDPOINT
# --------------------------------------------------------------------------
@app.post("/predict", response_model=PredictionOutput)
async def predict(request: Request):
    if app.state.regression_pipeline is None or app.state.classification_pipeline is None:
        raise HTTPException(status_code=503, detail="Model pipelines not loaded.")

    try:
        input_data = await request.json()
        flight_uid = input_data.pop("flight_uid", None)
        ground_truth = input_data.pop("ground_truth", None)
        df = pd.DataFrame([input_data])

        # Measure pure prediction time
        t0 = time.time()

        # Regression
        reg_pred = app.state.regression_pipeline.predict(df)[0]

        # Classification
        class_pred = app.state.classification_pipeline.predict(df)[0]
        class_proba = app.state.classification_pipeline.predict_proba(df)[0, 1]

        pred_duration = time.time() - t0
        PREDICTION_DURATION_SECONDS.observe(pred_duration)

        # Measure DB write time
        log_to_db = input_data.copy()
        log_to_db["flight_uid"] = flight_uid
        t0_db = time.time()
        with engine.connect() as conn:
            conn.execute(
                text("""
                    INSERT INTO api.predictions
                        (flight_uid, input_features, prediction_reg, prediction_class,
                         model_version_reg, model_version_class, ground_truth,
                         prediction_class_proba)
                    VALUES (:flight_uid, :features, :pred_reg, :pred_class,
                            :version_reg, :version_class, :gt,
                            :pred_class_proba)
                """),
                {
                    "flight_uid": flight_uid,
                    "features": json.dumps(log_to_db),
                    "pred_reg": float(reg_pred),
                    "pred_class": int(class_pred),
                    "version_reg": app.state.regression_version,
                    "version_class": app.state.classification_version,
                    "gt": json.dumps(ground_truth) if ground_truth else None,
                    "pred_class_proba": float(class_proba),
                }
            )
            conn.commit()
        DB_WRITE_DURATION_SECONDS.observe(time.time() - t0_db)
        PREDICTION_ROWS.inc()
        PREDICTION_COUNT.inc()

        return PredictionOutput(
            regression_prediction=float(reg_pred),
            classification_prediction=int(class_pred),
            classification_proba=class_proba,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")


# --------------------------------------------------------------------------
# ADMIN ENDPOINTS
# --------------------------------------------------------------------------

@app.post("/admin/reload-model")
async def reload_model():
    """Reloads both models from the registry and updates metrics (age, version)."""
    from datetime import datetime, timezone

    client = MlflowClient()
    start_reload = time.time()

    # Reload models
    for task, cfg in API_MODELS.items():
        model_uri = f"models:/{cfg['model_name']}@{cfg['alias']}"
        try:
            if task == "classification":
                pipeline = mlflow.sklearn.load_model(model_uri)   # for predict_proba
            else:
                pipeline = mlflow.pyfunc.load_model(model_uri)

            mv = client.get_model_version_by_alias(cfg["model_name"], cfg["alias"])
            version_str = f"{cfg['model_name']}_v{mv.version}@{cfg['alias']}"
            setattr(app.state, f"{task}_pipeline", pipeline)
            setattr(app.state, f"{task}_version", version_str)
            print(f"Model '{task}' reloaded: {version_str}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Reload failed for {task}: {e}")

    # Model load time
    MODEL_LOAD_DURATION_SECONDS.set(time.time() - start_reload)

    # Update champion model info (labels)
    CHAMPION_MODEL_INFO.clear()
    if app.state.regression_pipeline is not None:
        CHAMPION_MODEL_INFO.labels(
            model="regressor",
            version=app.state.regression_version
        ).set(1)
    if app.state.classification_pipeline is not None:
        CHAMPION_MODEL_INFO.labels(
            model="classifier",
            version=app.state.classification_version
        ).set(1)

    # Recalculate model ages (both models)
    for model_name, gauge in [('regressor', MODEL_AGE_HOURS_REGRESSOR),
                               ('classifier', MODEL_AGE_HOURS_CLASSIFIER)]:
        try:
            mv = client.get_model_version_by_alias(model_name, 'champion')
            run = client.get_run(mv.run_id)
            start_time = run.info.start_time / 1000.0   # milliseconds → seconds
            age_seconds = datetime.now(timezone.utc).timestamp() - start_time
            gauge.set(age_seconds / 3600.0)             # hours
        except Exception as e:
            print(f"WARNING: Could not calculate age for {model_name}: {e}")
            gauge.set(0.0)

    return {"status": "reloaded"}


@app.post("/admin/drift-metrics")
async def update_drift_metrics(data: dict):
    """Update all drift and performance metrics from the drift flow."""
    try:
        DRIFT_SCORE.set(float(data.get("drift_score", 0.0)))
        DRIFT_MAE.set(float(data.get("mae", 0.0)))
        DRIFT_ACTUAL_RATE.set(float(data.get("actual_rate", 0.0)))
        DRIFT_PREDICTED_RATE.set(float(data.get("predicted_rate", 0.0)))
        DRIFT_CLASS_F1.set(float(data.get("class_f1", 0.0)))
        DRIFT_CLASS_ROC_AUC.set(float(data.get("class_roc_auc", 0.0)))
        DRIFT_RATE_DELTA.set(float(data.get("rate_delta", 0.0)))
        DRIFT_REGRESSOR_RMSE.set(float(data.get("rmse", 0.0)))
        DRIFT_CLASS_ACCURACY.set(float(data.get("class_accuracy", 0.0)))
        DRIFT_CLASS_PRECISION.set(float(data.get("class_precision", 0.0)))
        DRIFT_CLASS_RECALL.set(float(data.get("class_recall", 0.0)))
        DRIFT_CLASS_SPECIFICITY.set(float(data.get("class_specificity", 0.0)))
        TOP_DELAY_AIRPORT.set(float(data.get("top_delay_airport", 0.0)))
        DRIFT_REGRESSOR_R2.set(float(data.get("r2", 0.0)))
        DRIFT_CLASS_CONFIDENCE_MEAN.set(float(data.get("class_confidence_mean", 0.0)))
        DRIFT_RESIDUAL_SKEWNESS.set(float(data.get("residual_skewness", 0.0)))
        DRIFT_PREDICTION_STDDEV_ROLLING.set(float(data.get("stddev_rolling", 0.0)))
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/admin/champion-metrics")
async def update_champion_metrics(data: dict):
    """Sets the reference metrics of the current champion (for comparison lines in Grafana)."""
    try:
        CHAMPION_REGRESSOR_RMSE.set(float(data.get("regressor_rmse", 0.0)))
        CHAMPION_REGRESSOR_MAE.set(float(data.get("regressor_mae", 0.0)))
        CHAMPION_REGRESSOR_R2.set(float(data.get("regressor_r2", 0.0)))
        CHAMPION_REGRESSOR_RESIDUAL_SKEWNESS.set(float(data.get("regressor_residual_skewness", 0.0)))

        CHAMPION_CLASSIFIER_F1.set(float(data.get("classifier_f1", 0.0)))
        CHAMPION_CLASSIFIER_ROC_AUC.set(float(data.get("classifier_roc_auc", 0.0)))
        CHAMPION_CLASSIFIER_ACCURACY.set(float(data.get("classifier_accuracy", 0.0)))
        CHAMPION_CLASSIFIER_PRECISION.set(float(data.get("classifier_precision", 0.0)))
        CHAMPION_CLASSIFIER_RECALL.set(float(data.get("classifier_recall", 0.0)))
        CHAMPION_CLASSIFIER_SPECIFICITY.set(float(data.get("classifier_specificity", 0.0)))
        CHAMPION_CLASSIFIER_CONFIDENCE_MEAN.set(float(data.get("classifier_confidence_mean", 0.0)))
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/admin/data-stats")
async def update_data_stats():
    """Refresh training and prediction row counts."""
    with engine.connect() as conn:
        TRAIN_ROWS.set(conn.execute(text("SELECT COUNT(*) FROM dbt_staging.retrain")).scalar())
        PREDICTION_ROWS.set(conn.execute(text("SELECT COUNT(*) FROM api.predictions")).scalar())
    return {"status": "ok"}


@app.post("/admin/top-airlines")
async def update_top_airlines(data: dict):
    """Set per-airline delay rates for the top airlines panel."""
    airlines = data.get("airlines", [])
    for entry in airlines:
        TOP_AIRLINE_DELAY_RATE.labels(
            rank=str(entry["rank"]),
            airline=entry["airline"]
        ).set(entry["rate"])
    return {"status": "ok"}


@app.post("/admin/baseline")
async def set_baseline(data: dict):
    """Sets the dynamic drift baseline (for demo purposes)."""
    value = float(data.get("value", 0.15))
    DRIFT_BASELINE_DYNAMIC.set(value)
    return {"status": "ok", "baseline": value}


@app.post("/admin/retrain")
async def trigger_retrain():
    """Appends current predictions to dbt_staging.retrain, then clears predictions and starts retraining."""
    import subprocess
    from sqlalchemy import text as sa_text

    target_table = "retrain"                    # Fixed table name
    full_table = f"dbt_staging.{target_table}"  # dbt_staging.retrain

    with engine.connect() as conn:
        # 1. Create table if it does not exist (copy structure from pre_covid_test)
        exists = conn.execute(
            sa_text("SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_schema='dbt_staging' AND table_name=:tbl)"),
            {"tbl": target_table}
        ).scalar()
        if not exists:
            conn.execute(sa_text(f"CREATE TABLE {full_table} (LIKE dbt_staging.pre_covid_100k)"))
            conn.commit()

        # 2. Determine column structure
        cols = conn.execute(
            sa_text(f"SELECT column_name, data_type FROM information_schema.columns WHERE table_schema='dbt_staging' AND table_name=:tbl ORDER BY ordinal_position"),
            {"tbl": target_table}
        ).fetchall()
        col_names = [c[0] for c in cols]

        type_cast_map = {
            'integer': 'bigint', 'bigint': 'bigint', 'smallint': 'bigint',
            'double precision': 'float', 'real': 'float', 'numeric': 'float',
            'text': 'text', 'character varying': 'text',
            'date': 'date', 'timestamp without time zone': 'timestamp',
            'timestamp with time zone': 'timestamptz', 'boolean': 'boolean'
        }

        select_parts = []
        for col_name, col_type in cols:
            pg_type = col_type.lower()
            if col_name == 'flight_uid':
                select_parts.append('flight_uid')
            elif col_name == 'flight_date':
                select_parts.append('timestamp::date')
            elif col_name == 'arr_delay_minutes':
                select_parts.append("(ground_truth->>'arr_delay_minutes')::float")
            elif col_name == 'arr_del15':
                select_parts.append("(ground_truth->>'arr_del15')::int")
            else:
                cast = type_cast_map.get(pg_type, 'text')
                select_parts.append(f"(input_features->>'{col_name}')::{cast}")

        # 3. Append current predictions to the table
        insert_sql = f"""
            INSERT INTO {full_table} ({', '.join(col_names)})
            SELECT {', '.join(select_parts)}
            FROM api.predictions
            WHERE ground_truth IS NOT NULL
        """
        conn.execute(sa_text(insert_sql))
        conn.commit()

        # 4. Clear predictions so they are not inserted again next time
        conn.execute(sa_text("TRUNCATE TABLE api.predictions RESTART IDENTITY"))
        conn.commit()

    # 5. Start retraining asynchronously
    def run_training(config_name):
        subprocess.Popen(
            ["python", "flows/train_flow.py", config_name],
            cwd="/app",
            env={**os.environ, "PYTHONPATH": "/app", "PYTHONUNBUFFERED": "1"}
        )

    run_training("DRIFT_RETRAIN_REG")
    run_training("DRIFT_RETRAIN_CLASS")
    return {"status": "retraining_started", "message": "Predictions appended to dbt_staging.retrain, predictions cleared, training launched."}


@app.post("/admin/retrain-status")
async def set_retrain_status(data: dict):
    """Set retraining status gauge (1 if new champion promoted)."""
    RETRAIN_STATUS.set(int(data.get("new_champion", 0)))
    return {"status": "ok"}


@app.post("/admin/drift-alarm")
async def set_drift_alarm(data: dict):
    """Set drift alarm active gauge and reset retrain status."""
    active = int(data.get("active", 0))
    DRIFT_ALARM_ACTIVE.set(active)
    # retrain_status must be set to 0 here
    # Set retrain_status to 0 as soon as the alarm is set
    RETRAIN_STATUS.set(0)
    
    return {"status": "ok", "active": active}

@app.post("/admin/init-champion-metrics")
async def init_champion_metrics():
    """Sets all champion metrics and initializes drift metrics with champion values."""
    client = MlflowClient()
    try:
        for model_name in ['regressor', 'classifier']:
            mv = client.get_model_version_by_alias(model_name, 'champion')
            run = client.get_run(mv.run_id)
            metrics = run.data.metrics
            if model_name == 'regressor':
                rmse = metrics.get('rmse', 0.0)
                mae = metrics.get('mae', 0.0)
                r2 = metrics.get('r2', 0.0)
                res_skew = metrics.get('residual_skewness', 0.0)

                CHAMPION_REGRESSOR_RMSE.set(rmse)
                CHAMPION_REGRESSOR_MAE.set(mae)
                CHAMPION_REGRESSOR_R2.set(r2)
                CHAMPION_REGRESSOR_RESIDUAL_SKEWNESS.set(res_skew)

                # Drift metrics start at champion level
                DRIFT_REGRESSOR_RMSE.set(rmse)
                DRIFT_MAE.set(mae)
                DRIFT_REGRESSOR_R2.set(r2)
                DRIFT_RESIDUAL_SKEWNESS.set(res_skew)

            else:   # classifier
                f1 = metrics.get('f1', 0.0)
                roc_auc = metrics.get('roc_auc', 0.0)
                acc = metrics.get('accuracy', 0.0)
                prec = metrics.get('precision', 0.0)
                rec = metrics.get('recall', 0.0)
                spec = metrics.get('specificity', 0.0)
                conf_mean = metrics.get('confidence_mean', 0.0)

                CHAMPION_CLASSIFIER_F1.set(f1)
                CHAMPION_CLASSIFIER_ROC_AUC.set(roc_auc)
                CHAMPION_CLASSIFIER_ACCURACY.set(acc)
                CHAMPION_CLASSIFIER_PRECISION.set(prec)
                CHAMPION_CLASSIFIER_RECALL.set(rec)
                CHAMPION_CLASSIFIER_SPECIFICITY.set(spec)
                CHAMPION_CLASSIFIER_CONFIDENCE_MEAN.set(conf_mean)

                # Drift metrics start at champion level
                DRIFT_CLASS_F1.set(f1)
                DRIFT_CLASS_ROC_AUC.set(roc_auc)
                DRIFT_CLASS_ACCURACY.set(acc)
                DRIFT_CLASS_PRECISION.set(prec)
                DRIFT_CLASS_RECALL.set(rec)
                DRIFT_CLASS_SPECIFICITY.set(spec)
                DRIFT_CLASS_CONFIDENCE_MEAN.set(conf_mean)

        # Load champion model info from MLflow (so the panel immediately shows the correct names)
        CHAMPION_MODEL_INFO.clear()
        try:
            for model_name in ['regressor', 'classifier']:
                mv = client.get_model_version_by_alias(model_name, 'champion')
                version_str = f"{model_name}_v{mv.version}@champion"
                CHAMPION_MODEL_INFO.labels(model=model_name, version=version_str).set(1)
        except Exception as e:
            print(f"WARNING: Could not load champion model info – {e}")

        DRIFT_SCORE.set(0.05)
        RETRAIN_STATUS.set(1)

        return {"status": "ok", "message": "Champion metrics loaded and drift metrics initialized"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --------------------------------------------------------------------------
# HEALTH CHECK ENDPOINT
# --------------------------------------------------------------------------
@app.get("/health")
def health_check():
    """Health check: returns whether both models are loaded."""
    return {
        "regression_loaded": app.state.regression_pipeline is not None,
        "classification_loaded": app.state.classification_pipeline is not None,
    }