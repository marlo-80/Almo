# src/api.py
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

# --- Umgebungsvariablen ---------------------------------------------------
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
DB_URI = os.environ.get("DB_URI", "postgresql://vikmar:vikmar@postgres:5432/fastapi_db")
engine = create_engine(DB_URI)


# --- Lifespan: Modelle beim Start laden ------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lädt Regressor und Classifier aus der MLflow‑Registry."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    # ----- Modellladezeit messen -----
    start_load = time.time()

    for task, cfg in API_MODELS.items():
        model_name = cfg["model_name"]
        alias = cfg["alias"]
        model_uri = f"models:/{model_name}@{alias}"
        try:
            if task == "classification":
                pipeline = mlflow.sklearn.load_model(model_uri)   # sklearn-Pipeline mit predict_proba
            else:
                pipeline = mlflow.pyfunc.load_model(model_uri)
            mv = client.get_model_version_by_alias(model_name, alias)
            version_str = f"{model_name}_v{mv.version}@{alias}"
            setattr(app.state, f"{task}_pipeline", pipeline)
            setattr(app.state, f"{task}_version", version_str)
            print(f"Modell '{task}' geladen: {version_str}")
        except Exception as e:
            print(f"WARNUNG: Modell '{task}' nicht geladen – {e}")
            setattr(app.state, f"{task}_pipeline", None)
            setattr(app.state, f"{task}_version", "not_loaded")

    load_duration = time.time() - start_load
    MODEL_LOAD_DURATION_SECONDS.set(load_duration)

    # ----- Champion‑Baselines aus MLflow laden -----
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
        print("Champion‑Baselines aus MLflow geladen.")
    except Exception as e:
        print(f"WARNUNG: Konnte Champion‑Baselines nicht laden – {e}")

    # ----- Zeilenanzahlen initial setzen -----
    with engine.connect() as conn:
        TRAIN_ROWS.set(conn.execute(text("SELECT COUNT(*) FROM dbt_staging.retrain")).scalar())
        PREDICTION_ROWS.set(conn.execute(text("SELECT COUNT(*) FROM api.predictions")).scalar())

    # ----- Modell‑Alter berechnen (beide Modelle) -----
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

    # ----- Champion‑Modell‑Info in Prometheus bereitstellen -----
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

    # ----- Drift‑Baseline (konstant) -----
    # DRIFT_BASELINE.set(0.05)

    yield

    # Aufräumen
    for task in API_MODELS:
        delattr(app.state, f"{task}_pipeline")


# --- FastAPI‑App -----------------------------------------------------------
app = FastAPI(
    title="Flight Delay Prediction API",
    description="Liefert Regressions- und Klassifikationsvorhersage",
    version="2.0",
    lifespan=lifespan,
)
Instrumentator().instrument(app).expose(app)

# --- Prometheus Metriken ---------------------------------------------------
# – Drift‑Metriken
DRIFT_SCORE = Gauge("data_drift_score", "Overall data drift score (0=no drift, 1=full drift)")
DRIFT_ACTUAL_RATE = Gauge("prediction_drift_actual_rate", "Actual fraction of delayed flights in current batch")
DRIFT_PREDICTED_RATE = Gauge("prediction_drift_predicted_rate", "Predicted fraction of delayed flights in current batch")
DRIFT_RATE_DELTA = Gauge("prediction_drift_rate_delta", "Predicted Delay Rate minus Actual Delay Rate")
DRIFT_CLASS_F1 = Gauge("prediction_drift_class_f1", "F1-Score des Klassifikators im aktuellen Batch")
DRIFT_CLASS_ROC_AUC = Gauge("prediction_drift_class_roc_auc", "ROC-AUC des Klassifikators im aktuellen Batch")
DRIFT_CLASS_ACCURACY = Gauge("prediction_drift_class_accuracy", "Accuracy des Klassifikators im aktuellen Batch")
DRIFT_CLASS_PRECISION = Gauge("prediction_drift_class_precision", "Precision des Klassifikators im aktuellen Batch")
DRIFT_CLASS_RECALL = Gauge("prediction_drift_class_recall", "Recall des Klassifikators im aktuellen Batch")
DRIFT_CLASS_SPECIFICITY = Gauge("prediction_drift_class_specificity", "Specificity (True Negative Rate) des Klassifikators im aktuellen Batch")
DRIFT_MAE = Gauge("prediction_drift_mae", "Mean Absolute Error between regression prediction and actual")
DRIFT_REGRESSOR_RMSE = Gauge("prediction_drift_rmse", "RMSE des Regressors im aktuellen Batch")
DRIFT_REGRESSOR_R2 = Gauge("prediction_drift_r2", "R² des Regressors im aktuellen Batch")
DRIFT_CLASS_CONFIDENCE_MEAN = Gauge("prediction_drift_class_confidence_mean", "Mittlere predicted probability (Klasse 1) im aktuellen Batch")
DRIFT_RESIDUAL_SKEWNESS = Gauge("prediction_drift_residual_skewness", "Schiefe der Residuen (true - prediction) im aktuellen Batch")
DRIFT_PREDICTION_STDDEV_ROLLING = Gauge("prediction_stddev_rolling", "Rollierende Standardabweichung der letzten 100 Regressionsvorhersagen")

# – Champion‑Baselines Regressor
CHAMPION_REGRESSOR_RMSE = Gauge("champion_regressor_rmse", "RMSE des aktuellen Champion-Regressors")
CHAMPION_REGRESSOR_MAE  = Gauge("champion_regressor_mae",  "MAE des aktuellen Champion-Regressors")
CHAMPION_REGRESSOR_R2   = Gauge("champion_regressor_r2",   "R² des aktuellen Champion-Regressors")
CHAMPION_REGRESSOR_RESIDUAL_SKEWNESS = Gauge("champion_regressor_residual_skewness", "Residuen-Schiefe des aktuellen Champion-Regressors")

# – Champion‑Baselines Classifier
CHAMPION_CLASSIFIER_F1      = Gauge("champion_classifier_f1",      "F1-Score des aktuellen Champion-Classifiers")
CHAMPION_CLASSIFIER_ROC_AUC = Gauge("champion_classifier_roc_auc", "ROC-AUC des aktuellen Champion-Classifiers")
CHAMPION_CLASSIFIER_ACCURACY = Gauge("champion_classifier_accuracy", "Accuracy des aktuellen Champion-Classifiers")
CHAMPION_CLASSIFIER_PRECISION = Gauge("champion_classifier_precision", "Precision des aktuellen Champion-Classifiers")
CHAMPION_CLASSIFIER_RECALL    = Gauge("champion_classifier_recall",    "Recall des aktuellen Champion-Classifiers")
CHAMPION_CLASSIFIER_SPECIFICITY = Gauge("champion_classifier_specificity", "Specificity des aktuellen Champion-Classifiers")
CHAMPION_CLASSIFIER_CONFIDENCE_MEAN = Gauge("champion_classifier_confidence_mean", "Mittlere Confidence (Klasse 1) des aktuellen Champion-Classifiers")

# – Sonstige Metriken
TRAIN_ROWS = Gauge("train_rows", "Anzahl Zeilen im Trainingsdatensatz")
PREDICTION_ROWS = Gauge("prediction_rows", "Anzahl Zeilen in api.predictions")
TOP_DELAY_AIRPORT = Gauge("top_delay_airport_id", "Origin‑Airport‑ID mit den meisten Verspätungen im aktuellen Batch")
MODEL_AGE_HOURS_REGRESSOR = Gauge("model_age_hours_regressor", "Age of the current champion regressor in hours")
MODEL_AGE_HOURS_CLASSIFIER = Gauge("model_age_hours_classifier", "Age of the current champion classifier in hours")
TOP_AIRLINE_DELAY_RATE = Gauge("top_airline_delay_rate", "Predicted delay rate per airline", ["rank", "airline"])
PREDICTION_COUNT = Counter("predictions_total", "Total number of prediction requests served")
DRIFT_BASELINE = Gauge("drift_baseline", "Baseline drift score (expected noise level)")

# – Betriebsmetriken
PREDICTION_DURATION_SECONDS = Histogram("prediction_duration_seconds", "Model prediction time (excl. DB write)")
DB_WRITE_DURATION_SECONDS = Histogram("db_write_duration_seconds", "Duration of INSERT into api.predictions")
MODEL_LOAD_DURATION_SECONDS = Gauge("model_load_duration_seconds", "Time to load models from MLflow at startup")

# – Neue Metriken für Demo / Retraining
RETRAIN_STATUS = Gauge("retrain_status", "1 if new champion was promoted after drift retraining")
DRIFT_BASELINE_DYNAMIC = Gauge("drift_baseline_dynamic", "Monatlich angepasste Drift-Baseline")
CHAMPION_MODEL_INFO = Gauge(
    "champion_model_info",
    "Current champion model version",
    ["model", "version"]
)


DRIFT_ALARM_ACTIVE = Gauge("drift_alarm_active", "1 while drift alarm is active, 0 otherwise")

class PredictionOutput(BaseModel):
    regression_prediction: float
    classification_prediction: int
    classification_proba: float | None = None


# --- Endpunkte -------------------------------------------------------------
@app.post("/predict", response_model=PredictionOutput)
async def predict(request: Request):
    if app.state.regression_pipeline is None or app.state.classification_pipeline is None:
        raise HTTPException(status_code=503, detail="Model pipelines not loaded.")

    try:
        input_data = await request.json()
        flight_uid = input_data.pop("flight_uid", None)
        ground_truth = input_data.pop("ground_truth", None)
        df = pd.DataFrame([input_data])

        # Reine Vorhersagezeit messen
        t0 = time.time()

        # Regression
        reg_pred = app.state.regression_pipeline.predict(df)[0]

        # Klassifikation
        class_pred = app.state.classification_pipeline.predict(df)[0]
        class_proba = app.state.classification_pipeline.predict_proba(df)[0, 1]

        pred_duration = time.time() - t0
        PREDICTION_DURATION_SECONDS.observe(pred_duration)

        # DB‑Schreibzeit messen
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


@app.post("/admin/reload-model")
async def reload_model():
    """Lädt beide Modelle neu aus der Registry und aktualisiert Metriken (Alter, Version)."""
    from datetime import datetime, timezone

    client = MlflowClient()
    start_reload = time.time()

    # Modelle neu laden
    for task, cfg in API_MODELS.items():
        model_uri = f"models:/{cfg['model_name']}@{cfg['alias']}"
        try:
            if task == "classification":
                pipeline = mlflow.sklearn.load_model(model_uri)   # für predict_proba
            else:
                pipeline = mlflow.pyfunc.load_model(model_uri)

            mv = client.get_model_version_by_alias(cfg["model_name"], cfg["alias"])
            version_str = f"{cfg['model_name']}_v{mv.version}@{cfg['alias']}"
            setattr(app.state, f"{task}_pipeline", pipeline)
            setattr(app.state, f"{task}_version", version_str)
            print(f"Modell '{task}' neu geladen: {version_str}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Reload failed for {task}: {e}")

    # Ladezeit der Modelle
    MODEL_LOAD_DURATION_SECONDS.set(time.time() - start_reload)

    # Champion‑Modell‑Info (Labels) aktualisieren
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

    # Modellalter (beide Modelle) neu berechnen
    for model_name, gauge in [('regressor', MODEL_AGE_HOURS_REGRESSOR),
                               ('classifier', MODEL_AGE_HOURS_CLASSIFIER)]:
        try:
            mv = client.get_model_version_by_alias(model_name, 'champion')
            run = client.get_run(mv.run_id)
            start_time = run.info.start_time / 1000.0   # Millisekunden → Sekunden
            age_seconds = datetime.now(timezone.utc).timestamp() - start_time
            gauge.set(age_seconds / 3600.0)             # Stunden
        except Exception as e:
            print(f"WARNUNG: Konnte Alter für {model_name} nicht berechnen: {e}")
            gauge.set(0.0)

    return {"status": "reloaded"}


@app.post("/admin/drift-metrics")
async def update_drift_metrics(data: dict):
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
    """Setzt die Referenz-Metriken des aktuellen Champions (für Vergleichslinien in Grafana)."""
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
    with engine.connect() as conn:
        TRAIN_ROWS.set(conn.execute(text("SELECT COUNT(*) FROM dbt_staging.retrain")).scalar())
        PREDICTION_ROWS.set(conn.execute(text("SELECT COUNT(*) FROM api.predictions")).scalar())
    return {"status": "ok"}


@app.post("/admin/top-airlines")
async def update_top_airlines(data: dict):
    airlines = data.get("airlines", [])
    for entry in airlines:
        TOP_AIRLINE_DELAY_RATE.labels(
            rank=str(entry["rank"]),
            airline=entry["airline"]
        ).set(entry["rate"])
    return {"status": "ok"}


@app.post("/admin/baseline")
async def set_baseline(data: dict):
    """Setzt die dynamische Baseline (für Demo‑Zwecke)."""
    value = float(data.get("value", 0.15))
    DRIFT_BASELINE_DYNAMIC.set(value)
    return {"status": "ok", "baseline": value}


@app.post("/admin/retrain")
async def trigger_retrain():
    """Hängt die aktuellen Predictions an die Tabelle dbt_staging.retrain an, löscht sie dann und startet Retraining."""
    import subprocess
    from sqlalchemy import text as sa_text

    target_table = "retrain"                    # Fester Tabellenname
    full_table = f"dbt_staging.{target_table}"  # dbt_staging.retrain

    with engine.connect() as conn:
        # 1. Tabelle anlegen, falls nicht vorhanden (Struktur aus pre_covid_test kopieren)
        exists = conn.execute(
            sa_text("SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_schema='dbt_staging' AND table_name=:tbl)"),
            {"tbl": target_table}
        ).scalar()
        if not exists:
            conn.execute(sa_text(f"CREATE TABLE {full_table} (LIKE dbt_staging.pre_covid_test)"))
            conn.commit()

        # 2. Spaltenstruktur ermitteln
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

        # 3. Aktuelle Predictions an die Tabelle anhängen
        insert_sql = f"""
            INSERT INTO {full_table} ({', '.join(col_names)})
            SELECT {', '.join(select_parts)}
            FROM api.predictions
            WHERE ground_truth IS NOT NULL
        """
        conn.execute(sa_text(insert_sql))
        conn.commit()

        # 4. Predictions löschen, damit sie beim nächsten Mal nicht doppelt eingefügt werden
        conn.execute(sa_text("TRUNCATE TABLE api.predictions RESTART IDENTITY"))
        conn.commit()

    # 5. Retraining asynchron starten
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
    RETRAIN_STATUS.set(int(data.get("new_champion", 0)))
    return {"status": "ok"}


@app.post("/admin/drift-alarm")
async def set_drift_alarm(data: dict):
    active = int(data.get("active", 0))
    DRIFT_ALARM_ACTIVE.set(active)
    # Hier muss retrain_status auf 0 gesetzt werden!!!
    # Setze retrain_status auf 0, sobald der Alarm gesetzt wird
    RETRAIN_STATUS.set(0)
    
    return {"status": "ok", "active": active}

@app.post("/admin/init-champion-metrics")
async def init_champion_metrics():
    """Setzt alle Champion-Metriken und initialisiert Drift-Metriken mit Champion-Werten."""
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

                # Drift-Metriken starten auf Champion-Niveau
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

                # Drift-Metriken starten auf Champion-Niveau
                DRIFT_CLASS_F1.set(f1)
                DRIFT_CLASS_ROC_AUC.set(roc_auc)
                DRIFT_CLASS_ACCURACY.set(acc)
                DRIFT_CLASS_PRECISION.set(prec)
                DRIFT_CLASS_RECALL.set(rec)
                DRIFT_CLASS_SPECIFICITY.set(spec)
                DRIFT_CLASS_CONFIDENCE_MEAN.set(conf_mean)

        # Champion-Modell-Info aus MLflow laden (damit das Panel sofort die richtigen Namen zeigt)
        CHAMPION_MODEL_INFO.clear()
        try:
            for model_name in ['regressor', 'classifier']:
                mv = client.get_model_version_by_alias(model_name, 'champion')
                version_str = f"{model_name}_v{mv.version}@champion"
                CHAMPION_MODEL_INFO.labels(model=model_name, version=version_str).set(1)
        except Exception as e:
            print(f"WARNUNG: Konnte Champion-Modell-Info nicht laden – {e}")

        DRIFT_SCORE.set(0.05)
        RETRAIN_STATUS.set(1)

        return {"status": "ok", "message": "Champion metrics loaded and drift metrics initialized"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.get("/health")
def health_check():
    return {
        "regression_loaded": app.state.regression_pipeline is not None,
        "classification_loaded": app.state.classification_pipeline is not None,
    }