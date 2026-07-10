# flows/drift_flow.py
import numpy as np
np.float_ = np.float64          # Workaround für NumPy‑2‑Kompatibilität von Evidently

import mlflow
import pandas as pd
from prefect import flow, task
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
import requests
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    r2_score,
)
from scipy.stats import skew
import os
from sqlalchemy import create_engine, text

DB_URI = "postgresql://testuser:testuser@postgres:5432/fastapi_db"
MLFLOW_URI = "http://mlflow:5000"


@task
def load_reference_data(month: int):
    """Lädt Referenzdaten nur für den angegebenen Monat (1‑12) aus pre_covid_100k."""
    engine = create_engine(DB_URI)
    query = f"""
        SELECT *
        FROM dbt_staging.pre_covid_100k
        WHERE month = {month}
    """
    df = pd.read_sql(query, engine)
    drop_cols = [
        "arr_delay_minutes", "arr_del15", "arr_delay", "dep_delay",
        "dep_delay_minutes", "flight_uid", "flight_date",
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    print(f"Referenzdaten geladen (Monat {month}): {df.shape[0]} Zeilen, {df.shape[1]} Spalten")
    return df


@task
def load_current_features():
    """Lädt nur die input_features (JSONB) aus api.predictions (für Evidently)."""
    engine = create_engine(DB_URI)
    query = """
        SELECT input_features
        FROM api.predictions
        ORDER BY timestamp DESC
        LIMIT 5000
    """
    df = pd.read_sql(query, engine)
    records = df["input_features"].dropna().apply(pd.Series)
    if records.empty:
        raise ValueError("Keine Predictions in api.predictions gefunden.")
    print(f"Aktuelle Daten geladen: {records.shape[0]} Zeilen, {records.shape[1]} Spalten")
    return records


    # Faire Referenz: Reduziere den größeren Datensatz auf die Größe des kleineren
    min_len = min(len(reference), len(current))
    if len(reference) > min_len:
        reference = reference.sample(n=min_len, random_state=42)
    if len(current) > min_len:
        current = current.sample(n=min_len, random_state=24)

    print(f"Angeglichene Größen: Referenz {len(reference)}, Aktuell {len(current)}")


@task
def compute_drift_report(reference: pd.DataFrame, current: pd.DataFrame) -> Report:
    """Erstellt einen Evidently Data Drift Report."""
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current, column_mapping=None)
    print("Drift-Report erstellt.")
    return report


@task
def log_report(report: Report):
    """Speichert den Report als HTML in MLflow."""
    mlflow.set_tracking_uri(MLFLOW_URI)
    with mlflow.start_run(run_name="drift_report"):
        report.save_html("/tmp/drift_report.html")
        mlflow.log_artifact("/tmp/drift_report.html", "drift_reports")
    print("Drift-Report in MLflow geloggt.")


@task
def load_current_predictions():
    """Lädt prediction_reg, prediction_class, prediction_class_proba, ground_truth sowie input_features aus api.predictions."""
    engine = create_engine(DB_URI)
    query = """
        SELECT prediction_reg, prediction_class, prediction_class_proba, ground_truth, input_features
        FROM api.predictions
        ORDER BY timestamp DESC
        LIMIT 5000
    """
    df = pd.read_sql(query, engine)
    df["true_reg"]   = df["ground_truth"].apply(lambda x: x.get("arr_delay_minutes") if x else None)
    df["true_class"] = df["ground_truth"].apply(lambda x: x.get("arr_del15") if x else None)
    # Extrahiere origin_airport_id aus den gespeicherten Features (JSONB)
    df["origin_airport_id"] = df["input_features"].apply(
        lambda x: x.get("origin_airport_id") if isinstance(x, dict) else None
    )
    df = df.dropna(subset=["true_reg", "true_class"])
    print(f"Validierungsdaten geladen: {df.shape[0]} Zeilen")
    return df


@task
def compute_and_send_metrics(preds_df: pd.DataFrame):
    """Berechnet alle Drift‑ und Performance‑Metriken des aktuellen Batches."""
    if preds_df.empty:
        mae = rmse = actual_rate = predicted_rate = 0.0
        class_f1 = class_roc_auc = class_accuracy = 0.0
        class_precision = class_recall = class_specificity = 0.0
        rate_delta = 0.0
        top_origin = 0.0
        r2 = residual_skewness = rolling_std = 0.0
        class_confidence_mean = 0.0
    else:
        y_true_reg = preds_df["true_reg"]
        y_pred_reg = preds_df["prediction_reg"]
        y_true_cls = preds_df["true_class"]
        y_pred_cls = preds_df["prediction_class"]

        # Regression
        mae = float(np.mean(np.abs(y_pred_reg - y_true_reg)))
        rmse = float(np.sqrt(np.mean((y_pred_reg - y_true_reg) ** 2)))
        r2 = float(r2_score(y_true_reg, y_pred_reg))
        residuals = y_true_reg - y_pred_reg
        residual_skewness = float(skew(residuals))

        # Rollierende Standardabweichung der letzten 100 Regressionsvorhersagen
        rolling_std = float(preds_df["prediction_reg"].tail(100).std()) if len(preds_df) >= 100 else float(preds_df["prediction_reg"].std())

        # Raten
        actual_rate    = float(y_true_cls.mean())
        predicted_rate = float(y_pred_cls.mean())
        rate_delta     = predicted_rate - actual_rate

        # Klassifikation (nur wenn beide Klassen vorkommen)
        if y_true_cls.nunique() == 2:
            class_f1 = float(f1_score(y_true_cls, y_pred_cls))
            class_roc_auc = float(roc_auc_score(y_true_cls, y_pred_cls))
            class_accuracy = float(accuracy_score(y_true_cls, y_pred_cls))
            class_precision = float(precision_score(y_true_cls, y_pred_cls))
            class_recall = float(recall_score(y_true_cls, y_pred_cls))
            # Specificity = True Negative Rate
            tp = ((y_true_cls == 1) & (y_pred_cls == 1)).sum()
            tn = ((y_true_cls == 0) & (y_pred_cls == 0)).sum()
            fp = ((y_true_cls == 0) & (y_pred_cls == 1)).sum()
            class_specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        else:
            class_f1 = class_roc_auc = class_accuracy = 0.0
            class_precision = class_recall = class_specificity = 0.0

        # Mittlere Vorhersagewahrscheinlichkeit (Klasse 1) aus der neuen Spalte
        if "prediction_class_proba" in preds_df.columns:
            class_confidence_mean = float(preds_df["prediction_class_proba"].mean())
        else:
            class_confidence_mean = 0.0

        # Flughafen mit den meisten Verspätungen (Origin)
        if 'origin_airport_id' in preds_df.columns:
            top_origin = float(preds_df["origin_airport_id"].value_counts().idxmax())
        else:
            top_origin = 0.0

    # NaN/Inf abfangen
    def clean(val):
        if val is None or np.isnan(val) or np.isinf(val):
            return 0.0
        return float(val)

    result = tuple(clean(v) for v in [
        mae, rmse, actual_rate, predicted_rate,
        class_f1, class_roc_auc, class_accuracy, class_precision,
        class_recall, class_specificity, rate_delta, top_origin,
        r2, residual_skewness, rolling_std, class_confidence_mean
    ])
    return result


@task
def compute_top_airlines(current_df: pd.DataFrame, cls_preds):
    """Ermittelt die Top‑3 Airlines nach vorhergesagter Verspätungsrate."""
    if current_df.empty or 'marketing_airline_network' not in current_df.columns:
        return []
    df = current_df[['marketing_airline_network']].copy()
    df['predicted_delay'] = cls_preds
    rates = df.groupby('marketing_airline_network')['predicted_delay'].mean().sort_values(ascending=False)
    top = []
    for rank, (airline, rate) in enumerate(rates.head(3).items(), 1):
        top.append({"rank": rank, "airline": airline, "rate": float(rate)})
    return top


@flow(name="drift-detection")
def drift_detection_flow():
    # Monat aus Umgebungsvariable DRIFT_MONTH, sonst Fallback auf aktuellen Monat
    month = int(os.environ.get("DRIFT_MONTH", 0))
    if month == 0:
        engine = create_engine(DB_URI)
        with engine.connect() as conn:
            res = conn.execute(text("SELECT EXTRACT(MONTH FROM MAX(timestamp)) FROM api.predictions"))
            month = int(res.scalar() or 1)
    print(f"Drift-Vergleich für Monat: {month}")

    reference = load_reference_data(month)
    current = load_current_features()

    # Nur Spalten vergleichen, die in beiden Datensätzen vorkommen
    common_cols = list(set(reference.columns) & set(current.columns))
    reference = reference[common_cols]
    current = current[common_cols]
    print(f"Verglichene Spalten: {common_cols}")

    # Numerische Spalten mit minimaler Varianz behalten, Strings separat behandeln
    num_cols = current.select_dtypes(include=[np.number]).columns
    valid_num = [col for col in num_cols if current[col].std() > 0.01]
    str_cols = current.select_dtypes(include=[object]).columns.tolist()
    valid_cols = valid_num + str_cols
    reference = reference[valid_cols]
    current = current[valid_cols]
    print(f"Spalten für Drift-Analyse: {valid_cols}")

    # --- Drift Booster für Demo (aktivierbar über Umgebungsvariable) -------
    if os.environ.get("DRIFT_BOOST", "0") == "1":
        print("☢️ Nuklearer Drift-Boost aktiviert: Alle numerischen Features werden massiv verändert.")
        num_cols = current.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            current[col] = current[col] * 10
        str_cols = current.select_dtypes(include=[object]).columns
        for col in str_cols:
            current[col] = "DRIFTED_" + current[col].astype(str)

    report = compute_drift_report(reference, current)

    # Zusätzliche Metriken aus den Predictions
    preds_df = load_current_predictions()
    (mae, rmse, actual_rate, predicted_rate,
     class_f1, class_roc_auc, class_accuracy, class_precision,
     class_recall, class_specificity, rate_delta, top_origin,
     r2, residual_skewness, rolling_std, class_confidence_mean) = compute_and_send_metrics(preds_df)

    # Top‑Airlines berechnen
    cls_preds = preds_df["prediction_class"].values if not preds_df.empty else []
    top_airlines = compute_top_airlines(current, cls_preds)

    # Drift‑Score aus Evidently holen
    drift_dict = report.as_dict()["metrics"][0]["result"]
    drift_score = drift_dict.get("share_of_drifted_columns", 0.0)
    drift_score = 0.0 if np.isnan(drift_score) or np.isinf(drift_score) else drift_score

    # Alles zusammen an die API senden
    try:
        requests.post(
            "http://api:8000/admin/drift-metrics",
            json={
                "drift_score": drift_score,
                "mae": mae,
                "rmse": rmse,
                "actual_rate": actual_rate,
                "predicted_rate": predicted_rate,
                "class_f1": class_f1,
                "class_roc_auc": class_roc_auc,
                "class_accuracy": class_accuracy,
                "class_precision": class_precision,
                "class_recall": class_recall,
                "class_specificity": class_specificity,
                "rate_delta": rate_delta,
                "top_delay_airport": top_origin,
                "r2": r2,
                "residual_skewness": residual_skewness,
                "stddev_rolling": rolling_std,
                "class_confidence_mean": class_confidence_mean,
            },
            timeout=10,
        )
        print("Metriken an API gesendet.")
    except Exception as e:
        print(f"Fehler beim Senden der Metriken: {e}")

    # Top‑Airlines separat senden
    if top_airlines:
        try:
            requests.post(
                "http://api:8000/admin/top-airlines",
                json={"airlines": top_airlines},
                timeout=5,
            )
            print("Top‑Airlines an API gesendet.")
        except Exception as e:
            print(f"Fehler beim Senden der Top-Airlines: {e}")

    # Report in MLflow loggen
    log_report(report)

    # Exakte prediction_rows aus der Datenbank setzen
    try:
        requests.post("http://api:8000/admin/data-stats", timeout=5)
        print("Data-Stats aktualisiert.")
    except Exception as e:
        print(f"Fehler beim Aktualisieren der Data-Stats: {e}")


    if drift_score > 0.5:  # oder dein dynamischer Schwellwert
        print("Drift-Alarm! Triggere Retraining…")
        requests.post("http://api:8000/admin/drift-alarm", json={"active": 1})
        requests.post("http://api:8000/admin/retrain", timeout=5)


if __name__ == "__main__":
    drift_detection_flow()