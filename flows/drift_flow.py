# flows/drift_flow.py
"""
Drift Detection Flow – Evidently & Prefect Orchestration

This Prefect flow detects data drift between a pre‑COVID reference dataset
(monthly slices from `dbt_staging.pre_covid_100k`) and the most recent
prediction data stored in `api.predictions`. It computes a drift score
using Evidently's DataDriftPreset, along with regression/classification
performance metrics, and sends all results to the FastAPI admin endpoints.

If the drift score exceeds 0.5, the flow triggers an automatic retraining
via the `/admin/retrain` endpoint, which appends current predictions to
`dbt_staging.retrain` and launches a new training pipeline.

------------------------------------------------------------------------------
Workflow
------------------------------------------------------------------------------
1. Determine the target month (from `DRIFT_MONTH` env var, else from latest
   prediction timestamp).
2. Load reference data for that month (all features except targets/IDs).
3. Load current prediction features (from `api.predictions.input_features`).
4. Align columns and filter constant numeric columns.
5. Compute Evidently data drift report.
6. Load current predictions with ground truth and calculate:
   - Regression: MAE, RMSE, R², residual skewness, rolling std.
   - Classification: actual/predicted delay rates, F1, ROC‑AUC, accuracy,
     precision, recall, specificity, confidence mean.
   - Top delay airport (origin) and top 3 airlines by predicted delay rate.
7. Send metrics to `/admin/drift-metrics` and `/admin/top-airlines`.
8. Log the drift report as an HTML artifact to MLflow.
9. If drift_score > 0.5:
   - Set drift alarm via `/admin/drift-alarm`.
   - Trigger retraining via `/admin/retrain`.

------------------------------------------------------------------------------
Environment Variables
------------------------------------------------------------------------------
DRIFT_MONTH          : Integer (1‑12). If not set, derived from latest prediction.
DRIFT_SEED           : Random seed for reproducibility (used in batch_inject).
PREFECT_LOGGING_LEVEL: Prefect log level (default: INFO).

------------------------------------------------------------------------------
Dependencies
------------------------------------------------------------------------------
- Prefect (flow, task decorators)
- Evidently (DataDriftPreset)
- scikit‑learn (metrics)
- scipy.stats (skew)
- SQLAlchemy (PostgreSQL connection)
- MLflow (tracking and logging)
- Requests (API calls)

------------------------------------------------------------------------------
Usage
------------------------------------------------------------------------------
    python flows/drift_flow.py

Typically invoked by the demo script (`covid_data_drift_demo.sh`) with
`DRIFT_MONTH` set for each monthly batch.

------------------------------------------------------------------------------
Notes
------------------------------------------------------------------------------
- The reference table `pre_covid_100k` must be built by dbt before running.
- The flow expects the FastAPI to be available at `http://api:8000`.
- Warning: numpy divide-by-zero warnings are suppressed by `np.seterr` and
  `warnings.filterwarnings` to avoid cluttering the logs.
- The `DRIFT_BOOST` environment variable (when set to "1") artificially
  amplifies drift for demo purposes.
"""

import numpy as np
np.float_ = np.float64          # Workaround for NumPy‑2‑compatibility of Evidently

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
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
np.seterr(divide='ignore', invalid='ignore')

# --------------------------------------------------------------------------
# CONFIGURATION
# --------------------------------------------------------------------------
DB_URI = "postgresql://testuser:testuser@postgres:5432/fastapi_db"
MLFLOW_URI = "http://mlflow:5000"

# --------------------------------------------------------------------------
# DATA LOADING TASKS
# --------------------------------------------------------------------------

@task
def load_reference_data(month: int):
    """Load reference data only for the given month (1‑12) from pre_covid_100k."""
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
    print(f"Reference data loaded (Month {month}): {df.shape[0]} rows, {df.shape[1]} columns")
    return df


@task
def load_current_features():
    """Load only the input_features (JSONB) from api.predictions (for Evidently)."""
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
        raise ValueError("No predictions found in api.predictions.")
    print(f"Current data loaded: {records.shape[0]} rows, {records.shape[1]} columns")
    return records


    # Fair reference: Reduce the larger dataset to the size of the smaller one
    min_len = min(len(reference), len(current))
    if len(reference) > min_len:
        reference = reference.sample(n=min_len, random_state=42)
    if len(current) > min_len:
        current = current.sample(n=min_len, random_state=24)

    print(f"Adjusted sizes: Reference {len(reference)}, Current {len(current)}")

# --------------------------------------------------------------------------
# DRIFT REPORT TASKS
# --------------------------------------------------------------------------

@task
def compute_drift_report(reference: pd.DataFrame, current: pd.DataFrame) -> Report:
    """Create an Evidently Data Drift Report."""
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current, column_mapping=None)
    print("Drift report created.")
    return report


@task
def log_report(report: Report):
    """Save the report as HTML in MLflow."""
    mlflow.set_tracking_uri(MLFLOW_URI)
    with mlflow.start_run(run_name="drift_report"):
        report.save_html("/tmp/drift_report.html")
        mlflow.log_artifact("/tmp/drift_report.html", "drift_reports")
    print("Drift report logged to MLflow.")

# --------------------------------------------------------------------------
# METRICS TASKS
# --------------------------------------------------------------------------

@task
def load_current_predictions():
    """Load prediction_reg, prediction_class, prediction_class_proba, ground_truth and input_features from api.predictions."""
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
    # Extract origin_airport_id from the stored features (JSONB)
    df["origin_airport_id"] = df["input_features"].apply(
        lambda x: x.get("origin_airport_id") if isinstance(x, dict) else None
    )
    df = df.dropna(subset=["true_reg", "true_class"])
    print(f"Validation data loaded: {df.shape[0]} rows")
    return df


@task
def compute_and_send_metrics(preds_df: pd.DataFrame):
    """Compute all drift and performance metrics of the current batch."""
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

        # Rolling standard deviation of the last 100 regression predictions
        rolling_std = float(preds_df["prediction_reg"].tail(100).std()) if len(preds_df) >= 100 else float(preds_df["prediction_reg"].std())

        # Rates
        actual_rate    = float(y_true_cls.mean())
        predicted_rate = float(y_pred_cls.mean())
        rate_delta     = predicted_rate - actual_rate

        # Classification (only if both classes are present)
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

        # Mean prediction probability (class 1) from the new column
        if "prediction_class_proba" in preds_df.columns:
            class_confidence_mean = float(preds_df["prediction_class_proba"].mean())
        else:
            class_confidence_mean = 0.0

        # Airport with the most delays (Origin)
        if 'origin_airport_id' in preds_df.columns:
            top_origin = float(preds_df["origin_airport_id"].value_counts().idxmax())
        else:
            top_origin = 0.0

    # Catch NaN/Inf
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
    """Determine the top‑3 airlines by predicted delay rate."""
    if current_df.empty or 'marketing_airline_network' not in current_df.columns:
        return []
    df = current_df[['marketing_airline_network']].copy()
    df['predicted_delay'] = cls_preds
    rates = df.groupby('marketing_airline_network')['predicted_delay'].mean().sort_values(ascending=False)
    top = []
    for rank, (airline, rate) in enumerate(rates.head(3).items(), 1):
        top.append({"rank": rank, "airline": airline, "rate": float(rate)})
    return top

# --------------------------------------------------------------------------
# MAIN FLOW
# --------------------------------------------------------------------------

@flow(name="drift-detection")
def drift_detection_flow():
    # Month from environment variable DRIFT_MONTH, otherwise fallback to current month
    month = int(os.environ.get("DRIFT_MONTH", 0))
    if month == 0:
        engine = create_engine(DB_URI)
        with engine.connect() as conn:
            res = conn.execute(text("SELECT EXTRACT(MONTH FROM MAX(timestamp)) FROM api.predictions"))
            month = int(res.scalar() or 1)
    print(f"Drift comparison for month: {month}")

    reference = load_reference_data(month)
    current = load_current_features()

    # Only compare columns that exist in both datasets
    common_cols = list(set(reference.columns) & set(current.columns))
    reference = reference[common_cols]
    current = current[common_cols]
    print(f"Compared columns: {common_cols}")

    # Keep numeric columns with minimal variance, handle strings separately
    num_cols = current.select_dtypes(include=[np.number]).columns
    valid_num = [col for col in num_cols if current[col].std() > 0.01]
    str_cols = current.select_dtypes(include=[object]).columns.tolist()
    valid_cols = valid_num + str_cols
    reference = reference[valid_cols]
    current = current[valid_cols]
    print(f"Columns for drift analysis: {valid_cols}")

    # --- Drift Booster for demo (activatable via environment variable) -------
    if os.environ.get("DRIFT_BOOST", "0") == "1":
        print("☢️ Nuclear drift boost activated: All numeric features are massively altered.")
        num_cols = current.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            current[col] = current[col] * 10
        str_cols = current.select_dtypes(include=[object]).columns
        for col in str_cols:
            current[col] = "DRIFTED_" + current[col].astype(str)

    report = compute_drift_report(reference, current)

    # Additional metrics from predictions
    preds_df = load_current_predictions()
    (mae, rmse, actual_rate, predicted_rate,
     class_f1, class_roc_auc, class_accuracy, class_precision,
     class_recall, class_specificity, rate_delta, top_origin,
     r2, residual_skewness, rolling_std, class_confidence_mean) = compute_and_send_metrics(preds_df)

    # Top airlines calculation
    cls_preds = preds_df["prediction_class"].values if not preds_df.empty else []
    top_airlines = compute_top_airlines(current, cls_preds)

    # Get drift score from Evidently
    drift_dict = report.as_dict()["metrics"][0]["result"]
    drift_score = drift_dict.get("share_of_drifted_columns", 0.0)
    drift_score = 0.0 if np.isnan(drift_score) or np.isinf(drift_score) else drift_score

    # Send everything to the API
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
        print("Metrics sent to API.")
    except Exception as e:
        print(f"Error sending metrics: {e}")

    # Send top airlines separately
    if top_airlines:
        try:
            requests.post(
                "http://api:8000/admin/top-airlines",
                json={"airlines": top_airlines},
                timeout=5,
            )
            print("Top airlines sent to API.")
        except Exception as e:
            print(f"Error sending top airlines: {e}")

    # Log report to MLflow
    log_report(report)

    # Set exact prediction_rows from the database
    try:
        requests.post("http://api:8000/admin/data-stats", timeout=5)
        print("Data stats updated.")
    except Exception as e:
        print(f"Error updating data stats: {e}")


    if drift_score > 0.5:  # or your dynamic threshold
        print("Drift alarm! Triggering retraining...")
        requests.post("http://api:8000/admin/drift-alarm", json={"active": 1})
        requests.post("http://api:8000/admin/retrain", timeout=5)


if __name__ == "__main__":
    drift_detection_flow()