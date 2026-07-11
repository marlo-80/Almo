# flows/train_flow.py
"""
Training Flow – Prefect Orchestration for Model Training and Promotion

This Prefect flow trains regression and classification models for flight
delay prediction. It loads data from PostgreSQL, applies preprocessing,
trains a scikit‑learn model, logs metrics to MLflow, and optionally promotes
the model to "champion" if it outperforms the currently registered version.

The flow is driven by configuration dictionaries from `flows/config.py`,
which define features, preprocessing strategies, hyperparameters, and
promotion criteria.

------------------------------------------------------------------------------
Workflow
------------------------------------------------------------------------------
1. Load training data via SQL query (from `dataset_query` in config).
2. Convert integer columns to float for compatibility.
3. Perform chronological split (70% train, 15% validation, 15% test).
4. Build a preprocessing pipeline based on the config:
   - Imputation (numeric/categorical)
   - Encoding (one‑hot, ordinal, target, frequency)
   - Cyclic transformations (sin/cos)
   - Scaling (standard, log‑transform)
5. Create and train the model (RandomForestRegressor/Classifier).
6. Log parameters, metrics, and artifacts to MLflow.
7. Compute the primary evaluation metric (e.g., RMSE, F1).
8. Compare against the current champion model:
   - If better: register as new version with the specified alias ("champion").
   - If not: optionally register without alias (if `register: true`).
9. On promotion:
   - Reset drift alarm via API (`/admin/drift-alarm`).
   - Update champion metrics via API (`/admin/champion-metrics`).
   - Trigger API reload via `/admin/reload-model`.
   - Send retrain status via `/admin/retrain-status`.

------------------------------------------------------------------------------
Configuration (from `flows/config.py`)
------------------------------------------------------------------------------
The flow expects a configuration dictionary with the following structure:

- run_name            : MLflow run name
- task                : 'regression' or 'classification'
- target_type         : 'continuous' or 'binary'
- impute_num          : numeric imputation strategy
- impute_cat          : categorical imputation strategy
- low_card_strategy   : encoding for low‑cardinality features
- high_card_strategy  : encoding for high‑cardinality features
- target              : target column name
- low_cardinality_cols: list of low‑cardinality feature columns
- high_cardinality_cols: list of high‑cardinality feature columns
- cyclic_cols         : cyclic features (sin/cos transformation)
- numeric_cols        : standard numeric features
- skewed_numeric_cols : numeric features with log transformation
- model_type          : e.g., 'RandomForestRegressor' or 'RandomForestClassifier'
- model_params        : hyperparameters (as dict)
- dataset_query       : SQL query for training data
- register            : whether to register in MLflow
- model_name          : registered model name
- alias               : alias for the model (e.g., 'champion')
- promotion_metric    : metric used for comparison
- promotion_mode      : 'minimize' or 'maximize'

For tuning configurations, `param_ranges` and `fixed_model_params` are used
instead of `model_params`.

------------------------------------------------------------------------------
Dependencies
------------------------------------------------------------------------------
- Prefect (flow and task decorators)
- MLflow (tracking and model registry)
- scikit‑learn (RandomForest, preprocessing)
- SQLAlchemy (PostgreSQL connection)
- Requests (API calls)

------------------------------------------------------------------------------
Usage
------------------------------------------------------------------------------
Run the flow with a configuration name:

    python flows/train_flow.py REG       # regression training
    python flows/train_flow.py CLASS     # classification training
    python flows/train_flow.py DRIFT_RETRAIN_REG  # drift retraining

If no argument is given, `DEFAULT_CONFIG` is used (fallback only).

------------------------------------------------------------------------------
Notes
------------------------------------------------------------------------------
- The flow uses a chronological split (not random) to preserve time‑order.
- The `promote_if_better` function handles idempotent registration.
- All database connections use the `testuser` credentials (from .env).
- The API endpoints are expected to be available at `http://api:8000`.
"""


import pandas as pd
from prefect import flow, task
from src.data import load_subset_table
from src.preprocessing import build_preprocessor
from src.train import train_and_log
from sklearn.ensemble import RandomForestRegressor

from flows.config import DEFAULT_CONFIG
from src.data import load_subset_table, convert_integers_to_float

import requests
from mlflow.tracking import MlflowClient

import httpx
from mlflow.tracking import MlflowClient

from src.train import create_model

import mlflow
import requests


@task
def load_and_clean_data(query: str, numeric_cols: list[str]) -> pd.DataFrame:
    """
    Load data from PostgreSQL and convert integer columns to float.

    Args:
        query (str): SQL query to execute.
        numeric_cols (list[str]): List of column names to convert to float64.

    Returns:
        pd.DataFrame: Cleaned DataFrame with converted columns.
    """
    df = load_subset_table(query)
    df = convert_integers_to_float(df, numeric_cols)
    return df

@task
def split_data(df: pd.DataFrame, target: str) -> tuple:
    """
    Split data chronologically into train, validation, and test sets.

    The DataFrame is assumed to be sorted by `flight_date` (ascending).
    Split ratios: 70% train, 15% validation, 15% test.

    Args:
        df (pd.DataFrame): Full dataset.
        target (str): Target column name (unused, kept for consistency).

    Returns:
        tuple: (train_df, val_df, test_df)
    """
    
    n = len(df)
    train_end = int(n * 0.7)
    val_end   = int(n * 0.85)
    train = df.iloc[:train_end]
    val   = df.iloc[train_end:val_end]
    test  = df.iloc[val_end:]
    return train, val, test

@task
def run_training(train_df, val_df, config: dict):
    """
    Build preprocessor, train model, and log to MLflow.

    Args:
        train_df (pd.DataFrame): Training data.
        val_df (pd.DataFrame): Validation data.
        config (dict): Configuration dictionary (see module docstring).

    Returns:
        tuple: (pipeline, score, run_id, artifact_name)
    """
    # Preprocessor und Modell erstellen
    preprocessor = build_preprocessor(
        low_card_cols=config.get("low_cardinality_cols", []),
        high_card_cols=config.get("high_cardinality_cols", []),
        cyclic_cols=config.get("cyclic_cols", []),
        numeric_cols=config.get("numeric_cols", []),
        skewed_numeric_cols=config.get("skewed_numeric_cols", []),
        low_card_strategy=config.get("low_card_strategy", "onehot"),
        high_card_strategy=config.get("high_card_strategy", "target"),
        impute_num=config.get("impute_num", "median"),
        impute_cat=config.get("impute_cat", "most_frequent"),
        target_type=config.get("target_type", "continuous"),
    )
    model = create_model(config["model_type"], config["model_params"])

    pipeline, score, run_id, artifact_name = train_and_log(train_df, val_df, preprocessor, model, config)
    return pipeline, score, run_id, artifact_name

@task
def promote_if_better(config: dict, new_score: float, run_id: str, artifact_name: str):
    """
    Compare the new model against the current champion and promote if better.

    If the new model outperforms the current champion (based on the specified
    `promotion_metric` and `promotion_mode`), it is registered with the alias
    and promoted to champion. Additional API calls are made to update the
    drift alarm, champion metrics, and trigger an API reload.

    Args:
        config (dict): Configuration containing model_name, alias,
                       promotion_metric, promotion_mode.
        new_score (float): Evaluation score of the newly trained model.
        run_id (str): MLflow run ID of the new model.
        artifact_name (str): Artifact path under which the model is logged.

    Returns:
        None
    """
    model_name = config.get("model_name")
    alias = config.get("alias")
    if not alias or not model_name:
        return

    client = MlflowClient()
    metric_name = config.get("promotion_metric", "rmse")
    mode = config.get("promotion_mode", "minimize")
    current_score = None

    try:
        current_mv = client.get_model_version_by_alias(model_name, alias)
        current_run = client.get_run(current_mv.run_id)
        current_score = current_run.data.metrics.get(metric_name)
    except Exception:
        pass

# --------------------------------------------------------------------------
# COMPARISON
# --------------------------------------------------------------------------
    if current_score is None:
        is_better = True
        comp_str = f"{new_score:.4f} (noch kein Champion)"
    elif mode == "minimize":
        is_better = new_score < current_score
        comp_str = f"{new_score:.4f} vs. Champion {current_score:.4f}"
    else:
        is_better = new_score > current_score
        comp_str = f"{new_score:.4f} vs. Champion {current_score:.4f}"

    if is_better:
        print(f"Besser: {metric_name} {comp_str} → wird registriert und zum Champion.")
        model_uri = f"runs:/{run_id}/{artifact_name}"
        try:
            client.get_model_version_by_alias(model_name, alias)
        except Exception:
            pass
        registered = mlflow.register_model(model_uri, model_name)

        # Get run metrics
        run = client.get_run(run_id)
        metrics = run.data.metrics

        # Set tags and description
        important = ["rmse", "mae", "f1", "accuracy", "r2", "specificity"]
        for key in important:
            if key in metrics:
                client.set_model_version_tag(model_name, registered.version, key, str(metrics[key]))
        desc_parts = [f"{k}={metrics[k]:.4f}" for k in important if k in metrics]
        client.update_model_version(model_name, registered.version, description=", ".join(desc_parts))

        client.set_registered_model_alias(model_name, alias, registered.version)
        print(f"New Champion: {model_name} v{registered.version}")

        # Reset drift alarm after new champion has been set
        try:
            requests.post("http://api:8000/admin/drift-alarm", json={"active": 0}, timeout=5)
            
            print("Drift alarm reset")
        except Exception as e:
            print(f"Error: Drift alarm reset: {e}")

        # Send champion metrics to API
        champion_payload = {}
        # Regressor
        for key, api_key in [("rmse", "regressor_rmse"), ("mae", "regressor_mae"),
                             ("r2", "regressor_r2"), ("residual_skewness", "regressor_residual_skewness")]:
            if key in metrics:
                champion_payload[api_key] = metrics[key]
        # Classifier
        for key, api_key in [("f1", "classifier_f1"), ("roc_auc", "classifier_roc_auc"),
                             ("accuracy", "classifier_accuracy"), ("precision", "classifier_precision"),
                             ("recall", "classifier_recall"), ("specificity", "classifier_specificity"),
                             ("confidence_mean", "classifier_confidence_mean")]:
            if key in metrics:
                champion_payload[api_key] = metrics[key]

        if champion_payload:
            try:
                requests.post("http://api:8000/admin/champion-metrics", json=champion_payload, timeout=5)
                print("Send champion metrics to API.")
            except Exception as e:
                print(f"Error: Setting champion metrics not possible: {e}")

        # Trigger API‑reload
        try:
            requests.post("http://api:8000/admin/reload-model", timeout=5)
            # Send retrain status to Grafana
            try:
                requests.post("http://api:8000/admin/retrain-status", json={"new_champion": 1}, timeout=5)
                print("Retrain status sent")
            except Exception as e:
                print(f"Error: Sending of retrain status: {e}")

        except Exception as e:
            print(f"Webhook failed: {e}")

    else:
        print(f"Not better: {metric_name} {comp_str}.")
        if config.get("register", False):
            model_uri = f"runs:/{run_id}/{artifact_name}"
            registered = mlflow.register_model(model_uri, model_name)
            print(f"Model registrated(without alias): {model_name} v{registered.version}")
        else:
            print("Model registration not active. Model will not be registrated.")


@flow(name="flight-delay-training")
def training_pipeline(config: dict = DEFAULT_CONFIG):
    """
    Main Prefect flow for model training and promotion.

    Orchestrates data loading, cleaning, splitting, preprocessing, training,
    and (optionally) promotion of the new model to champion.

    Args:
        config (dict): Configuration dictionary (see module docstring).

    Returns:
        sklearn.pipeline.Pipeline: The trained scikit‑learn pipeline.
    """
    all_cols = (
        config.get("low_cardinality_cols", []) +
        config.get("high_cardinality_cols", []) +
        config.get("cyclic_cols", []) +
        config.get("numeric_cols", []) +
        config.get("skewed_numeric_cols", [])
    )
    df = load_and_clean_data(config["dataset_query"], all_cols)
    train, val, test = split_data(df, config["target"])
    pipeline, score, run_id, artifact_name = run_training(train, val, config)
    if config.get("alias"):
        promote_if_better(config, score, run_id, artifact_name)
    return pipeline


if __name__ == "__main__":
    import sys
    from flows.config import DEFAULT_CONFIG

    config_name = sys.argv[1] if len(sys.argv) > 1 else "DEFAULT_CONFIG"
    import flows.config as cfg_module
    config = getattr(cfg_module, config_name, DEFAULT_CONFIG)
    training_pipeline(config)