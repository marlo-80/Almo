# src/train.py
"""
Training Module – Model Training and MLflow Logging

This module provides the core training logic for the flight delay prediction
models. It handles:

- Training a scikit‑learn pipeline (preprocessor + model) on the training set
- Computing a comprehensive set of regression or classification metrics
- Logging parameters, metrics, artifacts, and confusion matrices to MLflow
- Registering the model with MLflow (called by the flow)

The module supports both regression and classification tasks. For regression,
it additionally computes classification metrics (accuracy, precision, recall,
F1, ROC‑AUC) using a threshold of >15 minutes to define a delay.

------------------------------------------------------------------------------
File Location
------------------------------------------------------------------------------
- Host:   ./docker/src/train.py
- Container: /app/src/train.py

------------------------------------------------------------------------------
Dependencies
------------------------------------------------------------------------------
- MLflow (tracking, logging)
- scikit‑learn (metrics, pipeline)
- numpy, scipy.stats (skew)
- matplotlib, seaborn (confusion matrix plots)

------------------------------------------------------------------------------
Usage
------------------------------------------------------------------------------
The main entry point is `train_and_log()`, which is called by the training
flows (`train_flow.py` and `tune_flow.py`).

    from src.train import train_and_log, create_model

    model = create_model("RandomForestRegressor", {"n_estimators": 20})
    pipeline, score, run_id, artifact_name = train_and_log(
        train_df, val_df, preprocessor, model, config
    )
"""

import mlflow
import mlflow.sklearn
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    mean_squared_error,
    median_absolute_error,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_auc_score,
)
import os
import numpy as np
from scipy.stats import skew
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================================
# Constants
# ============================================================================
MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
EXPERIMENT_NAME = "flight-delay-prediction"


# ============================================================================
# Functions
# ============================================================================
def train_and_log(
    train_df,
    val_df,
    preprocessor,
    model,
    config: dict,
):
    """
    Train the full pipeline, compute metrics, and log everything to MLflow.

    This function:
    1. Splits the target from features.
    2. Builds a scikit‑learn Pipeline with the given preprocessor and model.
    3. Fits the pipeline on the training data.
    4. Predicts on the validation set.
    5. Computes a comprehensive set of metrics (depending on task).
    6. Logs all metrics, parameters, and artifacts (confusion matrix plots)
       to MLflow.
    7. Returns the trained pipeline, the primary evaluation score, the MLflow
       run ID, and the artifact path.

    For regression tasks, it also calculates classification metrics using a
    threshold of 15 minutes to define a delayed flight (arr_del15 = 1 if
    arr_delay_minutes > 15).

    Args:
        train_df (pd.DataFrame): Training data (includes target column).
        val_df (pd.DataFrame): Validation data (includes target column).
        preprocessor (sklearn.compose.ColumnTransformer): Preprocessing pipeline.
        model (sklearn.base.BaseEstimator): Unfitted scikit‑learn model.
        config (dict): Configuration dictionary containing:
            - "target": Target column name.
            - "task": "regression" or "classification".
            - "run_name" (optional): MLflow run name.
            - "dataset_name" (optional): Name for MLflow dataset logging.
            - "dataset_source" (optional): Source table name.
            - "promotion_metric" (optional): Metric used for promotion.
            - "tuning_metric" (optional): Metric used for tuning (overrides promotion_metric).
            - Feature column lists for signature inference.

    Returns:
        tuple: (pipeline, score_value, run_id, artifact_name)
            - pipeline: Trained scikit‑learn Pipeline.
            - score_value (float): The primary evaluation metric.
            - run_id (str): MLflow run ID of the logged run.
            - artifact_name (str): The artifact path used for the model.

    Raises:
        ValueError: If the model type is unsupported (handled by create_model).
    """
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    target = config["target"]
    task = config.get("task", "regression")

    # Split features and target
    X_train = train_df.drop(columns=[target]).copy()
    y_train = train_df[target]
    X_val = val_df.drop(columns=[target])
    y_val = val_df[target]

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_val)

    # Placeholders for confusion matrix figures
    fig_cm, fig_cm_norm = None, None

    if task == "classification":
        # --- Classification Metrics ---
        y_val_class = y_val.astype(int)
        y_pred_class = preds.astype(int)
        y_pred_proba = pipeline.predict_proba(X_val)[:, 1]

        accuracy = accuracy_score(y_val_class, y_pred_class)
        precision = precision_score(y_val_class, y_pred_class)
        recall = recall_score(y_val_class, y_pred_class)
        f1 = f1_score(y_val_class, y_pred_class)
        roc_auc = roc_auc_score(y_val_class, y_pred_proba)

        # Specificity (True Negative Rate)
        tp = ((y_val_class == 1) & (y_pred_class == 1)).sum()
        tn = ((y_val_class == 0) & (y_pred_class == 0)).sum()
        fp = ((y_val_class == 0) & (y_pred_class == 1)).sum()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        # Mean confidence (predicted probability for class 1)
        confidence_mean = float(np.mean(y_pred_proba))

        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": roc_auc,
            "specificity": specificity,
            "confidence_mean": confidence_mean,
        }

        if "tuning_metric" in config:
            score = config["tuning_metric"]
        else: #regression
            score = config.get("promotion_metric", "f1")
        score_value = metrics.get(score, list(metrics.values())[0])

        # Confusion Matrix
        cm = confusion_matrix(y_val_class, y_pred_class)
        fig_cm, ax_cm = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm)
        ax_cm.set_xlabel("Predicted")
        ax_cm.set_ylabel("True")
        ax_cm.set_title(f"Confusion Matrix – {config.get('run_name')}")

        cm_norm = confusion_matrix(y_val_class, y_pred_class, normalize="true")
        fig_cm_norm, ax_cm_norm = plt.subplots()
        sns.heatmap(cm_norm, annot=True, fmt=".2%", cmap="Greens", ax=ax_cm_norm)
        ax_cm_norm.set_xlabel("Predicted")
        ax_cm_norm.set_ylabel("True")
        ax_cm_norm.set_title(f"Normalized Confusion Matrix – {config.get('run_name')}")

    else:   # regression
        mae = mean_absolute_error(y_val, preds)
        mse = mean_squared_error(y_val, preds)
        rmse = mse ** 0.5
        medae = median_absolute_error(y_val, preds)
        r2 = r2_score(y_val, preds)

        # Residual skewness
        residuals = y_val - preds
        residual_skewness = float(skew(residuals))

        y_true_bin = val_df["arr_del15"].astype(int)
        y_pred_bin = (preds > 15).astype(int)
        precision = precision_score(y_true_bin, y_pred_bin)
        recall = recall_score(y_true_bin, y_pred_bin)
        f1 = f1_score(y_true_bin, y_pred_bin)
        accuracy = accuracy_score(y_true_bin, y_pred_bin)
        roc_auc  = roc_auc_score(y_true_bin, preds)

        metrics = {
            "mae": mae,
            "mse": mse,
            "rmse": rmse,
            "medae": medae,
            "r2": r2,
            "residual_skewness": residual_skewness,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": roc_auc,
        }

        if "tuning_metric" in config:
            score = config["tuning_metric"]
        else:
            score = config.get("promotion_metric", "rmse")
        score_value = metrics.get(score, list(metrics.values())[0])

    # --- MLflow Logging ---
    with mlflow.start_run(run_name=config.get("run_name", "custom_run")):
        # Log dataset information
        dataset = mlflow.data.from_pandas(
            train_df,
            name=config.get("dataset_name", "unnamed_dataset"),
            targets=target,
            source=config.get("dataset_source", "unknown_source"),
        )
        mlflow.log_input(dataset, context="training")

        # Log all config parameters (flattening lists/dicts)
        flat_params = {}
        for key, value in config.items():
            if isinstance(value, (list, dict)):
                flat_params[key] = str(value)
            else:
                flat_params[key] = value
        mlflow.log_params(flat_params)

        # Log metrics
        mlflow.log_metrics(metrics)

        # Log confusion matrix figures if available
        if fig_cm is not None:
            mlflow.log_figure(fig_cm, "confusion_matrix.png")
        if fig_cm_norm is not None:
            mlflow.log_figure(fig_cm_norm, "confusion_matrix_norm.png")
        plt.close("all")

        # Log model with signature (only on feature columns)
        feature_cols = (
            config.get("low_cardinality_cols", []) +
            config.get("high_cardinality_cols", []) +
            config.get("cyclic_cols", []) +
            config.get("numeric_cols", []) +
            config.get("skewed_numeric_cols", [])
        )
        X_train[config["numeric_cols"]] = X_train[config["numeric_cols"]].astype(float)
        X_train_signature = X_train[feature_cols]

        artifact_name = config.get("run_name", "full_pipeline")
        mlflow.sklearn.log_model(
            pipeline,
            artifact_path=artifact_name,
            signature=mlflow.models.infer_signature(X_train_signature, y_train),
        )

        run_id = mlflow.active_run().info.run_id
        
    return pipeline, score_value, run_id, artifact_name


def create_model(model_type: str, model_params: dict):
    """
    Instantiate a scikit‑learn model based on the given type and parameters.

    This is a factory function that returns an unfitted model instance.
    Supported model types are:
        - "RandomForestRegressor"
        - "RandomForestClassifier"

    Args:
        model_type (str): The class name of the model.
        model_params (dict): Keyword arguments to pass to the model constructor.

    Returns:
        sklearn.base.BaseEstimator: An unfitted model instance.

    Raises:
        ValueError: If the model_type is not supported.
    """
    if model_type == "RandomForestRegressor":
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(**model_params)
    elif model_type == "RandomForestClassifier":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(**model_params)
    raise ValueError(f"Unknown model_type: {model_type}")