# src/train.py
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

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
EXPERIMENT_NAME = "flight-delay-prediction"


def train_and_log(
    train_df,
    val_df,
    preprocessor,
    model,
    config: dict,
):
    """Trainiert die vollständige Pipeline, loggt Metriken und Artifacts nach MLflow."""
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    target = config["target"]
    task = config.get("task", "regression")

    # Zielspalte abtrennen
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

    # Platzhalter für eventuelle Confusion-Matrix-Figuren
    fig_cm, fig_cm_norm = None, None

    if task == "classification":
        y_val_class = y_val.astype(int)
        y_pred_class = preds.astype(int)
        y_pred_proba = pipeline.predict_proba(X_val)[:, 1]

        accuracy = accuracy_score(y_val_class, y_pred_class)
        precision = precision_score(y_val_class, y_pred_class)
        recall = recall_score(y_val_class, y_pred_class)
        f1 = f1_score(y_val_class, y_pred_class)
        roc_auc = roc_auc_score(y_val_class, y_pred_proba)

        # Wichtig: metrics zuerst bauen ...
        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": roc_auc,
        }

        # ... dann den Score daraus ableiten
        if "tuning_metric" in config:
            score = config["tuning_metric"]
        else:
            score = config.get("promotion_metric", "f1")

        score_value = metrics.get(score, list(metrics.values())[0])

        # Confusion Matrix erstellen
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

        y_true_bin = val_df["arr_del15"].astype(int)
        y_pred_bin = (preds > 15).astype(int)
        precision = precision_score(y_true_bin, y_pred_bin)
        recall = recall_score(y_true_bin, y_pred_bin)
        f1 = f1_score(y_true_bin, y_pred_bin)
        accuracy = accuracy_score(y_true_bin, y_pred_bin)
        roc_auc  = roc_auc_score(y_true_bin, preds)   # preds sind die stetigen Scores

        # Wichtig: metrics zuerst bauen ...
        metrics = {
            "mae": mae,
            "mse": mse,
            "rmse": rmse,
            "medae": medae,
            "r2": r2,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": roc_auc,
        }

        # ... dann den Score daraus ableiten
        if "tuning_metric" in config:
            score = config["tuning_metric"]
        else:
            score = config.get("promotion_metric", "rmse")

        score_value = metrics.get(score, list(metrics.values())[0])

    # Alles in einem Run loggen (nur ein start_run!)
    with mlflow.start_run(run_name=config.get("run_name", "custom_run")):
        # Dataset-Logging
        dataset = mlflow.data.from_pandas(
            train_df,
            name=config.get("dataset_name", "unnamed_dataset"),
            targets=target,
            source=config.get("dataset_source", "unknown_source"),
        )
        mlflow.log_input(dataset, context="training")

        # Parameter flach loggen
        flat_params = {}
        for key, value in config.items():
            if isinstance(value, (list, dict)):
                flat_params[key] = str(value)
            else:
                flat_params[key] = value
        mlflow.log_params(flat_params)

        # Metriken loggen
        mlflow.log_metrics(metrics)

        # Confusion-Matrix-Figuren loggen (nur wenn vorhanden)
        if fig_cm is not None:
            mlflow.log_figure(fig_cm, "confusion_matrix.png")
        if fig_cm_norm is not None:
            mlflow.log_figure(fig_cm_norm, "confusion_matrix_norm.png")
        plt.close("all")

        # Signatur nur mit echten Feature-Spalten
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
    if model_type == "RandomForestRegressor":
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(**model_params)
    elif model_type == "RandomForestClassifier":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(**model_params)
    raise ValueError(f"Unknown model_type: {model_type}")