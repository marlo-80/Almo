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
)
import os

MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
EXPERIMENT_NAME = "flight-delay-prediction"


def train_and_log(
    train_df,
    val_df,
    preprocessor,
    model,
    config: dict,
) -> Pipeline:
    """Trainiert die vollständige Pipeline, loggt Metriken und Artifacts nach MLflow."""
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    target = config["target"]

    # Zielspalte abtrennen
    X_train = train_df.drop(columns=[target])
    y_train = train_df[target]
    X_val = val_df.drop(columns=[target])
    y_val = val_df[target]

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_val)

    # Metriken
    mae = mean_absolute_error(y_val, preds)
    mse = mean_squared_error(y_val, preds)
    rmse = mse ** 0.5
    medae = median_absolute_error(y_val, preds)
    r2 = r2_score(y_val, preds)

    # Optional: binäre Klassifikationsmetriken (Schwellwert 15 Min.)
    threshold = config.get("delay_threshold", 15)
    y_true_bin = (y_val > threshold).astype(int)
    y_pred_bin = (preds > threshold).astype(int)
    precision = precision_score(y_true_bin, y_pred_bin)
    recall = recall_score(y_true_bin, y_pred_bin)
    f1 = f1_score(y_true_bin, y_pred_bin)

    with mlflow.start_run(run_name=config.get("run_name", "custom_run")):
        # Dataset-Logging mit dem vollständigen Trainings-DataFrame
        dataset = mlflow.data.from_pandas(
            train_df,
            name=config.get("dataset_name", "unnamed_dataset"),
            targets=target,
            source=config.get("dataset_source", "unknown_source"),
        )
        mlflow.log_input(dataset, context="training")

        flat_params = {}
        for key, value in config.items():
            if isinstance(value, (list, dict)):
                flat_params[key] = str(value)
            else:
                flat_params[key] = value
        mlflow.log_params(flat_params)
        mlflow.log_metrics({
            "mae": mae,
            "mse": mse,
            "rmse": rmse,
            "medae": medae,
            "r2": r2,
            "precision_15": precision,
            "recall_15": recall,
            "f1_15": f1,
        })
        mlflow.sklearn.log_model(
            pipeline,
            name=config.get("run_name", "full_pipeline"),
            )

        # Optional registrieren, falls in config gewünscht
        if config.get("register", False):
            run_id = mlflow.active_run().info.run_id
            model_uri = f"runs:/{run_id}/full_pipeline"
            registered = mlflow.register_model(
                model_uri,
                config.get("model_name", "flight-delay-baseline"),
            )
            if config.get("alias"):
                client = mlflow.tracking.MlflowClient()
                client.set_registered_model_alias(
                    config["model_name"],
                    config["alias"],
                    registered.version,
                )

    return pipeline, rmse



def create_model(model_type: str, model_params: dict):
    if model_type == "RandomForestRegressor":
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(**model_params)
    # später z. B. "KerasRegressor" oder "PyTorchRegressor" ergänzen
    raise ValueError(f"Unknown model_type: {model_type}")