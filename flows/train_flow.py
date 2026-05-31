# flows/train_flow.py
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
    df = load_subset_table(query)
    df = convert_integers_to_float(df, numeric_cols)
    return df

@task
def split_data(df: pd.DataFrame, target: str) -> tuple:
    """Chronologischer Split (Tabelle ist bereits nach flight_date sortiert)."""
    n = len(df)
    train_end = int(n * 0.7)
    val_end   = int(n * 0.85)
    train = df.iloc[:train_end]
    val   = df.iloc[train_end:val_end]
    test  = df.iloc[val_end:]
    return train, val, test

@task
def run_training(train_df, val_df, config: dict):
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

    # Vergleich durchführen (immer)
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

        # Run-Metriken holen
        run = client.get_run(run_id)
        metrics = run.data.metrics

        # Tags und Beschreibung setzen
        important = ["rmse", "mae", "f1", "accuracy", "r2", "specificity"]
        for key in important:
            if key in metrics:
                client.set_model_version_tag(model_name, registered.version, key, str(metrics[key]))
        desc_parts = [f"{k}={metrics[k]:.4f}" for k in important if k in metrics]
        client.update_model_version(model_name, registered.version, description=", ".join(desc_parts))

        client.set_registered_model_alias(model_name, alias, registered.version)
        print(f"Neuer Champion: {model_name} v{registered.version}")

        # Champion-Metriken dynamisch an die API senden (nur vorhandene)
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
                print("Champion-Metriken an API gesendet.")
            except Exception as e:
                print(f"Fehler beim Setzen der Champion-Metriken: {e}")

        # API‑Reload triggern
        try:
            requests.post("http://api:8000/admin/reload-model", timeout=5)
        except Exception as e:
            print(f"Webhook failed: {e}")

    else:
        print(f"Nicht besser: {metric_name} {comp_str}.")
        if config.get("register", False):
            model_uri = f"runs:/{run_id}/{artifact_name}"
            registered = mlflow.register_model(model_uri, model_name)
            print(f"Modell registriert (ohne Alias): {model_name} v{registered.version}")
        else:
            print("Registrierung nicht aktiv – Modell wird nicht registriert.")


@flow(name="flight-delay-training")
def training_pipeline(config: dict = DEFAULT_CONFIG):
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