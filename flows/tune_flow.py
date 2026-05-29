# flows/tune_flow.py
import sys
import mlflow                             
import pandas as pd
import optuna
from prefect import flow, task
from src.data import load_subset_table, convert_integers_to_float
from src.preprocessing import build_preprocessor
from src.train import train_and_log, create_model
from flows.config import DEFAULT_CONFIG
from flows.train_flow import promote_if_better
from mlflow.tracking import MlflowClient

import flows.config as cfg_module


@task
def load_and_prepare_data(config: dict) -> tuple:
    """Lädt die Rohdaten, konvertiert Integer und splittet chronologisch."""
    df = load_subset_table(config["dataset_query"])
    df = convert_integers_to_float(df, config["numeric_cols"])
    n = len(df)
    train_end = int(n * 0.7)
    val_end   = int(n * 0.85)
    train = df.iloc[:train_end]
    val   = df.iloc[train_end:val_end]
    test  = df.iloc[val_end:]
    return train, val, test

@task
def build_preprocessor_task(config: dict):
    return build_preprocessor(
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

@task
def run_optuna_study(train_df, val_df, preprocessor, config: dict):
    def objective(trial):
        model_params = {}
        for pname, prange in config["param_ranges"].items():
            if prange["type"] == "int":
                model_params[pname] = trial.suggest_int(pname, prange["low"], prange["high"])
            elif prange["type"] == "float":
                log = prange.get("log", False)
                model_params[pname] = trial.suggest_float(pname, prange["low"], prange["high"], log=log)

        fixed = config.get("fixed_model_params", {})
        model_params.update(fixed)

        trial_config = {**config, "model_params": model_params,
                        "run_name": f"{config['run_name']}_trial{trial.number}"}

        model = create_model(config["model_type"], model_params)
        pipeline, score, run_id, artifact_name = train_and_log(train_df, val_df, preprocessor, model, trial_config)

        trial.set_user_attr("run_id", run_id)
        trial.set_user_attr("run_name", trial_config["run_name"])
        trial.set_user_attr("artifact_name", artifact_name)

        return score

    study = optuna.create_study(
        study_name=config["run_name"],
        direction=config["tuning_direction"],
    )
    study.optimize(objective, n_trials=config["n_trials"])

    best_trial = study.best_trial
    best_score = best_trial.value
    best_params = best_trial.params
    best_run_id = best_trial.user_attrs["run_id"]
    best_run_name = best_trial.user_attrs["run_name"]
    best_artifact_name = best_trial.user_attrs["artifact_name"]

    # Markierung in MLflow setzen
    client = MlflowClient()
    client.set_tag(best_run_id, "mlflow.runName", f"!!!_{best_run_name}")

    # Immer Champion-Prüfung durchführen (mit Registrierung, wenn besser)
    if config.get("alias"):
        promote_if_better(config, best_score, best_run_id, best_artifact_name)

    return best_params, best_score

@flow(name="optuna-flight-delay-tuning")
def tuning_pipeline(config: dict = DEFAULT_CONFIG):
    train, val, test = load_and_prepare_data(config)
    preprocessor = build_preprocessor_task(config)
    best_params, best_score = run_optuna_study(train, val, preprocessor, config)
    return best_params, best_score

if __name__ == "__main__":
    # Config-Name über Kommandozeile wählen, z. B. DEFAULT_CONFIG oder MY_TUNE
    config_name = sys.argv[1] if len(sys.argv) > 1 else "DEFAULT_CONFIG"
    config = getattr(cfg_module, config_name, DEFAULT_CONFIG)
    tuning_pipeline(config)