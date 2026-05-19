# flows/tune_flow.py
import pandas as pd
import optuna
from prefect import flow, task
from src.data import load_subset_table, convert_integers_to_float
from src.preprocessing import build_preprocessor
from src.train import train_and_log, create_model
from flows.config import OPTUNA_CONFIG

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
    """Baut den Preprocessor aus der Config."""
    return build_preprocessor(
        numeric_cols=config["numeric_cols"],
        categorical_cols=config["categorical_cols"],
        impute_num=config.get("impute_num", "median"),
        impute_cat=config.get("impute_cat", "most_frequent"),
    )

@task
def run_optuna_study(train_df, val_df, preprocessor, config: dict):
    """Führt die Optuna-Studie durch, gibt bestes Modell und Score zurück."""
    def objective(trial):
        # Trial-spezifische Parameter vorschlagen
        model_params = {}
        for pname, prange in config["param_ranges"].items():
            if prange["type"] == "int":
                model_params[pname] = trial.suggest_int(pname, prange["low"], prange["high"])
            elif prange["type"] == "float":
                log = prange.get("log", False)
                model_params[pname] = trial.suggest_float(pname, prange["low"], prange["high"], log=log)
            # weitere Typen bei Bedarf ergänzen

        # Config für diesen Trial zusammenbauen
        trial_config = {**config, "model_params": model_params,
                        "run_name": f"{config['run_name']}_trial{trial.number}"}

        model = create_model(config["model_type"], model_params)
        pipeline, rmse = train_and_log(train_df, val_df, preprocessor, model, trial_config)
        return rmse

    study = optuna.create_study(
        study_name=config["run_name"],
        direction=config["direction"],
    )
    study.optimize(objective, n_trials=config["n_trials"])

    best_params = study.best_params
    best_score = study.best_value
    print(f"Optuna abgeschlossen. Beste Parameter: {best_params}, Bester RMSE: {best_score:.2f}")

    # Bestes Modell einmal final trainieren und ggf. registrieren
    final_config = {**config,
                    "run_name": f"{config['run_name']}_best",
                    "model_params": best_params}
    if config.get("register", False) and config.get("alias"):
        final_config["register"] = True
        final_config["alias"] = config["alias"]
    else:
        final_config["register"] = False

    model = create_model(config["model_type"], best_params)
    pipeline_best, _ = train_and_log(train_df, val_df, preprocessor, model, final_config)
    return best_params, best_score

@flow(name="optuna-flight-delay-tuning")
def tuning_pipeline(config: dict = OPTUNA_CONFIG):
    train, val, test = load_and_prepare_data(config)
    preprocessor = build_preprocessor_task(config)
    best_params, best_score = run_optuna_study(train, val, preprocessor, config)
    return best_params, best_score

if __name__ == "__main__":
    tuning_pipeline()