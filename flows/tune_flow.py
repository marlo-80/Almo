# flows/tune_flow.py
"""
Tuning Flow – Optuna Hyperparameter Optimization with Prefect

This Prefect flow performs hyperparameter tuning using Optuna for the flight
delay prediction models. It evaluates multiple hyperparameter combinations
via cross-validation on the training set, logs each trial to MLflow, and
automatically promotes the best-performing model to "champion" if it
outperforms the currently registered version.

The flow is driven by configuration dictionaries from `flows/config.py`
that contain search space definitions (`param_ranges`) and fixed parameters.

------------------------------------------------------------------------------
Workflow
------------------------------------------------------------------------------
1. Load training data via SQL query (from `dataset_query` in config).
2. Convert integer columns to float.
3. Perform chronological split (70% train, 15% validation, 15% test).
4. Build a preprocessing pipeline (same as in `train_flow.py`).
5. Run Optuna study with n_trials:
   - For each trial: sample hyperparameters from `param_ranges`.
   - Train and evaluate the model on the validation set.
   - Log trial parameters, metrics, and artifacts to MLflow.
   - Store run_id and artifact_path in trial user attributes.
6. After the study: retrieve the best trial's run_id and score.
7. Apply `promote_if_better` logic to register the best model as champion
   if it outperforms the current champion.
8. Return the best parameters and score.

------------------------------------------------------------------------------
Configuration (from `flows/config.py`)
------------------------------------------------------------------------------
The flow expects a configuration dictionary with the following fields:

- run_name            : MLflow run name (used as study name)
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
- param_ranges        : search space for hyperparameters (dict)
- fixed_model_params  : fixed hyperparameters (dict)
- n_trials            : number of Optuna trials
- tuning_metric       : metric to optimize (e.g., 'rmse', 'precision')
- tuning_direction    : 'minimize' or 'maximize'
- dataset_query       : SQL query for training data
- register            : whether to register in MLflow
- model_name          : registered model name
- alias               : alias for the model (e.g., 'champion')
- promotion_metric    : metric used for comparison (same as tuning_metric)
- promotion_mode      : 'minimize' or 'maximize'

------------------------------------------------------------------------------
Dependencies
------------------------------------------------------------------------------
- Prefect (flow and task decorators)
- Optuna (hyperparameter search)
- MLflow (tracking and model registry)
- scikit‑learn (RandomForest, preprocessing)
- SQLAlchemy (PostgreSQL connection)

------------------------------------------------------------------------------
Usage
------------------------------------------------------------------------------
Run the flow with a configuration name:

    python flows/tune_flow.py OPTUNA_REG       # regression tuning
    python flows/tune_flow.py OPTUNA_CLASS     # classification tuning

If no argument is given, `DEFAULT_CONFIG` is used (fallback only).

------------------------------------------------------------------------------
Notes
------------------------------------------------------------------------------
- The flow uses a chronological split (not random) to preserve time‑order.
- Each trial is logged as a separate MLflow run with the naming pattern:
  `<run_name>_trial<N>`.
- The best trial's run name is prefixed with "!!!" for easy identification
  in the MLflow UI.
- The `promote_if_better` function is reused from `train_flow.py` to ensure
  consistent promotion logic across both flows.
"""

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
    """
    Load training data and perform chronological split.

    Args:
        config (dict): Configuration dictionary containing `dataset_query`
                       and `numeric_cols`.

    Returns:
        tuple: (train_df, val_df, test_df) split as 70/15/15.
    """
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
    """
    Build the preprocessing pipeline based on the configuration.

    Args:
        config (dict): Configuration dictionary with feature and strategy
                       definitions.

    Returns:
        sklearn.compose.ColumnTransformer: The preprocessor pipeline.
    """
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
    """
    Run an Optuna hyperparameter optimization study.

    Each trial samples hyperparameters from the `param_ranges` defined in the
    config, trains a model, evaluates it on the validation set, and logs
    results to MLflow. The best trial is automatically promoted to champion
    if it outperforms the currently registered model.

    Args:
        train_df (pd.DataFrame): Training data.
        val_df (pd.DataFrame): Validation data.
        preprocessor (sklearn.compose.ColumnTransformer): Preprocessing pipeline.
        config (dict): Configuration dictionary with `param_ranges`,
                       `n_trials`, `tuning_direction`, `model_type`, and
                       promotion-related fields.

    Returns:
        tuple: (best_params, best_score) – the optimal hyperparameters and
               their corresponding evaluation score.
    """
    def objective(trial):
        """
        Objective function for a single Optuna trial.

        This function is called by Optuna for each trial. It samples
        hyperparameters from the search space defined in `config.param_ranges`,
        trains a model using the sampled parameters, evaluates it on the
        validation set, and returns the metric value to be optimized.

        The function also logs the trial to MLflow by calling `train_and_log`
        and stores the resulting run_id, run_name, and artifact_name in the
        trial's user attributes for later retrieval after the study completes.

        Args:
            trial (optuna.trial.Trial): The current trial object, used to suggest
                hyperparameter values.

        Returns:
            float: The evaluation score (e.g., RMSE, F1) of the trained model.
                The direction (minimize/maximize) is determined by
                `config.tuning_direction`.
        """
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

    # Set tag in MLFlow
    client = MlflowClient()
    client.set_tag(best_run_id, "mlflow.runName", f"!!!_{best_run_name}")

    # Always check champion
    if config.get("alias"):
        promote_if_better(config, best_score, best_run_id, best_artifact_name)

    return best_params, best_score

@flow(name="optuna-flight-delay-tuning")
def tuning_pipeline(config: dict = DEFAULT_CONFIG):
    """
    Main Prefect flow for hyperparameter tuning with Optuna.

    Orchestrates data loading, preprocessing, and the Optuna study.
    After tuning, the best model is automatically promoted to champion.

    Args:
        config (dict): Configuration dictionary (see module docstring).

    Returns:
        tuple: (best_params, best_score) – the optimal hyperparameters and
               their corresponding evaluation score.
    """
    train, val, test = load_and_prepare_data(config)
    preprocessor = build_preprocessor_task(config)
    best_params, best_score = run_optuna_study(train, val, preprocessor, config)
    return best_params, best_score

if __name__ == "__main__":
    config_name = sys.argv[1] if len(sys.argv) > 1 else "DEFAULT_CONFIG"
    config = getattr(cfg_module, config_name, DEFAULT_CONFIG)
    tuning_pipeline(config)