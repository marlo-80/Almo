# flows/config.py
"""
Configuration Module – Training, Tuning & Drift Retraining

This module contains all configuration dictionaries for the flight delay
prediction pipeline. Each dictionary defines a complete experiment setup,
including:

- Preprocessing strategies (imputation, encoding, scaling)
- Feature selection (low/high cardinality, cyclic, numeric, skewed)
- Model type and hyperparameters
- Dataset query and metadata
- MLflow registration and promotion settings

The configurations are used by:
- `flows/train_flow.py`  → for training new models
- `flows/tune_flow.py`   → for hyperparameter optimization (Optuna)
- `flows/drift_flow.py`  → for retraining when drift is detected

------------------------------------------------------------------------------
Config Structure
------------------------------------------------------------------------------

All configurations share a common schema:

Key                       | Type               | Description
--------------------------|--------------------|---------------------------------------------
run_name                  | str                | MLflow run name
task                      | str                | 'regression' or 'classification'
target_type               | str                | 'continuous' or 'binary'
impute_num                | str                | Numeric imputation: 'median', 'mean', 'most_frequent'
impute_cat                | str                | Categorical imputation: 'most_frequent'
low_card_strategy         | str                | 'onehot' or 'ordinal'
high_card_strategy        | str                | 'target', 'ordinal', or 'frequency'
target                    | str                | Target column name
low_cardinality_cols      | list[str]          | Columns with few unique values
high_cardinality_cols     | list[str]          | Columns with many unique values
cyclic_cols               | list[str]          | Cyclic features (sin/cos transformed)
numeric_cols              | list[str]          | Standard numeric features
skewed_numeric_cols       | list[str]          | Numeric features with log transformation
model_type                | str                | e.g., 'RandomForestRegressor'
model_params              | dict               | Hyperparameters (for training)
param_ranges              | dict               | Hyperparameter search space (for tuning)
fixed_model_params        | dict               | Fixed hyperparameters (for tuning)
dataset_query             | str                | SQL query for training data
dataset_name              | str                | Dataset identifier (logging)
dataset_start_date        | str                | Start date (logging)
dataset_end_date          | str                | End date (logging)
dataset_sample_size       | int                | Sample size (logging)
dataset_random_seed       | float              | Random seed (logging)
dataset_source            | str                | Source table name (logging)
register                  | bool               | Register model in MLflow?
model_name                | str                | Registered model name
alias                     | str                | Model alias (e.g., 'champion')
promotion_metric          | str                | Metric for promotion comparison
promotion_mode            | str                | 'minimize' or 'maximize'

------------------------------------------------------------------------------
Available Configurations
------------------------------------------------------------------------------

Name               | Purpose
-------------------|---------------------------------------------------------
DEFAULT_CONFIG     | Fallback for missing configurations (minimal)
REG                | Simple regression with RandomForestRegressor
CLASS              | Simple classification with RandomForestClassifier
OPTUNA_REG         | Optuna tuning for regression (3 trials)
OPTUNA_CLASS       | Optuna tuning for classification (3 trials)
DRIFT_RETRAIN_REG  | Regression retraining on drift (uses dbt_staging.retrain)
DRIFT_RETRAIN_CLASS| Classification retraining on drift (uses dbt_staging.retrain)

------------------------------------------------------------------------------
API Models
------------------------------------------------------------------------------

`API_MODELS` defines which models are loaded by the FastAPI service:
- regressor@champion  → regression model
- classifier@champion → classification model

These are loaded via MLflow at startup.

------------------------------------------------------------------------------
Usage in Flows
------------------------------------------------------------------------------

Training:
    from flows.config import REG
    python flows/train_flow.py REG

Tuning:
    from flows.config import OPTUNA_REG
    python flows/tune_flow.py OPTUNA_REG

Drift Retraining:
    from flows.config import DRIFT_RETRAIN_REG
    python flows/train_flow.py DRIFT_RETRAIN_REG

------------------------------------------------------------------------------
Notes
------------------------------------------------------------------------------
- All configurations use lowercase column names (matching dbt transformations).
- The `pre_covid_100K` table is the default training source.
- `register` is set to False for most configs to avoid automatic registration
  during tuning; promotion happens via `promote_if_better()` in train_flow.py.
- The `DRIFT_RETRAIN_*` configs inherit from REG/CLASS but override
  `dataset_query` to use the `dbt_staging.retrain` table.
"""


# ============================================================================
#  Default Values for Prediction API
# ============================================================================
API_MODELS = {
    "regression": {
        "model_name": "regressor",
        "alias": "champion",
    },
    "classification": {
        "model_name": "classifier",
        "alias": "champion",
    },
}





######################################################################################################
#                                              Default CONFIG                                        #
######################################################################################################
# This is only a fallback with default values. But it needs to be defined

# Minimale Fallback‑Konfiguration – wird nur geladen, falls kein anderer Name übergeben wird.
DEFAULT_CONFIG = {
    "run_name": "fallback",
    "task": "regression",
    "target_type": "continuous",
    "impute_num": "median",
    "impute_cat": "most_frequent",
    "low_card_strategy": "onehot",
    "high_card_strategy": "target",

    "target": "arr_delay_minutes",
    "low_cardinality_cols": [],
    "high_cardinality_cols": [],
    "cyclic_cols": [],
    "numeric_cols": [],
    "skewed_numeric_cols": [],

    "model_type": "RandomForestRegressor",
    "model_params": {},

    "register": False,
    "model_name": "fallback",
    "alias": "",
}


######################################################################################################
#                                            Simple Regression                                       #
######################################################################################################

# flows/config.py (nur die vier angepassten Configs)

REG = {
    "run_name": "rf_reg",
    "task": "regression",
    "target_type": "continuous",          # für TargetEncoder & Metriken

    # Preprocessing
    "impute_num": "median",
    "impute_cat": "most_frequent",
    "low_card_strategy": "onehot",
    "high_card_strategy": "target",

    # Feature-Spalten
    "target": "arr_delay_minutes",
    "low_cardinality_cols": [
        "year", "quarter", "month", "day_of_month", "day_of_week",
        "distance_group", "dep_time_blk",
    ],
    "high_cardinality_cols": [
        "origin_airport_id", "dest_airport_id",
        "flight_number_marketing_airline", "flight_number_operating_airline",
        "tail_number",
    ],
    "cyclic_cols": [
        "crs_dep_time", "crs_arr_time",
    ],
    "numeric_cols": [
        "crs_elapsed_time",
    ],
    "skewed_numeric_cols": [
        "distance",
    ],

    # Modell
    "model_type": "RandomForestRegressor",
    "model_params": {"n_estimators": 20, "max_depth": 5, "random_state": 42},

    # Daten (nur Logging)
    "dataset_query": "SELECT * FROM dbt_staging.pre_covid_100K",
    "dataset_name": "pre_covid_100K",
    "dataset_start_date": "2018-01-01",
    "dataset_end_date": "2020-01-01",
    "dataset_sample_size": 100000,
    "dataset_random_seed": 0.42,
    "dataset_source": "dbt_staging.pre_covid_100K",

    # Registrierung & Promotion
    "register": False,
    "model_name": "regressor",
    "alias": "champion",
    "promotion_metric": "rmse",
    "promotion_mode": "minimize",
}

######################################################################################################
#                                            Simple Classification                                   #
######################################################################################################

CLASS = {
    "run_name": "rf_class",
    "task": "classification",
    "target_type": "binary",

    # Preprocessing
    "impute_num": "median",
    "impute_cat": "most_frequent",
    "low_card_strategy": "onehot",
    "high_card_strategy": "target",

    # Feature-Spalten
    "target": "arr_del15",
    "low_cardinality_cols": [
        "year", "quarter", "month", "day_of_month", "day_of_week",
        "distance_group", "dep_time_blk",
    ],
    "high_cardinality_cols": [
        "origin_airport_id", "dest_airport_id",
        "flight_number_marketing_airline", "flight_number_operating_airline",
        "tail_number",
    ],
    "cyclic_cols": [
        "crs_dep_time", "crs_arr_time",
    ],
    "numeric_cols": [
        "crs_elapsed_time",
    ],
    "skewed_numeric_cols": [
        "distance",
    ],

    # Modell
    "model_type": "RandomForestClassifier",
    "model_params": {"n_estimators": 20, "max_depth": 2, "class_weight": "balanced", "random_state": 42},

    # Daten (nur Logging)
    "dataset_query": "SELECT * FROM dbt_staging.pre_covid_100K",
    "dataset_name": "pre_covid_100k",
    "dataset_start_date": "2018-01-01",
    "dataset_end_date": "2020-01-01",
    "dataset_sample_size": 100000,
    "dataset_random_seed": 0.42,
    "dataset_source": "dbt_staging.pre_covid_100K",

    # Registrierung & Promotion
    "register": False,
    "model_name": "classifier",
    "alias": "champion",
    "promotion_metric": "f1",
    "promotion_mode": "maximize",
}

######################################################################################################
#                                         Optuna Regression                                          #
######################################################################################################

OPTUNA_REG = {
    "run_name": "optuna_rf_reg",
    "task": "regression",
    "target_type": "continuous",

    # Optuna
    "n_trials": 3,
    "tuning_metric": "rmse",
    "tuning_direction": "minimize",

    # Preprocessing
    "impute_num": "median",
    "impute_cat": "most_frequent",
    "low_card_strategy": "onehot",
    "high_card_strategy": "target",

    # Feature-Spalten
    "target": "arr_delay_minutes",
    "low_cardinality_cols": [
        "year", "quarter", "month", "day_of_month", "day_of_week",
        "distance_group", "dep_time_blk",
    ],
    "high_cardinality_cols": [
        "origin_airport_id", "dest_airport_id",
        "flight_number_marketing_airline", "flight_number_operating_airline",
        "tail_number",
    ],
    "cyclic_cols": [
        "crs_dep_time", "crs_arr_time",
    ],
    "numeric_cols": [
        "crs_elapsed_time",
    ],
    "skewed_numeric_cols": [
        "distance",
    ],

    # Modell
    "model_type": "RandomForestRegressor",
    "param_ranges": {
        "n_estimators": {"type": "int", "low": 5, "high": 30},
        "max_depth":     {"type": "int", "low": 2, "high": 5},
    },
    "fixed_model_params": {"random_state": 42},

    # Daten (nur Logging)
    "dataset_query": "SELECT * FROM dbt_staging.pre_covid_100K",
    "dataset_name": "pre_covid_100k",
    "dataset_start_date": "2018-01-01",
    "dataset_end_date": "2020-01-01",
    "dataset_sample_size": 100000,
    "dataset_random_seed": 0.42,
    "dataset_source": "dbt_staging.pre_covid_100K",

    # Registrierung & Promotion
    "register": False,
    "model_name": "regressor",
    "alias": "champion",
    "promotion_metric": "rmse",
    "promotion_mode": "minimize",
}

######################################################################################################
#                                         Optuna Classification                                      #
######################################################################################################

OPTUNA_CLASS = {
    "run_name": "optuna_rf_class",
    "task": "classification",
    "target_type": "binary",

    # Optuna
    "n_trials": 3,
    "tuning_metric": "precision",
    "tuning_direction": "maximize",

    # Preprocessing
    "impute_num": "median",
    "impute_cat": "most_frequent",
    "low_card_strategy": "onehot",
    "high_card_strategy": "target",

    # Feature-Spalten
    "target": "arr_del15",
    "low_cardinality_cols": [
        "year", "quarter", "month", "day_of_month", "day_of_week",
        "distance_group", "dep_time_blk",
    ],
    "high_cardinality_cols": [
        "origin_airport_id", "dest_airport_id",
        "flight_number_marketing_airline", "flight_number_operating_airline",
        "tail_number",
    ],
    "cyclic_cols": [
        "crs_dep_time", "crs_arr_time",
    ],
    "numeric_cols": [
        "crs_elapsed_time",
    ],
    "skewed_numeric_cols": [
        "distance",
    ],

    # Modell
    "model_type": "RandomForestClassifier",
    "param_ranges": {
        "n_estimators": {"type": "int", "low": 5, "high": 40},
        "max_depth":     {"type": "int", "low": 1, "high": 5},
    },
    "fixed_model_params": {"class_weight": "balanced", "random_state": 42},

    # Daten (nur Logging)
    "dataset_query": "SELECT * FROM dbt_staging.pre_covid_100K",
    "dataset_name": "pre_covid_100k",
    "dataset_start_date": "2018-01-01",
    "dataset_end_date": "2020-01-01",
    "dataset_sample_size": 100000,
    "dataset_random_seed": 0.42,
    "dataset_source": "dbt_staging.pre_covid_100K",

    # Registrierung & Promotion
    "register": False,
    "model_name": "classifier",
    "alias": "champion",
    "promotion_metric": "precision",
    "promotion_mode": "maximize",
}




DRIFT_RETRAIN_REG = {
    **REG,
    "run_name": "drift_retrain_reg",
    "dataset_query": "SELECT * FROM dbt_staging.retrain",
    "target_table": "dbt_staging.retrain",   
}

DRIFT_RETRAIN_CLASS = {
    **CLASS,
    "run_name": "drift_retrain_class",
    "dataset_query": "SELECT * FROM dbt_staging.retrain",
    "target_table": "dbt_staging.retrain",
}







