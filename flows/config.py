# flows/config.py

# Default values for prediction model
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
    "target_table": "dbt_staging.retrain",   # <-- NEU
}

DRIFT_RETRAIN_CLASS = {
    **CLASS,
    "run_name": "drift_retrain_class",
    "dataset_query": "SELECT * FROM dbt_staging.retrain",
    "target_table": "dbt_staging.retrain",   # <-- NEU
}







