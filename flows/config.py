# flows/config.py
######################################################################################################
#                                            Simple Flow                                             #
######################################################################################################

DEFAULT_CONFIG = {
    "run_name": "simple_rf_no_preprocessing",

    # Feature Definition
    "target": "arr_delay",
    "numeric_cols": [
        "crs_dep_time", "crs_arr_time",
        "dep_delay", "dep_delay_minutes",
        "origin_airport_id", "dest_airport_id", "flight_number",
    ],
    "categorical_cols": [],
    "impute_num": "median",
    "impute_cat": "most_frequent",

    # Model Definition
    "model_type": "RandomForestRegressor",
    "model_params": {"n_estimators": 50, "max_depth": 10, "random_state": 42},

    # Data Definition
    "dataset_query": "SELECT * FROM dbt_staging.flights_subset",
    "dataset_source": "dbt_staging.flights_subset",
    "dataset_name": "flights_subset_2019-2020",
    "dataset_start_date": "2019-01-01",
    "dataset_end_date": "2020-01-01",
    "dataset_sample_size": 100000,
    "dataset_random_seed": 0.42,

    # Registration and Evaluation
    "register": False,
    "model_name": "flight-delay-baseline",
    "alias": "champion",
    "delay_threshold": 15,
}


SMALL_TREE = {
    "run_name": "simple_rf_no_preprocessing",
        
    # Feature Definition
    "target": "arr_delay",
    "numeric_cols": [
        "crs_dep_time", "crs_arr_time",
        "dep_delay", "dep_delay_minutes",
        "origin_airport_id", "dest_airport_id", "flight_number",
    ],
    "categorical_cols": [],
    "impute_num": "median",
    "impute_cat": "most_frequent",

    # Model Definition
    "model_type": "RandomForestRegressor",
    "model_params": {"n_estimators": 20, "max_depth": 50, "random_state": 42},

    # Data Definition
    "dataset_query": "SELECT * FROM dbt_staging.flights_subset",
    "dataset_source": "dbt_staging.flights_subset",
    "dataset_name": "flights_subset_2019-2020",
    "dataset_start_date": "2019-01-01",
    "dataset_end_date": "2020-01-01",
    "dataset_sample_size": 100000,
    "dataset_random_seed": 0.42,

    # Registration and Evaluation
    "register": False,
    "model_name": "flight-delay-baseline",
    "alias": "champion",
    "delay_threshold": 15,
}

######################################################################################################
#                                            Optuna Flow                                             #
######################################################################################################

# flows/config.py
OPTUNA_CONFIG = {
    # Experiment & Tuning Control
    "run_name": "optuna-rf-tuning",
    "n_trials": 30,
    "direction": "minimize",          # "minimize" für RMSE

    # Feature Definition
    "target": "arr_delay",
    "numeric_cols": [
        "crs_dep_time", "crs_arr_time",
        "dep_delay", "dep_delay_minutes",
        "origin_airport_id", "dest_airport_id", "flight_number",
    ],
    "categorical_cols": [],
    "impute_num": "median",
    "impute_cat": "most_frequent",
    
    # Model Definition
    "model_type": "RandomForestRegressor",
    "param_ranges": {
        "n_estimators": {"type": "int", "low": 50, "high": 300},
        "max_depth":     {"type": "int", "low": 5, "high": 20},
        # später erweiterbar: "learning_rate": {"type": "float", "low": 1e-3, "high": 0.1, "log": True}
    },

    # Data Definition
    "dataset_query": "SELECT * FROM dbt_staging.flights_subset",
    "dataset_source": "dbt_staging.flights_subset",
    "dataset_name": "flights_subset_2019-2020",
    "dataset_start_date": "2019-01-01",
    "dataset_end_date": "2020-01-01",
    "dataset_sample_size": 100000,
    "dataset_random_seed": 0.42,
    

    # Registration and Evaluation
    "register": False,
    "model_name": "flight-delay-baseline",
    "alias": "",
    "delay_threshold": 15,
}










