# flows/config.py
DEFAULT_CONFIG = {
    "run_name": "simple_rf_no_preprocessing",
    # Data parameter
    "dataset_query": "SELECT * FROM dbt_staging.flights_subset",
    "target": "arr_delay",
    "numeric_cols": [
        "crs_dep_time", "crs_arr_time",
        "dep_delay", "dep_delay_minutes",
        "origin_airport_id", "dest_airport_id", "flight_number",
    ],
    "categorical_cols": [],
    "impute_num": "median",
    "impute_cat": "most_frequent",
    # Model parameter
    "model_type": "RandomForestRegressor",
    "model_params": {"n_estimators": 50, "max_depth": 10, "random_state": 42},
    "register": False,
    "model_name": "flight-delay-baseline",
    "alias": "champion",
    "delay_threshold": 15,
    # Parameter for manual logging of used data sets 
    "dataset_name": "flights_subset_2019-2020",
    "dataset_start_date": "2019-01-01",
    "dataset_end_date": "2020-01-01",
    "dataset_sample_size": 100000,
    "dataset_random_seed": 0.42,
    # Parameter for logging of used data by MLFlow
    "dataset_name": "flights_subset_2019-2020",
    "dataset_source": "dbt_staging.flights_subset",
}


SMALL_TREE = {
    "run_name": "simple_rf_no_preprocessing",
    # Data parameter
    "dataset_query": "SELECT * FROM dbt_staging.flights_subset",
    "target": "arr_delay",
    "numeric_cols": [
        "crs_dep_time", "crs_arr_time",
        "dep_delay", "dep_delay_minutes",
        "origin_airport_id", "dest_airport_id", "flight_number",
    ],
    "categorical_cols": [],
    "impute_num": "median",
    "impute_cat": "most_frequent",
    # Model parameter
    "model_type": "RandomForestRegressor",
    "model_params": {"n_estimators": 20, "max_depth": 5, "random_state": 42},
    "register": False,
    "model_name": "flight-delay-baseline",
    "alias": "",
    "delay_threshold": 15,
    # Parameter for manual logging of used data sets 
    "dataset_name": "flights_subset_2019-2020",
    "dataset_start_date": "2019-01-01",
    "dataset_end_date": "2020-01-01",
    "dataset_sample_size": 100000,
    "dataset_random_seed": 0.42,
    # Parameter for logging of used data by MLFlow
    "dataset_name": "flights_subset_2019-2020",
    "dataset_source": "dbt_staging.flights_subset",
}