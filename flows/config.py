# flows/config.py
######################################################################################################
#                                            Simple Training                                         #
######################################################################################################


DEFAULT_REGRESSION_CONFIG = {
    "active ": True,
    "run_name": "small_rf",
    "dataset_query": "SELECT * FROM dbt_staging.flights_subset",
    "target": "arr_delay_minutes",
    "numeric_cols": [
        "year", "quarter", "month", "day_of_month", "day_of_week",
        "crs_dep_time", "crs_arr_time", "crs_elapsed_time",
        "distance", "distance_group",
    ],
    "categorical_cols": [
    ],
    "impute_num": "median",
    "impute_cat": "most_frequent",
    "model_type": "RandomForestRegressor",
    "model_params": {"n_estimators": 50, "max_depth": 10, "random_state": 42}, 
    "register": False,
    "model_name": "flight-delay-baseline",
    "alias": "",
    "delay_threshold": 15,
    "dataset_name": "flights_subset_2019-2020",
    "dataset_start_date": "2019-01-01",
    "dataset_end_date": "2020-01-01",
    "dataset_sample_size": 100000,
    "dataset_random_seed": 0.42,
    "dataset_source": "dbt_staging.flights_subset",
}

DEFAULT_CLASSIFICATION_CONFIG = {
    "active ": False,
    "run_name": "small_rf_classification",
    "dataset_query": "SELECT * FROM dbt_staging.flights_subset",
    "target": "arr_del_groups",
    "numeric_cols": [
        "year", "quarter", "month", "day_of_month", "day_of_week",
        "crs_dep_time", "crs_arr_time", "crs_elapsed_time",
        "distance", #"distance_group",
    ],
    "categorical_cols": [
        "airline", "operating_airline", "marketing_airline_network",
        "origin", "origin_city_name",
        "dest", "dest_city_name", 
        "distance_group", "tail_number", "flight_number_marketing_airline", "flight_number_operating_airline",
    ],
    "impute_num": "median",
    "impute_cat": "most_frequent",
    "model_type": "RandomForestClassifier",
    "classifier_model_params": {"n_estimators": 50, "max_features": "sqrt", "max_depth": 10, "random_state": 42},
    "dataset_name": "flights_subset_2019-2020",
    "dataset_start_date": "2019-01-01",
    "dataset_end_date": "2020-01-01",
    "dataset_sample_size": 100000,
    "dataset_random_seed": 0.42,
    "dataset_source": "dbt_staging.flights_subset",
}




######################################################################################################
#                                         Optuna Training                                            #
######################################################################################################

# flows/config.py
OPTUNA_CONFIG = {
    # Experiment & Tuning Control
    "run_name": "rf-baseline-tuning",
    "n_trials": 30,
    "direction": "minimize",          # "minimize" für RMSE

    # Feature Definition
    "target": "arr_delay",
    "numeric_cols": [
        "year", "quarter", "month", "day_of_month", "day_of_week",
        "crs_dep_time", "crs_arr_time", "crs_elapsed_time",
        "distance", "distance_group",
    ],
    "categorical_cols": [
        "origin", "origin_city_name", "origin_state",
        "dest", "dest_city_name", "dest_state",
        "marketing_airline_network", "operating_airline",
        "flight_number_marketing_airline", "flight_number_operating_airline",
        "tail_number",
        "dep_time_blk",
    ],
    "impute_num": "median",
    "impute_cat": "most_frequent",
    
    # Model Definition
    "model_type": "RandomForestRegressor",
    "param_ranges": {
        "n_estimators": {"type": "int", "low": 5, "high": 50},
        "max_depth":     {"type": "int", "low": 2, "high": 6},
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


######################################################################################################
#                                             Tests                                                  #
######################################################################################################


SMALL_TREE = {
    "run_name": "small_rf",
    "dataset_query": "SELECT * FROM dbt_staging.flights_subset",
    "target": "arr_delay_minutes",
    "numeric_cols": [
        "year", "quarter", "month", "day_of_month", "day_of_week",
        "crs_dep_time", "crs_arr_time", "crs_elapsed_time",
        "distance", "distance_group","origin_airport_id", "dest_airport_id",
    ],
    "categorical_cols": [
    ],
    "impute_num": "median",
    "impute_cat": "most_frequent",
    "model_type": "RandomForestRegressor",
    "model_params": {"n_estimators": 20, "max_depth": 5, "random_state": 42},
    "register": False,
    "model_name": "flight-delay-baseline",
    "alias": "",
    "delay_threshold": 15,
    "dataset_name": "flights_subset_2019-2020",
    "dataset_start_date": "2019-01-01",
    "dataset_end_date": "2020-01-01",
    "dataset_sample_size": 100000,
    "dataset_random_seed": 0.42,
    "dataset_source": "dbt_staging.flights_subset",
}



OPTUNA_TEST = {
    # Experiment & Tuning Control
    "run_name": "rf-baseline-tuning",
    "n_trials": 5,
    "direction": "minimize",          # "minimize" für RMSE

    # Feature Definition
    "target": "arr_delay",
    "numeric_cols": [
        "crs_dep_time", "crs_arr_time",
        #"dep_delay", "dep_delay_minutes",
        "origin_airport_id", "dest_airport_id",
    ],
    "categorical_cols": [],
    "impute_num": "median",
    "impute_cat": "most_frequent",
    
    # Model Definition
    "model_type": "RandomForestRegressor",
    "param_ranges": {
        "n_estimators": {"type": "int", "low": 5, "high": 10},
        "max_depth":     {"type": "int", "low": 2, "high": 5},
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








