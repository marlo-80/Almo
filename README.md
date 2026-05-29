# Capstone2-Delay_Prediction_For_US_Flights_2013-2018
This is the capstone project of Viktor and Markus. The projects goal is to predict flight delays for domestic flights in the US as a use case for a complete machine learning engineering setup. We analyze millions of historical U.S. flight records to predict arrival delays, both as a continuous value (minutes) and as a binary yes/no. Our pipeline, orchestrated by Prefect, automatically transforms raw data with dbt, trains models tracked in MLflow, and serves predictions through a FastAPI endpoint. A traffic simulator continuously sends requests to mimic real-world usage, while Prometheus and Grafana monitor the API's health and performance. The entire system runs in Docker, ensuring it's reproducible and ready to detect data drift with Evidently, which will trigger retraining when the world (or the data) changes.

## Prerequisites
- Docker & Docker Compose installed
- Project cloned, `docker/.env` contains at least:
  - `POSTGRES_USER`
  - `POSTGRES_PASSWORD`
- `Terminal with prompt at repo root`

<br><br>

## Initialization
To make sure that there are no conflicts when creating our docker containers, delete all volumes defined in the compose.yml first. You can use this command: <br>
```bash
docker compose -f docker/compose.yml down -v
```

All services needed to run in a Docker-based local stack. To start the local services execute this command: <br>
```bash
docker compose -f docker/compose.yml up -d
```

When the stack is running, the local endpoints are:
- `FastAPI/Uvicorn`: `http://127.0.0.1:8000`
- `Grafana`: `http://127.0.0.1:4200`
- `MLflow`: `http://127.0.0.1:5001`
- `Prefect`: `http://127.0.0.1:4200`
- `Postgres`: `http://127.0.0.1:5432`
- `Prometheus`: `http://127.0.0.1:9090`

### First Start only
At the first start some bootstrapping is needed to dowload the data and setup Postgres SQL. After all services from the initialization have been established execute:<br>
```bash
docker compose -f docker/compose.yml exec api python docker/scripts/bootstrap_db.py
```

Data will be downloaded to \repofolder\flight_data and Postgres will be initialised with those data. The process can take a long time, wait until you see the output: <br>
 ```bash
 "Import abgeschlossen. XXXX Zeilen in raw.flights eingefügt."
```
<br>

To check if the initialisation is still running you can watch the size of the Postgres database.: <br>
Linux/"Mac"
```bash
watch -n 5 "docker compose -f docker/compose.yml exec postgres psql -U vikmar -d fastapi_db -c \"SELECT pg_size_pretty(pg_database_size('fastapi_db')) AS size;\""
```
Windows
Linux/"Mac"
```bash
while -n 5 "docker compose -f docker/compose.yml exec postgres psql -U vikmar -d fastapi_db -c \"SELECT pg_size_pretty(pg_database_size('fastapi_db')) AS size;\""
```


As long as the values are growing while no other process writes to Postgres the process is still running.
 
You can verify the table with:<br>
```bash
docker compose -f docker/compose.yml exec postgres psql -U vikmar -d fastapi_db -c "SELECT COUNT(*) FROM raw.flights;"
```
<br><br>

## Creation of dbt models
Run dbt. Several data models will be build:<br>
```bash
docker compose -f docker/compose.yml exec api dbt run --project-dir /app/dbt --profiles-dir /app/dbt
```

You can improve the performance by removing the first double dash (--) in `--AND random() < 0.1` in every file that is in `/dbt/models/training`. However, this will lead to slightly different data in the resulting dbt models. 

<br>

Verification of the dbt model `flights_subset.sql`:<br>
```bash 
docker compose -f docker/compose.yml exec postgres psql -U vikmar -d fastapi_db -c "SELECT COUNT(*), MIN(flight_date), MAX(flight_date) FROM dbt_staging.flights_subset;"
```
<br><br>

## How to Train a New Model

All training logic is driven by a configuration dictionary. You do not need to modify any Python code – just change the values in the config.

### 1. Prerequisites

- All Docker containers are running (`docker compose -f docker/compose.yml up -d`)
- The database contains the `dbt_staging.flights_subset` table (or another table of your choice)
- MLflow is reachable at `http://localhost:5001`

### 2. Define your training configuration

Open `config.py` from `repo/flows/` and add your a dictionary that will define your model. Here are the available keys:

| Key | Type | Description | Example / Possible values |
|-----|------|-------------|---------------------------|
| `run_name` | `str` | Name of the MLflow run | `"simple_rf_no_preprocessing"` |
| `dataset_query` | `str` | SQL query to load training data | `"SELECT * FROM dbt_staging.flights_subset"` |
| `target` | `str` | Target column to predict | `"arr_delay"` |
| `numeric_cols` | `list[str]` | Numeric feature columns | `["crs_dep_time", "dep_delay_minutes", …]` |
| `categorical_cols` | `list[str]` | Categorical feature columns | `["airline", "origin"]` |
| `impute_num` | `str` | Imputation strategy for numeric columns | `"median"`, `"mean"`, `"most_frequent"` |
| `impute_cat` | `str` | Imputation strategy for categorical columns | `"most_frequent"` |
| `model_type` | `str` | Model class to use | `"RandomForestRegressor"` |
| `model_params` | `dict` | Hyperparameters passed to the model | `{"n_estimators": 50, "max_depth": 10}` |
| `register` | `bool` | Whether to register the model in MLflow | `true` / `false` |
| `model_name` | `str` | Registered model name in MLflow | `"flight-delay-baseline"` |
| `alias` | `str` | Alias to assign after registration | `"champion"`, `"staging"` |
| `delay_threshold` | `int` | Threshold (minutes) for binary classification metrics | `15` |
| `dataset_name` | `str` | Human‑readable dataset identifier shown in MLflow UI | `"flights_subset_2019-2020"` |
| `dataset_source` | `str` | Source table or view in PostgreSQL | `"dbt_staging.flights_subset"` |
| `dataset_start_date` | `str` | Start date of the underlying dbt sample | `"2019-01-01"` |
| `dataset_end_date` | `str` | End date of the underlying dbt sample | `"2020-01-01"` |
| `dataset_sample_size` | `int` | Number of rows in the dbt sample | `100000` |
| `dataset_random_seed` | `float` | Random seed used for dbt sampling | `0.42` |

All keys except `run_name`, `dataset_query`, `target`, `numeric_cols`, `categorical_cols`, `model_type`, and `model_params` are optional and fall back to sensible defaults.

### 3. Run the training flow
Assuming your configuration dictionary is called `NEW_MODEL` you can execute the whole training and logging pipeline with this command:
```bash
docker compose -f docker/compose.yml exec -e PYTHONPATH=/app -e PYTHONUNBUFFERED=1 api python flows/train_flow.py NEW_MODEL
```
Results of the training run can  be found in [MLFlow](http://127.0.0.1:5001). Metrics about the model usage can be found in [Grafana](http://127.0.0.1:4200).


<br><br>
## How to Tune a New Model
Our modeling pipeline is also capable of hyper parameter tuning with Optuna. Everything that needs to be defined for an Optuna run can also be done via the `config.py`. The principle is very much the same, but instead of defining model parameters you need to define parameter ranges for the model and some new parameter to control Optunas behaviour. These are the additional parameters for Optuna:


| Key | Type | Description | Example / Possible values |
|-----|------|-------------|---------------------------|
| `n_trials` | `int` | Number of hyperparameter trials | `5`, `30` |
| `direction` | `str` | Optimization direction for the metric | `"minimize"`, `"maximize"` |
| `param_ranges` | `dict` | Search spaces for hyperparameters (replaces `model_params`) | `{"n_estimators": {"type": "int", "low": 50, "high": 300}, "max_depth": {"type": "int", "low": 5, "high": 20}}` |

The flow of parameter tuning with Optuna is defined in `flows/tune_flow.py`. Assuming your configuration dictionary is called `NEW_OPTUNA_MODEL` you can execute the whole training and logging pipeline with this command:
```bash
docker compose -f docker/compose.yml exec -e PYTHONPATH=/app -e PYTHONUNBUFFERED=1 api python flows/tune_flow.py NEW_OPTUNA_MODEL
``` 
Results of the training run can  be found in [MLFlow](http://127.0.0.1:5001). Metrics about the model usage can be found in [Grafana](http://127.0.0.1:4200).

<br><br>

## How to run the Covid data drift experiment
Create data sets for pre-Covid flights and intra-Covid flights.
```bash

```

<br><br>

## Miscellaneous

### Recreation of the api.predictions table in PostgreSQL

If you ever need to rebuild the api.predictions table in PostgreSQL, execute these commands: 
```bash
docker compose -f docker/compose.yml stop simulator
```

```bash
docker compose -f docker/compose.yml exec postgres psql -U vikmar -d fastapi_db -c "DROP TABLE IF EXISTS api.predictions CASCADE;"```
```

```bash
docker compose -f docker/compose.yml exec postgres psql -U vikmar -d fastapi_db -c "
CREATE TABLE api.predictions (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    flight_uid TEXT,
    input_features JSONB,
    prediction_reg DOUBLE PRECISION,
    prediction_class INTEGER,
    model_version_reg TEXT,
    model_version_class TEXT,
    ground_truth DOUBLE PRECISION DEFAULT NULL
);
"
```

<br><br>


### New dbt models
If you want to create new data sets, define new dbt models in dbt/models/training.

To define which data set is used during model training, change the `dataset_query` in the model defining dictionary in `flows/config.py`.

All other fields in the data section of the config file just serve logging purposes in MLFlow

<br>

### Check of the prediction data base
To check the number of rows in api.predictions execute:
```bash
docker compose -f docker/compose.yml exec postgres psql -U vikmar -d fastapi_db -c "SELECT COUNT(*) FROM api.predictions;"
```

To see the last 20 line in api-predictions. execute:
```bash
docker compose -f docker/compose.yml exec postgres psql -U vikmar -d fastapi_db -c "SELECT * FROM api.predictions ORDER BY timestamp DESC LIMIT 20;"
```

<br>

### Filling the prediction data base fastly

If you want to fill api.predictions directly without using the API, you can use the `batch_injector.py` script.

It will use the intra-covid data set to write predictions directly into the data base. The command expects the start-date the end-date and the number of predictions you want to process.

This is an example: 
```bash
docker compose -f docker/compose.yml exec -e PYTHONPATH=/app api python docker/scripts/batch_inject.py 2020-04-01 2020-10-01 1000
```




