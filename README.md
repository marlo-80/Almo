# Capstone2-Delay_Prediction_For_US_Flights_2013-2018
This is the capstone project of Viktor and Markus. The projects goal is to predict flight delays for domestic flights in the US as a use case for a complete Machine Learning Engineering setup. 

## Prerequisites
- Docker & Docker Compose installed
- Project cloned, `docker/.env` contains at least:
  - `POSTGRES_USER`
  - `POSTGRES_PASSWORD`


## Initialization
To make sure that there are no conflicts when creating our docker containers, delete all volumes defined in the compose.yml first. You can use this command: <br>
```bash
docker compose -f docker/compose.yml down -v
```

All services needed run in a Docker-based local stack. To start the local services execute this from the repository root: <br>
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
```bash
watch -n 5 "docker compose -f docker/compose.yml exec postgres psql -U vikmar -d fastapi_db -c \"SELECT pg_size_pretty(pg_database_size('fastapi_db')) AS size;\""
```

As long as the values are growing while no other process writes to Postgres the process is still running.
 
You can verify the table with:<br>
```bash
docker compose -f docker/compose.yml exec postgres psql -U vikmar -d fastapi_db -c "SELECT COUNT(*) FROM raw.flights;"
```

## Creation of dbt models
Set a random seed to make data sampling reproducable. Unfortunately, dbt models don't have a seed parameter by themself.<br>
```bash
docker compose -f docker/compose.yml exec postgres psql -U vikmar -d fastapi_db -c "SELECT setseed(0.42);"
```
<br><br>
Run dbt with default values: Start: 2019-01-01 / Stopp: 2020-01-01 / Sample size: 100k rows):<br>
```bash
docker compose -f docker/compose.yml exec api dbt run --project-dir /app/dbt --profiles-dir /app/dbt
```
<br>

Verification of the dbt model:<br>
```bash 
docker compose -f docker/compose.yml exec postgres psql -U vikmar -d fastapi_db -c "SELECT COUNT(*), MIN(flight_date), MAX(flight_date) FROM dbt_staging.flights_subset;"`
```

## How to Train a New Model

All training logic is driven by a configuration dictionary. You do not need to modify any Python code – just change the values in the config.

### 1. Prerequisites

- All Docker containers are running (`docker compose -f docker/compose.yml up -d`)
- The database contains the `dbt_staging.flights_subset` table (or another table of your choice)
- MLflow is reachable at `http://localhost:5001`

### 2. Define your trainin configuration

Open `config.py` and add your a dictionary that will define your model. Here are the available keys:

| Key | Type | Description | Example / Possible values |
| --- | --- | --- | --- |
| `run_name` | `str` | Name of the MLflow run | `"my_experiment"` |
| `dataset_query` | `str` | SQL query that returns the training data | `"SELECT * FROM dbt_staging.flights_subset"` |
| `target` | `str` | Column to predict | `"arr_delay"` |
| `numeric_cols` | `list[str]` | Numeric feature columns | `["crs_dep_time", "dep_delay_minutes", ...]` |
| `categorical_cols` | `list[str]` | Categorical feature columns (can be empty) | `["airline", "origin"]` |
| `impute_num` | `str` | Imputation strategy for numeric columns | `"median"`, `"mean"`, `"most_frequent"` |
| `impute_cat` | `str` | Imputation strategy for categorical columns | `"most_frequent"` |
| `model_type` | `str` | Model class to use | `"RandomForestRegressor"` |
| `model_params` | `dict` | Hyperparameters passed to the model | `{"n_estimators": 100, "max_depth": 15}` |
| `register` | `bool` | Whether to register the model in MLflow | `true` / `false` |
| `model_name` | `str` | Name for the registered model | `"flight-delay-baseline"` |
| `alias` | `str` | Alias to assign after registration (e.g. "champion") | `"champion"`, `"staging"` |
| `delay_threshold` | `int` | Threshold (minutes) for binary classification metrics | `15` |

All keys except `run_name`, `dataset_query`, `target`, `numeric_cols`, `categorical_cols`, `model_type`, and `model_params` are optional and fall back to sensible defaults.

## 3. Run the training flow
Assuming your configuration dictionary is called `MY_MODEL` you cant execute the whole training and logging pipeline with this command:
```bash
docker compose -f docker/compose.yml exec -e PYTHONPATH=/app -e PYTHONUNBUFFERED=1 api python flows/train_flow.py YOUR_MODEL
```