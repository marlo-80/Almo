# Almo – A Modeling Framework for BTS Flight Data with Delay Prediction API

<p align="center">
<img src="Demo/Pipeline_metaphor6.png" alt="Almo Pipeline Overview" width="800">
</p>

Almo is a complete machine learning engineering framework for predicting domestic flight delays in the United States. It analyzes millions of historical flight records from the Bureau of Transportation Statistics to predict arrival delays—both as a continuous value (minutes) and as a binary yes/no decision. The pipeline, orchestrated by **Prefect**, automatically transforms raw data with **dbt**, trains models tracked in **MLflow**, and serves predictions through a **FastAPI** endpoint. **Evidently** monitors prediction data for drift, triggering alarms when the data diverges beyond a defined threshold. The entire system runs in **Docker**, ensuring reproducibility.

Two demos are included to showcase the system's capabilities:
- A **traffic simulator** that continuously sends prediction requests to the API, while **Prometheus** and **Grafana** monitor health and performance.
- A **COVID data drift demo** that generates monthly predictions from January 2020 onward. It shows how prediction data deviates from training data during the pandemic. When the drift score exceeds the threshold, new models are automatically trained and promoted to champion if they outperform the previous models.


## Table of Contents

- [What Almo Does](#what-almo-does)
- [How It Works (Pipeline Overview)](#how-it-works-pipeline-overview)
- [Prerequisites](#prerequisites)
- [Initialization](#initialization)
- [Creation of dbt Models](#creation-of-dbt-models)
- [How to Train a New Model](#how-to-train-a-new-model)
- [How to Tune a New Model](#how-to-tune-a-new-model)
- [How to Run the Traffic Simulator](#how-to-run-the-traffic-simulator)
- [How to Run the COVID Data Drift Experiment](#how-to-run-the-covid-data-drift-experiment)
- [Miscellaneous](#miscellaneous)
- [Project Structure](#project-structure)
- [Disclaimer](#disclaimer)



## What Almo Does

- **Ingests and models raw BTS flight data** – Raw CSV files are loaded into a PostgreSQL database, then transformed into clean, feature-rich tables using dbt.
- **Trains models with configurable preprocessing** – Using configuration dictionaries, you define features, preprocessing strategies (one‑hot encoding, target encoding, cyclic features, log transforms), and model hyperparameters. Both regression and classification models are supported.
- **Tracks experiments in MLflow** – Every training run logs parameters, metrics, artifacts, and dataset information to MLflow for full reproducibility.
- **Serves predictions via FastAPI** – A FastAPI endpoint accepts flight features and returns delay predictions. Both regressor and classifier models are loaded from the MLflow registry.
- **Detects data drift with Evidently** – The drift flow compares current prediction data against a pre‑COVID reference dataset. If the drift score exceeds a dynamic, monthly baseline, an alarm is triggered.
- **Automatically retrains on drift** – When an alarm fires, the system automatically appends the drifted predictions to the training dataset, retrains both models, and promotes the new model if it outperforms the current champion.
- **Monitors everything with Prometheus and Grafana** – Custom dashboards display drift scores, champion metrics, prediction rates, model ages, retraining status, and API performance in real time.



## How It Works (Pipeline Overview)

The entire process runs inside Docker and is orchestrated by Prefect. The main components are:

1. **Data Ingestion & Transformation** (`bootstrap_db.py`, `dbt`)  
   - Raw CSV files are downloaded via KaggleHub and loaded into `raw.flights`.  
   - dbt models clean and transform the data into staging and training tables (`stg_flights`, various subset tables).

2. **Model Training** (`train_flow.py`)  
   - A configuration dictionary defines all aspects of the training: features, preprocessing, model type, and hyperparameters.  
   - The pipeline builds a scikit‑learn `ColumnTransformer` for preprocessing and trains the model.  
   - Metrics, artifacts, and dataset info are logged to MLflow.  
   - If the new model outperforms the current champion, it is automatically registered and promoted.

3. **Prediction API** (`api.py`)  
   - A FastAPI app loads the champion models from MLflow at startup.  
   - Incoming prediction requests are processed and logged to `api.predictions` along with input features and ground truth.  
   - Prometheus metrics track prediction counts, durations, and model versions.

4. **Drift Detection** (`drift_flow.py`)  
   - Compares the feature distributions of recent predictions against a pre‑COVID reference (2018–2019).  
   - Computes a drift score using Evidently's data drift preset.  
   - Additional metrics (regression RMSE/MAE, classification F1/precision/recall) are calculated from the ground truth in `api.predictions`.  
   - All metrics are posted to the API, where they are exposed to Prometheus.

5. **Retraining on Drift**  
   - When the drift score exceeds the monthly baseline, the system triggers a retraining:  
     - Current predictions with ground truth are merged into the training table.  
     - Both regressor and classifier are retrained on the expanded dataset.  
     - If the new model is better, it becomes the champion and the API reloads its models.

6. **Monitoring** (Prometheus + Grafana)  
   - Prometheus scrapes the API's metrics endpoint every few seconds.  
   - Grafana dashboards visualize drift scores, champion baselines, prediction performance, model ages, and retraining status.  
   - Dynamic baselines and annotations are set during the COVID demo to provide context.



## Prerequisites
   - Docker and Docker Compose installed
   - **Windows users:** Use WSL2 (Ubuntu recommended). Clone the repository **inside** the WSL2 filesystem (e.g., `/home/username/almo`), not on the Windows host, to avoid I/O performance issues.
   - Terminal open in repository root

## System Requirements

- **RAM:** 16 GB recommended (8 GB minimum for small datasets)
- **Disk Space:** 50 GB free (for raw data + PostgreSQL + models)
- **CPU:** 4+ cores recommended (for dbt and training)
- **Docker Engine:** 20.10+ and Docker Compose 2.0+
- **OS:** Linux, macOS, or Windows 10/11 with WSL2



## Initialization
This will create docker container, download data, create dbt models and train prediction models. After the initialization everything is setup for model development as well as executing the Covid Data Drift Demo.
To start the setup you can use: 

### Option 1: Using `make` (recommended)
```bash
make setup
```


### Option 2: Using `setup.sh`
```bash
./setup
```


Downloading, extracting and check of data integrity takes a long time. You can test if data is still written to the PostgreSQL database by executing:

 Linux/macOS
```bash
watch -n 5 "docker compose -f docker/compose.yml exec postgres psql -U testuser -d fastapi_db -c \"SELECT pg_size_pretty(pg_database_size('fastapi_db')) AS size;\""
```

Windows
```bash
while ($true) { Clear-Host; docker compose -f docker/compose.yml exec postgres psql -U testuser -d fastapi_db -c "SELECT pg_size_pretty(pg_database_size('fastapi_db')) AS size;"; Start-Sleep -Seconds 5 }
```

Verify the import:

```bash
docker compose -f docker/compose.yml exec postgres psql -U testuser -d fastapi_db -c "SELECT COUNT(*) FROM raw.flights;"
```


Once running, the following endpoints are available:

| Service    | URL                          |
|------------|------------------------------|
| FastAPI    | [http://127.0.0.1:8000](http://127.0.0.1:8000/health)      |
| Grafana    | [http://127.0.0.1:3000](http://127.0.0.1:3000)      |
| MLflow     | [http://127.0.0.1:5001](http://127.0.0.1:5001)      |
| Prefect    | [http://127.0.0.1:4200](http://127.0.0.1:4200)      |
| PostgreSQL | [http://127.0.0.1:5432](http://127.0.0.1:5432)      |
| Prometheus | [http://127.0.0.1:9090](http://127.0.0.1:9090)      |





## Creation of dbt Models

Build all dbt models:

```bash
docker compose -f docker/compose.yml exec api dbt run --project-dir /app/dbt --profiles-dir /app/dbt
```

To build only models that have changed since the last run:

```bash
docker compose -f docker/compose.yml exec api dbt run --select state:modified+ --project-dir /app/dbt --profiles-dir /app/dbt
```

Verify a model:

```bash
docker compose -f docker/compose.yml exec postgres psql -U testuser -d fastapi_db -c "SELECT COUNT(*), MIN(flight_date), MAX(flight_date) FROM dbt_staging.flights_subset;"
```


## How to Train a New Model

All training is driven by configuration dictionaries in `flows/config.py`. No Python code changes are needed.

### 1. Prerequisites

<ul style="margin-top: -10px;">
  <li>All containers running (<code>docker compose -f docker/compose.yml up -d</code>)</li>
  <li>A populated training table (e.g., <code>dbt_staging.flights_subset</code>)</li>
  <li>MLflow reachable at <code>http://localhost:5001</code></li>
</ul>

### 2. Define Your Training Configuration

Add a new dictionary to `flows/config.py`. Available keys:

| Key | Type | Description | Example |
|-----|------|-------------|---------|
| `run_name` | `str` | MLflow run name | `"simple_rf_no_preprocessing"` |
| `dataset_query` | `str` | SQL query for training data | `"SELECT * FROM dbt_staging.flights_subset"` |
| `target` | `str` | Target column | `"arr_delay"` |
| `numeric_cols` | `list[str]` | Numeric features | `["crs_dep_time", "dep_delay_minutes", …]` |
| `categorical_cols` | `list[str]` | Categorical features | `["airline", "origin"]` |
| `impute_num` | `str` | Numeric imputation | `"median"`, `"mean"`, `"most_frequent"` |
| `impute_cat` | `str` | Categorical imputation | `"most_frequent"` |
| `model_type` | `str` | Model class | `"RandomForestRegressor"` |
| `model_params` | `dict` | Hyperparameters | `{"n_estimators": 50, "max_depth": 10}` |
| `register` | `bool` | Register in MLflow? | `true` / `false` |
| `model_name` | `str` | Registered model name | `"flight-delay-baseline"` |
| `alias` | `str` | Alias after registration | `"champion"`, `"staging"` |
| `dataset_name` | `str` | Dataset identifier (logging) | `"flights_subset_2019-2020"` |
| `dataset_source` | `str` | Source table or view | `"dbt_staging.flights_subset"` |
| `dataset_start_date` | `str` | Start date (logging) | `"2019-01-01"` |
| `dataset_end_date` | `str` | End date (logging) | `"2020-01-01"` |
| `dataset_sample_size` | `int` | Rows in the sample (logging) | `100000` |
| `dataset_random_seed` | `float` | Random seed (logging) | `0.42` |

Only `run_name`, `dataset_query`, `target`, `numeric_cols`, `categorical_cols`, `model_type`, and `model_params` are strictly required; the rest fall back to defaults.

### 3. Run the Training Flow

Assuming your config is named `NEW_MODEL` you can execute training with:

```bash
make train CONFIG=NEW_MODEL
```

View results in [MLflow](http://127.0.0.1:5001) and [Grafana](http://127.0.0.1:3000).


## How to Tune a New Model

Hyperparameter tuning with Optuna is also supported. Define parameter ranges instead of fixed values:

| Key | Type | Description | Example |
|-----|------|-------------|---------|
| `n_trials` | `int` | Number of trials | `5`, `30` |
| `direction` | `str` | Optimization direction | `"minimize"`, `"maximize"` |
| `param_ranges` | `dict` | Search spaces (replaces `model_params`) | `{"n_estimators": {"type": "int", "low": 50, "high": 300}, …}` |

Assuming your config is named `NEW_MODELS` you can execute training with:

```bash
make tune CONFIG=NEW_MODELS
```


## How to Run the Traffic Simulator

Start the simulator (requires trained models)
```bash
make simulator-up
```

Stop the simulator
```bash
make simulator-down
```
View simulator logs
```bash
make simulator-logs
```
---

## How to Run the COVID Data Drift Experiment
### Using MAKE
```bash
make demo
```


### Direct Execution
Make the demo script executable and run it:

```bash
chmod +x Demo/covid_data_drift_demo.sh
./Demo/covid_data_drift_demo.sh
```

The demo will:
- Empty the predictions table.
- Reset Prometheus and Grafana for a clean start.
- Inject monthly batches of 2020‑2022 flight data into the prediction API.
- Run the drift flow after each batch.
- When the drift score exceeds the threshold, automatically retrain both models and promote a new champion if improved.

---

## Miscellaneous

### Recreation of the `api.predictions` Table

```bash
docker compose -f docker/compose.yml stop simulator
docker compose -f docker/compose.yml exec postgres psql -U testuser -d fastapi_db -c "DROP TABLE IF EXISTS api.predictions CASCADE;"
docker compose -f docker/compose.yml exec postgres psql -U testuser -d fastapi_db -c "
CREATE TABLE api.predictions (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    flight_uid TEXT,
    input_features JSONB,
    prediction_reg DOUBLE PRECISION,
    prediction_class INTEGER,
    model_version_reg TEXT,
    model_version_class TEXT,
    ground_truth JSONB DEFAULT NULL,
    prediction_class_proba DOUBLE PRECISION DEFAULT NULL
);
"
```

### New dbt Models

Create new `.sql` files in `dbt/models/training`. Update `dataset_query` in your config to point to the new table or view.

### Check the Predictions Database

```bash
docker compose -f docker/compose.yml exec postgres psql -U testuser -d fastapi_db -c "SELECT COUNT(*) FROM api.predictions;"
docker compose -f docker/compose.yml exec postgres psql -U testuser -d fastapi_db -c "SELECT * FROM api.predictions ORDER BY timestamp DESC LIMIT 20;"
```

### Bulk Insert Predictions (Without the API)

```bash
docker compose -f docker/compose.yml exec -e PYTHONPATH=/app api python docker/scripts/batch_inject.py 2020-04-01 2020-10-01 1000
```

### Quick Database Maintenance

```bash
# Clear predictions with index reset
docker compose -f docker/compose.yml exec postgres psql -U testuser -d fastapi_db -c "TRUNCATE TABLE api.predictions RESTART IDENTITY;"

# Clear predictions without index reset
docker compose -f docker/compose.yml exec postgres psql -U testuser -d fastapi_db -c "TRUNCATE TABLE api.predictions;"

# Query drift score directly
curl -s "http://localhost:9090/api/v1/query?query=data_drift_score"
```



## Project Structure

```
almo/                                      # Project root (repository name)
├── .env                                   # Environment variables (POSTGRES_USER, etc.)
├── .gitignore                             # Git ignore file
├── .dockerignore                          # Docker ignore file
├── Makefile                               # Make commands (setup, train, demo, etc.)
├── setup.sh                               # Full setup script (build, start, import, dbt)
├── README.md                              # Project documentation (updated)
├── requirements.txt                       # Python dependencies
│
├── docker/                                # Docker-related files
│   ├── compose.yml                        # Main Docker Compose file (all services)
│   ├── init-db.sql                        # PostgreSQL init script (creates databases)
│   │
│   ├── dockerfiles/
│   │   └── dockerfile_fastAPI             # Dockerfile for the API service
│   │
│   ├── monitoring/
│   │   ├── prometheus.yml                 # Prometheus scrape config (1s interval)
│   │   └── grafana/
│   │       ├── grafana.ini                # Grafana config (min_refresh_interval=1s)
│   │       ├── dashboards/
│   │       │   └── flight-delay.json      # Main dashboard (UID: flight-delay)
│   │       └── provisioning/
│   │           ├── datasources/
│   │           │   └── prometheus.yaml    # Prometheus data source (UID: prometheus_ds)
│   │           └── dashboards/
│   │               └── dashboards.yml     # Dashboard provisioning config
│   │
│   ├── scripts/
│   │   ├── bootstrap_db.py                # Data import + Grafana token generation
│   │   ├── batch_inject.py                # Bulk prediction injection
│   │   └── generate_grafana_token.sh      # Service account token creation
│   │
│   └── simulator/
│       └── simulate_traffic.py            # Traffic simulator (continuous API calls)
│
├── src/                                   # Python source code
│   ├── __init__.py                        # (empty)
│   ├── api.py                             # FastAPI service (predictions, admin, metrics)
│   ├── data.py                            # Data loading (Kaggle, CSV, PostgreSQL)
│   ├── preprocessing.py                   # Feature engineering pipeline
│   └── train.py                           # Training logic + MLflow logging
│
├── flows/                                 # Prefect flows
│   ├── __init__.py                        # (empty)
│   ├── config.py                          # Model configs (REG, CLASS, OPTUNA, etc.)
│   ├── train_flow.py                      # Training flow (with promotion)
│   ├── tune_flow.py                       # Optuna tuning flow
│   └── drift_flow.py                      # Drift detection flow (Evidently)
│
├── dbt/                                   # dbt transformation models
│   ├── dbt_project.yml                    # dbt project config
│   ├── profiles.yml                       # dbt profiles (PostgreSQL connection)
│   └── models/
│       ├── staging/
│       │   ├── sources.yml                # Source definition (raw.flights)
│       │   └── stg_flights.sql            # Staging view
│       └── training/
│           ├── pre_covid.sql              # Pre‑COVID training table
│           ├── pre_covid_100k.sql         # Pre‑COVID sample (100k rows)
│           ├── intra_covid.sql            # Intra‑COVID table
│           ├── intra_covid_100k.sql       # Intra‑COVID sample (100k rows)
│           └── retrain.sql                # Retraining table (drift-triggered)
│
├── demo/                                  # Demo scripts
│   └── covid_data_drift_demo.sh           # COVID drift demo (36 months)
│
├── tests/                                 # Unit tests
│   ├── __init__.py                        # (empty)
│   ├── test_api.py                        # API endpoint test
│   └── test_preprocessing.py              # Preprocessor test
│
└── flight_data/                           # (created at runtime) – CSV files & Kaggle cache
```

---
## Makefile Commands

The project includes a `Makefile` for common tasks. Run `make help` to see all available commands:

| Command | Description |
|---------|-------------|
| `make setup` | Full setup (build, start, import, dbt, verify) |
| `make up` | Start all services (detached) |
| `make down` | Stop all services |
| `make logs` | Tail logs from all services |
| `make train` | Train a model (default: REG) |
| `make train CONFIG=CLASS` | Train with a specific config |
| `make tune` | Run Optuna tuning (default: OPTUNA_REG) |
| `make simulator-up` | Start the traffic simulator |
| `make simulator-down` | Stop the traffic simulator |
| `make demo` | Run the COVID data drift demo |
| `make clean` | Stop containers and remove volumes (⚠️ deletes all data!) |
| `make help` | Show this help message |

---

## Disclaimer

**Model performance for predicting flight delays is limited.** The BTS dataset does not contain enough information for reliable delay predictions. The models serve solely to demonstrate the surrounding machine learning engineering framework and its capabilities. This project is an educational showcase, not a production‑ready flight delay solution.