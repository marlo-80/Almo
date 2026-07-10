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
   - Windows Subsystem for Linux 2 installed
   - Terminal open in repository root



## Initialization

Make the `setup.sh` script executable and run it from the repository root
```bash
./setup.sh
```

Data will be downloaded to `flight_data/` and imported into `raw.flights`. This process can take a while.
To monitor progress, observe the database size:

```bash
# Linux/macOS
watch -n 5 "docker compose -f docker/compose.yml exec postgres psql -U vikmar -d fastapi_db -c \"SELECT pg_size_pretty(pg_database_size('fastapi_db')) AS size;\""

# Windows
while ($true) { Clear-Host; docker compose -f docker/compose.yml exec postgres psql -U vikmar -d fastapi_db -c "SELECT pg_size_pretty(pg_database_size('fastapi_db')) AS size;"; Start-Sleep -Seconds 5 }
```

Verify the import:

```bash
docker compose -f docker/compose.yml exec postgres psql -U vikmar -d fastapi_db -c "SELECT COUNT(*) FROM raw.flights;"
```


Once running, the following endpoints are available:

| Service    | URL                          |
|------------|------------------------------|
| FastAPI    | [http://127.0.0.1:8000](http://127.0.0.1:8000/health)      |
| Grafana    | [http://127.0.0.1:3000](http://127.0.0.1:3000)      |
| MLflow     | [http://127.0.0.1:5001](http://127.0.0.1:5001)      |
| Prefect    | [http://127.0.0.1:4200](http://127.0.0.1:4200)       |
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
docker compose -f docker/compose.yml exec postgres psql -U vikmar -d fastapi_db -c "SELECT COUNT(*), MIN(flight_date), MAX(flight_date) FROM dbt_staging.flights_subset;"
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

Assuming your config is named `NEW_MODEL`:

```bash
docker compose -f docker/compose.yml exec -e PYTHONPATH=/app -e PYTHONUNBUFFERED=1 api python flows/train_flow.py NEW_MODEL
```

View results in [MLflow](http://127.0.0.1:5001) and [Grafana](http://127.0.0.1:3000).


## How to Tune a New Model

Hyperparameter tuning with Optuna is also supported. Define parameter ranges instead of fixed values:

| Key | Type | Description | Example |
|-----|------|-------------|---------|
| `n_trials` | `int` | Number of trials | `5`, `30` |
| `direction` | `str` | Optimization direction | `"minimize"`, `"maximize"` |
| `param_ranges` | `dict` | Search spaces (replaces `model_params`) | `{"n_estimators": {"type": "int", "low": 50, "high": 300}, …}` |

Start the tuning:

```bash
docker compose -f docker/compose.yml exec -e PYTHONPATH=/app -e PYTHONUNBUFFERED=1 api python flows/tune_flow.py NEW_OPTUNA_MODEL
```


## How to Run the Traffic Simulator

Start the simulator container:

```bash
docker compose -f docker/compose.yml up -d simulator
```

Stop it:

```bash
docker compose -f docker/compose.yml down simulator
```

---

## How to Run the COVID Data Drift Experiment

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
docker compose -f docker/compose.yml exec postgres psql -U vikmar -d fastapi_db -c "DROP TABLE IF EXISTS api.predictions CASCADE;"
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
    ground_truth JSONB DEFAULT NULL,
    prediction_class_proba DOUBLE PRECISION DEFAULT NULL
);
"
```

### New dbt Models

Create new `.sql` files in `dbt/models/training`. Update `dataset_query` in your config to point to the new table or view.

### Check the Predictions Database

```bash
docker compose -f docker/compose.yml exec postgres psql -U vikmar -d fastapi_db -c "SELECT COUNT(*) FROM api.predictions;"
docker compose -f docker/compose.yml exec postgres psql -U vikmar -d fastapi_db -c "SELECT * FROM api.predictions ORDER BY timestamp DESC LIMIT 20;"
```

### Bulk Insert Predictions (Without the API)

```bash
docker compose -f docker/compose.yml exec -e PYTHONPATH=/app api python docker/scripts/batch_inject.py 2020-04-01 2020-10-01 1000
```

### Quick Database Maintenance

```bash
# Clear predictions with index reset
docker compose -f docker/compose.yml exec postgres psql -U vikmar -d fastapi_db -c "TRUNCATE TABLE api.predictions RESTART IDENTITY;"

# Clear predictions without index reset
docker compose -f docker/compose.yml exec postgres psql -U vikmar -d fastapi_db -c "TRUNCATE TABLE api.predictions;"

# Query drift score directly
curl -s "http://localhost:9090/api/v1/query?query=data_drift_score"
```



## Project Structure

```
Almo/
├── docker/
│   ├── compose.yml
│   ├── dockerfile_fastAPI
│   ├── .env
│   ├── init-db.sql
│   ├── monitoring/
│   │   ├── prometheus.yml
│   │   └── grafana/
│   │       ├── dashboards/
│   │       └── provisioning/
│   ├── scripts/
│   │   ├── bootstrap_db.py
│   │   ├── batch_inject.py
│   │   └── *.sh
│   └── simulator/
│       └── simulate_traffic.py
├── flows/
│   ├── config.py
│   ├── train_flow.py
│   ├── tune_flow.py
│   └── drift_flow.py
├── src/
│   ├── api.py
│   ├── data.py
│   ├── preprocessing.py
│   └── train.py
├── dbt/
│   ├── dbt_project.yml
│   ├── profiles.yml
│   └── models/
│       ├── staging/
│       └── training/
├── Demo/
│   ├── Pipeline_metaphor6.png
│   └── covid_data_drift_demo.sh
├── tests/
├── requirements.txt
└── README.md
```

---

## Disclaimer

**Model performance for predicting flight delays is limited.** The BTS dataset does not contain enough information for reliable delay predictions. The models serve solely to demonstrate the surrounding machine learning engineering framework and its capabilities. This project is an educational showcase, not a production‑ready flight delay solution.