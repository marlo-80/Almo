# Capstone2-Delay_Prediction_For_US_Flights_2013-2018
This is the capstone project of Viktor and Markus. The projects goal is to predict flight delays for domestic flights in the US as a use case for a complete Machine Learning Engineering setup. 

## Prerequisites
- Docker & Docker Compose installed
- Project cloned, `docker/.env` contains at least:
  - `POSTGRES_USER`
  - `POSTGRES_PASSWORD`


## Initialization
To make sure that there are no conflicts when creating our docker containers, delete all volumes defined in the compose.yml first. You can use this command: <br><br>
`docker compose down -v`

All services needed run in a Docker-based local stack. To start the local services execute this from the repository root: <br><br>
`docker compose -f docker/compose.yml up -d`

When the stack is running, the local endpoints are:
- `FastAPI/Uvicorn`: `http://127.0.0.1:8000`
- `Grafana`: `http://127.0.0.1:4200`
- `MLflow`: `http://127.0.0.1:5001`
- `Prefect`: `http://127.0.0.1:4200`
- `Postgres`: `http://127.0.0.1:5432`
- `Prometheus`: `http://127.0.0.1:9090`

### First Start only
At the first start some bootstrapping is needed to dowload the data and setup Postgres SQL. After all services from the initialization have been established execute:<br><br>
`docker compose -f docker/compose.yml --profile init up -d bootstrap`

Data will be downloaded to \repofolder\flight_data.<br>
Check the progress with:
<br><br>
`docker logs docker-bootstrap-1 -f`
<br><br>
Wait until the log shows "Import abgeschlossen. XXXX Zeilen in raw.flights eingefügt."<br>
You can verify the table with:
<br>
<br>
`docker compose -f docker/compose.yml exec postgres psql -U vikmar -d fastapi_db -c "SELECT COUNT(*) FROM raw.flights;"`

## Creation of dbt models
Set a random seed to make data sampling reproducable. Unfortunately, dbt models don't have a seed parameter by themself.
<br><br>
`docker compose -f docker/compose.yml exec postgres psql -U vikmar -d fastapi_db -c "SELECT setseed(0.42);"`
<br><br>
Run dbt with default values: Start: 2019-01-01 / Stopp: 2020-01-01 / Sample size: 100k rows):<br>
<br>
`docker compose -f docker/compose.yml exec api dbt run --project-dir /app/dbt --profiles-dir /app/dbt`
<br>

Verification of the dbt model:
<br>
`docker compose -f docker/compose.yml exec postgres psql -U vikmar -d fastapi_db -c "SELECT COUNT(*), MIN(flight_date), MAX(flight_date) FROM dbt_staging.flights_subset;"`

## Train and Log to MLFlow
For now, modeling and orchestration are still in one file. In the future everything regarding modeling will be in `\src` and everything regarding prefect in `\flows`. To start the `train_simple.py` example execute:

`docker compose -f docker/compose.yml exec api python flows/train_simple.py`

Open MLflow UI: http://localhost:5001<br>
Experiment flight-delay-prediction contains the run with metrics and model artifact.

