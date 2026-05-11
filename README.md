# Capstone2-Delay_Prediction_For_US_Flights_2013-2018
This is the capstone project of Viktor and Markus

This repository includes a Docker-based local stack for `Postgres`, `MLflow`, and `Prefect`. The command to start the services is: `docker compose docker/compose.yaml up -d` from the project root to start the local services used throughout the lessons.

Run `docker compose -f docker/compose.yaml up -d` from the project root to start the local services
When the stack is running, the local endpoints are:

- `FastAPI/Uvicorn`: `http://127.0.0.1:8000`
- `Grafana`: `http://127.0.0.1:4200`
- `MLflow`: `http://127.0.0.1:5001`
- `Prefect`: `http://127.0.0.1:4200`
- `Postgres`: `localhost:5432`
- `Prometheus`: `http://127.0.0.1:9090`
