#!/bin/bash
# setup.sh – Initialisiert die Datenbank mit den Flugdaten

set -e  # Bei Fehlern sofort abbrechen


echo "=============================================================================="
echo "                                  Almo Setup                                  "
echo "=============================================================================="
echo "Almo initialization started..."

echo ""
echo ""

echo "=============================================================================="
echo "                            Docker Initialization                             "
echo "=============================================================================="


echo "Building the docker images..."
echo ""
docker compose -f docker/compose.yml build
echo ""
echo "...docker images built"
echo ""
echo "Starting docker..."
echo ""
docker compose -f docker/compose.yml up -d api grafana mlflow prefect prometheus postgres
echo ""
echo "...docker started"

echo ""
echo ""




#echo "=============================================================================="
#echo "                               Data Import                                    "
#echo "=============================================================================="
#echo "Downloading and importing flight data..."
#echo ""
#docker compose -f docker/compose.yml run --rm api python docker/scripts/import_data.py
#echo ""
#echo "...Data import finished"
#echo ""
#echo ""


echo "=============================================================================="
echo "                           Database Initialization                            "
echo "=============================================================================="
echo "Creating database..."
echo ""
docker compose -f docker/compose.yml run --rm api python docker/scripts/bootstrap_db.py
echo ""
echo "...Database created"

echo ""

echo "Verifying database..."
echo ""
docker compose -f docker/compose.yml exec postgres psql -U testuser -d fastapi_db -c "SELECT COUNT(*) FROM raw.flights;"
echo ""
echo "...database verified"

echo ""
echo ""

echo "=============================================================================="
echo "                             DBT Model Creations                              "
echo "=============================================================================="
echo "DBT model creation started..."
echo ""
docker compose -f docker/compose.yml exec api dbt run --project-dir /app/dbt --profiles-dir /app/dbt
echo ""
echo "...DBT model creation finished"

echo ""

echo "DBT model verification started..."
echo ""
docker compose -f docker/compose.yml exec postgres psql -U testuser -d fastapi_db -c 'SELECT COUNT(*), MIN(flight_date), MAX(flight_date) FROM dbt_staging."pre_covid";'
docker compose -f docker/compose.yml exec postgres psql -U testuser -d fastapi_db -c 'SELECT COUNT(*), MIN(flight_date), MAX(flight_date) FROM dbt_staging."pre_covid_100k";'
docker compose -f docker/compose.yml exec postgres psql -U testuser -d fastapi_db -c 'SELECT COUNT(*), MIN(flight_date), MAX(flight_date) FROM dbt_staging."intra_covid";'
docker compose -f docker/compose.yml exec postgres psql -U testuser -d fastapi_db -c 'SELECT COUNT(*), MIN(flight_date), MAX(flight_date) FROM dbt_staging."intra_covid_100k";'
docker compose -f docker/compose.yml exec postgres psql -U testuser -d fastapi_db -c 'SELECT COUNT(*), MIN(flight_date), MAX(flight_date) FROM dbt_staging."retrain";'


echo ""
echo "...DBT model verification finished"

echo ""
echo ""

echo "=============================================================================="
echo "          Create Default Models for Regression and Classification             "
echo "=============================================================================="
echo "Model training started..."
echo ""
docker compose -f docker/compose.yml exec -e PYTHONPATH=/app -e PYTHONUNBUFFERED=1 api python flows/train_flow.py REG 
docker compose -f docker/compose.yml exec -e PYTHONPATH=/app -e PYTHONUNBUFFERED=1 api python flows/train_flow.py CLASS
echo ""
echo "Model training finished"

echo "=============================================================================="
echo "                             Almo Setup Finished                              "
echo "=============================================================================="
echo "...setup has finished. Almo is ready to be used."