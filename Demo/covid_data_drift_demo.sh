#!/bin/bash
#set -e

# --------------------------------------------------------------------------
# Konfiguration (lädt .env aus Root und docker/)
# --------------------------------------------------------------------------
for envfile in .env docker/.env; do
    if [ -f "$envfile" ]; then
        while IFS= read -r line || [ -n "$line" ]; do
            [[ "$line" =~ ^# ]] || [[ -z "$line" ]] && continue
            export "$line"
        done < "$envfile"
    fi
done

GRAFANA_URL="http://localhost:3000"
DASHBOARD_UID="flight-delay-xxl"
PANEL_ID="4"
API_URL="http://api:8000"
PROMETHEUS_URL="http://localhost:9090"
HUGE_ROWS=1000000000

TABLE_INTRA_COVID="dbt_staging.intra_covid_test"

# --------------------------------------------------------------------------
# Determinismus: globaler Seed für batch_inject & drift_flow
# --------------------------------------------------------------------------
export SEED=42
export DRIFT_SEED=42

# --------------------------------------------------------------------------
# Monatliche Baseline (angepasst an faire Vergleiche)
# --------------------------------------------------------------------------
get_baseline() {
    local month="$1"
    case $month in
        1) echo 0.15 ;;
        2) echo 0.20 ;;
        3) echo 0.18 ;;
        4) echo 0.20 ;;
        5) echo 0.20 ;;
        6) echo 0.35 ;;
        7) echo 0.38 ;;
        8) echo 0.30 ;;
        9) echo 0.10 ;;
        10) echo 0.10 ;;
        11) echo 0.10 ;;
        12) echo 0.10 ;;
        *) echo 0.10 ;;
    esac
}

# Schwellwerte
MAX_LOAD=6.0                 # 1‑Minuten‑Load (8 Kerne → 6 ist 75%)
MIN_FREE_MB=2000             # mindestens 2 GB freier RAM

wait_for_low_load() {
    local desc="$1"
    while true; do
        local load1=$(awk '{print $1}' /proc/loadavg)
        local free_mem=$(free -m | awk '/^Mem:/ {print $NF}')

        local overloaded=false
        if (( $(echo "$load1 > $MAX_LOAD" | bc -l) )); then
            echo "  ⚠️  CPU-Last zu hoch ($load1 > $MAX_LOAD) – warte 5s …"
            overloaded=true
        fi
        if [ "$free_mem" -lt "$MIN_FREE_MB" ]; then
            echo "  ⚠️  RAM knapp ($free_mem MB < $MIN_FREE_MB MB) – warte 5s …"
            overloaded=true
        fi

        if [ "$overloaded" = false ]; then
            break
        fi
        sleep 5
    done
}

# --------------------------------------------------------------------------
# Zeitmessung (robust gegen Locale)
# --------------------------------------------------------------------------
timed_step() {
    local desc="$1"
    shift
    echo -n "  → $desc ... "
    local start end dur
    start=$(date +%s.%N)
    "$@" > /dev/null 2>&1
    end=$(date +%s.%N)
    dur=$(awk "BEGIN { printf \"%.2f\", $end - $start }")
    echo "${dur}s"
}

# --------------------------------------------------------------------------
# Batch ausführen
# --------------------------------------------------------------------------
run_batch() {
    local label="$1"
    local start="$2"
    local end="$3"
    local month="$4"
    local baseline
    baseline=$(get_baseline "$month")

    echo ""
    echo "========================================="
    echo "  $label (Monat $month)"
    echo "  Baseline = $baseline"
    echo "========================================="

    # 1. Baseline setzen
    timed_step "Setze Baseline" docker compose -f docker/compose.yml exec api curl -s -X POST "$API_URL/admin/baseline" \
      -H "Content-Type: application/json" -d "{\"value\": $baseline}"
    sleep 2

    # 2. Grafana-Annotation
    if [ -n "$GRAFANA_API_KEY" ]; then
        timed_step "Grafana-Annotation" curl -s -X POST "$GRAFANA_URL/api/annotations" \
          -H "Authorization: Bearer $GRAFANA_API_KEY" \
          -H "Content-Type: application/json" \
          -d "{
                \"dashboardUID\": \"$DASHBOARD_UID\",
                \"panelId\": $PANEL_ID,
                \"time\": $(date +%s)000,
                \"tags\": [\"batch\"],
                \"text\": \"$label\"
              }"
        sleep 2
    else
        echo "  ⚠️  GRAFANA_API_KEY nicht gesetzt – Annotation übersprungen."
        sleep 2
    fi

    # 3. Batch-Daten injizieren (ganzer Monat, deterministisch)
    wait_for_low_load "Batch-Inject Monat $month"
    timed_step "Batch-Inject ($HUGE_ROWS Zeilen)" docker compose -f docker/compose.yml exec -e PYTHONPATH=/app -e SEED="$SEED" api python docker/scripts/batch_inject.py \
      "$start" "$end" "$HUGE_ROWS" "$TABLE_INTRA_COVID"
    sleep 2

    # 4. Drift-Flow mit Monatsangabe und Seed
    wait_for_low_load "Drift-Flow Monat $month"
    timed_step "Drift-Flow (fair)" docker compose -f docker/compose.yml exec -e PYTHONPATH=/app -e PYTHONUNBUFFERED=1 \
      -e DRIFT_MONTH="$month" -e DRIFT_SEED="$DRIFT_SEED" api python flows/drift_flow.py

    # Wichtig: Prometheus braucht einen Moment, um die neuen Metriken zu scrapen
    sleep 3

    # 5. Aktuellen Drift Score von Prometheus abrufen (Sicherer JSON-Parser via Heredoc)
    local raw_json
    raw_json=$(curl -s "$PROMETHEUS_URL/api/v1/query?query=data_drift_score")
    
    DRIFT_SCORE=$(python3 2>/dev/null <<EOF
import sys, json
try:
    d = json.loads('''$raw_json''')
    print(d['data']['result'][0]['value'][1])
except Exception:
    print('')
EOF
)
    
    if [ -n "$DRIFT_SCORE" ]; then
        echo "  📊 Aktueller Drift Score: $DRIFT_SCORE"
        
        # 6. Alarm-Prüfung mit bc
        if (( $(echo "$DRIFT_SCORE > 0.5" | bc -l) )); then
            echo "  🚨 Drift-Alarm! Starte Retraining …"
            docker compose -f docker/compose.yml exec -e PYTHONPATH=/app -e PYTHONUNBUFFERED=1 api python flows/train_flow.py DRIFT_RETRAIN_REG
            docker compose -f docker/compose.yml exec -e PYTHONPATH=/app -e PYTHONUNBUFFERED=1 api python flows/train_flow.py DRIFT_RETRAIN_CLASS
        fi
    else
        echo "  ⚠️  Konnte Drift-Score nicht abrufen – kein Alarm möglich."
    fi

    echo "  ✅ Monat $month abgeschlossen."

    # ---------- REPARIERTE PAUSEN-PRÜFUNG AM ENDE DES MONATS ----------
    local PAUSE_KEY=""
    read -t 0.1 -n 1 PAUSE_KEY 2>/dev/null || true
    if [ "$PAUSE_KEY" = "p" ]; then
        echo ""
        echo "  ⏸️  Pause angefordert. $label ist vollständig beendet."
        echo "     👉 Drücke [ENTER], um mit dem nächsten Monat fortzufahren..."
        read -s
        echo "  ▶️  Setze Demo fort..."
        echo ""
    fi
    # ------------------------------------------------------------------
}

# ==========================================================================
# Hauptprogramm
# ==========================================================================

echo "🔄 Setze Dashboard-Refresh auf 2s …"
curl -s -X PATCH "$GRAFANA_URL/api/dashboards/uid/$DASHBOARD_UID" \
  -H "Authorization: Bearer $GRAFANA_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"dashboard": {"refresh": "2s"}}' > /dev/null

echo "🧹 Lösche alte Prometheus-Daten …"
docker compose -f docker/compose.yml stop prometheus
docker compose -f docker/compose.yml rm -f prometheus
docker compose -f docker/compose.yml up -d prometheus

# --- NEU: Tabelle dbt_staging.retrain zurücksetzen ---
docker compose -f docker/compose.yml exec postgres psql -U vikmar -d fastapi_db -c "DROP TABLE IF EXISTS dbt_staging.retrain;"

docker compose -f docker/compose.yml exec postgres psql -U vikmar -d fastapi_db -c "CREATE TABLE dbt_staging.retrain AS TABLE dbt_staging.pre_covid_test;"

# Champion-Metriken aus MLflow laden und initial setzen
docker compose -f docker/compose.yml exec api curl -s -X POST "$API_URL/admin/init-champion-metrics" \
  -H "Content-Type: application/json" > /dev/null

# Baseline auf Standardwert und dynamische Baseline initial setzen
#docker compose -f docker/compose.yml exec api curl -s -X POST "$API_URL/admin/baseline" \
#  -H "Content-Type: application/json" -d '{"value": 0.15}' > /dev/null

docker compose -f docker/compose.yml exec api curl -s -X POST http://api:8000/admin/drift-metrics -H "Content-Type: application/json" -d '{"drift_score": 0.15}'  

# Retrain-Status und Alarm zurücksetzen
docker compose -f docker/compose.yml exec api curl -s -X POST "$API_URL/admin/retrain-status" \
  -H "Content-Type: application/json" -d '{"new_champion": 1}' > /dev/null
docker compose -f docker/compose.yml exec api curl -s -X POST "$API_URL/admin/drift-alarm" \
  -H "Content-Type: application/json" -d '{"active": 0}' > /dev/null

# Alte Grafana-Annotationen entfernen
if [ -n "$GRAFANA_API_KEY" ]; then
    echo "🧹 Lösche alte Grafana-Annotationen …"
    docker compose -f docker/compose.yml exec -e GRAFANA_API_KEY="$GRAFANA_API_KEY" api python -c "
import requests, os
key = os.environ['GRAFANA_API_KEY']
headers = {'Authorization': f'Bearer {key}'}
resp = requests.get('http://grafana:3000/api/annotations?tags=batch&type=annotation', headers=headers)
if resp.status_code == 200:
    for ann in resp.json():
        requests.delete(f'http://grafana:3000/api/annotations/{ann[\"id\"]}', headers=headers)
    print('Erledigt.')
else:
    print(f'Fehler beim Abrufen: {resp.status_code}')
"
else
    echo "⚠️  GRAFANA_API_KEY nicht gesetzt – Annotationen können nicht gelöscht werden."
fi

echo "Leere Tabelle api.predictions …"
docker compose -f docker/compose.yml exec postgres psql -U vikmar -d fastapi_db \
  -c "TRUNCATE TABLE api.predictions RESTART IDENTITY;" > /dev/null

# Vorhersage-Zähler auf 0 setzen, damit das Panel sofort korrekt anzeigt
docker compose -f docker/compose.yml exec api curl -s -X POST "$API_URL/admin/data-stats" \
  -H "Content-Type: application/json" > /dev/null  

echo "Setze initiale Baseline auf 0.15 …"
#docker compose -f docker/compose.yml exec api curl -s -X POST "$API_URL/admin/baseline" \
 # -H "Content-Type: application/json" -d '{"value": 0.15}' > /dev/null

docker compose -f docker/compose.yml exec api curl -s -X POST http://api:8000/admin/drift-metrics -H "Content-Type: application/json" -d '{"drift_score": 0.18}'  


curl -s -X POST http://localhost:8000/admin/retrain-status -H "Content-Type: application/json" -d '{"new_champion": 1}'
curl -X POST http://localhost:8000/admin/reload-model

echo ""
read -p "✅ Alles bereit. Drücke ENTER, um die Demo zu starten ..."


echo ""
echo "Starte Demo (2020 – 2022) …"
echo ""

# --- 2020 ---
run_batch "Januar 2020"     "2020-01-01" "2020-02-01" 1
run_batch "Februar 2020"    "2020-02-01" "2020-03-01" 2
run_batch "März 2020"       "2020-03-01" "2020-04-01" 3
run_batch "April 2020"      "2020-04-01" "2020-05-01" 4
run_batch "Mai 2020"        "2020-05-01" "2020-06-01" 5
run_batch "Juni 2020"       "2020-06-01" "2020-07-01" 6
run_batch "Juli 2020"       "2020-07-01" "2020-08-01" 7
run_batch "August 2020"     "2020-08-01" "2020-09-01" 8
run_batch "September 2020"  "2020-09-01" "2020-10-01" 9
run_batch "Oktober 2020"    "2020-10-01" "2020-11-01" 10
run_batch "November 2020"   "2020-11-01" "2020-12-01" 11
run_batch "Dezember 2020"   "2020-12-01" "2021-01-01" 12

# --- 2021 ---
run_batch "Januar 2021"     "2021-01-01" "2021-02-01" 1
run_batch "Februar 2021"    "2021-02-01" "2021-03-01" 2
run_batch "März 2021"       "2021-03-01" "2021-04-01" 3
run_batch "April 2021"      "2021-04-01" "2021-05-01" 4
run_batch "Mai 2021"        "2021-05-01" "2021-06-01" 5
run_batch "Juni 2021"       "2021-06-01" "2021-07-01" 6
run_batch "Juli 2021"       "2021-07-01" "2021-08-01" 7
run_batch "August 2021"     "2021-08-01" "2021-09-01" 8
run_batch "September 2021"  "2021-09-01" "2021-10-01" 9
run_batch "Oktober 2021"    "2021-10-01" "2021-11-01" 10
run_batch "November 2021"   "2021-11-01" "2021-12-01" 11
run_batch "Dezember 2021"   "2021-12-01" "2022-01-01" 12

# --- 2022 ---
run_batch "Januar 2022"     "2022-01-01" "2022-02-01" 1
run_batch "Februar 2022"    "2022-02-01" "2022-03-01" 2
run_batch "März 2022"       "2022-03-01" "2022-04-01" 3
run_batch "April 2022"      "2022-04-01" "2022-05-01" 4
run_batch "Mai 2022"        "2022-05-01" "2022-06-01" 5
run_batch "Juni 2022"       "2022-06-01" "2022-07-01" 6
run_batch "Juli 2022"       "2022-07-01" "2022-08-01" 7
run_batch "August 2022"     "2022-08-01" "2022-09-01" 8
run_batch "September 2022"  "2022-09-01" "2022-10-01" 9
run_batch "Oktober 2022"    "2022-10-01" "2022-11-01" 10
run_batch "November 2022"   "2022-11-01" "2022-12-01" 11
run_batch "Dezember 2022"   "2022-12-01" "2023-01-01" 12

echo ""
echo "=============================================="
echo " Demo beendet. Metriken in Grafana verfügbar."
echo "=============================================="
