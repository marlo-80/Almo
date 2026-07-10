#!/bin/bash
# docker/scripts/generate_grafana_token.sh

# 1. Relativer Pfad auf dem Host-System (ausgehend vom Projekt-Root / docker-Ordner)
# Wenn du im docker/ Ordner bist, schreibt das die Datei direkt dorthin
# KORREKT: Schreibt direkt in das vom Host gespiegelte Root-Verzeichnis
TOKEN_FILE="/app/docker/monitoring/grafana/grafana_token.txt"


# Falls Token schon existiert, sofort aufhören
if [ -f "$TOKEN_FILE" ]; then
    echo "✅ Token bereits vorhanden."
    exit 0
fi

echo "⏳ Generiere Grafana Token vom Host-System aus..."

# 2. Warten auf Grafana über GRAFANA (weil wir auf dem Host sind!)
echo "⏳ Warte auf Grafana unter grafana:3000..."
for i in {1..15}; do
    if curl -s --connect-timeout 2 http://grafana:3000/api/health > /dev/null; then
        echo "✅ Grafana ist bereit."
        break
    fi
    sleep 2
    if [ $i -eq 15 ]; then
        echo "❌ Grafana nicht erreichbar unter grafana:3000 (Timeout)."
        exit 1
    fi
done

# 3. Service Account erstellen via grafana
SA_RESPONSE=$(curl -s --connect-timeout 5 -X POST http://admin:admin@grafana:3000/api/serviceaccounts \
    -H "Content-Type: application/json" \
    -d '{"name":"demo-sa","role":"Admin"}')

# ID des Service Accounts extrahieren
SA_ID=$(echo "$SA_RESPONSE" | python3 -c "import sys,json; print(json.load(sys.stdin).get('id',''))" 2>/dev/null)

# 4. Falls er schon existiert, über die Suche holen via grafana
if [ -z "$SA_ID" ] || [ "$SA_ID" = "None" ]; then
    SA_SEARCH=$(curl -s --connect-timeout 5 "http://admin:admin@grafana:3000/api/serviceaccounts/search")
    SA_ID=$(echo "$SA_SEARCH" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    match = [sa['id'] for sa in d.get('serviceAccounts', []) if sa.get('name') == 'demo-sa']
    print(match[0] if match else '')
except Exception:
    print('')
" 2>/dev/null)
fi

if [ -z "$SA_ID" ] || [ "$SA_ID" = "None" ]; then
    echo "❌ Service Account konnte nicht ermittelt werden."
    exit 1
fi

# 5. Token generieren via grafana
TOKEN_RESPONSE=$(curl -s --connect-timeout 5 -X POST "http://admin:admin@grafana:3000/api/serviceaccounts/${SA_ID}/tokens" \
    -H "Content-Type: application/json" \
    -d "{\"name\":\"demo-token-$(date +%s)\"}")

TOKEN=$(echo "$TOKEN_RESPONSE" | python3 -c "import sys,json; print(json.load(sys.stdin).get('key',''))" 2>/dev/null)

if [ -n "$TOKEN" ] && [ "$TOKEN" != "None" ]; then
    # Ordnerstruktur auf dem Host erstellen, falls sie fehlt
    mkdir -p "$(dirname "$TOKEN_FILE")"
    # Token schreiben
    echo "$TOKEN" > "$TOKEN_FILE"
    echo "✅ Token erfolgreich auf dem Host gespeichert: $TOKEN_FILE"
    exit 0
else
    echo "❌ Token-Generierung fehlgeschlagen."
    echo "   Antwort: $TOKEN_RESPONSE"
    exit 1
fi
