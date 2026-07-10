#!/bin/bash
# --------------------------------------------------------------------------
# Automatische Grafana Service Account & Token Generierung
# --------------------------------------------------------------------------
echo "⏳ Generiere Grafana Service-Account und API-Token..."

# 1. Warten, bis Grafana bereit ist
while ! curl -s http://localhost:3000/api/health > /dev/null; do
    echo "  → Warte auf Grafana..."
    sleep 2
done

# 2. Service Account erstellen (oder vorhandenen suchen)
SA_RESPONSE=$(curl -s -X POST http://admin:admin@localhost:3000/api/serviceaccounts \
  -H "Content-Type: application/json" \
  -d '{"name":"demo-sa","role":"Admin"}')

# Extrahiere die ID via Python (kein jq nötig)
SA_ID=$(python3 -c "import sys, json; data=json.loads('$SA_RESPONSE'); print(data.get('id', ''))" 2>/dev/null)

# Falls der Account schon existiert, suche ihn
if [ -z "$SA_ID" ]; then
    echo "  → Service Account existiert bereits, suche ID..."
    SA_SEARCH=$(curl -s http://admin:admin@localhost:3000/api/serviceaccounts/search?query=demo-sa)
    SA_ID=$(python3 -c "import sys, json; data=json.loads('$SA_SEARCH'); print(data.get('serviceAccounts', [{'id':''}])[0]['id'])" 2>/dev/null)
fi

if [ -z "$SA_ID" ]; then
    echo "❌ Fehler: Service-Account konnte nicht erstellt oder gefunden werden."
    exit 1
fi

echo "  ✅ Service-Account ID: $SA_ID"

# 3. Token für den Service Account generieren
TOKEN_RESPONSE=$(curl -s -X POST http://admin:admin@localhost:3000/api/serviceaccounts/${SA_ID}/tokens \
  -H "Content-Type: application/json" \
  -d '{"name":"demo-token-'$(date +%s)'"}')

GRAFANA_API_TOKEN=$(python3 -c "import sys, json; data=json.loads('$TOKEN_RESPONSE'); print(data.get('key', ''))" 2>/dev/null)

if [ -n "$GRAFANA_API_TOKEN" ]; then
    echo "✅ Grafana API-Token erfolgreich generiert."
    # --- AUSGABE DES TOKENS (Lösung 2) ---
    echo "🔑 Token: $GRAFANA_API_TOKEN"
    # Optional: Token in Datei speichern (falls später benötigt)
    echo "$GRAFANA_API_TOKEN" > /tmp/grafana_token.txt
    echo "   (Token auch in /tmp/grafana_token.txt gespeichert)"
else
    echo "❌ Fehler: API-Token konnte nicht erstellt werden."
    echo "   Antwort: $TOKEN_RESPONSE"
    exit 1
fi

# Exportiere den Token für die folgenden Schritte (nur relevant, wenn das Skript gesourct wird)
export GRAFANA_API_TOKEN


docker compose -f docker/compose.yml exec api bash -c "echo 'export GRAFANA_API_KEY=$GRAFANA_API_KEY' > /etc/profile.d/grafana_token.sh"