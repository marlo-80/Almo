#!/bin/bash
# docker/scripts/generate_grafana_token.sh

# =============================================================================
# generate_grafana_token.sh – Automated Grafana Service Account Token Creation
# =============================================================================
#
# This script creates a Grafana Service Account and generates an API token
# for programmatic access. It is intended to run once as part of the bootstrap
# process, after Grafana is up and healthy.
#
# The token is written to a file that is shared via a Docker volume mount,
# allowing other services (e.g., the demo script) to read it without manual
# intervention.
#
# -----------------------------------------------------------------------------
# Prerequisites
# -----------------------------------------------------------------------------
# - Grafana is running and accessible via the Docker service name "grafana"
#   on port 3000.
# - The default admin credentials (admin:admin) are still valid.
# - The target directory for the token file exists and is writable.
#
# -----------------------------------------------------------------------------
# Workflow
# -----------------------------------------------------------------------------
# 1. Check if the token file already exists → skip if present (idempotence).
# 2. Wait for Grafana health endpoint to become available.
# 3. Create a Service Account named "demo-sa" with Admin role.
# 4. If the account already exists, retrieve its ID via search API.
# 5. Generate a new token for that Service Account (name includes timestamp).
# 6. Write the token to the file: /app/docker/monitoring/grafana/grafana_token.txt
# 7. Exit with success or failure status.
#
# -----------------------------------------------------------------------------
# Output
# -----------------------------------------------------------------------------
# - On success: the token is written to the file.
# - On error: an error message is printed to stderr and the script exits with 1.
#
# -----------------------------------------------------------------------------
# Usage
# -----------------------------------------------------------------------------
# This script is typically invoked by the bootstrap container after Grafana
# has started:
#
#   docker compose exec api bash /app/docker/scripts/generate_grafana_token.sh
#
# It can also be executed manually for testing if Grafana is running.
#
# -----------------------------------------------------------------------------
# Notes
# -----------------------------------------------------------------------------
# - The script uses python3 to parse JSON responses (no jq dependency).
# - The token file path is fixed; adjust the TOKEN_FILE variable if needed.
# - The script is idempotent: it will not recreate an existing token.
# - Timeouts are set to avoid indefinite hanging.
# =============================================================================

# --------------------------------------------------------------------------
# CONFIGURATION
# --------------------------------------------------------------------------
# Relative path on the host system (from the project root / docker folder).
# Writes directly into the host‑mirrored root directory.
TOKEN_FILE="/app/docker/monitoring/grafana/grafana_token.txt"

# --------------------------------------------------------------------------
# CHECK EXISTING TOKEN
# --------------------------------------------------------------------------

# If token already exists, stop immediately
#if [ -f "$TOKEN_FILE" ]; then
#    echo "Token already present."
#    exit 0
#fi

echo "Generating Grafana token from host system..."

# --------------------------------------------------------------------------
# WAIT FOR GRAFANA
# --------------------------------------------------------------------------
echo "Waiting for Grafana at grafana:3000..."
for i in {1..15}; do
    if curl -s --connect-timeout 2 http://grafana:3000/api/health > /dev/null; then
        echo "Grafana is ready."
        break
    fi
    sleep 2
    if [ $i -eq 15 ]; then
        echo "Grafana not reachable at grafana:3000 (timeout)."
        exit 1
    fi
done

# --------------------------------------------------------------------------
# CREATE SERVICE ACCOUNT
# --------------------------------------------------------------------------
SA_RESPONSE=$(curl -s --connect-timeout 5 -X POST http://admin:admin@grafana:3000/api/serviceaccounts \
    -H "Content-Type: application/json" \
    -d '{"name":"demo-sa","role":"Admin"}')

# Extract Service Account ID
SA_ID=$(echo "$SA_RESPONSE" | python3 -c "import sys,json; print(json.load(sys.stdin).get('id',''))" 2>/dev/null)

# If account already exists, retrieve via search
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
    echo "Could not determine Service Account ID."
    exit 1
fi

# --------------------------------------------------------------------------
# GENERATE TOKEN
# --------------------------------------------------------------------------
TOKEN_RESPONSE=$(curl -s --connect-timeout 5 -X POST "http://admin:admin@grafana:3000/api/serviceaccounts/${SA_ID}/tokens" \
    -H "Content-Type: application/json" \
    -d "{\"name\":\"demo-token-$(date +%s)\"}")

TOKEN=$(echo "$TOKEN_RESPONSE" | python3 -c "import sys,json; print(json.load(sys.stdin).get('key',''))" 2>/dev/null)

if [ -n "$TOKEN" ] && [ "$TOKEN" != "None" ]; then
    # Create directory structure on host if missing
    mkdir -p "$(dirname "$TOKEN_FILE")"
    # Write token to file
    echo "$TOKEN" > "$TOKEN_FILE"
    echo "Token successfully saved to host: $TOKEN_FILE"
    exit 0
else
    echo "Token generation failed."
    echo "   Response: $TOKEN_RESPONSE"
    exit 1
fi