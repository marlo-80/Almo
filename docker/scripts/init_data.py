#!/bin/bash
set -e

DATA_DIR="/app/flight_data"
ZIP_FILE="$DATA_DIR/flight-delay-dataset-20182022.zip"

echo "=== Initialisiere Daten ==="

# Prüfen, ob bereits CSVs existieren
if ls $DATA_DIR/Combined_Flights_*.csv 1> /dev/null 2>&1; then
    echo "✅ CSVs bereits vorhanden – überspringe Download."
    exit 0
fi

# Kaggle-CLI installieren, falls nicht vorhanden
if ! command -v kaggle &> /dev/null; then
    echo "📦 Installiere Kaggle-CLI..."
    pip install kaggle
fi

# Daten herunterladen (falls noch nicht geschehen)
if [ ! -f "$ZIP_FILE" ]; then
    echo "📥 Lade Dataset von Kaggle herunter..."
    kaggle datasets download -d robikscube/flight-delay-dataset-20182022 -p "$DATA_DIR"
else
    echo "📦 ZIP bereits vorhanden – überspringe Download."
fi

# Entpacken
echo "📂 Entpacke ZIP-Datei..."
unzip -o "$ZIP_FILE" -d "$DATA_DIR"

# ZIP löschen (optional)
rm -f "$ZIP_FILE"

echo "✅ Datenbereitstellung abgeschlossen."