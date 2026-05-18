# flows/train_simple.py
import pandas as pd
from sqlalchemy import create_engine
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn
from prefect import flow, task

# ------------------------------------------------------------------
# Konfiguration – hartcodiert für den ersten Durchlauf
# ------------------------------------------------------------------
DB_URI = "postgresql://vikmar:vikmar@postgres:5432/fastapi_db"
MLFLOW_URI = "http://mlflow:5000"          # Container‑interner Hostname
EXPERIMENT_NAME = "flight-delay-prediction"

TARGET_COL = "arr_delay"

# Numerische Spalten, die sicher im flights_subset vorhanden sind
# (ohne Preprocessing können wir keine kategorialen Spalten verwenden)
NUMERIC_FEATURES = [
    "crs_dep_time", "crs_arr_time",
    "dep_delay", "arr_delay", "dep_delay_minutes",
    "origin_airport_id", "dest_airport_id", "flight_number"
]

# ------------------------------------------------------------------
# Tasks
# ------------------------------------------------------------------
@task(retries=2)
def load_subset_table() -> pd.DataFrame:
    """Lädt die komplette Tabelle flights_subset (bereits sortiert)."""
    engine = create_engine(DB_URI)
    df = pd.read_sql("SELECT * FROM dbt_staging.flights_subset", engine)
    return df

@task
def chronological_split(df: pd.DataFrame, target: str = TARGET_COL):
    """
    Chronologischer Split basierend auf der Reihenfolge im DataFrame.
    Wir gehen davon aus, dass die Tabelle nach flight_date aufsteigend sortiert ist.
    """
    n = len(df)
    train_end = int(n * 0.7)
    val_end   = int(n * 0.85)

    train_df = df.iloc[:train_end]
    val_df   = df.iloc[train_end:val_end]
    test_df  = df.iloc[val_end:]

    return train_df, val_df, test_df

@task
def prepare_features(df: pd.DataFrame) -> (pd.DataFrame, pd.Series):
    """Extrahiert Features und Zielvariable."""
    X = df[NUMERIC_FEATURES].copy()
    y = df[TARGET_COL].copy()
    return X, y

@task
def train_and_log(train_df: pd.DataFrame, val_df: pd.DataFrame):
    """Trainiert RandomForest und loggt alles in MLflow."""
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Features direkt mit der Funktion trennen
    X_train = train_df[NUMERIC_FEATURES].copy()
    y_train = train_df[TARGET_COL].copy()
    X_val = val_df[NUMERIC_FEATURES].copy()
    y_val = val_df[TARGET_COL].copy()

    with mlflow.start_run(run_name="simple_rf_no_preprocessing"):
        mlflow.log_param("features", NUMERIC_FEATURES)
        mlflow.log_param("dataset", "dbt_staging.flights_subset")

        model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
        model.fit(X_train, y_train)

        preds = model.predict(X_val)
        mae = mean_absolute_error(y_val, preds)
        r2  = r2_score(y_val, preds)



        # Schwellwert definieren
        DELAY_THRESHOLD = 15   # Minuten

        y_true_bin = (y_val > DELAY_THRESHOLD).astype(int)
        y_pred_bin = (preds > DELAY_THRESHOLD).astype(int)

        precision = precision_score(y_true_bin, y_pred_bin)
        recall    = recall_score(y_true_bin, y_pred_bin)
        f1        = f1_score(y_true_bin, y_pred_bin)

        mlflow.log_metrics({
            "precision_15": precision,
            "recall_15": recall,
            "f1_15": f1,
        })



        mlflow.log_metrics({"mae_val": mae, "r2_val": r2, "precision_15": precision,"recall_15": recall, "f1_15": f1})
        mlflow.sklearn.log_model(model, "simple_rf_model")

        # Registrieren & Alias vergeben
        run_id = mlflow.active_run().info.run_id
        model_uri = f"runs:/{run_id}/simple_rf_model"
        registered = mlflow.register_model(model_uri, "flight-delay-baseline")
        client = mlflow.tracking.MlflowClient()
        client.set_registered_model_alias("flight-delay-baseline", "champion", registered.version)

        print(f"MAE: {mae:.2f}, R²: {r2:.4f}")
        return model

# ------------------------------------------------------------------
# Haupt‑Flow
# ------------------------------------------------------------------
@flow(name="flight-delay-simple-training")
def training_pipeline():
    df = load_subset_table()
    train, val, test = chronological_split(df)
    train_and_log(train, val)

if __name__ == "__main__":
    training_pipeline()