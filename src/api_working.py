from fastapi import FastAPI
from prometheus_client import Counter, Histogram, generate_latest
from prometheus_fastapi_instrumentator import Instrumentator
import mlflow.pyfunc


app = FastAPI()

# Prometheus Instrumentator – fügt automatisch Metriken zu allen Endpunkten hinzu
Instrumentator().instrument(app).expose(app)

# Optional: Eigene Metriken definieren
PREDICTION_REQUESTS = Counter(
    "prediction_requests_total",
    "Total number of prediction requests",
    ["model_version"]
)
PREDICTION_DURATION = Histogram(
    "prediction_duration_seconds",
    "Time spent processing prediction"
)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(data: dict):
    # Später wird hier das Modell aus MLflow geladen und die Vorhersage gemacht
    return {"delay": 0.0}

@app.get("/metrics")
async def metrics():
    """Expose Prometheus metrics."""
    return generate_latest(REGISTRY)