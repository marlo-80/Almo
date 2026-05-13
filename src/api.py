from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI()

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict(data: dict):
    return {"delay": 0.0}

# Instrumentator am Ende einbinden (manchmal nötig)
Instrumentator().instrument(app).expose(app)