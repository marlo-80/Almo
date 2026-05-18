import pandas as pd
import os

from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field
from typing import List

from contextlib import asynccontextmanager
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram
from prometheus_fastapi_instrumentator import Instrumentator

import mlflow


MODEL_NAME = "flight-delay-baseline"      # identisch mit dem Namen aus mlflow.register_model()
MODEL_ALIAS = "champion"                  # identisch mit dem Alias aus set_registered_model_alias()
model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"

mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000"))

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model_pipeline = mlflow.pyfunc.load_model(model_uri)
    print("Model pipeline successfully attached to app state.")
    
    yield  # Usual python generator logic. The application runs and processes requests here    

    # SHUTDOWN: Executed when the application closes down
    print("Cleaning up resources...")
    del app.state.model_pipeline


app = FastAPI(
    title="Flight Delay Prediction API",
    description="API for Predicting Flight Delays",
    version="1.0",
    lifespan=lifespan
)
Instrumentator().instrument(app).expose(app)

# Define the expected incoming JSON structure using Pydantic
class Flight(BaseModel):
    # Categorical target encoded features
    Origin: str = Field(..., examples=["JFK"])
    Dest: str = Field(..., examples=["LAX"])
    OriginAirportID: int = Field(..., examples=[12478])
    DestAirportID: int = Field(..., examples=[12892])
    Airline: str = Field(..., examples=["AA"])
    Operating_Airline: str = Field(..., examples=["AA"])
    Flight_Number_Marketing_Airline: int = Field(..., examples=[101])
    Tail_Number: str = Field(..., examples=["N789AA"])
    
    # Numeric features
    Year: int = Field(..., examples=[2026])
    Month: int = Field(..., examples=[5])
    DayofMonth: int = Field(..., examples=[13])
    DayOfWeek: int = Field(..., examples=[3])
    CRSDeptHrs: int = Field(..., examples=[14])
    CRSDepMins: int = Field(..., examples=[30])
    CRSArrHrs: int = Field(..., examples=[17])
    CRSArrMins: int = Field(..., examples=[45])
    Distance: float = Field(..., examples=[2475.0])

class PredictionOutput(BaseModel):
    prediction: float

class BatchFlights(BaseModel):
    flights: List[Flight]

# Schema for outgoing batch responses
class BatchPredictionOutput(BaseModel):
    predictions: List[float]    

@app.post("/predict", response_model=PredictionOutput)
def predict(payload: Flight):
    """Accepts single flight details and returns the model's regression prediction."""
    if app.state.model_pipeline is None:
        raise HTTPException(status_code=503, detail="Model pipeline is not loaded.")
    
    try:
        input_data = payload.model_dump()
        df = pd.DataFrame([input_data])        
        prediction = app.state.model_pipeline.predict(df)[0]        
        return PredictionOutput(prediction=float(prediction))
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")
    
@app.post("/predict_batch", response_model=BatchPredictionOutput)
def predict_batch(payload: BatchFlights):
    """Accepts an array of flight objects and returns a list of numerical predictions."""
    if app.state.model_pipeline is None:
        raise HTTPException(status_code=503, detail="Model pipeline is not loaded.")
    
    if not payload.flights:
        raise HTTPException(status_code=400, detail="The flight list cannot be empty.")
        
    try:
        # Convert list of Pydantic objects directly into a list of dictionaries
        data_dicts = [item.model_dump() for item in payload.flights]
        
        # Build a single Pandas DataFrame from the list of dictionaries
        # This preserves column names and order for the ColumnTransformer
        df = pd.DataFrame(data_dicts)        
        predictions = app.state.model_pipeline.predict(df)
        
        # Convert numpy array to native python floats for JSON serialization
        return BatchPredictionOutput(predictions=predictions.tolist())
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Batch prediction error: {str(e)}")    

@app.get("/health")
def health_check():    
    if app.state.model_pipeline is None:
        raise HTTPException(status_code=503, detail="Model pipeline is not loaded.")
    return {"status": "healthy", "model_loaded": True}
    

