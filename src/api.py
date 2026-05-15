import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
from prometheus_fastapi_instrumentator import Instrumentator

from prepare_data import MODEL_NAME, MODELS_DIR
from contextlib import asynccontextmanager

# tests run with: uvicorn api:app --reload
# $uvicorn api:app --reload --host 127.0.0.1 --port 8000
# AND (client)
# $curl -X 'POST' '127.0.0' -H 'Content-Type: application/json'   -d '{
# "flights": [
#   {
#      "Origin": "JFK", "Dest": "LAX", "OriginAirportID": 12478, "DestAirportID": 12892,
#      "Airline": "AA", "Operating_Airline": "AA", "Flight_Number_Marketing_Airline": 101, "Tail_Number": "N789AA",
#      "Year": 2026, "Month": 5, "DayofMonth": 13, "DayOfWeek": 3,
#      "CRSDeptHrs": 14, "CRSDepMins": 30, "CRSArrHrs": 17, "CRSArrMins": 45, "Distance": 2475.0
#    }
#  ]
#}'

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles pipeline loading on startup and frees up resources on shutdown."""
    model_path = f"./{MODELS_DIR}/{MODEL_NAME}"
    if not os.path.exists(model_path):
        print(f"Model file not found at {model_path}. Please ensure the model pipeline is saved at this location." )
        raise RuntimeError(f"Model file missing at: {model_path}")
        
    app.state.model_pipeline = joblib.load(model_path)
    print("Model pipeline successfully attached to app state.")
    
    yield  # Usual python generator logic. The application runs and processes requests here    

    # SHUTDOWN: Executed when the application closes down
    print("Cleaning up resources...")
    del app.state.model_pipeline


app = FastAPI(
    title="Flight Delay Prediction API",
    description="API for predicting flight delays using a HistGradientBoosting model.",
    version="1.0",
    lifespan=lifespan
)

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
        raise HTTPException(status_code=500, detail="Model pipeline is not loaded.")
    
    try:
        # Convert incoming Pydantic payload directly to a Python dictionary
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
        
        # Batch inference via Scikit-Learn
        predictions = model_pipeline.predict(df)
        
        # Convert numpy array to native python floats for JSON serialization
        return BatchPredictionOutput(predictions=predictions.tolist())
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Batch prediction error: {str(e)}")    

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": app.state.model_pipeline is not None}
