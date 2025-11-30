from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os
from typing import Dict
from app.metrics import observe_request, metrics_endpoint, REQUEST_COUNT, REQUEST_LATENCY

app = FastAPI(title="Model Server")

# Load model at startup
MODEL_PATH = os.environ.get("MODEL_PATH", "models/baseline.joblib")

class PredictRequest(BaseModel):
    x1: float
    x2: float

class PredictResponse(BaseModel):
    score: float
    model_version: str

@app.on_event("startup")
def load_model():
    global model, model_version
    try:
        saved = joblib.load(MODEL_PATH)
        model = saved["model"]
        model_version = saved.get("version", "v1.0")
        print(f"Loaded model from {MODEL_PATH} version={model_version}")
    except Exception as e:
        print("Failed to load model:", e)
        # re-raise so the server fails fast if model missing
        raise

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictResponse)
@observe_request
def predict(payload: PredictRequest):
    try:
        X = [[payload.x1, payload.x2]]
        proba = model.predict_proba(X)
        # For binary logistic regression, take probability of class 1
        score = float(proba[0][1])
        return {"score": score, "model_version": model_version}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
def metrics():
    return metrics_endpoint()