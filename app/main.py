from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
import time

# ----- Load model -----
model_data = joblib.load("models/baseline.joblib")
model = model_data['model']
model_version = model_data['version']

# ----- FastAPI app -----
app = FastAPI(title="Model Serving API")

# ----- Prometheus metrics -----
REQUEST_COUNT = Counter(
    "request_count", "Total number of requests", ["endpoint"]
)
REQUEST_LATENCY = Histogram(
    "request_latency_seconds", "Request latency in seconds", ["endpoint"]
)

# ----- Input data schema -----
class InputData(BaseModel):
    x1: float
    x2: float

# ----- Endpoints -----
@app.get("/health")
def health():
    REQUEST_COUNT.labels(endpoint="/health").inc()
    return {"status": "ok"}

@app.post("/predict")
def predict(input_data: InputData):
    start_time = time.time()
    
    x1 = input_data.x1
    x2 = input_data.x2
    
    # Prediction
    score = model.predict_proba([[x1, x2]])[:, 1][0]
    
    # Record metrics
    REQUEST_COUNT.labels(endpoint="/predict").inc()
    REQUEST_LATENCY.labels(endpoint="/predict").observe(time.time() - start_time)
    
    return {"score": float(score), "model_version": model_version}

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)