Model Serving with REST API and Batch Inference
Overview

This project provides a minimal FastAPI-based model server that supports:

REST API predictions (/predict)

Batch inference via a script (batch_infer.py)

Prometheus metrics exposed at /metrics

Docker packaging for easy deployment

The included model is pre-trained (models/baseline.joblib) and ready for use.

Setup
1) Clone the repository and create a virtual environment
git clone <your-repo-url>
cd W9D1_MiniProject_Sookram
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows
.venv\Scripts\activate
pip install -r requirements.txt

2) Run batch inference
python batch_infer.py data/input.csv data/predictions.csv


Input: data/input.csv (must include columns x1 and x2)

Output: data/predictions.csv (predicted probabilities added)

Example output:

Processed 8 rows in 0.00 seconds
Output saved to data/predictions.csv

3) Run the API locally
uvicorn app.main:app --reload


Available endpoints:

GET /health → Returns server health status ({"status": "ok"})

POST /predict → Accepts JSON {"x1": 1.0, "x2": 2.0} and returns predicted probability

GET /metrics → Prometheus metrics for request counts and latency

4) Run the server with Docker
docker build -t model-server:v1 .
docker run -p 8000:8000 model-server:v1


API accessible at http://localhost:8000

5) Example API requests

Health check

curl http://localhost:8000/health


Prediction

curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"x1":1.0,"x2":2.0}'


Windows PowerShell users can use Invoke-RestMethod:

Invoke-RestMethod -Uri http://localhost:8000/predict `
                  -Method POST `
                  -Body '{"x1":1.0,"x2":2.0}' `
                  -ContentType "application/json"

Notes

Python 3.11+ is recommended.

The pre-trained model is stored in models/baseline.joblib.

Prometheus metrics provide request count and latency for monitoring.

Ensure all dependencies from requirements.txt are installed.
