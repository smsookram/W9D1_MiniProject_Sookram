# Model Serving with REST API and Batch Inference

## Overview
Minimal FastAPI-based model server that supports:
- REST predictions (`/predict`)
- Batch inference script (`batch_infer.py`)
- Prometheus metrics exposed at `/metrics`
- Docker packaging

## Setup

### 1) Clone and install
```bash
git clone <your-repo-url>
cd your-repo
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
