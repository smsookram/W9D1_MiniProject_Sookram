from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
import time
from fastapi.responses import Response

REQUEST_COUNT = Counter("predict_requests_total", "Total number of /predict requests")
REQUEST_LATENCY = Histogram("predict_request_latency_seconds", "Latency of /predict requests in seconds")

def observe_request(func):
    """Decorator to measure latency and increment counter for predict endpoint"""
    def wrapper(*args, **kwargs):
        REQUEST_COUNT.inc()
        start = time.time()
        try:
            return func(*args, **kwargs)
        finally:
            REQUEST_LATENCY.observe(time.time() - start)
    wrapper.__name__ = getattr(func, "__name__", "wrapped")
    return wrapper

def metrics_endpoint():
    # Returns prometheus metrics bytes and proper content-type
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)