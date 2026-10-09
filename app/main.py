"""FastAPI model-serving application.

Run:  uvicorn app.main:app --reload
Docs: http://localhost:8000/docs
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

from app import db
from app.model_service import model_service
from app.schemas import (BatchRequest, BatchResponse, FeedbackRequest, PredictRequest,
                         PredictResponse)
from src.config import LOG_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s",
                    handlers=[logging.StreamHandler(), logging.FileHandler(LOG_DIR / "app.log")])
log = logging.getLogger("spam-api")

# ---------- Prometheus metrics (scraped from /metrics) ----------
PRED_COUNTER = Counter("spam_predictions_total", "Predictions served", ["label"])
LATENCY = Histogram("spam_prediction_latency_seconds", "Model inference latency",
                    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1))
PROBA = Histogram("spam_probability", "Distribution of predicted spam probability",
                  buckets=[i / 10 for i in range(1, 11)])
ERRORS = Counter("spam_prediction_errors_total", "Failed prediction requests")
FEEDBACK = Counter("spam_feedback_total", "Ground-truth labels received", ["correct"])
MODEL_LOADED = Gauge("spam_model_loaded", "1 if a model is loaded")


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    ok = model_service.load()
    MODEL_LOADED.set(1 if ok else 0)
    log.info("Model loaded: %s (version %s)", ok, model_service.version)
    yield


app = FastAPI(title="Email Spam Detection API", version="1.0.0", lifespan=lifespan)


def _require_model():
    if not model_service.loaded:
        ERRORS.inc()
        raise HTTPException(503, "Model not loaded. Train one first: python -m src.train")


def _run(texts: list[str]) -> list[PredictResponse]:
    _require_model()
    try:
        results = model_service.predict(texts)
        ids = db.log_predictions(results)          # prediction logging
    except Exception as exc:
        ERRORS.inc()
        log.exception("prediction failed")
        raise HTTPException(500, f"Prediction failed: {exc}")
    out = []
    for pid, r in zip(ids, results):
        label = "spam" if r["prediction"] else "ham"
        PRED_COUNTER.labels(label).inc()
        LATENCY.observe(r["latency_ms"] / 1000)
        PROBA.observe(r["probability"])
        out.append(PredictResponse(prediction_id=pid, label=label, is_spam=bool(r["prediction"]),
                                   spam_probability=round(r["probability"], 4),
                                   model_version=r["model_version"], latency_ms=round(r["latency_ms"], 3)))
    return out


@app.get("/health")
def health():
    return {"status": "ok" if model_service.loaded else "degraded", "model_loaded": model_service.loaded,
            "model_version": model_service.version}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    return _run([req.text])[0]


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(req: BatchRequest):
    return BatchResponse(predictions=_run(req.texts))


@app.post("/feedback")
def feedback(req: FeedbackRequest):
    """Send the true label of an earlier prediction. This is what makes performance monitoring possible."""
    correct = db.add_feedback(req.prediction_id, req.true_label)
    if correct is None:
        raise HTTPException(404, "prediction_id not found")
    FEEDBACK.labels(str(correct).lower()).inc()
    return {"status": "recorded", "model_was_correct": correct}


@app.get("/model/info")
def model_info():
    _require_model()
    return model_service.metadata


@app.get("/monitoring/summary")
def monitoring_summary():
    return db.summary()


@app.post("/admin/reload")
def reload_model():
    """Hot-reload model.joblib after retraining, without restarting the server."""
    ok = model_service.load()
    MODEL_LOADED.set(1 if ok else 0)
    if not ok:
        raise HTTPException(404, "No model file found")
    return {"status": "reloaded", "model_version": model_service.version}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
