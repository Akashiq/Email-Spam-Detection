"""Loads the trained pipeline once and serves predictions."""
import threading
import time

import joblib

from src.config import METADATA_PATH, MODEL_PATH
from src.modeling import load_metadata


class ModelService:
    def __init__(self):
        self._lock = threading.Lock()
        self.pipeline = None
        self.metadata: dict = {}

    @property
    def loaded(self) -> bool:
        return self.pipeline is not None

    @property
    def version(self) -> str:
        return self.metadata.get("version", "unknown")

    def load(self) -> bool:
        if not MODEL_PATH.exists():
            return False
        pipe = joblib.load(MODEL_PATH)
        with self._lock:                     # swap atomically so hot-reload is safe
            self.pipeline = pipe
            self.metadata = load_metadata()
        return True

    def predict(self, texts: list[str]) -> list[dict]:
        if not self.loaded:
            raise RuntimeError("Model not loaded")
        start = time.perf_counter()
        probs = self.pipeline.predict_proba(texts)[:, 1]
        per_item_ms = (time.perf_counter() - start) * 1000 / len(texts)
        return [{"text": t, "prediction": int(p >= 0.5), "probability": float(p),
                 "latency_ms": per_item_ms, "model_version": self.version}
                for t, p in zip(texts, probs)]


model_service = ModelService()
