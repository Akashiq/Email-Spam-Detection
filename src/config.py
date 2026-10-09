"""Central configuration. Every value can be overridden with an environment variable,
which is how Docker / docker-compose changes behaviour without touching code."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = Path(os.getenv("MODEL_DIR", BASE_DIR / "models"))
MODEL_PATH = MODEL_DIR / "model.joblib"
METADATA_PATH = MODEL_DIR / "metadata.json"

LOG_DIR = Path(os.getenv("LOG_DIR", BASE_DIR / "logs"))
DB_PATH = Path(os.getenv("DB_PATH", LOG_DIR / "predictions.db"))

REPORT_DIR = Path(os.getenv("REPORT_DIR", BASE_DIR / "reports"))
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{BASE_DIR / 'mlflow.db'}")
TRAIN_EXPERIMENT = os.getenv("TRAIN_EXPERIMENT", "spam-detection")
MONITOR_EXPERIMENT = os.getenv("MONITOR_EXPERIMENT", "spam-monitoring")

# Degradation thresholds used by src/monitor.py
F1_DROP_THRESHOLD = float(os.getenv("F1_DROP_THRESHOLD", "0.05"))   # absolute drop vs baseline
PSI_THRESHOLD = float(os.getenv("PSI_THRESHOLD", "0.20"))           # >0.2 = significant drift
OOV_INCREASE_THRESHOLD = float(os.getenv("OOV_INCREASE_THRESHOLD", "0.10"))
MIN_LABELED_SAMPLES = int(os.getenv("MIN_LABELED_SAMPLES", "30"))

for d in (MODEL_DIR, LOG_DIR, REPORT_DIR, DATA_DIR):
    d.mkdir(parents=True, exist_ok=True)
