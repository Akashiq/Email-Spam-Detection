"""Point the app at a throw-away directory BEFORE importing it, and train a small model once."""
import os
import tempfile

_tmp = tempfile.mkdtemp(prefix="spam_test_")
for var, sub in {"MODEL_DIR": "models", "LOG_DIR": "logs", "REPORT_DIR": "reports", "DATA_DIR": "data"}.items():
    os.environ[var] = os.path.join(_tmp, sub)
os.environ["DB_PATH"] = os.path.join(_tmp, "logs", "predictions.db")

import pytest  # noqa: E402

from src.data_utils import generate_dataset  # noqa: E402
from src.modeling import baseline_stats, build_pipeline, compute_metrics, save_model  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def trained_model():
    df = generate_dataset(1500, seed=1)
    y = (df["label"] == "spam").astype(int)
    pipe = build_pipeline("logreg").fit(df["text"], y)
    proba = pipe.predict_proba(df["text"])[:, 1]
    return save_model(pipe, {"model_name": "logreg", "metrics": compute_metrics(y, (proba >= .5).astype(int), proba),
                             "baseline": baseline_stats(pipe, df["text"], proba)})


@pytest.fixture(scope="session")
def client(trained_model):
    from fastapi.testclient import TestClient
    from app.main import app
    with TestClient(app) as c:      # `with` runs the startup (lifespan) code
        yield c
