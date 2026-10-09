"""Model building, evaluation and baseline statistics (no MLflow here, so it is easy to test)."""
import json
from datetime import datetime, timezone

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.config import METADATA_PATH, MODEL_PATH

MODEL_NAMES = ["logreg", "naive_bayes", "linear_svm"]


def build_pipeline(name: str, max_features: int = 20000, ngram_max: int = 2) -> Pipeline:
    tfidf = TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, ngram_max),
                            max_features=max_features, sublinear_tf=True)
    if name == "logreg":
        clf = LogisticRegression(max_iter=1000, C=5.0, class_weight="balanced")
    elif name == "naive_bayes":
        clf = MultinomialNB(alpha=0.1)
    elif name == "linear_svm":
        # LinearSVC has no predict_proba, so calibrate it to get probabilities
        clf = CalibratedClassifierCV(LinearSVC(class_weight="balanced"), cv=3)
    else:
        raise ValueError(f"Unknown model '{name}', choose from {MODEL_NAMES}")
    return Pipeline([("tfidf", tfidf), ("clf", clf)])


def compute_metrics(y_true, y_pred, y_proba=None) -> dict:
    m = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    if y_proba is not None and len(set(y_true)) > 1:
        m["roc_auc"] = roc_auc_score(y_true, y_proba)
    return {k: float(v) for k, v in m.items()}


def oov_rate(pipe: Pipeline, texts) -> float:
    """Fraction of tokens (uni-grams) the vectorizer has never seen. Rises when the language drifts."""
    tfidf = pipe.named_steps["tfidf"]
    vocab = tfidf.vocabulary_
    analyzer = tfidf.build_analyzer()
    total = unknown = 0
    for t in texts:
        for tok in analyzer(str(t)):
            if " " in tok:        # skip bigrams, look at unigrams only
                continue
            total += 1
            unknown += tok not in vocab
    return float(unknown / total) if total else 0.0


def prob_histogram(probs, bins: int = 10) -> list:
    counts, _ = np.histogram(probs, bins=bins, range=(0, 1))
    return (counts / max(counts.sum(), 1)).tolist()


def baseline_stats(pipe: Pipeline, X_test, y_proba) -> dict:
    """Reference statistics saved with the model; monitoring compares live traffic against these."""
    return {
        "spam_rate_pred": float(np.mean(np.asarray(y_proba) >= 0.5)),
        "mean_text_length": float(np.mean([len(str(t)) for t in X_test])),
        "oov_rate": oov_rate(pipe, X_test),
        "prob_histogram": prob_histogram(y_proba),
    }


def save_model(pipe: Pipeline, metadata: dict) -> None:
    metadata = {**metadata, "version": datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")}
    joblib.dump(pipe, MODEL_PATH)
    METADATA_PATH.write_text(json.dumps(metadata, indent=2))
    return metadata


def load_metadata() -> dict:
    return json.loads(METADATA_PATH.read_text()) if METADATA_PATH.exists() else {}
