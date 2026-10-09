"""Prediction store (SQLite). Every prediction is logged; ground-truth labels arrive later via /feedback.
SQLite keeps the project dependency-free; in production swap this for Postgres / a log pipeline."""
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone

import pandas as pd

from src.config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id            TEXT PRIMARY KEY,
    ts            TEXT NOT NULL,
    text          TEXT NOT NULL,
    prediction    INTEGER NOT NULL,
    probability   REAL NOT NULL,
    latency_ms    REAL NOT NULL,
    model_version TEXT,
    true_label    INTEGER,
    feedback_ts   TEXT
);
CREATE INDEX IF NOT EXISTS idx_predictions_ts ON predictions(ts);
"""


@contextmanager
def _conn():
    con = sqlite3.connect(DB_PATH, timeout=30)
    try:
        yield con
        con.commit()
    finally:
        con.close()


def init_db():
    with _conn() as con:
        con.executescript(SCHEMA)


def _now():
    return datetime.now(timezone.utc).isoformat()


def log_predictions(rows: list[dict]) -> list[str]:
    """rows: dicts with text, prediction, probability, latency_ms, model_version. Returns new ids."""
    ids = [str(uuid.uuid4()) for _ in rows]
    with _conn() as con:
        con.executemany(
            "INSERT INTO predictions (id, ts, text, prediction, probability, latency_ms, model_version) "
            "VALUES (?,?,?,?,?,?,?)",
            [(i, _now(), r["text"], r["prediction"], r["probability"], r["latency_ms"], r["model_version"])
             for i, r in zip(ids, rows)])
    return ids


def add_feedback(prediction_id: str, true_label: int):
    """Returns None if the id is unknown, else True/False = was the model's prediction correct."""
    with _conn() as con:
        row = con.execute("SELECT prediction FROM predictions WHERE id=?", (prediction_id,)).fetchone()
        if row is None:
            return None
        con.execute("UPDATE predictions SET true_label=?, feedback_ts=? WHERE id=?",
                    (true_label, _now(), prediction_id))
        return row[0] == true_label


def read_predictions(limit: int | None = None, since: str | None = None) -> pd.DataFrame:
    q, params = "SELECT * FROM predictions", []
    if since:
        q += " WHERE ts >= ?"
        params.append(since)
    q += " ORDER BY ts DESC"
    if limit:
        q += " LIMIT ?"
        params.append(limit)
    with _conn() as con:
        return pd.read_sql_query(q, con, params=params)


def summary() -> dict:
    with _conn() as con:
        row = con.execute(
            "SELECT COUNT(*), COALESCE(SUM(prediction),0), COALESCE(AVG(latency_ms),0), "
            "COUNT(true_label), COALESCE(SUM(prediction = true_label),0) FROM predictions").fetchone()
    total, spam, avg_lat, labeled, correct = row
    return {
        "total_predictions": total,
        "predicted_spam": spam,
        "spam_rate": (spam / total) if total else 0.0,
        "avg_latency_ms": round(avg_lat, 3),
        "labeled_predictions": labeled,
        "live_accuracy": (correct / labeled) if labeled else None,
    }
