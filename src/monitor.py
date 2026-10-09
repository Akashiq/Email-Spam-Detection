"""Performance-degradation & drift evaluation on logged production traffic.

Compares live traffic (logs/predictions.db) with the baseline stored next to the model
(models/metadata.json) and flags:

  1. PERFORMANCE degradation : F1 on labelled predictions dropped > F1_DROP_THRESHOLD vs baseline
  2. PREDICTION drift        : PSI of the spam-probability distribution > PSI_THRESHOLD
  3. DATA drift              : out-of-vocabulary token rate rose > OOV_INCREASE_THRESHOLD
  4. Informational           : spam-rate shift, text-length shift, latency

Usage:
    python -m src.monitor                 # analyse everything
    python -m src.monitor --last 500      # only the last 500 predictions
    python -m src.monitor --all-versions  # include predictions from older model versions
    python -m src.monitor --no-mlflow
Exit code 1 if any alert fires (handy for cron / CI / alerting).
"""
import argparse
import json
from datetime import datetime, timezone

import joblib
import numpy as np

from app import db
from src.config import (F1_DROP_THRESHOLD, METADATA_PATH, MIN_LABELED_SAMPLES, MLFLOW_TRACKING_URI,
                        MODEL_PATH, MONITOR_EXPERIMENT, OOV_INCREASE_THRESHOLD, PSI_THRESHOLD,
                        REPORT_DIR)
from src.modeling import compute_metrics, oov_rate, prob_histogram


def psi(expected, actual, eps: float = 1e-4) -> float:
    """Population Stability Index between two binned distributions.
    <0.1 stable, 0.1-0.2 moderate shift, >0.2 significant shift."""
    e = np.clip(np.asarray(expected, dtype=float), eps, None)
    a = np.clip(np.asarray(actual, dtype=float), eps, None)
    e, a = e / e.sum(), a / a.sum()
    return float(np.sum((a - e) * np.log(a / e)))


def analyse(last: int | None = None, all_versions: bool = False) -> dict:
    meta = json.loads(METADATA_PATH.read_text())
    base = meta["baseline"]
    df = db.read_predictions(limit=last)
    if not all_versions:                      # judge the model that is deployed now, not its predecessors
        df = df[df["model_version"] == meta["version"]]
    report = {"generated_at": datetime.now(timezone.utc).isoformat(), "model_version": meta["version"],
              "n_predictions": int(len(df)), "alerts": [], "metrics": {}}
    if df.empty:
        report["alerts"].append("NO_DATA: no predictions logged yet")
        return report

    pipe = joblib.load(MODEL_PATH)
    m = report["metrics"]

    # ---- 1. performance on labelled rows ----
    labeled = df.dropna(subset=["true_label"])
    m["n_labeled"] = int(len(labeled))
    m["baseline_f1"] = base_f1 = meta["metrics"]["f1"]
    if len(labeled) >= MIN_LABELED_SAMPLES:
        live = compute_metrics(labeled["true_label"].astype(int), labeled["prediction"], labeled["probability"])
        m["live"] = live
        m["f1_drop"] = base_f1 - live["f1"]
        if m["f1_drop"] > F1_DROP_THRESHOLD:
            report["alerts"].append(
                f"PERFORMANCE_DEGRADATION: live F1 {live['f1']:.3f} vs baseline {base_f1:.3f} "
                f"(drop {m['f1_drop']:.3f} > {F1_DROP_THRESHOLD})")
    else:
        report["alerts"].append(f"INFO: only {len(labeled)} labelled rows (< {MIN_LABELED_SAMPLES}); "
                                "performance check skipped. Send more /feedback.")

    # ---- 2. prediction drift (PSI on spam probability) ----
    m["psi_probability"] = score = psi(base["prob_histogram"], prob_histogram(df["probability"]))
    if score > PSI_THRESHOLD:
        report["alerts"].append(f"PREDICTION_DRIFT: PSI={score:.3f} > {PSI_THRESHOLD}")

    # ---- 3. data drift (OOV rate) ----
    m["baseline_oov_rate"] = base["oov_rate"]
    m["live_oov_rate"] = live_oov = oov_rate(pipe, df["text"])
    if live_oov - base["oov_rate"] > OOV_INCREASE_THRESHOLD:
        report["alerts"].append(f"DATA_DRIFT: OOV rate {live_oov:.3f} vs baseline {base['oov_rate']:.3f}")

    # ---- 4. informational stats ----
    m["live_spam_rate"] = float(df["prediction"].mean())
    m["baseline_spam_rate"] = base["spam_rate_pred"]
    m["live_mean_text_length"] = float(df["text"].str.len().mean())
    m["baseline_mean_text_length"] = base["mean_text_length"]
    m["avg_latency_ms"] = float(df["latency_ms"].mean())
    m["p95_latency_ms"] = float(np.percentile(df["latency_ms"], 95))
    if abs(m["live_spam_rate"] - m["baseline_spam_rate"]) > 0.15:
        report["alerts"].append(f"SPAM_RATE_SHIFT: {m['live_spam_rate']:.2%} vs baseline {m['baseline_spam_rate']:.2%}")

    report["degraded"] = any(not a.startswith("INFO") and not a.startswith("NO_DATA") for a in report["alerts"])
    return report


def log_to_mlflow(report: dict):
    import mlflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MONITOR_EXPERIMENT)
    with mlflow.start_run(run_name=f"monitor-{datetime.now().strftime('%Y%m%d-%H%M%S')}"):
        mlflow.set_tag("model_version", report["model_version"])
        mlflow.set_tag("degraded", str(report.get("degraded", False)))
        flat = {}
        for k, v in report["metrics"].items():
            if isinstance(v, dict):
                flat.update({f"live_{kk}": vv for kk, vv in v.items()})
            else:
                flat[k] = v
        mlflow.log_metrics({k: float(v) for k, v in flat.items()})
        mlflow.log_param("n_predictions", report["n_predictions"])
        mlflow.log_dict(report, "monitoring_report.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--last", type=int, default=None, help="only analyse the N most recent predictions")
    ap.add_argument("--all-versions", action="store_true", help="include predictions from older model versions")
    ap.add_argument("--no-mlflow", action="store_true")
    args = ap.parse_args()

    report = analyse(args.last, args.all_versions)
    path = REPORT_DIR / f"monitoring_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"\nReport saved to {path}")
    if not args.no_mlflow and report["n_predictions"]:
        try:
            log_to_mlflow(report)
            print("Logged to MLflow experiment:", MONITOR_EXPERIMENT)
        except Exception as exc:
            print("MLflow logging skipped:", exc)
    print("\nSTATUS:", "DEGRADED - action needed" if report.get("degraded") else "HEALTHY")
    raise SystemExit(1 if report.get("degraded") else 0)


if __name__ == "__main__":
    main()
