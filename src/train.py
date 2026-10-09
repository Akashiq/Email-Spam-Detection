"""Train several candidate models, track every run in MLflow, save the best one.

Usage:
    python -m src.train                       # synthetic data (auto-generated)
    python -m src.train --data data/spam.csv  # real data
    python -m src.train --no-mlflow           # skip tracking
"""
import argparse
import tempfile
from pathlib import Path

from sklearn.model_selection import train_test_split

from src.config import DATA_DIR, MLFLOW_TRACKING_URI, TRAIN_EXPERIMENT
from src.data_utils import generate_dataset, load_dataset
from src.modeling import (MODEL_NAMES, baseline_stats, build_pipeline, compute_metrics,
                          save_model)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="CSV/TSV with label,text columns")
    ap.add_argument("--extra", nargs="*", default=[], help="extra labelled CSVs to add (e.g. data/drifted.csv) - used for retraining")
    ap.add_argument("--models", nargs="+", default=MODEL_NAMES, choices=MODEL_NAMES)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-mlflow", action="store_true")
    args = ap.parse_args()

    # ---- 1. data ----
    if args.data:
        df = load_dataset(args.data)
        data_source = str(args.data)
    else:
        df = generate_dataset(3000, seed=args.seed)
        df["label"] = (df["label"] == "spam").astype(int)
        df.to_csv(DATA_DIR / "emails_train_used.csv", index=False)
        data_source = "synthetic"
    if args.extra:
        import pandas as pd
        df = pd.concat([df] + [load_dataset(p) for p in args.extra], ignore_index=True).drop_duplicates("text")
        data_source += " + " + ", ".join(args.extra)
    print(f"Loaded {len(df)} rows, spam ratio = {df['label'].mean():.2%}")

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=args.test_size, stratify=df["label"], random_state=args.seed)

    mlflow = None
    if not args.no_mlflow:
        import mlflow
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(TRAIN_EXPERIMENT)
        print(f"MLflow tracking URI: {MLFLOW_TRACKING_URI}")

    # ---- 2. train + evaluate every candidate ----
    best = None
    for name in args.models:
        pipe = build_pipeline(name)
        run_ctx = mlflow.start_run(run_name=name) if mlflow else None
        try:
            pipe.fit(X_train, y_train)
            proba = pipe.predict_proba(X_test)[:, 1]
            pred = (proba >= 0.5).astype(int)
            metrics = compute_metrics(y_test, pred, proba)
            print(f"{name:12s} " + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

            run_id = None
            if mlflow:
                run_id = run_ctx.info.run_id
                mlflow.log_params({"model": name, "data_source": data_source, "n_train": len(X_train),
                                   "n_test": len(X_test), "test_size": args.test_size, "seed": args.seed,
                                   "max_features": pipe.named_steps["tfidf"].max_features})
                mlflow.log_metrics(metrics)
            if best is None or metrics["f1"] > best["metrics"]["f1"]:
                best = {"name": name, "pipe": pipe, "metrics": metrics, "proba": proba, "run_id": run_id}
        finally:
            if mlflow:
                mlflow.end_run()

    # ---- 3. save the winner with baseline stats for monitoring ----
    metadata = save_model(best["pipe"], {
        "model_name": best["name"],
        "data_source": data_source,
        "n_train": len(X_train),
        "metrics": best["metrics"],
        "baseline": baseline_stats(best["pipe"], X_test, best["proba"]),
        "mlflow_run_id": best["run_id"],
    })
    print(f"\nBest model: {best['name']} (F1={best['metrics']['f1']:.4f}) -> saved, version {metadata['version']}")

    if mlflow:  # attach the final artifacts to the winning run
        from src.config import METADATA_PATH, MODEL_PATH
        with mlflow.start_run(run_id=best["run_id"]):
            mlflow.set_tag("deployed", "true")
            mlflow.log_artifact(str(MODEL_PATH))
            mlflow.log_artifact(str(METADATA_PATH))


if __name__ == "__main__":
    main()
