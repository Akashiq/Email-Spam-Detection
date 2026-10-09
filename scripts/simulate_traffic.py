"""Send realistic traffic to the running API and report ground truth via /feedback.

    python scripts/simulate_traffic.py --mode normal --n 300   # traffic like the training data
    python scripts/simulate_traffic.py --mode drift  --n 300   # new spam style -> triggers alerts
"""
import argparse
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data_utils import generate_dataset  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--mode", choices=["normal", "drift"], default="normal")
    ap.add_argument("--feedback-rate", type=float, default=0.8,
                    help="fraction of predictions for which we send the true label")
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    df = generate_dataset(args.n, seed=args.seed, drift=(args.mode == "drift"))
    sent = correct = fb = 0
    for i, row in enumerate(df.itertuples()):
        r = requests.post(f"{args.url}/predict", json={"text": row.text}, timeout=10)
        r.raise_for_status()
        pred = r.json()
        truth = 1 if row.label == "spam" else 0
        sent += 1
        correct += int(pred["is_spam"]) == truth
        if (i / args.n) < args.feedback_rate:
            requests.post(f"{args.url}/feedback",
                          json={"prediction_id": pred["prediction_id"], "true_label": truth}, timeout=10)
            fb += 1
    print(f"mode={args.mode} sent={sent} feedback={fb} accuracy={correct/sent:.3f}")


if __name__ == "__main__":
    main()
