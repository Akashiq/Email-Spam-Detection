# Email Spam Detection – Model Serving, Tracking & Monitoring

An end-to-end MLOps project: train a spam classifier, track experiments with **MLflow**, serve it through a
**FastAPI** REST API, ship it with **Docker**, **log every prediction**, expose **Prometheus metrics**, and
**detect performance degradation / data drift** on live traffic.

```
                 ┌───────────────┐   metrics+models   ┌──────────────┐
 data ─► train.py ───────────────► MLflow (5000)      │ Prometheus   │
          │ saves model.joblib + metadata.json         │   (9090)     │
          ▼                                            └──────▲───────┘
   models/ ──load──► FastAPI (8000) ── /metrics ─────────────┘
                       │  /predict  /feedback
                       ▼
               logs/predictions.db  (SQLite: every prediction + later ground-truth)
                       │
                       ▼
               monitor.py  ─► compares with baseline ─► alerts + MLflow "spam-monitoring"
```

## 1. Project structure

| Path | What it is |
|---|---|
| `src/config.py` | All settings (paths, MLflow URI, alert thresholds). Overridable by env vars. |
| `src/data_utils.py` | Loads real datasets (UCI / Kaggle formats) and **generates synthetic data** (normal + drifted). |
| `src/modeling.py` | Pipelines (TF-IDF + LogReg / NaiveBayes / LinearSVM), metrics, baseline statistics, save/load. |
| `src/train.py` | Trains all candidates, logs each as an **MLflow run**, saves the best model. |
| `src/monitor.py` | **Degradation evaluation**: F1 vs baseline, PSI drift, OOV drift, spam-rate shift, latency. |
| `app/main.py` | FastAPI app: `/predict`, `/predict/batch`, `/feedback`, `/health`, `/metrics`, … |
| `app/db.py` | SQLite **prediction logging** + feedback (ground truth). |
| `app/model_service.py` | Loads the model once, thread-safe hot reload. |
| `app/schemas.py` | Pydantic request/response validation. |
| `scripts/simulate_traffic.py` | Sends normal or drifted traffic (+ feedback) to the API. |
| `tests/` | pytest tests for the API and monitoring logic. |
| `Dockerfile`, `Dockerfile.mlflow`, `docker-compose.yml` | Containers: api, mlflow, prometheus, trainer, monitor. |
| `monitoring/prometheus.yml` | Prometheus scrape config. |

## 2. Run locally (step by step)

**Prerequisites:** Python 3.10+ (3.11 recommended). Docker Desktop for section 3.

```bash
# 2.1  create environment
python -m venv .venv
source .venv/bin/activate            # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2.2  create dataset (synthetic, works offline)  -> data/emails.csv, data/drifted.csv
python -m src.data_utils

# 2.3  train (3 models, all logged to MLflow, best saved to models/)
python -m src.train

# 2.4  look at the experiments
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000     # open http://localhost:5000

# 2.5  start the API (new terminal, venv active)
uvicorn app.main:app --reload --port 8000                         # docs: http://localhost:8000/docs
```

Try it:

```bash
curl -X POST localhost:8000/predict -H "Content-Type: application/json" \
     -d '{"text":"WINNER!! Claim your free iPhone at http://bit.ly/x9Z"}'
# -> {"prediction_id":"...","label":"spam","is_spam":true,"spam_probability":0.98,...}

curl -X POST localhost:8000/feedback -H "Content-Type: application/json" \
     -d '{"prediction_id":"<id from above>","true_label":1}'

curl localhost:8000/monitoring/summary
curl localhost:8000/metrics | head -30
```

### Using a REAL dataset (recommended for your report)
Download the **SMS Spam Collection** (UCI) or Kaggle "SMS Spam Collection Dataset" (`spam.csv`), put it in `data/`, then:
```bash
python -m src.train --data data/spam.csv
```
Supported: Kaggle `spam.csv` (`v1,v2`), UCI `SMSSpamCollection` (tab separated), or any CSV with `label,text`.
For emails, e.g. Enron-Spam, convert to `label,text` columns.

## 3. Run with Docker

```bash
python -m src.train --no-mlflow          # (or step 3.2) – the API needs a trained model in ./models
docker compose up -d --build             # starts api:8000, mlflow:5000, prometheus:9090
docker compose --profile tools run --rm trainer   # retrain INSIDE docker, logging to the MLflow container
curl -X POST localhost:8000/admin/reload          # hot-load the new model, no restart
docker compose --profile tools run --rm monitor   # run the degradation report
docker compose logs -f api
docker compose down                      # stop (add -v to delete MLflow data)
```
URLs: API docs http://localhost:8000/docs · MLflow http://localhost:5000 · Prometheus http://localhost:9090

Plain Docker (without compose):
```bash
docker build -t spam-api .
docker run -p 8000:8000 -v "$PWD/logs:/app/logs" spam-api
```

## 4. Demonstrate performance degradation (the key experiment)

```bash
# 1. healthy traffic (same distribution as training) with ground-truth feedback
python scripts/simulate_traffic.py --mode normal --n 300
python -m src.monitor                    # -> STATUS: HEALTHY

# 2. drifted traffic: a new, obfuscated spam style ("fr33 cr yp70 b0nus ...") never seen in training
python scripts/simulate_traffic.py --mode drift --n 300
python -m src.monitor                    # -> PERFORMANCE_DEGRADATION, DATA_DRIFT, PREDICTION_DRIFT ... (exit code 1)

# 3. fix it: retrain including the new data, reload, verify
python -m src.train --extra data/drifted.csv
curl -X POST localhost:8000/admin/reload
python scripts/simulate_traffic.py --mode normal --n 200
python scripts/simulate_traffic.py --mode drift  --n 200
python -m src.monitor                    # -> HEALTHY again (monitor only judges the current model version)
```
Each monitor run is saved to `reports/*.json` **and** logged to the MLflow experiment `spam-monitoring`
(so you can chart F1, PSI and OOV rate over time).

### What the monitor checks
| Check | How | Alert when |
|---|---|---|
| Performance degradation | F1/precision/recall on predictions that received `/feedback` vs the F1 saved at training time | F1 drop > `F1_DROP_THRESHOLD` (0.05) |
| Prediction drift | **PSI** between training-time and live spam-probability histograms | PSI > 0.2 (0.1–0.2 = moderate) |
| Data drift | **OOV rate**: share of live words the vocabulary has never seen | rise > 0.10 |
| Spam-rate shift | live predicted spam % vs baseline | > 15 points |
| Latency | mean / p95 inference time | informational |

Why both label-based and label-free checks? Ground truth (user "this is spam" reports) arrives late or rarely;
PSI/OOV warn you **immediately**, F1 confirms the damage once labels arrive.

Tune thresholds with env vars, e.g. `F1_DROP_THRESHOLD=0.03 python -m src.monitor`.

## 5. API reference

| Method & path | Purpose |
|---|---|
| `POST /predict` `{"text": "..."}` | classify one message, logs it, returns `prediction_id` |
| `POST /predict/batch` `{"texts": [...]}` | up to 100 messages |
| `POST /feedback` `{"prediction_id","true_label"}` | attach ground truth (1=spam, 0=ham) |
| `GET /health` | liveness + model loaded? (used by Docker HEALTHCHECK) |
| `GET /model/info` | version, training metrics, baseline stats |
| `GET /monitoring/summary` | totals, spam rate, latency, live accuracy |
| `GET /metrics` | Prometheus format (`spam_predictions_total`, latency histogram, errors, feedback) |
| `POST /admin/reload` | hot-reload the model after retraining (protect this in production!) |

## 6. Tests
```bash
pytest -q
```

## 7. How each piece works (short version)
* **Model**: `TfidfVectorizer` turns text into weighted word/bi-gram features; a linear classifier outputs
  P(spam). Wrapped in one sklearn `Pipeline`, so the saved file contains preprocessing + model (no train/serve skew).
* **MLflow**: `train.py` opens one *run* per candidate, logging parameters and metrics; the winner gets the tag
  `deployed=true` and its `model.joblib` + `metadata.json` are stored as artifacts.
* **Baseline**: at training time we store test-set F1, probability histogram, OOV rate, text length in `metadata.json`.
  This is the "normal" the monitor compares against.
* **Serving**: model loads once at startup (FastAPI lifespan); endpoints are sync functions so FastAPI runs the
  CPU-bound sklearn code in a thread-pool.
* **Logging**: each prediction row = id, timestamp, text, label, probability, latency, model version; `/feedback` fills `true_label`.
* **Docker**: image installs deps first (cached layer), copies code, runs uvicorn; compose wires api + mlflow + prometheus;
  `./models` and `./logs` are volumes so models can be swapped and logs survive restarts.

## 8. Troubleshooting
* `503 Model not loaded` → run `python -m src.train`, then restart or `POST /admin/reload`.
* `monitor` says *NO_DATA* → no predictions for the **current model version** yet; send traffic (or use `--all-versions`).
* `INFO: only N labelled rows` → send more `/feedback` (needs ≥ 30, `MIN_LABELED_SAMPLES`).
* MLflow UI empty → use the same URI: `--backend-store-uri sqlite:///mlflow.db` from the project folder.
* Windows without `make` → use the raw commands above (the Makefile is only a shortcut).
* Port already in use → change the left side of the port mapping or `--port`.

## 9. Ideas to extend
Grafana dashboard on Prometheus, scheduled monitor job (cron / Airflow) that triggers retraining, MLflow Model
Registry staging→production, transformer model (DistilBERT), authentication on `/admin/*`, Postgres instead of SQLite, CI with GitHub Actions running `pytest` + `docker build`.

> Privacy note: this project stores raw message text for monitoring. In production, hash/redact PII or store only features.
