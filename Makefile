.PHONY: install data train serve test monitor mlflow-ui drift retrain docker-up docker-train docker-monitor

install:        ; pip install -r requirements.txt
data:           ; python -m src.data_utils
train:          ; python -m src.train
serve:          ; uvicorn app.main:app --reload --port 8000
test:           ; pytest -q
monitor:        ; python -m src.monitor
mlflow-ui:      ; mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
drift:          ; python scripts/simulate_traffic.py --mode drift --n 300
retrain:        ; python -m src.train --extra data/drifted.csv
docker-up:      ; docker compose up -d --build
docker-train:   ; docker compose --profile tools run --rm trainer
docker-monitor: ; docker compose --profile tools run --rm monitor
