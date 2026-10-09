SPAM = "WINNER!! You have been selected to receive a $1000 gift card. Click http://bit.ly/x9Z now!"
HAM = "Hi Sam, can we move the budget meeting to Friday?"


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["model_loaded"] is True


def test_predict_spam_and_ham(client):
    spam = client.post("/predict", json={"text": SPAM}).json()
    ham = client.post("/predict", json={"text": HAM}).json()
    assert spam["label"] == "spam" and spam["is_spam"] is True
    assert ham["label"] == "ham"
    assert 0 <= spam["spam_probability"] <= 1
    assert spam["prediction_id"]


def test_validation_error(client):
    assert client.post("/predict", json={"text": ""}).status_code == 422
    assert client.post("/predict", json={}).status_code == 422


def test_batch(client):
    r = client.post("/predict/batch", json={"texts": [SPAM, HAM]})
    assert r.status_code == 200
    assert [p["label"] for p in r.json()["predictions"]] == ["spam", "ham"]


def test_feedback_flow(client):
    pid = client.post("/predict", json={"text": SPAM}).json()["prediction_id"]
    r = client.post("/feedback", json={"prediction_id": pid, "true_label": 1})
    assert r.status_code == 200 and r.json()["model_was_correct"] is True
    assert client.post("/feedback", json={"prediction_id": "nope", "true_label": 1}).status_code == 404


def test_monitoring_summary_and_metrics(client):
    client.post("/predict", json={"text": HAM})
    s = client.get("/monitoring/summary").json()
    assert s["total_predictions"] >= 1
    m = client.get("/metrics")
    assert m.status_code == 200 and "spam_predictions_total" in m.text


def test_model_info(client):
    info = client.get("/model/info").json()
    assert "metrics" in info and "baseline" in info
