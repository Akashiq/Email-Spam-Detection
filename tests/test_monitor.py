from app import db
from app.model_service import model_service
from src.data_utils import generate_dataset
from src.monitor import analyse, psi


def test_psi_identical_is_zero_and_shift_is_large():
    same = [0.1] * 10
    assert psi(same, same) < 1e-6
    assert psi([0.9] + [0.1 / 9] * 9, [0.1 / 9] * 9 + [0.9]) > 0.2


def _simulate(drift: bool, n=150, flip=False):
    db.init_db()
    model_service.load()
    df = generate_dataset(n, seed=99, drift=drift)
    results = model_service.predict(list(df["text"]))
    for pid, lab in zip(db.log_predictions(results), df["label"]):
        truth = 1 if lab == "spam" else 0
        db.add_feedback(pid, truth)


def test_healthy_traffic_is_not_flagged():
    _simulate(drift=False)
    report = analyse()
    assert report["degraded"] is False


def test_drifted_traffic_is_flagged():
    _simulate(drift=True, n=300)
    report = analyse(last=300)
    assert report["degraded"] is True
    assert any(a.startswith("DATA_DRIFT") for a in report["alerts"])
