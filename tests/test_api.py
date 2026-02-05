from fastapi.testclient import TestClient
from src.serving.app import app

client = TestClient(app)


def test_root():
    r = client.get("/")
    assert r.status_code == 200
    assert "message" in r.json()


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict():
    r = client.post("/predict", json={"features": [1.0, 2.5, 3.5]})
    assert r.status_code == 200
    body = r.json()
    assert "prediction" in body
    assert body["prediction"] == 7.0
    assert "model_version" in body
