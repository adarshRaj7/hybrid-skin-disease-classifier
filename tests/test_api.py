import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import api.main as api_main


@pytest.fixture()
def client(tiny_checkpoint, monkeypatch):
    # Point the API's normal startup flow at the tiny test checkpoint,
    # instead of bypassing it, so these tests exercise the real
    # "load once at startup" code path.
    monkeypatch.setenv("MODEL_CHECKPOINT_PATH", str(tiny_checkpoint))
    monkeypatch.delenv("HF_MODEL_REPO_ID", raising=False)
    with TestClient(api_main.app) as test_client:
        yield test_client


def _png_bytes(size=(64, 64), color=(100, 50, 50)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


def test_health_endpoint_reports_loaded_model(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["num_classes"] == 4


def test_predict_with_valid_image_returns_full_payload(client):
    files = {"file": ("test.png", _png_bytes(), "image/png")}
    response = client.post("/predict", files=files)
    assert response.status_code == 200
    body = response.json()
    for key in ("predicted_class", "confidence", "top_3_predictions", "severity", "model_version", "disclaimer"):
        assert key in body
    assert 0.0 <= body["confidence"] <= 1.0


def test_predict_rejects_unsupported_content_type(client):
    files = {"file": ("test.txt", b"hello", "text/plain")}
    response = client.post("/predict", files=files)
    assert response.status_code == 415


def test_predict_rejects_invalid_image_bytes(client):
    files = {"file": ("test.png", b"not a real png", "image/png")}
    response = client.post("/predict", files=files)
    assert response.status_code == 422
    # No internal details/stack traces should leak to the client.
    assert "Traceback" not in response.text


def test_predict_gradcam_returns_overlay(client):
    files = {"file": ("test.png", _png_bytes(), "image/png")}
    response = client.post("/predict/gradcam", files=files)
    assert response.status_code == 200
    body = response.json()
    assert "gradcam_overlay_png_base64" in body
    assert len(body["gradcam_overlay_png_base64"]) > 0


def test_health_reports_degraded_when_model_not_loaded(monkeypatch):
    monkeypatch.setenv("MODEL_CHECKPOINT_PATH", "/nonexistent/path/model.pt")
    monkeypatch.delenv("HF_MODEL_REPO_ID", raising=False)
    with TestClient(api_main.app) as test_client:
        response = test_client.get("/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is False


def test_predict_returns_503_when_model_not_loaded(monkeypatch):
    monkeypatch.setenv("MODEL_CHECKPOINT_PATH", "/nonexistent/path/model.pt")
    monkeypatch.delenv("HF_MODEL_REPO_ID", raising=False)
    with TestClient(api_main.app) as test_client:
        files = {"file": ("test.png", _png_bytes(), "image/png")}
        response = test_client.post("/predict", files=files)
    assert response.status_code == 503
