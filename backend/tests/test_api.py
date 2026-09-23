from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from app.main import app

client = TestClient(app)


def make_png():
    image = Image.new("RGB", (64, 64), color=(180, 140, 120))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert "demo_mode" in body


def test_rejects_non_image():
    response = client.post(
        "/predict",
        files={"file": ("bad.txt", b"hello", "text/plain")},
    )
    assert response.status_code == 400


def test_demo_mode_does_not_fake_prediction():
    response = client.post(
        "/predict",
        files={"file": ("lesion.png", make_png(), "image/png")},
    )
    assert response.status_code == 200
    body = response.json()

    if body["demo_mode"]:
        assert body["top_class"] == "demo"
        assert body["confidence"] == 0.0
        assert body["heatmap_data_url"] is None


def test_research_status_is_transparent():
    response = client.get("/research-status")
    assert response.status_code == 200
    body = response.json()
    assert "model_loaded" in body
    assert "evaluation_available" in body
    assert body["implemented"]["lesion_level_split"] is True
    if not body["evaluation_available"]:
        assert body["evaluation"] is None


def test_stress_test_does_not_fake_demo_results():
    response = client.post(
        "/stress-test",
        files={"file": ("lesion.png", make_png(), "image/png")},
    )
    assert response.status_code == 200
    body = response.json()

    if body["demo_mode"]:
        assert body["stability"] is None
        assert body["results"] == []


def test_rejects_tiny_image():
    image = Image.new("RGB", (8, 8), color=(120, 120, 120))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    response = client.post(
        "/predict",
        files={"file": ("tiny.png", buffer.getvalue(), "image/png")},
    )
    assert response.status_code == 400
