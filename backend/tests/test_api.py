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
