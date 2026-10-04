"""Upload hardening: size limits, decompression bombs, oversized bodies."""

from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from app import main
from app.main import app

client = TestClient(app)


def png(width, height):
    buffer = BytesIO()
    Image.new("RGB", (width, height), (180, 140, 120)).save(buffer, "PNG", optimize=True)
    return buffer.getvalue()


def test_rejects_decompression_bomb_before_decoding():
    data = png(9000, 9000)  # ~0.25 MB file, 81 megapixels decoded
    assert len(data) < main.MAX_UPLOAD_BYTES
    for path in ("/predict", "/stress-test"):
        response = client.post(path, files={"file": ("bomb.png", data, "image/png")})
        assert response.status_code == 413
        assert "too large" in response.json()["detail"]


def test_rejects_oversized_upload():
    data = b"\x89PNG" + b"0" * (main.MAX_UPLOAD_BYTES + 10)
    response = client.post("/predict", files={"file": ("big.png", data, "image/png")})
    assert response.status_code == 413


def test_large_but_valid_image_is_accepted():
    response = client.post("/predict", files={"file": ("photo.png", png(3000, 2000), "image/png")})
    assert response.status_code == 200


def test_truncated_image_is_rejected():
    data = png(200, 200)[:120]
    response = client.post("/predict", files={"file": ("cut.png", data, "image/png")})
    assert response.status_code == 400


def test_cors_does_not_allow_credentials():
    response = client.options(
        "/predict",
        headers={"Origin": "http://localhost:3000", "Access-Control-Request-Method": "POST"},
    )
    assert response.headers.get("access-control-allow-credentials") != "true"
