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


def test_only_jpeg_png_webp_are_decoded():
    for fmt in ("TIFF", "GIF", "BMP", "ICO", "PPM"):
        buffer = BytesIO()
        Image.new("RGB", (300, 300), 120).save(buffer, fmt)
        response = client.post("/predict", files={"file": ("x.png", buffer.getvalue(), "image/png")})
        assert response.status_code == 400, fmt


def test_chunked_oversized_body_is_cut_off():
    big = b"\x89PNG" + b"\0" * (12 * 1024 * 1024)
    body = (
        b'--B\r\nContent-Disposition: form-data; name="file"; filename="a.png"\r\n'
        b"Content-Type: image/png\r\n\r\n" + big + b"\r\n--B--\r\n"
    )
    chunks = iter([body[i : i + 65536] for i in range(0, len(body), 65536)])
    response = client.post("/predict", content=chunks, headers={"content-type": "multipart/form-data; boundary=B"})
    assert response.status_code == 413


def test_api_responses_carry_security_headers():
    response = client.get("/health")
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["cache-control"] == "no-store"
    assert "default-src 'none'" in response.headers["content-security-policy"]


def test_rate_limit_returns_429_with_retry_after():
    from starlette.applications import Starlette
    from starlette.responses import PlainTextResponse
    from starlette.routing import Route

    from app.guards import RequestGuard

    async def ok(request):
        return PlainTextResponse("ok")

    inner = Starlette(routes=[Route("/predict", ok, methods=["POST"])])
    guarded = RequestGuard(inner, max_body_bytes=1024, limited_paths=("/predict",), rate_limit=2, rate_window_seconds=60)
    limited = TestClient(guarded)
    assert [limited.post("/predict").status_code for _ in range(3)] == [200, 200, 429]
    blocked = limited.post("/predict")
    assert int(blocked.headers["retry-after"]) >= 1


def test_rate_limit_uses_rightmost_forwarded_ip_when_trusted():
    from app.guards import RequestGuard

    guard = RequestGuard(None, 1, (), 1, 60, trust_proxy_headers=True)
    scope = {"client": ("10.0.0.1", 1)}
    headers = {b"x-forwarded-for": b"6.6.6.6, 203.0.113.9"}
    assert guard.client_id(scope, headers) == "203.0.113.9"  # spoofed left entry ignored
    assert RequestGuard(None, 1, (), 1, 60).client_id(scope, headers) == "10.0.0.1"
