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


def test_research_status_reports_every_version(two_versions, monkeypatch):
    from app import main
    from app.registry import ModelRegistry

    monkeypatch.setattr(main, "registry", ModelRegistry(two_versions))
    body = client.get("/research-status").json()
    assert body["default_model"] == "v2"
    assert [m["version"] for m in body["models"]] == ["v2", "v1"]
    assert body["evaluation"]["balanced_accuracy"] == 0.73  # top level = default
    v1 = body["models"][1]
    assert v1["evaluation"]["balanced_accuracy"] == 0.74
    assert v1["evaluation"]["test_images"] == 8
    assert v1["evaluation"]["confusion_matrix"] == [[3, 1], [0, 4]]
    assert v1["evaluation"]["per_class"][0] == {"label": "mel", "sensitivity": 0.75, "specificity": 1.0, "support": 4}
    assert [row["epoch"] for row in v1["training"]] == [1, 2]
    assert "val_loss" not in v1["training"][0]


def test_predict_uses_requested_version(two_versions, monkeypatch):
    from app import main
    from app.registry import ModelRegistry

    monkeypatch.setattr(main, "registry", ModelRegistry(two_versions))
    files = {"file": ("lesion.png", make_png(), "image/png")}
    assert client.post("/predict", files=files).json()["model_version"] == "v2"
    assert client.post("/predict?model=v1", files=files).json()["model_version"] == "v1"
    assert client.post("/stress-test?model=v1", files=files).json()["model_version"] == "v1"
    missing = client.post("/predict?model=v9", files=files)
    assert missing.status_code == 404 and "v2, v1" in missing.json()["detail"]
    assert client.get("/health").json()["models"] == ["v2", "v1"]


def test_metrics_hidden_when_measured_on_other_weights(tmp_path, monkeypatch):
    from app import main
    from app.registry import ModelRegistry
    from conftest import make_version

    make_version(tmp_path / "models" / "v1", evaluation_for_other_weights=True)
    monkeypatch.setattr(main, "registry", ModelRegistry(tmp_path / "models"))
    body = client.get("/research-status").json()
    assert body["model_loaded"] is True
    assert body["evaluation_available"] is False and body["evaluation"] is None


def test_no_models_means_demo_and_no_metrics(tmp_path, monkeypatch):
    from app import main
    from app.registry import ModelRegistry

    monkeypatch.setattr(main, "registry", ModelRegistry(tmp_path / "empty"))
    body = client.get("/research-status").json()
    assert body["model_loaded"] is False and body["training"] is None and body["evaluation"] is None
    files = {"file": ("lesion.png", make_png(), "image/png")}
    assert client.post("/predict", files=files).json()["demo_mode"] is True
