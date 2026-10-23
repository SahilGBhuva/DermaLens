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


def test_scores_only_prediction_skips_grad_cam(two_versions, monkeypatch):
    from app import main
    from app.registry import ModelRegistry

    monkeypatch.setattr(main, "registry", ModelRegistry(two_versions))
    files = {"file": ("lesion.png", make_png(), "image/png")}
    full = client.post("/predict?model=v1", files=files).json()
    scores = client.post("/predict?model=v1&explain=false", files=files).json()
    assert full["heatmap_data_url"].startswith("data:image/")
    assert scores["heatmap_data_url"] is None
    assert scores["top_class"] == full["top_class"]
    assert scores["probabilities"] == full["probabilities"]


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


def test_regular_photos_are_accepted_and_flagged():
    files = {"file": ("lesion.png", make_png(), "image/png")}
    photo = client.post("/predict?image_type=photo", files=files).json()
    assert photo["image_type"] == "photo"
    assert "demonstration only" in photo["reliability_note"]
    derm = client.post("/predict", files=files).json()
    assert derm["image_type"] == "dermoscopy" and derm["reliability_note"] is None
    assert client.post("/stress-test?image_type=photo", files=files).json()["image_type"] == "photo"
    assert client.post("/predict?image_type=selfie", files=files).status_code == 400


def test_photos_go_to_the_photo_model_when_it_is_served(two_versions, monkeypatch):
    from app import main
    from app.registry import ModelRegistry

    monkeypatch.setattr(main, "registry", ModelRegistry(two_versions))
    files = {"file": ("lesion.png", make_png(), "image/png")}

    # Photo model not served (default "v3"): photos fall back to the default model.
    fallback = client.post("/predict?image_type=photo", files=files).json()
    assert fallback["model_version"] == "v2"
    assert "dermoscopy images only" in fallback["reliability_note"]
    assert client.get("/health").json()["photo_model"] == "v2"

    monkeypatch.setattr(main, "PHOTO_MODEL", "v1")
    monkeypatch.setitem(main.PHOTO_NOTES, "v1", "v1 photo note")
    photo = client.post("/predict?image_type=photo", files=files).json()
    assert photo["model_version"] == "v1" and photo["reliability_note"] == "v1 photo note"
    assert client.post("/stress-test?image_type=photo", files=files).json()["model_version"] == "v1"
    # Dermoscopy keeps the default, and an explicit choice always wins.
    assert client.post("/predict", files=files).json()["model_version"] == "v2"
    assert client.post("/predict?image_type=photo&model=v2", files=files).json()["model_version"] == "v2"
    assert client.get("/health").json()["photo_model"] == "v1"


def test_concern_flag_is_returned_only_for_models_that_define_one(tmp_path, monkeypatch):
    import json

    from app import main
    from app.registry import ModelRegistry
    from conftest import make_version

    make_version(tmp_path / "v1", seed=1)
    make_version(tmp_path / "v4", seed=4)
    config_path = tmp_path / "v4" / "model_config.json"
    config = json.loads(config_path.read_text())
    config["concern"] = {"classes": ["akiec", "bcc", "mel"], "threshold": 0.0, "target_sensitivity": 0.9}
    config_path.write_text(json.dumps(config))
    monkeypatch.setattr(main, "registry", ModelRegistry(tmp_path, default="v4"))
    files = {"file": ("lesion.png", make_png(), "image/png")}

    flagged = client.post("/predict?model=v4", files=files).json()
    concern = sum(flagged["probabilities"][c] for c in ("akiec", "bcc", "mel"))
    assert flagged["needs_closer_look"] is True  # threshold 0: always flagged
    assert abs(flagged["concern_score"] - concern) < 1e-6

    plain = client.post("/predict?model=v1", files=files).json()
    assert plain["needs_closer_look"] is None and plain["concern_score"] is None


def test_concern_flag_uses_the_threshold_for_the_photo_type(tmp_path, monkeypatch):
    import json

    from app import main
    from app.registry import ModelRegistry
    from conftest import make_version

    make_version(tmp_path / "v4", seed=4)
    config_path = tmp_path / "v4" / "model_config.json"
    config = json.loads(config_path.read_text())
    # Dermoscopy: threshold 0 (always flagged). Phone: above any possible score (never flagged).
    config["concern"] = {"classes": ["akiec", "bcc", "mel"], "threshold": 0.0,
                         "thresholds": {"dermoscopy": 0.0, "phone": 1.01}, "target_sensitivity": 0.9}
    config_path.write_text(json.dumps(config))
    monkeypatch.setattr(main, "registry", ModelRegistry(tmp_path, default="v4"))
    monkeypatch.setattr(main, "PHOTO_MODEL", "v4")
    files = {"file": ("lesion.png", make_png(), "image/png")}
    assert client.post("/predict?model=v4", files=files).json()["needs_closer_look"] is True
    assert client.post("/predict?model=v4&image_type=photo", files=files).json()["needs_closer_look"] is False


def test_per_class_flag_matches_the_ml_rule(tmp_path, monkeypatch):
    import json
    import sys
    from pathlib import Path

    from app import main
    from app.registry import ModelRegistry
    from conftest import make_version

    make_version(tmp_path / "v4", seed=4)
    config_path = tmp_path / "v4" / "model_config.json"
    config = json.loads(config_path.read_text())
    config["concern"] = {"classes": ["akiec", "bcc", "mel"], "threshold": 1.01, "target_sensitivity": 0.9,
                         "class_thresholds": {"dermoscopy": {"mel": 0.0, "bcc": 1.01, "akiec": 1.01},
                                              "phone": {"mel": 1.01, "bcc": 1.01, "akiec": 1.01}}}
    config_path.write_text(json.dumps(config))
    monkeypatch.setattr(main, "registry", ModelRegistry(tmp_path, default="v4"))
    monkeypatch.setattr(main, "PHOTO_MODEL", "v4")
    files = {"file": ("lesion.png", make_png(), "image/png")}
    derm = client.post("/predict?model=v4", files=files).json()
    phone = client.post("/predict?model=v4&image_type=photo", files=files).json()
    assert derm["needs_closer_look"] is True and phone["needs_closer_look"] is False

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ml"))
    try:
        from model import CLASSES as ML_CLASSES, concern_flags_for
    finally:
        sys.path.pop(0)
    probs = [[derm["probabilities"][c] for c in ML_CLASSES]]
    assert bool(concern_flags_for(probs, config["concern"], "dermoscopy")[0]) is derm["needs_closer_look"]
    assert bool(concern_flags_for(probs, config["concern"], "phone")[0]) is phone["needs_closer_look"]
