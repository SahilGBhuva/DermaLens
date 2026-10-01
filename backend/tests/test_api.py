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


def test_research_status_serves_real_evaluation(tmp_path, monkeypatch):
    import json

    from app import main

    evaluation = tmp_path / "evaluation.json"
    evaluation.write_text(
        json.dumps(
            {
                "accuracy": 0.8,
                "balanced_accuracy": 0.7,
                "macro_f1": 0.65,
                "weighted_f1": 0.79,
                "macro_ovr_roc_auc": None,  # undefined when a class is missing
                "expected_calibration_error": 0.05,
                "multiclass_brier_score": 0.3,
                "confusion_matrix": [[3, 1], [0, 4]],
                "per_class_sensitivity_specificity": {
                    "mel": {"sensitivity": 0.75, "specificity": 1.0},
                    "nv": {"sensitivity": 1.0, "specificity": 0.75},
                },
                "classification_report": {"mel": {"support": 4.0}, "nv": {"support": 4.0}},
            }
        )
    )
    monkeypatch.setattr(main, "EVALUATION_PATH", evaluation)

    response = client.get("/research-status")
    assert response.status_code == 200
    body = response.json()
    assert body["evaluation_available"] is True
    assert body["evaluation"]["balanced_accuracy"] == 0.7
    assert body["evaluation"]["macro_ovr_roc_auc"] is None
    assert body["evaluation"]["test_images"] == 8
    assert body["evaluation"]["per_class"][0] == {
        "label": "mel",
        "sensitivity": 0.75,
        "specificity": 1.0,
        "support": 4,
    }
