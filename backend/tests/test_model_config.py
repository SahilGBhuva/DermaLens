"""Model settings: integrity checks, calibration and preprocessing parity."""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torchvision import models

from app.model import CLASSES, DermaLensModel

ROOT = Path(__file__).resolve().parents[2]


def save_random_weights(path: Path) -> str:
    torch.manual_seed(0)
    net = models.efficientnet_b0(weights=None)
    net.classifier[1] = torch.nn.Linear(net.classifier[1].in_features, len(CLASSES))
    torch.save(net.state_dict(), path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def weights(tmp_path_factory):
    path = tmp_path_factory.mktemp("w") / "model.pt"
    return path, save_random_weights(path)


def write_config(tmp_path, **values):
    path = tmp_path / "model_config.json"
    path.write_text(json.dumps(values))
    return path


def test_refuses_weights_with_wrong_hash(tmp_path, weights):
    model_path, _ = weights
    config = write_config(tmp_path, weights_sha256="0" * 64)
    model = DermaLensModel(model_path, config)
    assert model.demo_mode is True
    assert model.status == "weights failed the integrity check"


def test_loads_weights_with_matching_hash(tmp_path, weights):
    model_path, digest = weights
    model = DermaLensModel(model_path, write_config(tmp_path, weights_sha256=digest))
    assert model.demo_mode is False
    assert model.info()["status"] == "loaded"


def test_default_settings_are_plain_softmax(tmp_path, weights):
    model_path, _ = weights
    model = DermaLensModel(model_path, tmp_path / "missing.json")
    x = torch.randn(1, 3, 224, 224, device=model.device)
    with torch.inference_mode():
        expected = torch.softmax(model.model(x), dim=1)
        actual = model._probabilities(x)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)


def test_logit_bias_shifts_toward_melanoma(tmp_path, weights):
    model_path, _ = weights
    bias = [0.0] * len(CLASSES)
    bias[CLASSES.index("mel")] = 2.0
    plain = DermaLensModel(model_path, tmp_path / "missing.json")
    biased = DermaLensModel(model_path, write_config(tmp_path, logit_bias=bias))
    x = torch.randn(1, 3, 224, 224, device=plain.device)
    with torch.inference_mode():
        mel = CLASSES.index("mel")
        assert biased._probabilities(x)[0, mel] > plain._probabilities(x)[0, mel]


def test_preprocessing_matches_training_pipeline(tmp_path, weights):
    sys.path.insert(0, str(ROOT / "ml"))
    try:
        from model import eval_transform  # ml/model.py
    finally:
        sys.path.pop(0)
    model_path, _ = weights
    api = DermaLensModel(model_path, tmp_path / "missing.json")
    rng = np.random.default_rng(1)
    image = Image.fromarray(rng.integers(0, 255, (317, 451, 3), dtype=np.uint8))
    torch.testing.assert_close(api.transform(image), eval_transform()(image))
