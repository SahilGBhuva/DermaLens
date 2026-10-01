"""Exercise the loaded-model code path without trained weights.

These tests use a randomly initialised network, so they check mechanics
(shapes, probability sums, Grad-CAM output, stress-test structure), never
accuracy.
"""

import base64
from io import BytesIO

import numpy as np
import pytest
import torch
from PIL import Image

from app.gradcam import HEAT_COLORS, OVERLAY_SIZE, colorize, overlay_heatmap
from app.model import CLASSES, DermaLensModel


@pytest.fixture(scope="module")
def live_model():
    torch.manual_seed(0)
    model = DermaLensModel()
    model.model.eval()
    model.demo_mode = False
    return model


def lesion_image():
    rng = np.random.default_rng(0)
    pixels = rng.integers(90, 200, size=(96, 96, 3), dtype=np.uint8)
    pixels[30:66, 30:66] = (70, 40, 30)
    return Image.fromarray(pixels)


def decode_data_url(url: str) -> Image.Image:
    assert url.startswith("data:image/png;base64,")
    return Image.open(BytesIO(base64.b64decode(url.split(",", 1)[1])))


def test_colorize_matches_site_scale_endpoints():
    colors = colorize(np.array([0.0, 1.0]))
    np.testing.assert_allclose(colors[0], HEAT_COLORS[0], atol=1e-6)
    np.testing.assert_allclose(colors[1], HEAT_COLORS[-1], atol=1e-6)


def test_overlay_is_png_at_overlay_size():
    heat = np.zeros((7, 7), dtype=np.float32)
    heat[3, 3] = 1.0
    overlay = decode_data_url(overlay_heatmap(lesion_image(), heat))
    assert overlay.size == (OVERLAY_SIZE, OVERLAY_SIZE)
    assert overlay.mode == "RGB"


def test_live_prediction_is_well_formed(live_model):
    result = live_model.predict(lesion_image())
    assert result.demo_mode is False
    assert set(result.probabilities) == set(CLASSES)
    assert sum(result.probabilities.values()) == pytest.approx(1.0, abs=1e-4)
    assert result.top_class in CLASSES
    assert result.confidence == pytest.approx(max(result.probabilities.values()))
    assert result.uncertainty == pytest.approx(1 - result.confidence)
    assert 0.0 <= result.entropy <= 1.0 + 1e-6
    assert decode_data_url(result.heatmap_data_url).size == (OVERLAY_SIZE, OVERLAY_SIZE)


def test_live_stress_test_is_well_formed(live_model):
    body = live_model.stress_test(lesion_image())
    assert body["demo_mode"] is False
    names = [row["variant"] for row in body["results"]]
    assert names == ["Original", "Darker", "Brighter", "Lower contrast", "Blur"]
    assert body["original_class"] == body["results"][0]["top_class"]
    assert 0.2 <= body["stability"] <= 1.0
