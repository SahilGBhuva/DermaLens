import numpy as np
import torch
from sklearn.metrics import recall_score

from calibrate import MEL, fit_bias, fit_temperature, nll
from metrics import summarize_metrics
from model import CLASSES, apply_calibration


def synthetic(n=1400, seed=0, overconfident=3.0, mel_shrink=0.4):
    """Logits that are overconfident and under-call melanoma."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, len(CLASSES), n)
    logits = rng.normal(0, 1, (n, len(CLASSES)))
    logits[np.arange(n), y] += 1.5
    logits[:, MEL] -= mel_shrink * 3
    probs = torch.softmax(torch.tensor(logits * overconfident, dtype=torch.float32), dim=1)
    return probs, y


def test_identity_calibration_is_noop():
    probs, _ = synthetic()
    torch.testing.assert_close(apply_calibration(probs, 1.0, [0.0] * len(CLASSES)), probs)


def test_temperature_reduces_validation_nll_for_overconfident_model():
    probs, y = synthetic()
    t = fit_temperature(probs, y)
    assert t > 1.0
    assert nll(apply_calibration(probs, t, [0.0] * len(CLASSES)), y) < nll(probs, y)


def test_bias_raises_melanoma_sensitivity_to_target():
    probs, y = synthetic()
    before = recall_score(y == MEL, probs.argmax(1).numpy() == MEL)
    # Same order as the pipeline: fit temperature, then offsets.
    t = fit_temperature(probs, y)
    bias = fit_bias(probs, y, temperature=t, min_mel_sensitivity=0.8)
    after = recall_score(y == MEL, apply_calibration(probs, t, bias).argmax(1).numpy() == MEL)
    assert after > before
    assert after >= 0.8


def test_metrics_stay_seven_by_seven_when_classes_missing():
    y = [0, 1, 2, 4, 5, 5]
    p = np.full((6, 7), 0.01)
    p[np.arange(6), [0, 1, 2, 4, 5, 4]] = 0.94
    m = summarize_metrics(y, p, CLASSES)
    assert len(m["confusion_matrix"]) == 7
    assert m["macro_ovr_roc_auc"] is None or np.isfinite(m["macro_ovr_roc_auc"])
