import numpy as np
import pandas as pd
import pytest

from evaluate_external import summarize
from fit_decisions import CONCERN_INDEX, choose_threshold, concern_scores, flag_stats
from model import CLASSES


def test_threshold_is_the_highest_that_meets_the_floor_in_every_domain():
    # Dermoscopy: concerning scores 0.9, 0.8, 0.7, 0.6, 0.5 (5 lesions).
    # Phone: concerning scores 0.95, 0.4 (2 lesions).
    derm = (np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.3, 0.2]), np.array([1, 1, 1, 1, 1, 0, 0], bool))
    phone = (np.array([0.95, 0.4, 0.1]), np.array([1, 1, 0], bool))
    # 80% floor: dermoscopy alone would allow t = 0.6, but the phone domain
    # needs both of its lesions flagged, so t must drop to 0.4.
    assert choose_threshold({"derm": derm}, 0.8) == pytest.approx(0.6)
    assert choose_threshold({"derm": derm, "phone": phone}, 0.8) == pytest.approx(0.4)
    stats = flag_stats(*derm, 0.6)
    assert stats["sensitivity"] == pytest.approx(0.8) and stats["specificity"] == 1.0


def test_concern_score_sums_the_three_concerning_classes():
    probs = np.zeros((1, len(CLASSES)))
    probs[0, CLASSES.index("mel")] = 0.2
    probs[0, CLASSES.index("bcc")] = 0.15
    probs[0, CLASSES.index("akiec")] = 0.05
    probs[0, CLASSES.index("nv")] = 0.6
    assert concern_scores(probs)[0] == pytest.approx(0.4)
    assert sorted(CLASSES[i] for i in CONCERN_INDEX) == ["akiec", "bcc", "mel"]


def test_external_summary_uses_the_models_own_decision_rule():
    frame = pd.DataFrame({"diagnostic": ["MEL", "NEV"], "patient_id": ["P1", "P2"]})
    probs = np.zeros((2, len(CLASSES)))
    # The melanoma's top class is nevus, but 40% of its mass is concerning.
    probs[0, CLASSES.index("nv")], probs[0, CLASSES.index("mel")] = 0.6, 0.4
    probs[1, CLASSES.index("nv")], probs[1, CLASSES.index("mel")] = 0.9, 0.1
    old_rule = summarize(frame, probs)["concerning_lesions"]
    new_rule = summarize(frame, probs, {"classes": ["akiec", "bcc", "mel"], "threshold": 0.3})["concerning_lesions"]
    assert old_rule["sensitivity"] == 0.0  # top-class rule misses it
    assert new_rule["sensitivity"] == 1.0 and new_rule["specificity"] == 1.0
    assert new_rule["decision_rule"].startswith("P(akiec + bcc + mel) >= 0.3")


def test_external_summary_uses_the_datasets_own_domain_threshold():
    frame = pd.DataFrame({"diagnostic": ["NEV"], "patient_id": ["P1"]})
    probs = np.zeros((1, len(CLASSES)))
    probs[0, CLASSES.index("nv")], probs[0, CLASSES.index("mel")] = 0.8, 0.2
    concern = {"classes": ["akiec", "bcc", "mel"], "threshold": 0.1, "thresholds": {"phone": 0.3, "dermoscopy": 0.1}}
    # PAD-UFES-20 is phone photos: threshold 0.3, so this benign mole is not flagged.
    assert summarize(frame, probs, concern)["concerning_lesions"]["specificity"] == 1.0
    assert summarize(frame, probs, {**concern, "thresholds": {}})["concerning_lesions"]["specificity"] == 0.0


def test_per_class_rule_flags_a_rare_class_the_pooled_rule_misses():
    from model import concern_flags_for

    probs = np.zeros((2, len(CLASSES)))
    # A melanoma with modest P(mel) and little else concerning; a mole.
    probs[0, CLASSES.index("nv")], probs[0, CLASSES.index("mel")] = 0.75, 0.25
    probs[1, CLASSES.index("nv")], probs[1, CLASSES.index("mel")] = 0.95, 0.05
    pooled = {"classes": ["akiec", "bcc", "mel"], "threshold": 0.3, "thresholds": {"phone": 0.3}}
    per_class = {**pooled, "class_thresholds": {"phone": {"mel": 0.2, "bcc": 0.5, "akiec": 0.5}}}
    assert concern_flags_for(probs, pooled, "phone").tolist() == [False, False]
    assert concern_flags_for(probs, per_class, "phone").tolist() == [True, False]
    # Another photo type without per-class thresholds falls back to the pooled rule.
    assert concern_flags_for(probs, per_class, "dermoscopy").tolist() == [False, False]
