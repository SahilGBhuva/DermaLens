import numpy as np
import pandas as pd
import pytest

from evaluate_external import load_metadata, summarize
from model import CLASSES


def one_hot(names):
    probs = np.full((len(names), len(CLASSES)), 0.01)
    for row, name in enumerate(names):
        probs[row, CLASSES.index(name)] = 0.94
    return probs


def test_summary_maps_classes_and_keeps_scc_separate():
    frame = pd.DataFrame(
        {
            "diagnostic": ["MEL", "MEL", "NEV", "BCC", "SEK", "ACK", "SCC", "SCC"],
            "fitspatrick": [2, 2, 3, 5, None, 1, 2, 2],
        }
    )
    predicted = ["mel", "nv", "nv", "bcc", "bkl", "bkl", "bcc", "nv"]
    result = summarize(frame, one_hot(predicted))

    assert result["images_scored"] == 6  # the two SCC rows are excluded
    assert result["accuracy"] == pytest.approx(4 / 6)
    assert result["melanoma_sensitivity"] == pytest.approx(0.5)
    # mean recall over the five mapped classes: MEL .5, NEV 1, BCC 1, SEK 1, ACK 0
    assert result["balanced_accuracy"] == pytest.approx(3.5 / 5)
    assert result["scc_unmapped"] == {"images": 2, "flagged_as_concerning": 0.5}
    # concerning truth: MEL, MEL, BCC, ACK, SCC, SCC -> flagged mel, bcc, bcc = 3/6
    assert result["concerning_lesions"]["sensitivity"] == pytest.approx(0.5)
    assert result["concerning_lesions"]["specificity"] == pytest.approx(1.0)
    skin = result["accuracy_by_fitzpatrick_skin_type"]
    assert skin["I–II"] == {"images": 3, "accuracy": pytest.approx(1 / 3)}
    assert skin["not recorded"]["images"] == 1
    assert result["confusion_true_pad_vs_predicted"]["MEL"]["nv"] == 1


def test_metadata_rejects_unknown_codes_and_missing_images(tmp_path):
    (tmp_path / "a.png").write_bytes(b"")
    good = tmp_path / "meta.csv"
    pd.DataFrame({"img_id": ["a.png"], "diagnostic": ["MEL"]}).to_csv(good, index=False)
    assert len(load_metadata(good, tmp_path)) == 1

    bad_code = tmp_path / "bad.csv"
    pd.DataFrame({"img_id": ["a.png"], "diagnostic": ["XYZ"]}).to_csv(bad_code, index=False)
    with pytest.raises(ValueError, match="unexpected diagnostic"):
        load_metadata(bad_code, tmp_path)

    traversal = tmp_path / "traversal.csv"
    pd.DataFrame({"img_id": ["../meta.csv"], "diagnostic": ["NEV"]}).to_csv(traversal, index=False)
    with pytest.raises(ValueError, match="plain file name"):
        load_metadata(traversal, tmp_path)

    missing = tmp_path / "missing.csv"
    pd.DataFrame({"img_id": ["nope.png"], "diagnostic": ["NEV"]}).to_csv(missing, index=False)
    with pytest.raises(FileNotFoundError):
        load_metadata(missing, tmp_path)
