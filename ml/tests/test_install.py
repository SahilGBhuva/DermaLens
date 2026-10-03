import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from install_model import find_artifacts, verify
from model import CLASSES, sha256

ML = Path(__file__).resolve().parents[1]


def make_download(folder: Path, weights=b"v2-weights", suffix="", measured_on=None, mtime=None):
    folder.mkdir(parents=True, exist_ok=True)
    w = folder / f"dermalens_efficientnet_b0{suffix}.pt"
    w.write_bytes(weights)
    digest = sha256(w)
    files = {
        f"model_config{suffix}.json": {"classes": CLASSES, "weights_sha256": digest, "tta": True},
        f"evaluation{suffix}.json": {
            "accuracy": 0.82, "balanced_accuracy": 0.78, "macro_f1": 0.68,
            "per_class_sensitivity_specificity": {"mel": {"sensitivity": 0.81, "specificity": 0.88}},
            "settings": {"weights_sha256": measured_on or digest},
        },
        f"training_history{suffix}.json": [],
    }
    for name, body in files.items():
        (folder / name).write_text(json.dumps(body))
    if mtime:
        for p in folder.glob(f"*{suffix}.*"):
            os.utime(p, (mtime, mtime))
    return digest


def test_picks_newest_renamed_download(tmp_path):
    make_download(tmp_path, b"old", mtime=1_000_000)
    digest = make_download(tmp_path, b"new", suffix=" (1)", mtime=2_000_000)
    found = find_artifacts(tmp_path)
    assert found["weights"].name == "dermalens_efficientnet_b0 (1).pt"
    assert verify(found)[2] == digest


def test_rejects_evaluation_from_other_weights(tmp_path):
    make_download(tmp_path, measured_on="f" * 64)
    with pytest.raises(SystemExit, match="different weights"):
        verify(find_artifacts(tmp_path))


def test_rejects_settings_for_other_weights(tmp_path):
    make_download(tmp_path)
    (tmp_path / "dermalens_efficientnet_b0.pt").write_bytes(b"swapped")
    with pytest.raises(SystemExit, match="does not match"):
        verify(find_artifacts(tmp_path))


def test_reports_missing_files(tmp_path):
    make_download(tmp_path)
    (tmp_path / "model_config.json").unlink()
    with pytest.raises(SystemExit, match="model_config.json"):
        find_artifacts(tmp_path)


def test_install_backs_up_previous_and_records_evaluation_hash(tmp_path):
    models = tmp_path / "models"
    make_download(models, b"v1-weights")
    cfg = json.loads((models / "model_config.json").read_text())
    (models / "model_config.json").write_text(json.dumps({**cfg, "version": "v1"}))
    digest = make_download(tmp_path / "dl", b"v2-weights")

    subprocess.run(
        [sys.executable, str(ML / "install_model.py"), "--from", str(tmp_path / "dl"),
         "--to", str(models), "--version", "v2"],
        check=True, cwd=ML, capture_output=True,
    )
    assert (models / "previous" / "v1" / "dermalens_efficientnet_b0.pt").read_bytes() == b"v1-weights"
    config = json.loads((models / "model_config.json").read_text())
    assert config["version"] == "v2"
    assert config["weights_sha256"] == digest
    assert config["evaluation_sha256"] == sha256(models / "evaluation.json")
