import hashlib
import json

import numpy as np

from app.scope import ScopeCheck


def write_stats(folder, means, present, precision, threshold):
    path = folder / "scope_stats.npz"
    np.savez_compressed(path, means=means.astype(np.float32), present=present, precision=precision.astype(np.float32))
    return {"stats_file": path.name, "stats_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "threshold": threshold}


def test_scores_near_and_far_and_respects_missing_classes(tmp_path):
    means = np.array([[0.0, 0.0], [10.0, 0.0], [99.0, 99.0]])
    spec = write_stats(tmp_path, means, np.array([True, True, False]), np.eye(2), threshold=9.0)
    check = ScopeCheck.load(tmp_path, spec)
    assert check.score(np.array([10.5, 0.0])) == 0.25  # nearest present class
    assert check.in_scope(check.score(np.array([1.0, 1.0])))
    assert not check.in_scope(check.score(np.array([99.0, 99.0])))  # the absent class is ignored


def test_tampered_or_missing_statistics_disable_the_check(tmp_path):
    spec = write_stats(tmp_path, np.zeros((1, 2)), np.array([True]), np.eye(2), threshold=1.0)
    assert ScopeCheck.load(tmp_path, None) is None
    assert ScopeCheck.load(tmp_path, {**spec, "stats_sha256": "0" * 64}) is None
    assert ScopeCheck.load(tmp_path, {**spec, "stats_file": "absent.npz"}) is None


def test_api_reports_scope_only_for_models_with_statistics(tmp_path, monkeypatch):
    from app import main
    from app.registry import ModelRegistry
    from conftest import make_version
    from test_api import client, make_png

    make_version(tmp_path / "v1", seed=1)
    make_version(tmp_path / "v4", seed=4)
    features = 1280
    config_path = tmp_path / "v4" / "model_config.json"
    config = json.loads(config_path.read_text())
    # Threshold 0 with means far away: every image is out of scope.
    config["scope"] = write_stats(tmp_path / "v4", np.full((7, features), 50.0), np.ones(7, bool), np.eye(features), 0.0)
    config_path.write_text(json.dumps(config))
    monkeypatch.setattr(main, "registry", ModelRegistry(tmp_path, default="v4"))
    files = {"file": ("lesion.png", make_png(), "image/png")}

    checked = client.post("/predict?model=v4", files=files).json()
    assert checked["in_scope"] is False and checked["scope_score"] > 0
    plain = client.post("/predict?model=v1", files=files).json()
    assert plain["in_scope"] is None and plain["scope_score"] is None
