import hashlib

from app import registry as registry_module
from app.registry import ModelRegistry, build_registry, parse_releases
from conftest import make_version


def test_parse_releases_keeps_order_and_rejects_bad_names():
    value = "v2=https://x/model-v2/, v1=https://x/model-v1, ../evil=https://x/e, nonsense"
    assert parse_releases(value) == [("v2", "https://x/model-v2"), ("v1", "https://x/model-v1")]


def test_default_can_be_overridden(two_versions):
    assert ModelRegistry(two_versions).default == "v2"
    assert ModelRegistry(two_versions, default="v1").order == ["v1", "v2"]
    assert ModelRegistry(two_versions, default="nope").default == "v2"


def test_legacy_single_model_layout_still_served(tmp_path):
    make_version(tmp_path / "v1")
    for f in (tmp_path / "v1").iterdir():
        f.rename(tmp_path / f.name)  # files directly in models/
    registry = ModelRegistry(tmp_path)
    assert list(registry.entries) == ["v1"]
    assert registry.get(None).model.demo_mode is False


def test_build_registry_downloads_each_release(tmp_path, monkeypatch):
    hosted = tmp_path / "hosted"
    make_version(hosted / "model-v1", seed=1)
    digest = make_version(hosted / "model-v2", seed=2, history=False)
    monkeypatch.setenv("DERMALENS_ALLOW_FILE_URLS", "1")
    monkeypatch.setenv(
        "MODEL_RELEASES",
        f"v2={(hosted / 'model-v2').as_uri()},v1={(hosted / 'model-v1').as_uri()}",
    )
    monkeypatch.delenv("DEFAULT_MODEL", raising=False)
    monkeypatch.setattr(registry_module, "PINNED_SHA256", {})  # test models aren't the published ones
    registry = build_registry(tmp_path / "models")
    assert registry.order == ["v2", "v1"]
    served = tmp_path / "models" / "v2" / "dermalens_efficientnet_b0.pt"
    assert hashlib.sha256(served.read_bytes()).hexdigest() == digest
    assert registry.get("v2").training() is None  # optional file missing is fine
    assert registry.get("v1").evaluation() is not None


def test_release_files_that_do_not_match_the_pins_are_refused(tmp_path, monkeypatch):
    hosted = tmp_path / "hosted"
    make_version(hosted / "model-v1", seed=1)
    good = make_version(hosted / "model-v2", seed=2)
    monkeypatch.setenv("DERMALENS_ALLOW_FILE_URLS", "1")
    monkeypatch.setenv(
        "MODEL_RELEASES",
        f"v2={(hosted / 'model-v2').as_uri()},v1={(hosted / 'model-v1').as_uri()}",
    )
    monkeypatch.delenv("DEFAULT_MODEL", raising=False)
    # v2 is pinned to its real hash; v1 is pinned to a hash its weights don't have.
    monkeypatch.setattr(
        registry_module,
        "PINNED_SHA256",
        {"v2": {"dermalens_efficientnet_b0.pt": good}, "v1": {"dermalens_efficientnet_b0.pt": "0" * 64}},
    )
    registry = build_registry(tmp_path / "models")
    assert list(registry.entries) == ["v2"]
    assert not (tmp_path / "models" / "v1" / "dermalens_efficientnet_b0.pt").exists()
