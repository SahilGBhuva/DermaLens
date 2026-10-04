import hashlib

import pytest

from app.weights import ensure_file


def test_downloads_when_missing(tmp_path, monkeypatch):
    source = tmp_path / "hosted.pt"
    source.write_bytes(b"weights")
    target = tmp_path / "models" / "model.pt"
    monkeypatch.setenv("MODEL_URL", source.as_uri())
    monkeypatch.setenv("MODEL_SHA256", hashlib.sha256(b"weights").hexdigest())

    ensure_file(target, "MODEL_URL", "MODEL_SHA256")
    assert target.read_bytes() == b"weights"


def test_rejects_checksum_mismatch(tmp_path, monkeypatch):
    source = tmp_path / "hosted.pt"
    source.write_bytes(b"tampered")
    target = tmp_path / "model.pt"
    monkeypatch.setenv("MODEL_URL", source.as_uri())
    monkeypatch.setenv("MODEL_SHA256", "0" * 64)

    with pytest.raises(RuntimeError):
        ensure_file(target, "MODEL_URL", "MODEL_SHA256")
    assert not target.exists()


def test_no_url_means_no_download(tmp_path, monkeypatch):
    monkeypatch.delenv("MODEL_URL", raising=False)
    target = tmp_path / "model.pt"
    ensure_file(target, "MODEL_URL")
    assert not target.exists()


def test_rejects_insecure_url(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_URL", "http://example.com/model.pt")
    with pytest.raises(RuntimeError, match="https"):
        ensure_file(tmp_path / "model.pt", "MODEL_URL")
