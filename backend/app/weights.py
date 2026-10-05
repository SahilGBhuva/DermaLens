"""Fetch trained artifacts at startup when they are hosted elsewhere.

Model weights are deliberately kept out of git. A cloud deployment sets
MODEL_URL (and optionally EVALUATION_URL) to where the files are published,
for example a GitHub Release asset; they are downloaded once on startup.
Without those variables nothing is fetched and the API stays in demo mode.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import tempfile
import urllib.request
from pathlib import Path

log = logging.getLogger("dermalens.weights")


def ensure_file(path: Path, url_env: str, sha256_env: str | None = None) -> None:
    url = os.getenv(url_env, "").strip()
    if path.exists() or not url:
        return
    # https only; local file:// sources only when explicitly enabled (tests,
    # local container checks).
    allowed = ("https://", "file://") if os.getenv("DERMALENS_ALLOW_FILE_URLS") == "1" else ("https://",)
    if not url.startswith(allowed):
        raise RuntimeError(f"{url_env} must be an https:// URL")

    path.parent.mkdir(parents=True, exist_ok=True)
    log.info("Downloading %s from %s", path.name, url)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
        with urllib.request.urlopen(url, timeout=120) as response:  # nosec B310: scheme checked above
            shutil.copyfileobj(response, tmp)
        tmp_path = Path(tmp.name)

    expected = os.getenv(sha256_env, "").strip().lower() if sha256_env else ""
    if expected:
        actual = hashlib.sha256(tmp_path.read_bytes()).hexdigest()
        if actual != expected:
            tmp_path.unlink(missing_ok=True)
            raise RuntimeError(f"{path.name} checksum mismatch: expected {expected}, got {actual}")

    tmp_path.replace(path)
