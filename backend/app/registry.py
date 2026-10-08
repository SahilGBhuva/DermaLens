"""Serve several trained model versions side by side.

Layout on disk (one folder per version, each with the same four files):

    models/v1/dermalens_efficientnet_b0.pt
    models/v1/model_config.json
    models/v1/evaluation.json
    models/v1/training_history.json
    models/v2/...

Files placed directly in models/ (the original single-model layout) are still
served, under the version named in their model_config.json.

In the cloud, MODEL_RELEASES lists where to download each version, e.g.

    MODEL_RELEASES="v2=https://github.com/<owner>/<repo>/releases/download/model-v2,
                    v1=https://github.com/<owner>/<repo>/releases/download/model-v1"

The first entry is the default unless DEFAULT_MODEL says otherwise.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path

from .model import DermaLensModel, sha256
from .weights import download

log = logging.getLogger("dermalens.registry")

WEIGHTS = "dermalens_efficientnet_b0.pt"
CONFIG = "model_config.json"
EVALUATION = "evaluation.json"
HISTORY = "training_history.json"
VERSION_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,31}$")

# SHA-256 of each published release file, pinned in reviewed code. The weights
# hash inside model_config.json comes from the same release, so it only catches
# corruption; these pins also catch a release asset being swapped. Add a new
# version's hashes here when it is published.
PINNED_SHA256: dict[str, dict[str, str]] = {
    "v1": {
        WEIGHTS: "548cce02f6fd1fe98b5be28030dd0b560f208f29890535c0bb0701c6df377866",  # pragma: allowlist secret (public file hash)
        CONFIG: "1113df87271634a89508046ef0e38d82b66fad9ccf9ffccdd252c095af64579b",  # pragma: allowlist secret (public file hash)
        EVALUATION: "bbe393bee66f5a715115c90d1e94b5785f053e67c66a449493e9d6b1db370b06",  # pragma: allowlist secret (public file hash)
    },
    "v2": {
        WEIGHTS: "a9033a0c6d56c5af967ed828d4e6525eb558fce069342f7d3460a959290f720e",  # pragma: allowlist secret (public file hash)
        CONFIG: "144798dd5b4aa0deeb97f8d06a1e9d993f4fdb54c106552f73084cc4bf4aef6e",  # pragma: allowlist secret (public file hash)
        EVALUATION: "b2fd11540791d9a1e7eefb72632f789b8fdc12698ff3ea5f031b696a42c30fe7",  # pragma: allowlist secret (public file hash)
    },
}


@dataclass
class ModelEntry:
    version: str
    folder: Path
    model: DermaLensModel

    @property
    def evaluation_path(self) -> Path:
        return self.folder / EVALUATION

    @property
    def history_path(self) -> Path:
        return self.folder / HISTORY

    def evaluation(self) -> dict | None:
        """Held-out metrics, only if measured on exactly these weights."""
        if self.model.demo_mode or not self.evaluation_path.exists():
            return None
        try:
            raw = json.loads(self.evaluation_path.read_text())
        except (OSError, ValueError):
            return None
        expected_file = self.model.config.get("evaluation_sha256")
        if expected_file and sha256(self.evaluation_path) != expected_file:
            return None
        measured_on = (raw.get("settings") or {}).get("weights_sha256")
        if measured_on and measured_on != getattr(self.model, "weights_sha256", None):
            return None
        return raw

    def training(self) -> list | None:
        if self.model.demo_mode or not self.history_path.exists():
            return None
        try:
            rows = json.loads(self.history_path.read_text())
        except (OSError, ValueError):
            return None
        keys = ("epoch", "train_accuracy", "val_accuracy", "val_balanced_accuracy")
        return [{k: row.get(k) for k in keys} for row in rows if isinstance(row, dict)] or None


def parse_releases(value: str) -> list[tuple[str, str]]:
    releases = []
    for part in value.split(","):
        if "=" not in part:
            continue
        version, url = (p.strip() for p in part.split("=", 1))
        if VERSION_NAME.match(version) and url:
            releases.append((version, url.rstrip("/")))
    return releases


def fetch_releases(models_dir: Path, releases: list[tuple[str, str]]) -> None:
    """Download each version's files from its release URL if missing."""
    for version, base in releases:
        folder = models_dir / version
        pins = PINNED_SHA256.get(version, {})
        try:
            for name in (WEIGHTS, CONFIG, EVALUATION):
                download(f"{base}/{name}", folder / name, pins.get(name, ""))
        except RuntimeError as exc:
            # Fail closed: drop the whole version rather than serve a mix.
            log.error("Not serving %s: %s", version, exc)
            for name in (WEIGHTS, CONFIG, EVALUATION):
                (folder / name).unlink(missing_ok=True)
            continue
        try:
            download(f"{base}/{HISTORY}", folder / HISTORY)  # optional
        except Exception as exc:  # noqa: BLE001 - history only feeds a chart
            log.warning("No training history for %s: %s", version, exc)


class ModelRegistry:
    def __init__(self, models_dir: Path, default: str | None = None, order: list[str] | None = None):
        self.entries: dict[str, ModelEntry] = {}

        if (models_dir / WEIGHTS).exists():  # original single-model layout
            entry = self._load(models_dir, None)
            self.entries[entry.version] = entry
        if models_dir.exists():
            for folder in sorted((p for p in models_dir.iterdir() if p.is_dir()), reverse=True):
                if VERSION_NAME.match(folder.name) and (folder / WEIGHTS).exists() and folder.name not in self.entries:
                    self.entries[folder.name] = self._load(folder, folder.name)

        if not self.entries:  # nothing trained yet: one clearly labelled demo model
            self.entries["demo"] = ModelEntry("demo", models_dir, DermaLensModel(models_dir / WEIGHTS, models_dir / CONFIG))

        ranked = [v for v in (order or []) if v in self.entries] + [v for v in self.entries if v not in (order or [])]
        self.default = default if default in self.entries else ranked[0]
        self.order = [self.default] + [v for v in ranked if v != self.default]

    @staticmethod
    def _load(folder: Path, name: str | None) -> ModelEntry:
        model = DermaLensModel(folder / WEIGHTS, folder / CONFIG)
        version = name or model.config.get("version") or "default"
        return ModelEntry(str(version), folder, model)

    def get(self, version: str | None) -> ModelEntry | None:
        return self.entries.get(version or self.default)

    def __iter__(self):
        return (self.entries[v] for v in self.order)


def build_registry(models_dir: Path) -> ModelRegistry:
    releases = parse_releases(os.getenv("MODEL_RELEASES", ""))
    if releases:
        fetch_releases(models_dir, releases)
    return ModelRegistry(models_dir, os.getenv("DEFAULT_MODEL") or None, [v for v, _ in releases])
