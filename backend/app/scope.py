"""Out-of-scope check at serving time.

Statistics are fitted by ml/fit_scope.py (class means and a shared precision
matrix of the model's pooled features). An image whose smallest Mahalanobis
distance to any class mean is above the validation-set threshold does not look
like the skin photos the model learned from.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import numpy as np

log = logging.getLogger("dermalens.scope")


class ScopeCheck:
    def __init__(self, means, present, precision, threshold: float, thresholds: dict | None = None):
        self.means = means[present].astype(np.float32)
        self.precision = precision.astype(np.float32)
        self.threshold = float(threshold)  # fallback: the loosest per-domain threshold
        self.thresholds = {k: float(v) for k, v in (thresholds or {}).items()}

    @classmethod
    def load(cls, folder: Path, spec: dict | None) -> "ScopeCheck | None":
        """The fitted check, or None when absent or failing its integrity check."""
        if not spec:
            return None
        path = folder / spec["stats_file"]
        if not path.exists():
            log.warning("Scope statistics %s not found; out-of-scope check disabled", path.name)
            return None
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != spec.get("stats_sha256"):
            log.error("Refusing %s: sha256 %s does not match model_config.json", path.name, actual)
            return None
        with np.load(path, allow_pickle=False) as stats:
            return cls(stats["means"], stats["present"].astype(bool), stats["precision"], spec["threshold"],
                       spec.get("thresholds"))

    def score(self, features: np.ndarray) -> float:
        diff = self.means - features.astype(np.float32)[None, :]
        return float(np.einsum("ij,jk,ik->i", diff, self.precision, diff).min())

    def in_scope(self, score: float, domain: str | None = None) -> bool:
        """Compare with the threshold fitted for this domain ("dermoscopy" or "phone")."""
        return score <= self.thresholds.get(domain or "", self.threshold)
