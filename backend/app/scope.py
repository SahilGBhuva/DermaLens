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


def pixel_reason(image) -> str | None:
    """Obvious non-photos the feature check finds oddly hard: plain colour or pure noise.

    Measured on a 128 px greyscale copy. Real photos vary (std well above 2)
    and neighbouring pixels are strongly alike (correlation usually > 0.9);
    plain colour has almost no variation, random noise has ~0 correlation.
    Keep in sync with ml/evaluate_scope.py, which reports how often genuine
    test images trip these rules.
    """
    grey = np.asarray(image.convert("L").resize((128, 128)), dtype=np.float32)
    if grey.std() < 2.0:
        return "plain colour"
    left, right = grey[:, :-1].ravel(), grey[:, 1:].ravel()
    if np.corrcoef(left, right)[0, 1] < 0.3:
        return "noise"
    return None
