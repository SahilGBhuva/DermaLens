import hashlib
import json

import pytest
import torch
from torchvision import models

CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]


def make_version(folder, seed=0, balanced=0.7, evaluation_for_other_weights=False, history=True):
    """A model folder with random weights and matching settings/evaluation."""
    folder.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    net = models.efficientnet_b0(weights=None)
    net.classifier[1] = torch.nn.Linear(net.classifier[1].in_features, len(CLASSES))
    weights = folder / "dermalens_efficientnet_b0.pt"
    torch.save(net.state_dict(), weights)
    digest = hashlib.sha256(weights.read_bytes()).hexdigest()
    (folder / "model_config.json").write_text(json.dumps({"version": folder.name, "weights_sha256": digest}))
    (folder / "evaluation.json").write_text(json.dumps({
        "accuracy": 0.8, "balanced_accuracy": balanced, "macro_f1": 0.6, "weighted_f1": 0.8,
        "macro_ovr_roc_auc": None, "expected_calibration_error": 0.05, "multiclass_brier_score": 0.3,
        "confusion_matrix": [[3, 1], [0, 4]],
        "per_class_sensitivity_specificity": {"mel": {"sensitivity": 0.75, "specificity": 1.0}},
        "classification_report": {"mel": {"support": 4.0}},
        "settings": {"weights_sha256": "f" * 64 if evaluation_for_other_weights else digest},
    }))
    if history:
        (folder / "training_history.json").write_text(json.dumps([
            {"epoch": 1, "train_accuracy": 0.6, "val_accuracy": 0.65, "val_loss": 1.0},
            {"epoch": 2, "train_accuracy": 0.8, "val_accuracy": 0.75, "val_loss": 0.8},
        ]))
    return digest


@pytest.fixture
def two_versions(tmp_path):
    root = tmp_path / "models"
    make_version(root / "v1", seed=1, balanced=0.74)
    make_version(root / "v2", seed=2, balanced=0.73)
    return root
