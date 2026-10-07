import hashlib
import json
from pathlib import Path

import torch
from torchvision import models, transforms

CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
IMAGE_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def build_model(pretrained: bool = True):
    weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = models.efficientnet_b0(weights=weights)
    model.classifier[1] = torch.nn.Linear(
        model.classifier[1].in_features,
        len(CLASSES),
    )
    return model


def pick_device() -> torch.device:
    """Prefer CUDA, then Apple Silicon (MPS), then CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def eval_transform(image_size: int = IMAGE_SIZE):
    """The exact preprocessing used for validation, test and the live API."""
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )


def predict_proba_tta(model, x: torch.Tensor) -> torch.Tensor:
    """Average softmax over the image and its horizontal/vertical flips.

    Dermoscopy has no canonical orientation, so the flipped views are equally
    valid inputs; averaging them gives steadier probabilities. The API uses the
    same four views, so held-out scores match what the live site serves.
    """
    views = [x, x.flip(-1), x.flip(-2), x.flip(-1).flip(-2)]
    return torch.stack([torch.softmax(model(v), dim=1) for v in views]).mean(0)


def predict_proba(model, x: torch.Tensor, tta: bool) -> torch.Tensor:
    return predict_proba_tta(model, x) if tta else torch.softmax(model(x), dim=1)


def apply_calibration(probs: torch.Tensor, temperature: float, logit_bias) -> torch.Tensor:
    """Temperature scaling plus a per-class log-odds offset.

    With temperature 1 and zero bias this returns `probs` unchanged.
    """
    bias = torch.as_tensor(logit_bias, dtype=probs.dtype, device=probs.device)
    return torch.softmax(torch.log(probs.clamp_min(1e-12)) / temperature + bias, dim=1)


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def default_config() -> dict:
    """Settings that reproduce a plain, uncalibrated, single-view model."""
    return {
        "arch": "efficientnet_b0",
        "classes": CLASSES,
        "image_size": IMAGE_SIZE,
        "mean": MEAN,
        "std": STD,
        "tta": False,
        "temperature": 1.0,
        "logit_bias": [0.0] * len(CLASSES),
    }


def load_config(path) -> dict:
    config = default_config()
    if path and Path(path).exists():
        config.update(json.loads(Path(path).read_text()))
    return config
