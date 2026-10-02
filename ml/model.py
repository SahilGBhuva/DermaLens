import torch
from torchvision import models

CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]


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


def predict_proba_tta(model, x: torch.Tensor) -> torch.Tensor:
    """Average softmax over the image and its horizontal/vertical flips.

    Dermoscopy has no canonical orientation, so the flipped views are equally
    valid inputs; averaging them gives steadier probabilities. The API uses the
    same four views, so held-out scores match what the live site serves.
    """
    views = [x, x.flip(-1), x.flip(-2), x.flip(-1).flip(-2)]
    return torch.stack([torch.softmax(model(v), dim=1) for v in views]).mean(0)
