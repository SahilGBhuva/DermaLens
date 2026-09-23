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
