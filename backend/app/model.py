from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import torch
from PIL import Image
from torchvision import models, transforms

CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "dermalens_efficientnet_b0.pt"


@dataclass
class Prediction:
    probabilities: Dict[str, float]
    top_class: str
    confidence: float
    uncertainty: float
    demo_mode: bool


class DermaLensModel:
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        self.model = self._build_model()
        self.demo_mode = True

        if MODEL_PATH.exists():
            state = torch.load(MODEL_PATH, map_location=self.device)
            self.model.load_state_dict(state)
            self.model.eval()
            self.demo_mode = False

    def _build_model(self):
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, len(CLASSES))
        return model.to(self.device)

    @torch.inference_mode()
    def predict(self, image: Image.Image) -> Prediction:
        if self.demo_mode:
            # Deliberately non-diagnostic placeholder values.
            probs = {label: 1.0 / len(CLASSES) for label in CLASSES}
            return Prediction(
                probabilities=probs,
                top_class="demo",
                confidence=0.0,
                uncertainty=1.0,
                demo_mode=True,
            )

        x = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        logits = self.model(x)
        p = torch.softmax(logits, dim=1)[0]

        values, indices = torch.sort(p, descending=True)
        top_idx = int(indices[0].item())
        confidence = float(values[0].item())

        # Simple uncertainty proxy: 1 - max probability.
        uncertainty = 1.0 - confidence

        return Prediction(
            probabilities={CLASSES[i]: float(p[i].item()) for i in range(len(CLASSES))},
            top_class=CLASSES[top_idx],
            confidence=confidence,
            uncertainty=uncertainty,
            demo_mode=False,
        )
