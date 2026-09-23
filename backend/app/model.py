from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import torch
from PIL import Image, ImageEnhance, ImageFilter
from torchvision import models, transforms

from .gradcam import GradCAM, overlay_heatmap

CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "dermalens_efficientnet_b0.pt"


@dataclass
class Prediction:
    probabilities: Dict[str, float]
    top_class: str
    confidence: float
    uncertainty: float
    entropy: float
    demo_mode: bool
    heatmap_data_url: Optional[str]


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

    def _predict_summary(self, image: Image.Image):
        x = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            logits = self.model(x)
            p = torch.softmax(logits, dim=1)[0]

        top_idx = int(torch.argmax(p).item())
        confidence = float(p[top_idx].item())
        uncertainty = 1.0 - confidence
        entropy = float(
            (-(p * torch.log(p + 1e-12)).sum() / math.log(len(CLASSES))).item()
        )

        return {
            "x": x,
            "probabilities": {CLASSES[i]: float(p[i].item()) for i in range(len(CLASSES))},
            "top_class": CLASSES[top_idx],
            "top_idx": top_idx,
            "confidence": confidence,
            "uncertainty": uncertainty,
            "entropy": entropy,
        }

    def predict(self, image: Image.Image) -> Prediction:
        if self.demo_mode:
            probs = {label: 1.0 / len(CLASSES) for label in CLASSES}
            return Prediction(
                probabilities=probs,
                top_class="demo",
                confidence=0.0,
                uncertainty=1.0,
                entropy=1.0,
                demo_mode=True,
                heatmap_data_url=None,
            )

        original = image.convert("RGB")
        summary = self._predict_summary(original)

        target_layer = self.model.features[-1]
        cam = GradCAM(self.model, target_layer)
        try:
            heatmap = cam.generate(
                summary["x"].clone().detach().requires_grad_(True),
                summary["top_idx"],
            )
            heatmap_data_url = overlay_heatmap(original, heatmap)
        finally:
            cam.close()
            self.model.zero_grad(set_to_none=True)

        return Prediction(
            probabilities=summary["probabilities"],
            top_class=summary["top_class"],
            confidence=summary["confidence"],
            uncertainty=summary["uncertainty"],
            entropy=summary["entropy"],
            demo_mode=False,
            heatmap_data_url=heatmap_data_url,
        )

    def stress_test(self, image: Image.Image):
        if self.demo_mode:
            return {
                "demo_mode": True,
                "stability": None,
                "results": [],
            }

        base = image.convert("RGB")
        variants = {
            "Original": base,
            "Darker": ImageEnhance.Brightness(base).enhance(0.65),
            "Brighter": ImageEnhance.Brightness(base).enhance(1.35),
            "Lower contrast": ImageEnhance.Contrast(base).enhance(0.65),
            "Blur": base.filter(ImageFilter.GaussianBlur(radius=1.5)),
        }

        results = []
        for name, variant in variants.items():
            summary = self._predict_summary(variant)
            results.append(
                {
                    "variant": name,
                    "top_class": summary["top_class"],
                    "confidence": summary["confidence"],
                    "entropy": summary["entropy"],
                }
            )

        original_class = results[0]["top_class"]
        same_count = sum(row["top_class"] == original_class for row in results)

        return {
            "demo_mode": False,
            "stability": same_count / len(results),
            "original_class": original_class,
            "results": results,
        }
