from __future__ import annotations

import hashlib
import json
import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import torch
from PIL import Image, ImageEnhance, ImageFilter
from torchvision import models, transforms

from .gradcam import GradCAM, overlay_heatmap

CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
MODEL_PATH = MODELS_DIR / "dermalens_efficientnet_b0.pt"
CONFIG_PATH = MODELS_DIR / "model_config.json"

log = logging.getLogger("dermalens.model")

# Fewer intra-op threads keeps per-request memory down on small instances.
torch.set_num_threads(max(1, int(os.getenv("TORCH_THREADS", "2"))))

# Mirrors ml/model.py default_config(): a plain, uncalibrated, single-view model.
DEFAULT_CONFIG = {
    "classes": CLASSES,
    "image_size": 224,
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
    "tta": False,
    "temperature": 1.0,
    "logit_bias": [0.0] * len(CLASSES),
}


def load_config(path: Path = CONFIG_PATH) -> dict:
    config = dict(DEFAULT_CONFIG)
    if path.exists():
        config.update(json.loads(path.read_text()))
    if config["classes"] != CLASSES:
        raise ValueError(f"model_config.json classes {config['classes']} do not match {CLASSES}")
    return config


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    def __init__(self, model_path: Path = MODEL_PATH, config_path: Path = CONFIG_PATH) -> None:
        self.device = self._pick_device()
        self.config = load_config(config_path)
        # Must match ml/model.py eval_transform(): what validation/test used.
        self.transform = transforms.Compose(
            [
                transforms.Resize((self.config["image_size"], self.config["image_size"])),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.config["mean"], std=self.config["std"]),
            ]
        )
        self.model = self._build_model()
        self.demo_mode = True
        self.status = "no trained weights found"

        if model_path.exists():
            expected = self.config.get("weights_sha256")
            actual = sha256(model_path)
            if expected and expected != actual:
                # Never serve predictions from weights the settings weren't made for.
                self.status = "weights failed the integrity check"
                log.error("Refusing %s: sha256 %s, expected %s", model_path.name, actual, expected)
                return
            state = torch.load(model_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state)
            self.model.eval()
            # Inference never updates weights. Grad-CAM only needs gradients of the
            # target layer's activations (they flow from the input), so skipping
            # per-weight gradients saves memory and time on small servers.
            for parameter in self.model.parameters():
                parameter.requires_grad_(False)
            self.demo_mode = False
            self.status = "loaded"
            self.weights_sha256 = actual

    @staticmethod
    def _pick_device() -> torch.device:
        forced = os.getenv("DERMALENS_DEVICE")  # e.g. "cpu" to mimic the cloud server
        if forced:
            return torch.device(forced)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _build_model(self):
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, len(CLASSES))
        return model.to(self.device)

    def _predict_summary(self, image: Image.Image):
        x = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            p = self._probabilities(x)[0]

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

    def _probabilities(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the same settings ml/evaluate.py used for the published scores."""
        if self.config["tta"]:
            views = [x, x.flip(-1), x.flip(-2), x.flip(-1).flip(-2)]
            p = torch.stack([torch.softmax(self.model(v), dim=1) for v in views]).mean(0)
        else:
            p = torch.softmax(self.model(x), dim=1)
        bias = torch.as_tensor(self.config["logit_bias"], dtype=p.dtype, device=p.device)
        return torch.softmax(torch.log(p.clamp_min(1e-12)) / self.config["temperature"] + bias, dim=1)

    def info(self) -> dict:
        return {
            "status": self.status,
            "version": self.config.get("version"),
            "recipe": self.config.get("recipe"),
            "tta": bool(self.config["tta"]),
            "calibrated": self.config["temperature"] != 1.0 or any(self.config["logit_bias"]),
            "min_mel_sensitivity_target": self.config.get("min_mel_sensitivity_target"),
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
