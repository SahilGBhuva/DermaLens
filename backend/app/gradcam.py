from __future__ import annotations

import base64
from io import BytesIO

import numpy as np
import torch
from PIL import Image


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        self._forward_handle = target_layer.register_forward_hook(self._save_activations)
        self._backward_handle = target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, inputs, output):
        self.activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, x: torch.Tensor, class_index: int):
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        score = logits[:, class_index].sum()
        score.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture model activations/gradients.")

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1)
        cam = torch.relu(cam)

        cam_min = cam.amin(dim=(1, 2), keepdim=True)
        cam_max = cam.amax(dim=(1, 2), keepdim=True)
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        return cam[0].detach().cpu().numpy()

    def close(self):
        self._forward_handle.remove()
        self._backward_handle.remove()


# Matches the attention legend on the DermaLens site: low → high contribution.
HEAT_STOPS = np.array([0.0, 0.3, 0.55, 0.8, 1.0], dtype=np.float32)
HEAT_COLORS = (
    np.array(
        [
            [0x24, 0x62, 0xE0],  # blue
            [0x37, 0xB6, 0xFF],  # cyan
            [0xFF, 0xE4, 0x5C],  # yellow
            [0xFF, 0x9F, 0x1A],  # orange
            [0xFF, 0x3B, 0x2F],  # red
        ],
        dtype=np.float32,
    )
    / 255.0
)
OVERLAY_SIZE = 448


def colorize(heat: np.ndarray) -> np.ndarray:
    """Map a [0, 1] heatmap to RGB with the DermaLens attention scale."""
    heat = np.clip(heat, 0.0, 1.0)
    return np.stack(
        [np.interp(heat, HEAT_STOPS, HEAT_COLORS[:, channel]) for channel in range(3)],
        axis=-1,
    ).astype(np.float32)


def overlay_heatmap(image: Image.Image, heatmap: np.ndarray, size: int = OVERLAY_SIZE) -> str:
    image = image.convert("RGB").resize((size, size), Image.Resampling.LANCZOS)
    heat = Image.fromarray(np.uint8(np.clip(heatmap, 0, 1) * 255)).resize(
        (size, size), Image.Resampling.BICUBIC
    )
    heat_np = np.asarray(heat, dtype=np.float32) / 255.0

    # Low-attention areas keep a light blue tint so the scale reads end to end;
    # high-attention areas are mostly heat color.
    alpha = (0.18 + 0.5 * heat_np)[..., None]
    base = np.asarray(image, dtype=np.float32) / 255.0
    blended = base * (1 - alpha) + colorize(heat_np) * alpha
    blended = np.uint8(np.clip(blended, 0, 1) * 255)

    output = Image.fromarray(blended)
    buffer = BytesIO()
    output.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
