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


def overlay_heatmap(image: Image.Image, heatmap: np.ndarray) -> str:
    image = image.convert("RGB").resize((224, 224))
    heat = Image.fromarray(np.uint8(np.clip(heatmap, 0, 1) * 255)).resize((224, 224))
    heat_np = np.asarray(heat, dtype=np.float32) / 255.0

    # Simple red/yellow heatmap without an extra plotting dependency.
    color = np.zeros((224, 224, 3), dtype=np.float32)
    color[..., 0] = np.clip(heat_np * 1.6, 0, 1)
    color[..., 1] = np.clip((heat_np - 0.25) * 1.3, 0, 1)

    base = np.asarray(image, dtype=np.float32) / 255.0
    alpha = (heat_np[..., None] * 0.55)
    blended = base * (1 - alpha) + color * alpha
    blended = np.uint8(np.clip(blended, 0, 1) * 255)

    output = Image.fromarray(blended)
    buffer = BytesIO()
    output.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
