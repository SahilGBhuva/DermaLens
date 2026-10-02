"""Fit calibration and decision offsets on the VALIDATION split only.

1. Temperature scaling: one scalar T that makes confidence match accuracy
   (minimises negative log-likelihood on validation).
2. Per-class log-odds offsets: shift each class's score so the decision rule
   maximises validation balanced accuracy, subject to a minimum melanoma
   sensitivity. This trades some specificity for catching more melanomas.

The test split is never read here. ml/evaluate.py applies the saved settings
to the test split exactly once.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, recall_score
from torch.utils.data import DataLoader

from dataset import SkinLesionDataset
from model import CLASSES, apply_calibration, build_model, eval_transform, load_config, pick_device, predict_proba, sha256

MEL = CLASSES.index("mel")


def collect_probs(model, loader, device, tta: bool):
    model.eval()
    ys, probs = [], []
    with torch.inference_mode():
        for images, labels in loader:
            probs.append(predict_proba(model, images.to(device), tta).cpu())
            ys.append(labels)
    return torch.cat(ys).numpy(), torch.cat(probs)


def nll(probs: torch.Tensor, y: np.ndarray) -> float:
    return float(-torch.log(probs[torch.arange(len(y)), torch.as_tensor(y)].clamp_min(1e-12)).mean())


def fit_temperature(probs: torch.Tensor, y: np.ndarray) -> float:
    zeros = [0.0] * probs.shape[1]
    grid = np.round(np.arange(0.25, 3.01, 0.05), 2)
    return float(min(grid, key=lambda t: nll(apply_calibration(probs, float(t), zeros), y)))


def decision_score(probs, y, bias, temperature, min_mel_sensitivity):
    pred = apply_calibration(probs, temperature, bias).argmax(1).numpy()
    balanced = balanced_accuracy_score(y, pred)
    mel_sens = recall_score(y == MEL, pred == MEL, zero_division=0)
    shortfall = max(0.0, min_mel_sensitivity - mel_sens)
    return balanced - 2.0 * shortfall, balanced, mel_sens


def fit_bias(probs, y, temperature, min_mel_sensitivity=0.8, passes=3):
    """Coordinate ascent over per-class offsets in [-3, 3] (applied after temperature)."""
    bias = np.zeros(probs.shape[1])
    grid = np.round(np.arange(-3.0, 3.01, 0.1), 2)
    best = decision_score(probs, y, bias, temperature, min_mel_sensitivity)[0]
    for _ in range(passes):
        improved = False
        for k in range(len(bias)):
            for value in grid:
                trial = bias.copy()
                trial[k] = value
                score = decision_score(probs, y, trial, temperature, min_mel_sensitivity)[0]
                if score > best + 1e-9:
                    best, bias, improved = score, trial, True
        if not improved:
            break
    return bias


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="validation split")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--config", required=True, help="model_config.json to write")
    parser.add_argument("--tta", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-mel-sensitivity", type=float, default=0.8)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    config = load_config(args.config)
    config["tta"] = args.tta

    device = pick_device()
    model = build_model(pretrained=False).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    ds = SkinLesionDataset(args.csv, {c: i for i, c in enumerate(CLASSES)}, eval_transform(config["image_size"]))
    y, probs = collect_probs(model, DataLoader(ds, batch_size=args.batch_size), device, args.tta)

    temperature = fit_temperature(probs, y)
    bias = fit_bias(probs, y, temperature, args.min_mel_sensitivity)
    _, before_bal, before_mel = decision_score(probs, y, np.zeros(len(CLASSES)), 1.0, 0)
    _, after_bal, after_mel = decision_score(probs, y, bias, temperature, 0)

    config.update(
        {
            "temperature": temperature,
            "logit_bias": [round(float(b), 2) for b in bias],
            "min_mel_sensitivity_target": args.min_mel_sensitivity,
            "weights_sha256": sha256(args.weights),
            "validation": {
                "images": int(len(y)),
                "nll_before": nll(probs, y),
                "nll_after": nll(apply_calibration(probs, temperature, bias), y),
                "balanced_accuracy_before": before_bal,
                "balanced_accuracy_after": after_bal,
                "melanoma_sensitivity_before": before_mel,
                "melanoma_sensitivity_after": after_mel,
            },
        }
    )
    Path(args.config).parent.mkdir(parents=True, exist_ok=True)
    Path(args.config).write_text(json.dumps(config, indent=2))
    print(json.dumps(config["validation"], indent=2))
    print(f"temperature={temperature} logit_bias={config['logit_bias']}")


if __name__ == "__main__":
    main()
