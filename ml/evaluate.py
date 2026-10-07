"""Score the final model on the held-out TEST split.

Run this once, after training, model selection and calibration are finished
on the validation split. It applies the saved model_config.json exactly as the
API does, and records those settings in the output.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import SkinLesionDataset
from metrics import summarize_metrics
from model import CLASSES, apply_calibration, build_model, eval_transform, load_config, pick_device, predict_proba, sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--config", default=None, help="model_config.json from ml/calibrate.py")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", default="models/evaluation.json")
    args = parser.parse_args()

    config = load_config(args.config)
    weights_hash = sha256(args.weights)
    expected = config.get("weights_sha256")
    if expected and expected != weights_hash:
        raise SystemExit(f"weights do not match model_config.json (expected {expected}, got {weights_hash})")

    class_to_idx = {name: i for i, name in enumerate(CLASSES)}
    ds = SkinLesionDataset(args.csv, class_to_idx, eval_transform(config["image_size"]))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    device = pick_device()
    model = build_model(pretrained=False).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device, weights_only=True))
    model.eval()

    all_y, all_probs = [], []
    with torch.inference_mode():
        for images, labels in loader:
            probs = predict_proba(model, images.to(device), config["tta"])
            probs = apply_calibration(probs, config["temperature"], config["logit_bias"])
            all_y.extend(labels.numpy().tolist())
            all_probs.append(probs.cpu().numpy())

    metrics = summarize_metrics(all_y, np.concatenate(all_probs, axis=0), CLASSES)
    metrics["settings"] = {
        "weights_sha256": weights_hash,
        "tta": config["tta"],
        "temperature": config["temperature"],
        "logit_bias": config["logit_bias"],
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    # allow_nan=False fails loudly here rather than writing JSON the API can't serve.
    output.write_text(json.dumps(metrics, indent=2, allow_nan=False))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
