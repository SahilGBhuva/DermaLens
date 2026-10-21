"""Test the out-of-scope check fitted by ml/fit_scope.py.

Reports, for the fitted threshold:
- how many genuine TEST images are accepted, per domain (should be ~95%),
- how many non-skin images are refused (everyday photos from a folder, plus
  random-noise and plain-colour images generated here),
- the ROC-AUC for separating genuine from non-skin images.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

from fit_scope import extract_features, loader_for, scope_scores
from model import build_model, eval_transform, load_config, pick_device

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def pixel_reason(image) -> str | None:
    """Same rule as backend/app/scope.py (plain colour or pure noise); keep in sync."""
    grey = np.asarray(image.convert("L").resize((128, 128)), dtype=np.float32)
    if grey.std() < 2.0:
        return "plain colour"
    left, right = grey[:, :-1].ravel(), grey[:, 1:].ravel()
    if np.corrcoef(left, right)[0, 1] < 0.3:
        return "noise"
    return None


class FolderImages(Dataset):
    def __init__(self, paths, transform):
        self.paths, self.transform = paths, transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        return self.transform(Image.open(self.paths[idx]).convert("RGB")), -1


class SyntheticImages(Dataset):
    """Random noise and plain colours: obviously not skin lesions."""

    def __init__(self, count, transform, seed=0):
        rng = np.random.default_rng(seed)
        self.images = []
        for i in range(count):
            if i % 2:
                pixels = rng.integers(0, 256, (224, 224, 3), dtype=np.uint8)
            else:
                pixels = np.full((224, 224, 3), rng.integers(0, 256, 3), dtype=np.uint8)
            self.images.append(Image.fromarray(pixels))
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.transform(self.images[idx]), -1


def domain_name(path) -> str:
    """'folder/file' so two files both called val.csv stay distinct."""
    path = Path(path)
    return f"{path.parent.name}/{path.stem}"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--weights", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--in-domain-csv", required=True, nargs="+", help="test CSVs, one per domain")
    parser.add_argument("--ood-dir", type=Path, default=None, help="folder of non-skin photos")
    parser.add_argument("--ood-name", default="everyday photos")
    parser.add_argument("--max-ood", type=int, default=4000)
    parser.add_argument("--synthetic", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    config = load_config(args.config)
    scope = config.get("scope")
    if not scope:
        raise SystemExit("model_config.json has no 'scope' block; run ml/fit_scope.py first")
    stats = np.load(Path(args.config).parent / scope["stats_file"])
    means = stats["means"].astype(np.float64)
    means[~stats["present"]] = np.inf
    precision = stats["precision"].astype(np.float64)
    threshold = scope["threshold"]

    device = pick_device()
    model = build_model(pretrained=False).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device, weights_only=True))
    transform = eval_transform(config["image_size"])

    def scores_of(loader):
        features, _ = extract_features(model, loader, device)
        return scope_scores(features, means, precision)

    genuine = {domain_name(p): scores_of(loader_for(p, config, args.batch_size)) for p in args.in_domain_csv}
    outside = {"noise and plain colours": scores_of(DataLoader(SyntheticImages(args.synthetic, transform), batch_size=args.batch_size))}
    if args.ood_dir:
        paths = sorted(p for p in args.ood_dir.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
        rng = np.random.default_rng(0)
        if len(paths) > args.max_ood:
            paths = [paths[i] for i in sorted(rng.choice(len(paths), args.max_ood, replace=False))]
        outside[args.ood_name] = scores_of(DataLoader(FolderImages(paths, transform), batch_size=args.batch_size, num_workers=2))

    all_genuine = np.concatenate(list(genuine.values()))

    def rates_at(t):
        return {
            "genuine_accepted": {k: float((v <= t).mean()) for k, v in genuine.items()},
            "non_skin_refused": {k: float((v > t).mean()) for k, v in outside.items()},
        }

    result = {
        "per_domain_thresholds": {name: {"threshold": t, **rates_at(t)} for name, t in scope.get("thresholds", {}).items()},
        "threshold": threshold,
        "genuine_accepted": {k: {"images": int(len(v)), "accepted": float((v <= threshold).mean())} for k, v in genuine.items()},
        "non_skin_refused": {k: {"images": int(len(v)), "refused": float((v > threshold).mean())} for k, v in outside.items()},
        "roc_auc_genuine_vs_non_skin": {
            k: float(roc_auc_score(np.r_[np.zeros(len(all_genuine)), np.ones(len(v))], np.r_[all_genuine, v]))
            for k, v in outside.items()
        },
    }
    # The pixel rules on their own: genuine test images should never trip them.
    tripped = {}
    for p in args.in_domain_csv:
        paths = pd.read_csv(p)["image_path"]
        tripped[domain_name(p)] = float(np.mean([pixel_reason(Image.open(x)) is not None for x in paths]))
    synthetic = SyntheticImages(args.synthetic, transform)
    result["pixel_rules"] = {
        "genuine_tripped": tripped,
        "noise_and_plain_caught": float(np.mean([pixel_reason(im) is not None for im in synthetic.images])),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    # Every score, so other thresholds can be studied without re-running.
    np.savez_compressed(args.output.with_suffix(".scores.npz"),
                        **{f"genuine:{k}": v for k, v in genuine.items()}, **{f"outside:{k}": v for k, v in outside.items()})
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
