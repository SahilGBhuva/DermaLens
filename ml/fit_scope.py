"""Out-of-scope check: does this image look like the skin photos the model learned from?

Every DermaLens version always answers with one of seven classes, even for a
photo of a dog. This fits a simple, well-studied detector on the model's own
image features (the 1,280 numbers before its final layer):

1. On TRAINING images, fit one mean per class and a shared covariance
   (Ledoit-Wolf shrinkage, because 1,280 dimensions is a lot).
2. A new image's score is its smallest Mahalanobis distance to any class
   mean (Lee et al., 2018). Far from every class = probably not a skin lesion.
3. Thresholds are set on VALIDATION images: one per domain (named by the
   validation file's folder, e.g. "dermoscopy" and "phone"), each accepting
   `--accept-rate` of that domain's genuine images. The API uses the one for
   the photo type the visitor chose; "threshold" (the loosest) is the
   fallback.

ml/evaluate_scope.py then measures how many genuine test images are accepted
and how many non-skin images are refused. The test splits are never read here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.covariance import LedoitWolf
from torch.utils.data import DataLoader

from dataset import SkinLesionDataset
from model import CLASSES, build_model, eval_transform, load_config, pick_device


def extract_features(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    """Pooled 1,280-dimensional features (what the classifier layer sees)."""
    model.eval()
    feats, labels = [], []
    with torch.inference_mode():
        for images, y in loader:
            x = model.avgpool(model.features(images.to(device)))
            feats.append(torch.flatten(x, 1).cpu().numpy())
            labels.append(np.asarray(y))
    return np.concatenate(feats).astype(np.float64), np.concatenate(labels)


def fit_gaussian(features: np.ndarray, labels: np.ndarray, n_classes: int) -> tuple[np.ndarray, np.ndarray]:
    """Class means and the precision of a shared (tied) covariance."""
    means = np.zeros((n_classes, features.shape[1]))
    centred = np.empty_like(features)
    for k in range(n_classes):
        rows = labels == k
        if rows.any():
            means[k] = features[rows].mean(axis=0)
            centred[rows] = features[rows] - means[k]
    present = np.isin(np.arange(n_classes), np.unique(labels))
    means[~present] = np.inf  # a class never seen cannot be the nearest
    precision = LedoitWolf().fit(centred).precision_
    return means, precision


def scope_scores(features: np.ndarray, means: np.ndarray, precision: np.ndarray) -> np.ndarray:
    """Smallest squared Mahalanobis distance from each row to any class mean."""
    best = np.full(len(features), np.inf)
    for mean in means:
        if not np.all(np.isfinite(mean)):
            continue
        diff = features - mean
        best = np.minimum(best, np.einsum("ij,jk,ik->i", diff, precision, diff))
    return best


def loader_for(csv: str | Path, config: dict, batch_size: int, frame: pd.DataFrame | None = None) -> DataLoader:
    class_to_idx = {c: i for i, c in enumerate(CLASSES)}
    if frame is not None:
        csv = Path(csv)
        frame.to_csv(csv, index=False)
    ds = SkinLesionDataset(csv, class_to_idx, eval_transform(config["image_size"]))
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2)


def domain_name(path) -> str:
    """'folder/file' so two files both called val.csv stay distinct."""
    path = Path(path)
    return f"{path.parent.name}/{path.stem}"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--weights", required=True)
    parser.add_argument("--config", required=True, help="model_config.json to update with the 'scope' block")
    parser.add_argument("--train-csv", required=True, nargs="+")
    parser.add_argument("--val-csv", required=True, nargs="+", help="one CSV per domain")
    parser.add_argument("--max-train", type=int, default=6000, help="class-stratified sample of training images")
    parser.add_argument("--accept-rate", type=float, default=0.95)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--work-dir", default=None, help="scratch folder (default: a new private temp dir)")
    args = parser.parse_args()

    config = load_config(args.config)
    device = pick_device()
    model = build_model(pretrained=False).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device, weights_only=True))
    work = Path(args.work_dir) if args.work_dir else Path(tempfile.mkdtemp(prefix="dermalens-scope-"))
    work.mkdir(parents=True, exist_ok=True)

    train = pd.concat([pd.read_csv(p)[["image_path", "label"]] for p in args.train_csv], ignore_index=True)
    fraction = min(1.0, args.max_train / len(train))
    train = train.groupby("label").sample(frac=fraction, random_state=42)
    features, labels = extract_features(model, loader_for(work / "train.csv", config, args.batch_size, train), device)
    means, precision = fit_gaussian(features, labels, len(CLASSES))

    accepted, thresholds = {}, []
    val_scores = {}
    for path in args.val_csv:
        f, _ = extract_features(model, loader_for(path, config, args.batch_size), device)
        val_scores[domain_name(path)] = scope_scores(f, means, precision)
        thresholds.append(float(np.quantile(val_scores[domain_name(path)], args.accept_rate)))
    threshold = max(thresholds)  # fallback: every domain keeps at least the target acceptance
    per_domain = {Path(p).parent.name: round(t, 4) for p, t in zip(args.val_csv, thresholds)}
    for name, scores in val_scores.items():
        accepted[name] = float((scores <= threshold).mean())

    stats_path = Path(args.config).parent / "scope_stats.npz"
    finite = np.where(np.isfinite(means), means, 0.0)
    np.savez_compressed(stats_path, means=finite.astype(np.float32), present=np.isfinite(means).all(axis=1),
                        precision=precision.astype(np.float32))
    config["scope"] = {
        "method": "min class-conditional Mahalanobis distance on pooled features (tied Ledoit-Wolf covariance)",
        "stats_file": stats_path.name,
        "stats_sha256": hashlib.sha256(stats_path.read_bytes()).hexdigest(),
        "threshold": round(threshold, 4),
        "thresholds": per_domain,
        "accept_rate_target": args.accept_rate,
        "training_images": int(len(train)),
        "validation_accept": accepted,
    }
    Path(args.config).write_text(json.dumps(config, indent=2))
    print(json.dumps(config["scope"], indent=2))


if __name__ == "__main__":
    main()
