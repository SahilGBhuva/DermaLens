"""Split BCN20000 by lesion into train / val / test for v5.

v1-v4.2 never trained on BCN20000, which made it an outside dermoscopy test.
v5 pre-registers the opposite: train on part of it, so the model sees a second
hospital's cameras and patients. The cost is stated up front: once part of
BCN20000 is training data, no BCN20000 number is "outside" any more. Only the
held-out test lesions are scored, and they come from the same hospital as the
training lesions.

Rules, in the order they matter:
1. One lesion never spans two splits (images of the same lesion look alike).
2. If lesion ids are missing, each image is its own group, and the summary says so.
3. Invasive SCC has no DermaLens class, so it is kept in the test CSV (scored
   as flagged / not flagged, as before) but never used for training or
   validation labels.
4. Fixed seed, stratified by diagnosis, so the split is reproducible.

    python ml/split_bcn20000.py --metadata data/bcn/metadata.csv --images-dir data/bcn/images --output-dir data/bcn_splits
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

TRAIN_LABELS = {"AK": "akiec", "BCC": "bcc", "BKL": "bkl", "DF": "df", "MEL": "mel", "NV": "nv", "VASC": "vasc"}


def split_by_lesion(frame: pd.DataFrame, seed: int = 42) -> pd.Series:
    """'train', 'val' or 'test' per row; one lesion never spans two splits."""
    groups = frame["lesion_id"].where(frame["lesion_id"].notna(), "image:" + frame["img_id"].astype(str))
    folds = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    assignment = pd.Series("train", index=frame.index)
    for fold, (_, held) in enumerate(folds.split(frame, frame["diagnostic"], groups)):
        if fold == 0:
            assignment.iloc[held] = "test"
        elif fold == 1:
            assignment.iloc[held] = "val"
    return assignment


def build_splits(frame: pd.DataFrame, images_dir: Path, seed: int = 42) -> dict[str, pd.DataFrame]:
    frame = frame.copy().reset_index(drop=True)
    frame["split"] = split_by_lesion(frame, seed)
    frame["label"] = frame["diagnostic"].map(TRAIN_LABELS)
    frame["image_path"] = [str(images_dir / name) for name in frame["img_id"]]

    groups = frame["lesion_id"].where(frame["lesion_id"].notna(), "image:" + frame["img_id"].astype(str))
    lesions = {name: set(groups[frame["split"] == name]) for name in ("train", "val", "test")}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        if not lesions[a].isdisjoint(lesions[b]):
            raise ValueError(f"{a}/{b} share lesions")

    columns = ["image_path", "label", "img_id", "diagnostic", "lesion_id"]
    out = {}
    for name in ("train", "val", "test"):
        part = frame[frame["split"] == name]
        if name != "test":  # SCC has no class to learn or to choose an epoch with
            part = part[part["label"].notna()]
        out[name] = part[columns]
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--metadata", required=True, type=Path, help="metadata.csv from prepare_bcn20000.py")
    parser.add_argument("--images-dir", required=True, type=Path)
    parser.add_argument("--output-dir", default="data/bcn_splits", type=Path)
    args = parser.parse_args()

    frame = pd.read_csv(args.metadata)
    if "lesion_id" not in frame.columns:
        frame["lesion_id"] = None
    splits = build_splits(frame, args.images_dir)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {"lesion_ids_missing": int(frame["lesion_id"].isna().sum())}
    for name, part in splits.items():
        part.to_csv(args.output_dir / f"{name}.csv", index=False)
        summary[name] = {
            "images": int(len(part)),
            "lesions": int(part["lesion_id"].nunique()),
            "diagnostic_counts": {k: int(v) for k, v in part["diagnostic"].value_counts().items()},
        }
    (args.output_dir / "split_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
