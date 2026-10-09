"""Split PAD-UFES-20 by patient and shrink its photos for training.

Every image of a patient lands in exactly one of train / val / test (60/20/20,
stratified by diagnosis, fixed seed), so no person is seen in training and
then "recognised" at test time. The photos are large PNGs; each is shrunk
once to at most 512 px on its longer side (the model sees 224 px), which
makes training and evaluation much faster. File names are kept.

Training labels use the same mapping as ml/evaluate_external.py, plus
invasive SCC -> akiec: the nearest DermaLens class (HAM10000's akiec includes
intraepithelial carcinoma), so SCC photos teach "this needs a closer look".
Evaluation still reports SCC separately.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold

from evaluate_external import PAD_TO_DERMALENS, load_metadata

TRAIN_LABELS = {**PAD_TO_DERMALENS, "SCC": "akiec"}
MAX_SIDE = 512


def shrink(job: tuple[str, str]) -> None:
    source, target = job
    with Image.open(source) as image:
        image = image.convert("RGB")
        image.thumbnail((MAX_SIDE, MAX_SIDE))
        image.save(target, optimize=False)


def split_by_patient(frame: pd.DataFrame, seed: int = 42) -> pd.Series:
    """'train', 'val' or 'test' per row; one patient never spans two splits."""
    folds = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    assignment = pd.Series("train", index=frame.index)
    for fold, (_, held) in enumerate(folds.split(frame, frame["diagnostic"], frame["patient_id"])):
        if fold == 0:
            assignment.iloc[held] = "test"
        elif fold == 1:
            assignment.iloc[held] = "val"
    return assignment


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--images-dir", required=True, type=Path)
    parser.add_argument("--output-dir", default="data/pad_splits", type=Path)
    args = parser.parse_args()

    frame = load_metadata(args.metadata, args.images_dir)
    if "patient_id" not in frame.columns:
        raise ValueError("metadata needs a patient_id column to split by patient")

    out_images = args.output_dir / "images"
    out_images.mkdir(parents=True, exist_ok=True)
    jobs = [(str(args.images_dir / name), str(out_images / name)) for name in frame["img_id"]]
    with ProcessPoolExecutor() as pool:
        list(pool.map(shrink, jobs, chunksize=16))

    frame["split"] = split_by_patient(frame)
    frame["label"] = frame["diagnostic"].map(TRAIN_LABELS)
    frame["image_path"] = [str(out_images / name) for name in frame["img_id"]]

    # Belt and braces: the grouping guarantees this, but check it anyway.
    patients = {name: set(frame.loc[frame["split"] == name, "patient_id"]) for name in ("train", "val", "test")}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        if not patients[a].isdisjoint(patients[b]):
            raise ValueError(f"{a}/{b} share patients")

    columns = ["image_path", "label", "img_id", "diagnostic", "patient_id"]
    if "fitspatrick" in frame.columns:
        columns.append("fitspatrick")
    for name in ("train", "val", "test"):
        frame.loc[frame["split"] == name, columns].to_csv(args.output_dir / f"{name}.csv", index=False)

    summary = {
        name: {
            "images": int((frame["split"] == name).sum()),
            "patients": len(patients[name]),
            "diagnostic_counts": {k: int(v) for k, v in frame.loc[frame["split"] == name, "diagnostic"].value_counts().items()},
        }
        for name in ("train", "val", "test")
    }
    (args.output_dir / "split_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
