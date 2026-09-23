import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True, help="HAM10000_metadata.csv")
    parser.add_argument("--images-dir", required=True, help="Directory containing HAM10000 .jpg images")
    parser.add_argument("--output-dir", default="data/splits")
    args = parser.parse_args()

    metadata = pd.read_csv(args.metadata)
    required = {"image_id", "dx", "lesion_id"}
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(f"Metadata missing required columns: {sorted(missing)}")

    if not set(metadata["dx"]).issubset(CLASSES):
        raise ValueError("Unexpected diagnostic labels found in metadata.")

    images_dir = Path(args.images_dir)
    metadata["image_path"] = metadata["image_id"].map(lambda x: str(images_dir / f"{x}.jpg"))
    metadata["label"] = metadata["dx"]

    missing_images = metadata[~metadata["image_path"].map(lambda p: Path(p).exists())]
    if not missing_images.empty:
        example = missing_images.iloc[0]["image_path"]
        raise FileNotFoundError(
            f"{len(missing_images)} image files referenced by metadata were not found. "
            f"Example: {example}"
        )

    # Split by lesion_id so multiple images of the same physical lesion never
    # leak between train/validation/test sets.
    first = GroupShuffleSplit(n_splits=1, train_size=0.70, random_state=42)
    train_idx, temp_idx = next(first.split(metadata, groups=metadata["lesion_id"]))
    train = metadata.iloc[train_idx].copy()
    temp = metadata.iloc[temp_idx].copy()

    second = GroupShuffleSplit(n_splits=1, train_size=0.50, random_state=42)
    val_rel_idx, test_rel_idx = next(second.split(temp, groups=temp["lesion_id"]))
    val = temp.iloc[val_rel_idx].copy()
    test = temp.iloc[test_rel_idx].copy()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    cols = ["image_path", "label", "image_id", "lesion_id"]
    train[cols].to_csv(out / "train.csv", index=False)
    val[cols].to_csv(out / "val.csv", index=False)
    test[cols].to_csv(out / "test.csv", index=False)

    print(f"train={len(train)} val={len(val)} test={len(test)}")
    print("class counts (train):")
    print(train["label"].value_counts().sort_index())


if __name__ == "__main__":
    main()
