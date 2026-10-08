"""Score trained models, unchanged, on PAD-UFES-20 smartphone photos.

PAD-UFES-20 (Pacheco et al., 2020, CC BY 4.0) holds 2,298 clinical photos
taken with smartphones in Brazil: a different country, camera type and
imaging method from HAM10000's dermoscopy. Nothing here is tuned on it; each
model is run exactly as the API serves it (same preprocessing, TTA and
calibration from its model_config.json).

Five of its six diagnoses map onto DermaLens classes. Invasive squamous cell
carcinoma (SCC) has no DermaLens class, so it is left out of the class
metrics and reported separately: how often it is at least flagged as one of
the concerning classes.

    python ml/evaluate_external.py --metadata data/pad_ufes_20/metadata.csv \
        --images-dir data/pad_ufes_20/images --models models/v1,models/v2 \
        --output models/external/pad_ufes_20.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from metrics import expected_calibration_error
from model import CLASSES, apply_calibration, build_model, eval_transform, load_config, pick_device, predict_proba, sha256

# PAD-UFES-20 diagnostic code -> DermaLens class.
PAD_TO_DERMALENS = {
    "ACK": "akiec",  # actinic keratosis
    "BCC": "bcc",
    "MEL": "mel",
    "NEV": "nv",
    "SEK": "bkl",  # seborrheic keratosis is part of "benign keratosis-like"
}
UNMAPPED = "SCC"
# Predictions that would prompt a closer look: (pre-)malignant classes.
CONCERNING = {"akiec", "bcc", "mel"}
# Ground truth that warrants one: everything except moles and seborrheic keratoses.
CONCERNING_TRUTH = {"ACK", "BCC", "MEL", "SCC"}


class PadDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, images_dir: Path, transform):
        self.frame = frame.reset_index(drop=True)
        self.images_dir = images_dir
        self.transform = transform

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        image = Image.open(self.images_dir / self.frame.loc[idx, "img_id"]).convert("RGB")
        return self.transform(image), idx


def load_metadata(metadata: Path, images_dir: Path) -> pd.DataFrame:
    frame = pd.read_csv(metadata)
    missing = {"img_id", "diagnostic"} - set(frame.columns)
    if missing:
        raise ValueError(f"metadata missing columns: {sorted(missing)}")
    # img_id comes from a downloaded file: allow plain file names only, so a
    # crafted row cannot point outside the images folder.
    unsafe = [name for name in frame["img_id"] if Path(str(name)).name != str(name) or str(name).startswith(".")]
    if unsafe:
        raise ValueError(f"img_id must be a plain file name, got e.g. {unsafe[:3]}")
    unknown = set(frame["diagnostic"]) - set(PAD_TO_DERMALENS) - {UNMAPPED}
    if unknown:
        raise ValueError(f"unexpected diagnostic codes: {sorted(unknown)}")
    absent = [name for name in frame["img_id"] if not (images_dir / name).exists()]
    if absent:
        raise FileNotFoundError(f"{len(absent)} images missing from {images_dir}, e.g. {absent[:3]}")
    return frame


def run_model(model_dir: Path, frame: pd.DataFrame, images_dir: Path, batch_size: int) -> np.ndarray:
    config = load_config(model_dir / "model_config.json")
    weights = model_dir / "dermalens_efficientnet_b0.pt"
    if config.get("weights_sha256") and config["weights_sha256"] != sha256(weights):
        raise SystemExit(f"{weights} does not match its model_config.json")

    device = pick_device()
    model = build_model(pretrained=False).to(device)
    model.load_state_dict(torch.load(weights, map_location=device, weights_only=True))
    model.eval()

    loader = DataLoader(PadDataset(frame, images_dir, eval_transform(config["image_size"])), batch_size=batch_size)
    probs = np.zeros((len(frame), len(CLASSES)), dtype=np.float64)
    with torch.inference_mode():
        for images, idx in loader:
            batch = predict_proba(model, images.to(device), config["tta"])
            batch = apply_calibration(batch, config["temperature"], config["logit_bias"])
            probs[idx.numpy()] = batch.cpu().numpy()
            print(f"  {model_dir.name}: {min(int(idx[-1]) + 1, len(frame))}/{len(frame)}", end="\r", flush=True)
    print()
    return probs


def summarize(frame: pd.DataFrame, probs: np.ndarray) -> dict:
    """Metrics for one model's probabilities over the PAD-UFES-20 rows."""
    predicted = np.array([CLASSES[i] for i in probs.argmax(1)])
    diagnostic = frame["diagnostic"].to_numpy()
    mapped = np.isin(diagnostic, list(PAD_TO_DERMALENS))
    truth = np.array([PAD_TO_DERMALENS.get(d, "") for d in diagnostic])

    per_class = {}
    for code, name in PAD_TO_DERMALENS.items():
        rows = diagnostic == code
        per_class[code] = {
            "dermalens_class": name,
            "images": int(rows.sum()),
            "sensitivity": float((predicted[rows] == name).mean()) if rows.any() else None,
        }
    recalls = [v["sensitivity"] for v in per_class.values() if v["sensitivity"] is not None]

    class_index = {name: i for i, name in enumerate(CLASSES)}
    ece = expected_calibration_error([class_index[t] for t in truth[mapped]], probs[mapped])

    flagged = np.isin(predicted, list(CONCERNING))
    should_flag = np.isin(diagnostic, list(CONCERNING_TRUTH))
    scc = diagnostic == UNMAPPED

    confusion = {
        code: {name: int(((diagnostic == code) & (predicted == name)).sum()) for name in CLASSES}
        for code in [*PAD_TO_DERMALENS, UNMAPPED]
    }

    by_skin_type = {}
    if "fitspatrick" in frame.columns:  # sic: the dataset's spelling
        skin = pd.to_numeric(frame["fitspatrick"], errors="coerce").to_numpy()
        groups = {"I–II": (1, 2), "III–IV": (3, 4), "V–VI": (5, 6)}
        for label, (low, high) in groups.items():
            rows = mapped & (skin >= low) & (skin <= high)
            by_skin_type[label] = {
                "images": int(rows.sum()),
                "accuracy": float((predicted[rows] == truth[rows]).mean()) if rows.any() else None,
            }
        rows = mapped & np.isnan(skin)
        by_skin_type["not recorded"] = {
            "images": int(rows.sum()),
            "accuracy": float((predicted[rows] == truth[rows]).mean()) if rows.any() else None,
        }

    return {
        "images_scored": int(mapped.sum()),
        "accuracy": float((predicted[mapped] == truth[mapped]).mean()),
        "balanced_accuracy": float(np.mean(recalls)),
        "melanoma_sensitivity": per_class["MEL"]["sensitivity"],
        "expected_calibration_error": ece,
        "per_class": per_class,
        "concerning_lesions": {
            "description": "Flagged as akiec, bcc or mel when the truth is ACK, BCC, MEL or SCC",
            "sensitivity": float(flagged[should_flag].mean()) if should_flag.any() else None,
            "specificity": float((~flagged[~should_flag]).mean()) if (~should_flag).any() else None,
        },
        "scc_unmapped": {
            "images": int(scc.sum()),
            "flagged_as_concerning": float(flagged[scc].mean()) if scc.any() else None,
        },
        "accuracy_by_fitzpatrick_skin_type": by_skin_type,
        "confusion_true_pad_vs_predicted": confusion,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--images-dir", required=True, type=Path)
    parser.add_argument("--models", default="models/v1,models/v2", help="comma-separated model folders")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", default="models/external/pad_ufes_20.json", type=Path)
    args = parser.parse_args()

    frame = load_metadata(args.metadata, args.images_dir)
    print(f"PAD-UFES-20: {len(frame)} images, {frame['diagnostic'].value_counts().to_dict()}")

    results = {
        "dataset": {
            "name": "PAD-UFES-20",
            "citation": "Pacheco AGC et al. PAD-UFES-20: a skin lesion dataset composed of patient data and "
            "clinical images collected from smartphones. Data in Brief, 2020. doi:10.17632/zr7vgbcyr2.1",
            "licence": "CC BY 4.0",
            "images": len(frame),
            "diagnostic_counts": {k: int(v) for k, v in frame["diagnostic"].value_counts().items()},
            "mapping": PAD_TO_DERMALENS,
            "note": "Models are applied unchanged; nothing was tuned on this dataset.",
        },
        "models": {},
    }
    for folder in [Path(p.strip()) for p in args.models.split(",") if p.strip()]:
        probs = run_model(folder, frame, args.images_dir, args.batch_size)
        summary = summarize(frame, probs)
        summary["weights_sha256"] = sha256(folder / "dermalens_efficientnet_b0.pt")
        results["models"][folder.name] = summary
        print(f"{folder.name}: balanced accuracy {summary['balanced_accuracy']:.3f}, "
              f"melanoma sensitivity {summary['melanoma_sensitivity']:.3f}, ECE {summary['expected_calibration_error']:.3f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False, allow_nan=False))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
