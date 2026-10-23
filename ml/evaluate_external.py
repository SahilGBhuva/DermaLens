"""Score trained models, unchanged, on an outside dataset.

Datasets (``--dataset``): PAD-UFES-20 smartphone photos (default) and
BCN20000 dermoscopy from Hospital Clinic Barcelona (see DATASETS below and
ml/prepare_bcn20000.py).

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

from metrics import expected_calibration_error, grouped_bootstrap
from model import CLASSES, apply_calibration, concern_flags_for, describe_concern_rule, build_model, eval_transform, load_config, pick_device, predict_proba, sha256

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

# Every supported dataset: its diagnostic codes -> DermaLens classes, the code
# with no DermaLens class (scored only as "flagged as concerning"), and which
# true codes warrant a closer look.
DATASETS = {
    "pad_ufes_20": {
        "name": "PAD-UFES-20",
        "citation": "Pacheco AGC et al. PAD-UFES-20: a skin lesion dataset composed of patient data and "
        "clinical images collected from smartphones. Data in Brief, 2020. doi:10.17632/zr7vgbcyr2.1",
        "licence": "CC BY 4.0",
        "mapping": PAD_TO_DERMALENS,
        "unmapped": UNMAPPED,
        "concerning": CONCERNING_TRUTH,
        "decision_domain": "phone",
    },
    "bcn20000": {
        "name": "BCN20000 (Hospital Clinic Barcelona, via the ISIC Archive)",
        "citation": "Hernandez-Perez C. et al. BCN20000: Dermoscopic lesions in the wild. Scientific Data, 2024. "
        "ISIC Archive collection 249.",
        "licence": "CC BY-NC",
        "mapping": {
            "AK": "akiec", "BCC": "bcc", "BKL": "bkl", "DF": "df", "MEL": "mel", "NV": "nv", "VASC": "vasc",
        },
        "unmapped": "SCC",
        "concerning": {"AK", "BCC", "MEL", "SCC"},
        "decision_domain": "dermoscopy",
    },
}


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


def load_metadata(metadata: Path, images_dir: Path, dataset: str = "pad_ufes_20") -> pd.DataFrame:
    spec = DATASETS[dataset]
    frame = pd.read_csv(metadata)
    missing = {"img_id", "diagnostic"} - set(frame.columns)
    if missing:
        raise ValueError(f"metadata missing columns: {sorted(missing)}")
    # img_id comes from a downloaded file: allow plain file names only, so a
    # crafted row cannot point outside the images folder.
    unsafe = [name for name in frame["img_id"] if Path(str(name)).name != str(name) or str(name).startswith(".")]
    if unsafe:
        raise ValueError(f"img_id must be a plain file name, got e.g. {unsafe[:3]}")
    unknown = set(frame["diagnostic"]) - set(spec["mapping"]) - {spec["unmapped"]}
    if unknown:
        raise ValueError(f"unexpected diagnostic codes: {sorted(unknown)}")
    absent = [name for name in frame["img_id"] if not (images_dir / name).exists()]
    if absent:
        raise FileNotFoundError(f"{len(absent)} images missing from {images_dir}, e.g. {absent[:3]}")
    return frame


def run_model(model_dir: Path, frame: pd.DataFrame, images_dir: Path, batch_size: int) -> tuple[np.ndarray, dict]:
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
    return probs, config


def concern_threshold(concern: dict, domain: str | None) -> float:
    """The per-domain threshold (v4.1+) if the config has one, else the shared one."""
    return concern.get("thresholds", {}).get(domain or "", concern["threshold"])


def concern_flags(probs: np.ndarray, predicted: np.ndarray, concern: dict | None, domain: str | None = None) -> np.ndarray:
    """The model's own "needs a closer look" decision for each row.

    v4+ configs carry a `concern` threshold on P(akiec) + P(bcc) + P(mel);
    earlier versions flag when the top class is one of those classes.
    """
    if concern:
        return concern_flags_for(probs, concern, domain)
    return np.isin(predicted, list(CONCERNING))


def summarize(frame: pd.DataFrame, probs: np.ndarray, concern: dict | None = None, dataset: str = "pad_ufes_20") -> dict:
    """Metrics for one model's probabilities over the dataset's rows."""
    spec = DATASETS[dataset]
    mapping, unmapped, concerning_truth = spec["mapping"], spec["unmapped"], spec["concerning"]
    predicted = np.array([CLASSES[i] for i in probs.argmax(1)])
    diagnostic = frame["diagnostic"].to_numpy()
    mapped = np.isin(diagnostic, list(mapping))
    truth = np.array([mapping.get(d, "") for d in diagnostic])

    per_class = {}
    for code, name in mapping.items():
        rows = diagnostic == code
        per_class[code] = {
            "dermalens_class": name,
            "images": int(rows.sum()),
            "sensitivity": float((predicted[rows] == name).mean()) if rows.any() else None,
        }
    recalls = [v["sensitivity"] for v in per_class.values() if v["sensitivity"] is not None]

    class_index = {name: i for i, name in enumerate(CLASSES)}
    ece = expected_calibration_error([class_index[t] for t in truth[mapped]], probs[mapped])

    flagged = concern_flags(probs, predicted, concern, spec.get("decision_domain"))
    should_flag = np.isin(diagnostic, list(concerning_truth))
    scc = diagnostic == unmapped

    confusion = {
        code: {name: int(((diagnostic == code) & (predicted == name)).sum()) for name in CLASSES}
        for code in [*mapping, unmapped]
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

    # 95% intervals, resampling whole patients when the metadata has them.
    # Resample whole patients if every row has one, else whole lesions (BCN20000 publishes no patient ids).
    group_column = next((c for c in ("patient_id", "lesion_id") if c in frame.columns and frame[c].notna().all()), None)
    groups = frame[group_column].to_numpy() if group_column else np.arange(len(frame))

    def statistic(rows):
        d, p, t, m = diagnostic[rows], predicted[rows], truth[rows], mapped[rows]
        recalls = [float((p[d == code] == name).mean()) for code, name in mapping.items() if (d == code).any()]
        concerning = np.isin(d, list(concerning_truth))
        flags = flagged[rows]
        return {
            "accuracy": float((p[m] == t[m]).mean()) if m.any() else None,
            "balanced_accuracy": float(np.mean(recalls)) if len(recalls) == len(mapping) else None,
            "melanoma_sensitivity": float((p[d == "MEL"] == "mel").mean()) if (d == "MEL").any() else None,
            "concerning_sensitivity": float(flags[concerning].mean()) if concerning.any() else None,
            "concerning_specificity": float((~flags[~concerning]).mean()) if (~concerning).any() else None,
        }

    return {
        "confidence_intervals_95": grouped_bootstrap(statistic, groups),
        "images_scored": int(mapped.sum()),
        "accuracy": float((predicted[mapped] == truth[mapped]).mean()),
        "balanced_accuracy": float(np.mean(recalls)),
        "melanoma_sensitivity": per_class["MEL"]["sensitivity"],
        "expected_calibration_error": ece,
        "per_class": per_class,
        "concerning_lesions": {
            "description": "Flagged for a closer look when the truth is ACK, BCC, MEL or SCC",
            "decision_rule": (
                describe_concern_rule(concern, spec.get("decision_domain")) if concern
                else "top class is akiec, bcc or mel"
            ),
            "sensitivity": float(flagged[should_flag].mean()) if should_flag.any() else None,
            "specificity": float((~flagged[~should_flag]).mean()) if (~should_flag).any() else None,
        },
        # Each true code separately (pre-registration rule since v4.2).
        "flagged_by_code": {
            code: {
                "images": int((diagnostic == code).sum()),
                "flagged": float(flagged[diagnostic == code].mean()),
                "interval_95": grouped_bootstrap(
                    lambda rows, c=code: {"rate": flagged[rows][diagnostic[rows] == c].mean() if (diagnostic[rows] == c).any() else None},
                    groups,
                ).get("rate"),
            }
            for code in sorted(concerning_truth)
            if (diagnostic == code).any()
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
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="pad_ufes_20")
    parser.add_argument("--predictions", type=Path, default=None, help="also save every image's probabilities per model (CSV)")
    args = parser.parse_args()

    spec = DATASETS[args.dataset]
    frame = load_metadata(args.metadata, args.images_dir, args.dataset)
    print(f"{spec['name']}: {len(frame)} images, {frame['diagnostic'].value_counts().to_dict()}")

    results = {
        "dataset": {
            "name": spec["name"],
            "citation": spec["citation"],
            "licence": spec["licence"],
            "images": len(frame),
            "diagnostic_counts": {k: int(v) for k, v in frame["diagnostic"].value_counts().items()},
            "mapping": spec["mapping"],
            "note": "Models are applied unchanged; nothing was tuned on this dataset.",
        },
        "models": {},
    }
    keep = [c for c in ("img_id", "diagnostic", "patient_id", "lesion_id") if c in frame.columns]
    predictions = frame[keep].copy()
    for folder in [Path(p.strip()) for p in args.models.split(",") if p.strip()]:
        probs, config = run_model(folder, frame, args.images_dir, args.batch_size)
        summary = summarize(frame, probs, config.get("concern"), args.dataset)
        summary["weights_sha256"] = sha256(folder / "dermalens_efficientnet_b0.pt")
        for k, name in enumerate(CLASSES):
            predictions[f"{folder.name}_p_{name}"] = probs[:, k].round(6)
        results["models"][folder.name] = summary
        print(f"{folder.name}: balanced accuracy {summary['balanced_accuracy']:.3f}, "
              f"melanoma sensitivity {summary['melanoma_sensitivity']:.3f}, ECE {summary['expected_calibration_error']:.3f}")

    if args.predictions:
        args.predictions.parent.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(args.predictions, index=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False, allow_nan=False))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
