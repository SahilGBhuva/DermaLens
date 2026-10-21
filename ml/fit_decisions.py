"""v4 decision layer, fitted on the VALIDATION splits only.

v1-v3 bent the class scores with per-class offsets so the top class caught
more melanomas. That made the displayed percentages less honest and called
many harmless moles something else. v4 separates the two jobs:

1. Probabilities: temperature scaling only (minimum validation negative
   log-likelihood). No per-class offsets, so a score of 70% means what it says
   as far as validation can tell.
2. Decision: "needs a closer look" when P(akiec) + P(bcc) + P(mel) reaches a
   threshold. The threshold is the HIGHEST value that still flags at least
   `--target-sensitivity` of the concerning lesions in EVERY validation domain
   (dermoscopy and phone photos separately), so false alarms are as few as
   that safety floor allows.

The test splits are never read here.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score
from torch.utils.data import DataLoader

from calibrate import collect_probs, fit_temperature, nll
from dataset import SkinLesionDataset
from metrics import expected_calibration_error
from model import CLASSES, apply_calibration, build_model, eval_transform, load_config, pick_device, sha256

CONCERN_CLASSES = ["akiec", "bcc", "mel"]
CONCERN_INDEX = [CLASSES.index(name) for name in CONCERN_CLASSES]


def concern_scores(probs: np.ndarray) -> np.ndarray:
    return probs[:, CONCERN_INDEX].sum(axis=1)


def choose_threshold(domains: dict, target: float) -> float:
    """Highest threshold whose flag keeps sensitivity >= target in every domain.

    `domains` maps a name to (scores, is_concerning). Flagging is score >= t,
    so raising t can only lower sensitivity; scan the candidate cut points from
    the top down and stop at the first one that satisfies every domain.
    """
    candidates = np.unique(np.concatenate([scores for scores, _ in domains.values()]))[::-1]
    for t in candidates:
        if all((scores[truth] >= t).mean() >= target for scores, truth in domains.values() if truth.any()):
            return float(t)
    return float(candidates[-1])


def flag_stats(scores, truth, threshold):
    flags = scores >= threshold
    return {
        "images": int(len(scores)),
        "concerning": int(truth.sum()),
        "sensitivity": float(flags[truth].mean()) if truth.any() else None,
        "specificity": float((~flags[~truth]).mean()) if (~truth).any() else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--weights", required=True)
    parser.add_argument("--val-csv", required=True, nargs="+", help="one validation CSV per domain")
    parser.add_argument("--config", required=True, help="model_config.json to write (existing keys are kept)")
    parser.add_argument("--target-sensitivity", type=float, default=0.90)
    parser.add_argument("--tta", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    config = load_config(args.config)
    device = pick_device()
    model = build_model(pretrained=False).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device, weights_only=True))
    class_to_idx = {c: i for i, c in enumerate(CLASSES)}

    raw = {}
    for path in args.val_csv:
        ds = SkinLesionDataset(path, class_to_idx, eval_transform(config["image_size"]))
        raw[Path(path).stem] = collect_probs(model, DataLoader(ds, batch_size=args.batch_size), device, args.tta)

    y_all = np.concatenate([y for y, _ in raw.values()])
    p_all = torch.cat([p for _, p in raw.values()])
    temperature = fit_temperature(p_all, y_all)
    zeros = [0.0] * len(CLASSES)

    calibrated = {name: (y, apply_calibration(p, temperature, zeros).numpy()) for name, (y, p) in raw.items()}
    domains = {name: (concern_scores(p), np.isin(y, CONCERN_INDEX)) for name, (y, p) in calibrated.items()}
    threshold = choose_threshold(domains, args.target_sensitivity)
    # v4.1: also one threshold per domain (validation file name, e.g. "dermoscopy",
    # "phone"); the API uses the one for the photo type the visitor chose.
    per_domain = {name: choose_threshold({name: d}, args.target_sensitivity) for name, d in domains.items()}

    validation = {}
    for name, (y, p) in calibrated.items():
        validation[name] = {
            "balanced_accuracy": float(balanced_accuracy_score(y, p.argmax(1))),
            "expected_calibration_error": expected_calibration_error(y, p),
            "concern_flag": flag_stats(*domains[name], threshold),
            "concern_flag_own_threshold": flag_stats(*domains[name], per_domain[name]),
        }

    config.update(
        {
            "tta": args.tta,
            "temperature": temperature,
            "logit_bias": zeros,
            "concern": {
                "classes": CONCERN_CLASSES,
                "threshold": round(threshold, 6),
                "thresholds": {name: round(t, 6) for name, t in per_domain.items()},
                "target_sensitivity": args.target_sensitivity,
            },
            "weights_sha256": sha256(args.weights),
            "validation": {
                "images": int(len(y_all)),
                "nll_before": nll(p_all, y_all),
                "nll_after": nll(apply_calibration(p_all, temperature, zeros), y_all),
                "domains": validation,
            },
        }
    )
    config.pop("min_mel_sensitivity_target", None)
    Path(args.config).parent.mkdir(parents=True, exist_ok=True)
    Path(args.config).write_text(json.dumps(config, indent=2))
    print(json.dumps({"temperature": temperature, "threshold": threshold, "validation": validation}, indent=2))


if __name__ == "__main__":
    main()
