"""Selective prediction ("I don't know") from saved per-image probabilities.

Question: if the model only answers when it is confident enough and refers
the rest to a person, how much more accurate are the answers it does give,
and what happens to the melanomas?

Confidence is the top-class probability. For each target coverage (the share
of images answered) the cut-off is the matching quantile of confidence on the
same images, so no test labels are used to pick it. Reported per coverage:

* accuracy and balanced accuracy of the answered images (95% interval from
  resampling whole lesions or patients, cut-offs held fixed);
* what happened to every melanoma: answered correctly, answered wrongly, or
  referred. A system that "improves accuracy" by quietly referring cancers is
  not a safety gain, so this is reported next to the accuracy.

Inputs are the CSVs written by ml/evaluate.py (`p_<class>` columns, truth in
`label`) or ml/evaluate_external.py (`<model>_p_<class>`, truth in
`diagnostic`). Nothing here reads images or trains anything.

    python ml/selective.py --csv docs/results/rescored/v2_ham10000_test_predictions.csv \\
        --id-column image_id --truth label --group lesion_id --output selective_v2.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from metrics import grouped_bootstrap

CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
DEFAULT_COVERAGES = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5)
# Dataset diagnosis codes -> DermaLens classes (same as ml/evaluate_external.py,
# repeated here so this script needs no PyTorch). Codes with no class, such as
# invasive SCC, are dropped.
CODE_MAPS = {
    "bcn20000": {"AK": "akiec", "BCC": "bcc", "BKL": "bkl", "DF": "df", "MEL": "mel", "NV": "nv", "VASC": "vasc"},
    "pad_ufes_20": {"ACK": "akiec", "BCC": "bcc", "MEL": "mel", "NEV": "nv", "SEK": "bkl"},
}


def balanced_accuracy(truth: np.ndarray, pred: np.ndarray) -> float | None:
    present = np.unique(truth)
    if len(present) == 0:
        return None
    return float(np.mean([(pred[truth == k] == k).mean() for k in present]))


def selective_curve(
    probs: np.ndarray,
    truth: np.ndarray,
    groups: np.ndarray,
    coverages=DEFAULT_COVERAGES,
    n_boot: int = 1000,
) -> list[dict]:
    """One row per coverage. `truth` holds class indices into CLASSES."""
    probs, truth, groups = np.asarray(probs), np.asarray(truth), np.asarray(groups)
    confidence = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    mel = CLASSES.index("mel")
    rows = []
    for coverage in coverages:
        # Answer the `coverage` share with the highest confidence.
        cutoff = np.quantile(confidence, 1 - coverage) if coverage < 1 else -np.inf
        answered = confidence >= cutoff

        def statistic(idx, answered=answered):
            a = answered[idx]
            t, p = truth[idx][a], pred[idx][a]
            if len(t) == 0:
                return {"accuracy": None, "balanced_accuracy": None}
            return {"accuracy": float((t == p).mean()), "balanced_accuracy": balanced_accuracy(t, p)}

        point = statistic(np.arange(len(truth)))
        intervals = grouped_bootstrap(statistic, groups, n_boot=n_boot)
        is_mel = truth == mel
        n_mel = int(is_mel.sum())
        mel_correct = int((answered & is_mel & (pred == mel)).sum())
        mel_wrong = int((answered & is_mel & (pred != mel)).sum())
        rows.append(
            {
                "coverage": float(coverage),
                "answered": int(answered.sum()),
                "confidence_cutoff": None if np.isinf(cutoff) else round(float(cutoff), 4),
                "accuracy": point["accuracy"],
                "accuracy_interval_95": intervals.get("accuracy"),
                "balanced_accuracy": point["balanced_accuracy"],
                "balanced_accuracy_interval_95": intervals.get("balanced_accuracy"),
                "melanomas": n_mel,
                "melanomas_answered_correctly": mel_correct,
                "melanomas_answered_wrongly": mel_wrong,
                "melanomas_referred": n_mel - mel_correct - mel_wrong,
            }
        )
    return rows


def load_probabilities(frame: pd.DataFrame, prefix: str):
    columns = [f"{prefix}p_{c}" for c in CLASSES]
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ValueError(f"missing probability columns, e.g. {missing[:2]}")
    return frame[columns].to_numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--prefix", default="", help="model prefix in external CSVs, e.g. 'v3_'")
    parser.add_argument("--id-column", default="image_id")
    parser.add_argument("--truth", default="label", help="'label' (class names) or 'diagnostic' (dataset codes)")
    parser.add_argument(
        "--dataset",
        choices=sorted(CODE_MAPS),
        default=None,
        help="map 'diagnostic' codes to classes; codes with no class (e.g. SCC) are dropped",
    )
    parser.add_argument("--group", default="lesion_id")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--n-boot", type=int, default=1000)
    args = parser.parse_args()

    frame = pd.read_csv(args.csv)
    probs = load_probabilities(frame, args.prefix)
    names = frame[args.truth]
    if args.dataset:
        names = names.map(CODE_MAPS[args.dataset])
    keep = names.notna().to_numpy()
    truth = names[keep].map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
    result = {
        "source": args.csv.name,
        "model_prefix": args.prefix or None,
        "images": int(keep.sum()),
        "groups": int(frame[args.group][keep].nunique()),
        "confidence": "top-class probability",
        "curve": selective_curve(probs[keep], truth, frame[args.group].to_numpy()[keep], n_boot=args.n_boot),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    for row in result["curve"]:
        print(
            f"coverage {row['coverage']:.0%}: balanced accuracy {row['balanced_accuracy']:.3f}, "
            f"melanomas correct/wrong/referred "
            f"{row['melanomas_answered_correctly']}/{row['melanomas_answered_wrongly']}/{row['melanomas_referred']}"
        )


if __name__ == "__main__":
    main()
