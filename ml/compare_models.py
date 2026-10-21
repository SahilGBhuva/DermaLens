"""Paired comparison of two models from saved per-image probabilities.

Both models are scored on the SAME bootstrap resamples (whole lesions or
patients), so the interval is for the difference itself, which is much
tighter and fairer than comparing two separate intervals.

Inputs are the prediction CSVs written by ml/evaluate.py (`p_<class>`
columns) or ml/evaluate_external.py (`<model>_p_<class>` columns); rows are
matched on the image id.

    python ml/compare_models.py --a v2.csv --b v41.csv --a-prefix "" --b-prefix "" \\
        --truth label --group lesion_id --output paired.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_external import DATASETS
from metrics import grouped_bootstrap
from model import CLASSES


def load(path: Path, prefix: str, key: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    columns = [f"{prefix}p_{c}" for c in CLASSES]
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ValueError(f"{path} lacks columns {missing[:2]}")
    out = frame.drop(columns=[c for c in frame.columns if "_p_" in c or c.startswith("p_")])
    out["pred"] = frame[columns].to_numpy().argmax(1)
    return out.set_index(key)


def paired(truth: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray, groups: np.ndarray, n_boot: int = 2000) -> dict:
    """Point estimates and paired 95% intervals for B - A."""
    mel = CLASSES.index("mel")

    def balanced(pred, rows):
        t = truth[rows]
        return np.mean([(pred[rows][t == k] == k).mean() for k in np.unique(t)])

    def melanoma(pred, rows):
        m = truth[rows] == mel
        return (pred[rows][m] == mel).mean() if m.any() else None

    def statistic(rows):
        out = {"balanced_accuracy": balanced(pred_b, rows) - balanced(pred_a, rows)}
        ma, mb = melanoma(pred_a, rows), melanoma(pred_b, rows)
        out["melanoma_sensitivity"] = None if ma is None else mb - ma
        return out

    point = statistic(np.arange(len(truth)))
    intervals = grouped_bootstrap(statistic, groups, n_boot=n_boot)
    return {
        name: {
            "difference_b_minus_a": float(point[name]),
            "interval_95": intervals.get(name),
            "verdict": (
                "B better" if intervals.get(name) and intervals[name][0] > 0
                else "B worse" if intervals.get(name) and intervals[name][1] < 0
                else "no detectable difference"
            ),
        }
        for name in point
        if point[name] is not None
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--a", required=True, type=Path)
    parser.add_argument("--b", required=True, type=Path)
    parser.add_argument("--a-prefix", default="")
    parser.add_argument("--b-prefix", default="")
    parser.add_argument("--key", default="image_id", help="image id column (img_id for external datasets)")
    parser.add_argument("--truth", default="label", help="'label' (class names) or 'diagnostic' (dataset codes)")
    parser.add_argument("--dataset", choices=sorted(DATASETS), default=None, help="maps 'diagnostic' codes")
    parser.add_argument("--group", default="lesion_id")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    a, b = load(args.a, args.a_prefix, args.key), load(args.b, args.b_prefix, args.key)
    shared = a.index.intersection(b.index)
    if len(shared) < len(a) or len(shared) < len(b):
        raise SystemExit(f"image sets differ: {len(a)} vs {len(b)} rows, {len(shared)} shared")
    a, b = a.loc[shared], b.loc[shared]

    truth_names = a[args.truth]
    if args.dataset:  # dataset codes -> classes; codes with no class (e.g. SCC) are dropped
        truth_names = truth_names.map(DATASETS[args.dataset]["mapping"])
    keep = truth_names.notna().to_numpy()
    truth = truth_names[keep].map({c: i for i, c in enumerate(CLASSES)}).to_numpy()

    result = {
        "a": f"{args.a.name} {args.a_prefix}".strip(),
        "b": f"{args.b.name} {args.b_prefix}".strip(),
        "images": int(keep.sum()),
        "groups": int(a[args.group][keep].nunique()),
        **paired(truth, a["pred"].to_numpy()[keep], b["pred"].to_numpy()[keep], a[args.group].to_numpy()[keep]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
