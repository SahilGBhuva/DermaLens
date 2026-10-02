"""Full DermaLens pipeline with a strict train / validation / test discipline.

1. Split by lesion and check for leakage.
2. Train each candidate recipe; compare them on VALIDATION balanced accuracy.
3. Calibrate the chosen model on VALIDATION (temperature + per-class offsets).
4. Evaluate the final, calibrated model on TEST exactly once.
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

# name -> extra train.py arguments
VARIANTS = {
    "weighted_ce": ["--loss", "ce", "--sampler", "none"],
    "focal_balanced": ["--loss", "focal", "--sampler", "balanced"],
}


def run(command):
    print("\n$", " ".join(str(part) for part in command), flush=True)
    subprocess.run(command, check=True)


def main():
    parser = argparse.ArgumentParser(
        description="Run the full DermaLens data prep, training, calibration and evaluation pipeline."
    )
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--variants", default=",".join(VARIANTS), help=f"comma-separated subset of {list(VARIANTS)}")
    parser.add_argument("--min-mel-sensitivity", type=float, default=0.8)
    parser.add_argument("--splits-dir", default="data/splits")
    parser.add_argument("--weights", default="models/dermalens_efficientnet_b0.pt")
    parser.add_argument("--config", default="models/model_config.json")
    parser.add_argument("--evaluation", default="models/evaluation.json")
    args = parser.parse_args()

    splits = Path(args.splits_dir)
    train_csv, val_csv, test_csv = (splits / f"{name}.csv" for name in ("train", "val", "test"))
    out_dir = Path(args.weights).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    run([sys.executable, "ml/prepare_ham10000.py", "--metadata", args.metadata,
         "--images-dir", args.images_dir, "--output-dir", args.splits_dir])
    run([sys.executable, "ml/check_splits.py", "--train", str(train_csv),
         "--val", str(val_csv), "--test", str(test_csv)])

    # Train every candidate; the score that decides is validation-only.
    results = {}
    for name in [v.strip() for v in args.variants.split(",") if v.strip()]:
        weights = out_dir / f"candidate_{name}.pt"
        history = out_dir / f"training_history_{name}.json"
        run([sys.executable, "ml/train.py", "--train-csv", str(train_csv), "--val-csv", str(val_csv),
             "--epochs", str(args.epochs), "--batch-size", str(args.batch_size),
             "--output", str(weights), "--history", str(history), *VARIANTS[name]])
        rows = json.loads(history.read_text())
        results[name] = max(row["val_balanced_accuracy"] for row in rows)

    chosen = max(results, key=results.get)
    print("\nvalidation balanced accuracy by recipe:", json.dumps(results, indent=2))
    print(f"chosen recipe: {chosen}")
    shutil.copy(out_dir / f"candidate_{chosen}.pt", args.weights)
    shutil.copy(out_dir / f"training_history_{chosen}.json", out_dir / "training_history.json")

    run([sys.executable, "ml/calibrate.py", "--csv", str(val_csv), "--weights", args.weights,
         "--config", args.config, "--min-mel-sensitivity", str(args.min_mel_sensitivity)])

    config = json.loads(Path(args.config).read_text())
    config["recipe"] = chosen
    config["candidates_val_balanced_accuracy"] = results
    Path(args.config).write_text(json.dumps(config, indent=2))

    # The only time the test split is scored.
    run([sys.executable, "ml/evaluate.py", "--csv", str(test_csv), "--weights", args.weights,
         "--config", args.config, "--batch-size", str(args.batch_size), "--output", args.evaluation])

    print("\nDermaLens pipeline complete.")
    print(f"Weights: {args.weights}")
    print(f"Settings: {args.config}")
    print(f"Held-out evaluation: {args.evaluation}")


if __name__ == "__main__":
    main()
