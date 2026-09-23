import argparse
import subprocess
import sys
from pathlib import Path


def run(command):
    print("\n$", " ".join(str(part) for part in command), flush=True)
    subprocess.run(command, check=True)


def main():
    parser = argparse.ArgumentParser(
        description="Run the full DermaLens data prep, training, and evaluation pipeline."
    )
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--splits-dir", default="data/splits")
    parser.add_argument("--weights", default="models/dermalens_efficientnet_b0.pt")
    parser.add_argument("--evaluation", default="models/evaluation.json")
    args = parser.parse_args()

    splits = Path(args.splits_dir)
    train_csv = splits / "train.csv"
    val_csv = splits / "val.csv"
    test_csv = splits / "test.csv"

    run(
        [
            sys.executable,
            "ml/prepare_ham10000.py",
            "--metadata",
            args.metadata,
            "--images-dir",
            args.images_dir,
            "--output-dir",
            args.splits_dir,
        ]
    )

    run(
        [
            sys.executable,
            "ml/check_splits.py",
            "--train",
            str(train_csv),
            "--val",
            str(val_csv),
            "--test",
            str(test_csv),
        ]
    )

    run(
        [
            sys.executable,
            "ml/train.py",
            "--train-csv",
            str(train_csv),
            "--val-csv",
            str(val_csv),
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--output",
            args.weights,
        ]
    )

    run(
        [
            sys.executable,
            "ml/evaluate.py",
            "--csv",
            str(test_csv),
            "--weights",
            args.weights,
            "--batch-size",
            str(args.batch_size),
            "--output",
            args.evaluation,
        ]
    )

    print("\nDermaLens pipeline complete.")
    print(f"Weights: {args.weights}")
    print(f"Held-out evaluation: {args.evaluation}")


if __name__ == "__main__":
    main()
