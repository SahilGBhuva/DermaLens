"""v3: v2 fine-tuned on smartphone photos as well as dermoscopy.

Same discipline as ml/run_pipeline.py:

1. Split HAM10000 by lesion (identical to v1/v2's splits) and PAD-UFES-20
   by patient.
2. Train from v2's weights on PAD-UFES-20 train plus a replay sample of
   HAM10000 train, so dermoscopy is not forgotten. The epoch is chosen on
   VALIDATION only: the mean balanced accuracy of the two domains.
3. Calibrate on the two validation splits together (same method as v2).
4. Score each TEST split exactly once: HAM10000 test for v3, and the
   PAD-UFES-20 test patients for v1, v2 and v3 side by side.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def run(command):
    print("\n$", " ".join(str(part) for part in command), flush=True)
    subprocess.run([str(part) for part in command], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--ham-metadata", required=True)
    parser.add_argument("--ham-images", required=True)
    parser.add_argument("--pad-metadata", required=True)
    parser.add_argument("--pad-images", required=True)
    parser.add_argument("--init-weights", default="models/v2/dermalens_efficientnet_b0.pt")
    parser.add_argument("--compare-models", default="models/v1,models/v2")
    parser.add_argument("--ham-replay", type=int, default=3000, help="HAM10000 train images mixed in per run")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--work-dir", default="data/v3")
    parser.add_argument("--output-dir", default="models/v3")
    args = parser.parse_args()

    work, out = Path(args.work_dir), Path(args.output_dir)
    work.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    py, ml = sys.executable, Path(__file__).resolve().parent

    # 1. Splits.
    ham = work / "ham_splits"
    run([py, ml / "prepare_ham10000.py", "--metadata", args.ham_metadata, "--images-dir", args.ham_images, "--output-dir", ham])
    run([py, ml / "check_splits.py", "--train", ham / "train.csv", "--val", ham / "val.csv", "--test", ham / "test.csv"])
    pad = work / "pad_splits"
    run([py, ml / "prepare_pad_ufes_20.py", "--metadata", args.pad_metadata, "--images-dir", args.pad_images, "--output-dir", pad])

    # 2. Training mix: all PAD train + a class-stratified HAM replay sample.
    ham_train = pd.read_csv(ham / "train.csv")
    fraction = min(1.0, args.ham_replay / len(ham_train))
    replay = ham_train.groupby("label").sample(frac=fraction, random_state=42)
    pad_train = pd.read_csv(pad / "train.csv")
    mixed = pd.concat([pad_train[["image_path", "label"]], replay[["image_path", "label"]]], ignore_index=True)
    mixed.to_csv(work / "train_mixed.csv", index=False)
    print(f"training mix: {len(pad_train)} phone photos + {len(replay)} dermoscopy images", flush=True)

    # Name the validation files by domain; train.py labels history columns by file name.
    pd.read_csv(ham / "val.csv").to_csv(work / "dermoscopy.csv", index=False)
    pd.read_csv(pad / "val.csv")[["image_path", "label"]].to_csv(work / "phone.csv", index=False)
    weights = out / "dermalens_efficientnet_b0.pt"
    run([py, ml / "train.py", "--train-csv", work / "train_mixed.csv",
         "--val-csv", work / "dermoscopy.csv", work / "phone.csv", "--no-val-tta",
         "--init-weights", args.init_weights, "--loss", "ce", "--sampler", "balanced",
         "--epochs", args.epochs, "--patience", args.patience, "--lr", args.lr,
         "--batch-size", args.batch_size, "--output", weights, "--history", out / "training_history.json"])

    # 3. Calibrate on both validation splits.
    both_val = pd.concat([pd.read_csv(work / "dermoscopy.csv")[["image_path", "label"]], pd.read_csv(work / "phone.csv")])
    both_val.to_csv(work / "val_both.csv", index=False)
    config = out / "model_config.json"
    run([py, ml / "calibrate.py", "--csv", work / "val_both.csv", "--weights", weights, "--config", config,
         "--batch-size", args.batch_size])
    settings = json.loads(config.read_text())
    settings.update(
        {
            "version": "v3",
            "recipe": "v2 fine-tuned on PAD-UFES-20 train + HAM10000 replay",
            "training_data": {
                "phone_photos": int(len(pad_train)),
                "dermoscopy_replay": int(len(replay)),
                "pad_split_summary": json.loads((pad / "split_summary.json").read_text()),
            },
        }
    )
    config.write_text(json.dumps(settings, indent=2))

    # 4. The only time each test split is scored.
    run([py, ml / "evaluate.py", "--csv", ham / "test.csv", "--weights", weights, "--config", config,
         "--batch-size", args.batch_size, "--output", out / "evaluation.json"])
    run([py, ml / "evaluate_external.py", "--metadata", pad / "test.csv", "--images-dir", pad / "images",
         "--models", f"{args.compare_models},{out}", "--batch-size", args.batch_size,
         "--output", out / "pad_ufes_20_test.json"])

    print("\nv3 pipeline complete:", ", ".join(sorted(p.name for p in out.iterdir())))


if __name__ == "__main__":
    main()
