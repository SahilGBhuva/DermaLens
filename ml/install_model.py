"""Install a trained model downloaded from Colab into models/, safely.

Picks the newest copy of each artifact in the source folder (browsers add
" (1)" to repeated downloads), checks that weights, settings and evaluation
all belong together by SHA-256, backs up the currently installed model, then
copies the new one in and prints a before/after comparison.

    python ml/install_model.py --from ~/Downloads --version v2
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

from model import CLASSES, sha256

ARTIFACTS = {
    "weights": ("dermalens_efficientnet_b0", ".pt", "dermalens_efficientnet_b0.pt"),
    "config": ("model_config", ".json", "model_config.json"),
    "evaluation": ("evaluation", ".json", "evaluation.json"),
    "history": ("training_history", ".json", "training_history.json"),
}
COMPARE = ["accuracy", "balanced_accuracy", "macro_f1", "macro_ovr_roc_auc", "expected_calibration_error"]


def newest(folder: Path, stem: str, suffix: str):
    """Newest file named like `stem.suffix`, `stem (1).suffix`, `stem-2.suffix`."""
    matches = [
        p for p in folder.glob(f"{stem}*{suffix}")
        if p.stem == stem or p.stem[len(stem):].strip(" -_()0123456789") == ""
    ]
    return max(matches, key=lambda p: p.stat().st_mtime, default=None)


def find_artifacts(source: Path) -> dict:
    found = {key: newest(source, stem, suffix) for key, (stem, suffix, _) in ARTIFACTS.items()}
    missing = [ARTIFACTS[key][2] for key in ("weights", "config", "evaluation") if not found[key]]
    if missing:
        raise SystemExit(f"Missing in {source}: {', '.join(missing)}")
    return found


def verify(found: dict) -> tuple[dict, dict, str]:
    """Raise if the weights, settings and evaluation don't belong together."""
    weights_hash = sha256(found["weights"])
    config = json.loads(found["config"].read_text())
    evaluation = json.loads(found["evaluation"].read_text())

    if config.get("classes", CLASSES) != CLASSES:
        raise SystemExit(f"model_config.json classes {config.get('classes')} do not match {CLASSES}")
    if config.get("weights_sha256") != weights_hash:
        raise SystemExit(
            f"{found['weights'].name} does not match {found['config'].name}: "
            f"sha256 {weights_hash}, settings expect {config.get('weights_sha256')}"
        )
    measured_on = (evaluation.get("settings") or {}).get("weights_sha256")
    if measured_on != weights_hash:
        raise SystemExit(
            f"{found['evaluation'].name} was measured on different weights ({measured_on}); "
            "it cannot be published with this model"
        )
    return config, evaluation, weights_hash


def summary(evaluation: dict) -> dict:
    row = {key: evaluation.get(key) for key in COMPARE}
    per_class = evaluation.get("per_class_sensitivity_specificity") or {}
    row["melanoma_sensitivity"] = (per_class.get("mel") or {}).get("sensitivity")
    row["melanoma_specificity"] = (per_class.get("mel") or {}).get("specificity")
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--from", dest="source", default=str(Path.home() / "Downloads"))
    parser.add_argument("--to", dest="target", default="models")
    parser.add_argument("--version", default=None, help="label stored in model_config.json, e.g. v2")
    args = parser.parse_args()

    source, target = Path(args.source).expanduser(), Path(args.target)
    found = find_artifacts(source)
    for key, path in found.items():
        print(f"{key:10s} {path}" if path else f"{key:10s} (not found, optional)")
    config, evaluation, weights_hash = verify(found)
    print(f"\nchecks passed: weights sha256 {weights_hash}")

    previous = None
    if (target / "evaluation.json").exists():
        previous = json.loads((target / "evaluation.json").read_text())
    if (target / ARTIFACTS["weights"][2]).exists():
        old_config = target / "model_config.json"
        label = json.loads(old_config.read_text()).get("version", "previous") if old_config.exists() else "previous"
        backup = target / "previous" / label
        backup.mkdir(parents=True, exist_ok=True)
        for _, _, name in ARTIFACTS.values():
            if (target / name).exists():
                shutil.copy2(target / name, backup / name)
        print(f"backed up current model to {backup}")

    target.mkdir(parents=True, exist_ok=True)
    shutil.copy2(found["weights"], target / ARTIFACTS["weights"][2])
    shutil.copy2(found["evaluation"], target / ARTIFACTS["evaluation"][2])
    if found["history"]:
        shutil.copy2(found["history"], target / ARTIFACTS["history"][2])
    if args.version:
        config["version"] = args.version
    # Lets the API confirm the published evaluation file is the one installed here.
    config["evaluation_sha256"] = sha256(target / ARTIFACTS["evaluation"][2])
    (target / "model_config.json").write_text(json.dumps(config, indent=2))
    print(f"installed into {target}/")

    new = summary(evaluation)
    old = summary(previous) if previous else {}
    print(f"\n{'held-out test':30s} {'before':>8s} {'after':>8s}")
    for key, value in new.items():
        before = old.get(key)
        fmt = lambda v: "—" if v is None else f"{v:.3f}"
        print(f"{key:30s} {fmt(before):>8s} {fmt(value):>8s}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
