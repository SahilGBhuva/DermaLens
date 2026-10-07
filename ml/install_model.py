"""Install a trained model downloaded from Colab into models/, safely.

Picks the newest copy of each artifact in the source folder (browsers add
" (1)" to repeated downloads), checks that weights, settings and evaluation
all belong together by SHA-256, then installs them into models/<version>/
(moving any existing copy of that version to models/.backups/) and prints a
comparison with every installed version. The API serves each folder.

    python ml/install_model.py --from ~/Downloads --version v3
"""

import argparse
import json
import re
import shutil
import sys
import time
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
    parser.add_argument("--version", required=True, help="folder/label for this model, e.g. v3")
    args = parser.parse_args()

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,31}", args.version):
        raise SystemExit("--version must be letters, digits, '.', '_' or '-' (e.g. v3)")

    source, models_dir = Path(args.source).expanduser(), Path(args.target)
    found = find_artifacts(source)
    for key, path in found.items():
        print(f"{key:10s} {path}" if path else f"{key:10s} (not found, optional)")
    config, evaluation, weights_hash = verify(found)
    print(f"\nchecks passed: weights sha256 {weights_hash}")

    target = models_dir / args.version
    if target.exists():
        # Hidden folder: the API only serves folders whose names start with a letter or digit.
        backup = models_dir / ".backups" / f"{args.version}-{time.strftime('%Y%m%d-%H%M%S')}"
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(target), str(backup))
        print(f"moved the existing {args.version} to {backup}")

    target.mkdir(parents=True)
    shutil.copy2(found["weights"], target / ARTIFACTS["weights"][2])
    shutil.copy2(found["evaluation"], target / ARTIFACTS["evaluation"][2])
    if found["history"]:
        shutil.copy2(found["history"], target / ARTIFACTS["history"][2])
    config["version"] = args.version
    # Lets the API confirm the published evaluation file is the one installed here.
    config["evaluation_sha256"] = sha256(target / ARTIFACTS["evaluation"][2])
    (target / "model_config.json").write_text(json.dumps(config, indent=2))
    print(f"installed into {target}/")

    columns = {}
    for folder in sorted(p for p in models_dir.iterdir() if p.is_dir() and not p.name.startswith(".")):
        if (folder / "evaluation.json").exists():
            columns[folder.name] = summary(json.loads((folder / "evaluation.json").read_text()))
    fmt = lambda v: "—" if v is None else f"{v:.3f}"
    print(f"\n{'held-out test':30s}" + "".join(f"{name:>9s}" for name in columns))
    for key in summary(evaluation):
        print(f"{key:30s}" + "".join(f"{fmt(row.get(key)):>9s}" for row in columns.values()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
