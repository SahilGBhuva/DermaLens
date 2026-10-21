"""Fetch BCN20000 (Hospital Clinic Barcelona) from the ISIC Archive for testing.

BCN20000 is 18,946 dermoscopic images from a different hospital and country
than HAM10000, licensed CC BY-NC, mostly confirmed by histopathology. None of
it was used to train any DermaLens version, so it is an outside dermoscopy
test. Images are downloaded over https from the ISIC Archive API; each file
must decode as an image and be at least the size the archive lists (served
files carry a few extra kilobytes of metadata, so sizes never match exactly).

Diagnoses are mapped onto DermaLens classes by the archive's diagnosis_3
(diagnosis_2 for vascular lesions). Invasive SCC has no DermaLens class and is
kept as "SCC" (scored only as flagged / not flagged). Scars, melanoma
metastases and anything else without a class are left out and counted.

    python ml/prepare_bcn20000.py --output-dir data/bcn20000
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path

import pandas as pd
from PIL import Image

API = "https://api.isic-archive.com/api/v2/images/search/"
COLLECTION = 249  # BCN20000

BY_DIAGNOSIS_3 = {
    "Nevus": "NV",
    "Melanoma, NOS": "MEL",
    "Basal cell carcinoma": "BCC",
    "Seborrheic keratosis": "BKL",
    "Solar lentigo": "BKL",  # HAM10000's "benign keratosis-like" includes solar lentigo
    "Solar or actinic keratosis": "AK",
    "Squamous cell carcinoma, NOS": "SCC",
    "Dermatofibroma": "DF",
}
VASCULAR_DIAGNOSIS_2 = "Benign soft tissue proliferations - Vascular"


def code_for(clinical: dict) -> str | None:
    if clinical.get("diagnosis_2") == VASCULAR_DIAGNOSIS_2:
        return "VASC"
    return BY_DIAGNOSIS_3.get(clinical.get("diagnosis_3"))


def get_json(url: str) -> dict:
    if not url.startswith("https://"):
        raise ValueError(f"refusing non-https URL: {url[:60]}")
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:  # nosec B310: https checked above
                return json.load(response)
        except OSError:
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("unreachable")


def list_images(limit: int | None) -> list[dict]:
    rows, url = [], f"{API}?collections={COLLECTION}&limit=100"
    while url and (limit is None or len(rows) < limit):
        page = get_json(url)
        for image in page.get("results", []):
            clinical = image.get("metadata", {}).get("clinical", {})
            full = image.get("files", {}).get("full", {})
            rows.append(
                {
                    "isic_id": image["isic_id"],
                    "code": code_for(clinical),
                    "diagnosis_2": clinical.get("diagnosis_2"),
                    "diagnosis_3": clinical.get("diagnosis_3"),
                    "confirm_type": clinical.get("diagnosis_confirm_type"),
                    "lesion_id": clinical.get("lesion_id"),
                    "patient_id": clinical.get("patient_id"),
                    "image_type": image.get("metadata", {}).get("acquisition", {}).get("image_type"),
                    "licence": image.get("copyright_license"),
                    "url": full.get("url"),
                    "size": full.get("size"),
                }
            )
        url = page.get("next")
    return rows[:limit] if limit else rows


def valid_image(data: bytes, listed_size: int | None) -> bool:
    if listed_size and len(data) < listed_size:
        return False
    try:
        with Image.open(BytesIO(data)) as image:
            image.verify()
        return True
    except Exception:  # noqa: BLE001 - any decoding failure means a bad file
        return False


def download(row: dict, folder: Path) -> str | None:
    target = folder / f"{row['isic_id']}.jpg"
    if target.exists() and valid_image(target.read_bytes(), row["size"]):
        return None
    url = row["url"] or ""
    if not url.startswith("https://"):
        return f"{row['isic_id']}: non-https URL"
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:  # nosec B310: https checked above
                data = response.read()
            if not valid_image(data, row["size"]):
                raise OSError(f"not a valid image ({len(data)} bytes, listed {row['size']})")
            target.write_bytes(data)
            return None
        except OSError as exc:
            if attempt == 3:
                return f"{row['isic_id']}: {exc}"
            time.sleep(2 ** attempt)
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output-dir", default="data/bcn20000", type=Path)
    parser.add_argument("--limit", type=int, default=None, help="only the first N images (for quick checks)")
    parser.add_argument("--metadata-only", action="store_true", help="list and map, but download no images")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    rows = list_images(args.limit)
    frame = pd.DataFrame(rows)
    wrong = frame[(frame["image_type"] != "dermoscopic") | (frame["licence"] != "CC-BY-NC")]
    if len(wrong):
        raise SystemExit(f"{len(wrong)} images are not CC-BY-NC dermoscopy, e.g. {wrong['isic_id'].tolist()[:3]}")

    kept = frame[frame["code"].notna()].copy()
    excluded = frame[frame["code"].isna()]
    summary = {
        "listed": int(len(frame)),
        "kept": int(len(kept)),
        "kept_by_code": {k: int(v) for k, v in kept["code"].value_counts().items()},
        "excluded_by_diagnosis_3": {str(k): int(v) for k, v in excluded["diagnosis_3"].fillna("(none)").value_counts().items()},
        "lesions": int(kept["lesion_id"].nunique()),
        "patients": int(kept["patient_id"].nunique()),
        "histopathology_share": float((kept["confirm_type"] == "histopathology").mean()) if len(kept) else None,
    }
    print(json.dumps(summary, indent=2))

    out = args.output_dir
    (out / "images").mkdir(parents=True, exist_ok=True)
    if not args.metadata_only:
        with ThreadPoolExecutor(args.workers) as pool:
            failures = [f for f in pool.map(lambda r: download(r, out / "images"), kept.to_dict("records")) if f]
        if failures:
            raise SystemExit(f"{len(failures)} downloads failed, e.g. {failures[:3]}")

    kept["img_id"] = kept["isic_id"] + ".jpg"
    kept = kept.rename(columns={"code": "diagnostic"})
    columns = ["img_id", "diagnostic", "lesion_id", "patient_id", "diagnosis_3", "confirm_type"]
    if kept["patient_id"].isna().all():  # not published for BCN20000; group by lesion instead
        columns.remove("patient_id")
    kept[columns].to_csv(
        out / "metadata.csv", index=False
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
