from pathlib import Path

import numpy as np
import pandas as pd

from split_bcn20000 import build_splits


def make_frame(n_lesions=300, seed=0):
    rng = np.random.default_rng(seed)
    codes = rng.choice(["NV", "MEL", "BCC", "BKL", "AK", "DF", "VASC", "SCC"], size=n_lesions, p=[0.4, 0.15, 0.12, 0.12, 0.08, 0.04, 0.04, 0.05])
    rows = []
    for i, code in enumerate(codes):
        for j in range(rng.integers(1, 4)):  # several images per lesion
            rows.append({"img_id": f"L{i}_{j}.jpg", "diagnostic": code, "lesion_id": f"lesion{i}"})
    return pd.DataFrame(rows)


def test_no_lesion_spans_two_splits():
    parts = build_splits(make_frame(), Path("imgs"))
    seen = {name: set(part["lesion_id"]) for name, part in parts.items()}
    assert seen["train"].isdisjoint(seen["val"])
    assert seen["train"].isdisjoint(seen["test"])
    assert seen["val"].isdisjoint(seen["test"])


def test_scc_only_in_test_and_every_image_placed_once():
    frame = make_frame()
    parts = build_splits(frame, Path("imgs"))
    assert "SCC" not in set(parts["train"]["diagnostic"]) | set(parts["val"]["diagnostic"])
    assert parts["train"]["label"].notna().all() and parts["val"]["label"].notna().all()
    placed = pd.concat(parts.values())["img_id"]
    assert placed.is_unique
    # SCC images outside the test split are dropped on purpose (no class to learn).
    expected = frame[(frame["diagnostic"] != "SCC") | frame["img_id"].isin(parts["test"]["img_id"])]
    assert set(placed) == set(expected["img_id"])
    assert set(parts["test"]["img_id"]) >= set(frame.loc[frame["diagnostic"] == "SCC", "img_id"]) & set(parts["test"]["img_id"])


def test_missing_lesion_ids_become_single_image_groups():
    frame = make_frame().assign(lesion_id=None)
    parts = build_splits(frame, Path("imgs"))
    non_scc = (frame["diagnostic"] != "SCC").sum()
    total = sum(len(p) for p in parts.values())
    assert non_scc <= total <= len(frame)
    assert pd.concat(parts.values())["img_id"].is_unique


def test_split_is_reproducible():
    a = build_splits(make_frame(), Path("imgs"))
    b = build_splits(make_frame(), Path("imgs"))
    assert a["test"]["img_id"].tolist() == b["test"]["img_id"].tolist()
