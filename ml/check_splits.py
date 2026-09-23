import argparse

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--val", required=True)
    parser.add_argument("--test", required=True)
    args = parser.parse_args()

    train = pd.read_csv(args.train)
    val = pd.read_csv(args.val)
    test = pd.read_csv(args.test)

    for name, df in [("train", train), ("val", val), ("test", test)]:
        required = {"image_path", "label", "lesion_id"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{name} missing columns: {sorted(missing)}")

    train_lesions = set(train["lesion_id"])
    val_lesions = set(val["lesion_id"])
    test_lesions = set(test["lesion_id"])

    assert train_lesions.isdisjoint(val_lesions), "train/val lesion leakage"
    assert train_lesions.isdisjoint(test_lesions), "train/test lesion leakage"
    assert val_lesions.isdisjoint(test_lesions), "val/test lesion leakage"

    print("No lesion_id leakage detected.")
    print(f"train={len(train)} val={len(val)} test={len(test)}")


if __name__ == "__main__":
    main()
