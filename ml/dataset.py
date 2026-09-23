from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class SkinLesionDataset(Dataset):
    def __init__(self, csv_path, class_to_idx, transform=None):
        self.df = pd.read_csv(csv_path)
        self.class_to_idx = class_to_idx
        self.transform = transform

        required = {"image_path", "label"}
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"CSV missing required columns: {sorted(missing)}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(Path(row["image_path"])).convert("RGB")
        label = self.class_to_idx[row["label"]]

        if self.transform:
            image = self.transform(image)

        return image, label
