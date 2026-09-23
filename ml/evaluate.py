import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import SkinLesionDataset
from metrics import summarize_metrics
from model import CLASSES, build_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", default="models/evaluation.json")
    args = parser.parse_args()

    tf = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225],
            ),
        ]
    )

    class_to_idx = {name: i for i, name in enumerate(CLASSES)}
    ds = SkinLesionDataset(args.csv, class_to_idx, tf)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(pretrained=False).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    all_y = []
    all_probs = []

    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)

            all_y.extend(labels.numpy().tolist())
            all_probs.append(probs.cpu().numpy())

    metrics = summarize_metrics(
        all_y,
        np.concatenate(all_probs, axis=0),
        CLASSES,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
