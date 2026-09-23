import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import SkinLesionDataset
from model import CLASSES, build_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
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
    loader = DataLoader(ds, batch_size=args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(pretrained=False).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    confusion = torch.zeros(len(CLASSES), len(CLASSES), dtype=torch.int64)

    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            preds = logits.argmax(1).cpu()

            for y_true, y_pred in zip(labels, preds):
                confusion[y_true, y_pred] += 1

    accuracy = confusion.diag().sum().item() / confusion.sum().item()

    print(f"accuracy={accuracy:.4f}")
    print("confusion_matrix=")
    print(confusion)


if __name__ == "__main__":
    main()
