import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import SkinLesionDataset
from model import CLASSES, build_model


def make_transforms(train: bool):
    ops = [transforms.Resize((224, 224))]
    if train:
        ops.extend(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.15, contrast=0.15),
            ]
        )
    ops.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225],
            ),
        ]
    )
    return transforms.Compose(ops)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--val-csv", required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--output", default="models/dermalens_efficientnet_b0.pt")
    parser.add_argument("--history", default="models/training_history.json")
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    class_to_idx = {name: i for i, name in enumerate(CLASSES)}

    train_ds = SkinLesionDataset(args.train_csv, class_to_idx, make_transforms(True))
    val_ds = SkinLesionDataset(args.val_csv, class_to_idx, make_transforms(False))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    train_df = pd.read_csv(args.train_csv)
    labels = train_df["label"].map(class_to_idx).to_numpy()
    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(len(CLASSES)),
        y=labels,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(pretrained=True).to(device)

    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val = float("inf")
    history = []

    for epoch in range(args.epochs):
        model.train()
        train_loss_sum = 0.0
        train_correct = 0

        for images, labels_tensor in train_loader:
            images = images.to(device)
            labels_tensor = labels_tensor.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels_tensor)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * images.size(0)
            train_correct += (logits.argmax(1) == labels_tensor).sum().item()

        model.eval()
        val_loss_sum = 0.0
        val_correct = 0

        with torch.inference_mode():
            for images, labels_tensor in val_loader:
                images = images.to(device)
                labels_tensor = labels_tensor.to(device)
                logits = model(images)
                loss = criterion(logits, labels_tensor)

                val_loss_sum += loss.item() * images.size(0)
                val_correct += (logits.argmax(1) == labels_tensor).sum().item()

        scheduler.step()

        train_loss = train_loss_sum / len(train_loader.dataset)
        val_loss = val_loss_sum / len(val_loader.dataset)
        train_acc = train_correct / len(train_loader.dataset)
        val_acc = val_correct / len(val_loader.dataset)

        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        print(json.dumps(row))

        if val_loss < best_val:
            best_val = val_loss
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), output)
            print(f"saved best model to {output}")

        hist_path = Path(args.history)
        hist_path.parent.mkdir(parents=True, exist_ok=True)
        hist_path.write_text(json.dumps(history, indent=2))


if __name__ == "__main__":
    main()
