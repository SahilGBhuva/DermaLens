import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import SkinLesionDataset
from model import CLASSES, build_model, pick_device, predict_proba_tta

IMAGE_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def make_transforms(train: bool):
    if not train:
        return transforms.Compose(
            [
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(MEAN, STD),
            ]
        )

    # Dermoscopy has no fixed orientation, and capture conditions vary, so the
    # training images are varied in framing, rotation, flips and lighting. The
    # lighting jitter also covers the API's stress-test conditions (brightness
    # and contrast 0.65–1.35, mild blur), which should help top-class stability.
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.6, 1.0), ratio=(0.85, 1.18)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(30),
            transforms.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.2, hue=0.03),
            transforms.RandomApply([transforms.GaussianBlur(5, sigma=(0.1, 1.5))], p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )


def evaluate(model, loader, criterion, device):
    model.eval()
    loss_sum, ys, preds = 0.0, [], []
    with torch.inference_mode():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            loss_sum += criterion(model(images), labels).item() * images.size(0)
            preds.append(predict_proba_tta(model, images).argmax(1).cpu())
            ys.append(labels.cpu())
    y = torch.cat(ys).numpy()
    p = torch.cat(preds).numpy()
    return {
        "loss": loss_sum / len(loader.dataset),
        "accuracy": float((y == p).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y, p)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--val-csv", required=True)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--patience", type=int, default=6, help="stop after this many epochs without a better validation balanced accuracy")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--output", default="models/dermalens_efficientnet_b0.pt")
    parser.add_argument("--history", default="models/training_history.json")
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    class_to_idx = {name: i for i, name in enumerate(CLASSES)}
    train_ds = SkinLesionDataset(args.train_csv, class_to_idx, make_transforms(True))
    val_ds = SkinLesionDataset(args.val_csv, class_to_idx, make_transforms(False))

    device = pick_device()
    print(f"training on {device}", flush=True)
    workers = 4 if device.type == "cuda" else 2
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=workers,
        pin_memory=device.type == "cuda", drop_last=len(train_ds) > args.batch_size,
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=workers)

    labels = pd.read_csv(args.train_csv)["label"].map(class_to_idx).to_numpy()
    present = np.unique(labels)
    weights = np.ones(len(CLASSES))
    weights[present] = compute_class_weight(class_weight="balanced", classes=present, y=labels)

    model = build_model(pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32, device=device),
        label_smoothing=args.label_smoothing,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    # One warm-up epoch, then cosine decay over the rest.
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        [
            torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=1),
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs - 1, 1)),
        ],
        milestones=[1],
    )
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best, stale, history = -1.0, 0, []
    for epoch in range(args.epochs):
        model.train()
        loss_sum, correct = 0.0, 0
        for images, labels_tensor in train_loader:
            images, labels_tensor = images.to(device), labels_tensor.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, labels_tensor)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_sum += loss.item() * images.size(0)
            correct += (logits.argmax(1) == labels_tensor).sum().item()
        scheduler.step()

        seen = len(train_loader) * args.batch_size if train_loader.drop_last else len(train_ds)
        val = evaluate(model, val_loader, criterion, device)
        row = {
            "epoch": epoch + 1,
            "train_loss": loss_sum / seen,
            "train_accuracy": correct / seen,
            "val_loss": val["loss"],
            "val_accuracy": val["accuracy"],
            "val_balanced_accuracy": val["balanced_accuracy"],
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        print(json.dumps(row), flush=True)

        # Select on validation balanced accuracy: every class counts equally,
        # which is what matters for rare but serious classes like melanoma.
        if val["balanced_accuracy"] > best:
            best, stale = val["balanced_accuracy"], 0
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), output)
            print(f"saved best model (val balanced accuracy {best:.3f}) to {output}", flush=True)
        else:
            stale += 1

        hist_path = Path(args.history)
        hist_path.parent.mkdir(parents=True, exist_ok=True)
        hist_path.write_text(json.dumps(history, indent=2))

        if stale >= args.patience:
            print(f"early stop: no improvement for {args.patience} epochs", flush=True)
            break


if __name__ == "__main__":
    main()
