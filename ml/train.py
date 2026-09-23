import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from dataset import SkinLesionDataset
from model import CLASSES, build_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--output", default="models/dermalens_efficientnet_b0.pt")
    args = parser.parse_args()

    class_to_idx = {name: i for i, name in enumerate(CLASSES)}

    train_tf = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225],
            ),
        ]
    )

    dataset = SkinLesionDataset(args.csv, class_to_idx, train_tf)
    val_size = max(1, int(0.2 * len(dataset)))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(pretrained=True).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_val = float("inf")

    for epoch in range(args.epochs):
        model.train()
        running = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running += loss.item() * images.size(0)

        train_loss = running / len(train_loader.dataset)

        model.eval()
        val_running = 0.0
        correct = 0

        with torch.inference_mode():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                loss = criterion(logits, labels)

                val_running += loss.item() * images.size(0)
                correct += (logits.argmax(1) == labels).sum().item()

        val_loss = val_running / len(val_loader.dataset)
        val_acc = correct / len(val_loader.dataset)

        print(
            f"epoch={epoch + 1} "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} "
            f"val_acc={val_acc:.4f}"
        )

        if val_loss < best_val:
            best_val = val_loss
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), output)
            print(f"saved {output}")


if __name__ == "__main__":
    main()
