"""Training script for 5-Class DR Severity Classification on APTOS 2019."""

import argparse
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.datasets.aptos_dataset import APTOSDataset
from src.models.efficientnet import FundusClassifier


def train_dr_grading(
    manifest_path: str = "manifests/dr_grading/aptos_train.csv",
    epochs: int = 10,
    batch_size: int = 8,
    lr: float = 3e-4,
    image_size: int = 512,
    limit: int | None = None,
    output_checkpoint: str = "checkpoints/classification/best_dr_classifier.pth"
):
    print("===================================================================")
    print("[*] FUNDUS AI - Training EfficientNet-B0 DR Grading on APTOS 2019")
    print(f"    Manifest: {manifest_path} | Epochs: {epochs} | Batch size: {batch_size}")
    print("===================================================================\n")

    dataset = APTOSDataset(manifest_path=manifest_path, image_size=image_size, limit=limit)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training Device: {device} | Total training samples: {len(dataset)}")

    model = FundusClassifier(backbone="efficientnet_b0", num_classes=5, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    out_path = Path(output_checkpoint)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        avg_loss = train_loss / max(len(loader), 1)
        acc = (correct / max(total, 1)) * 100.0
        print(f"Epoch [{epoch:02d}/{epochs:02d}] - Loss: {avg_loss:.4f} | Accuracy: {acc:.2f}%")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), str(out_path))
            print(f"  [+] Saved new best DR classifier checkpoint to: {out_path}")

    print(f"\n[OK] APTOS Training complete! Checkpoint saved to: {out_path.resolve()}")


def main():
    parser = argparse.ArgumentParser(description="Train EfficientNet-B0 on APTOS 2019")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output_checkpoint", type=str, default="checkpoints/classification/best_dr_classifier.pth")
    args = parser.parse_args()

    train_dr_grading(epochs=args.epochs, batch_size=args.batch_size, limit=args.limit, output_checkpoint=args.output_checkpoint)


if __name__ == "__main__":
    main()
