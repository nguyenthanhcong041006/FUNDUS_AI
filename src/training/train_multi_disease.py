"""Training script for Multi-Label Disease Classification on ODIR-5K."""

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

from src.datasets.odir_dataset import ODIRDataset, ODIR_DISEASE_COLUMNS
from src.models.densenet import FundusMultiLabelDenseNet


def train_multi_disease(
    train_manifest: str = "manifests/multi_disease/odir_train.csv",
    val_manifest: str = "manifests/multi_disease/odir_val.csv",
    epochs: int = 10,
    batch_size: int = 8,
    lr: float = 3e-4,
    image_size: int = 512,
    limit: int | None = None,
    output_checkpoint: str = "checkpoints/classification/best_odir_classifier.pth"
):
    print("===================================================================")
    print("[*] FUNDUS AI - Training DenseNet-121 Multi-Disease on ODIR-5K")
    print(f"    Train manifest: {train_manifest} | Val manifest: {val_manifest}")
    print(f"    Target classes (8): {ODIR_DISEASE_COLUMNS}")
    print("===================================================================\n")

    train_dataset = ODIRDataset(manifest_path=train_manifest, image_size=image_size, limit=limit)
    val_dataset = ODIRDataset(manifest_path=val_manifest, image_size=image_size, limit=limit // 4 if limit else None)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training Device: {device} | Train eyes: {len(train_dataset)} | Val eyes: {len(val_dataset)}")

    model = FundusMultiLabelDenseNet(num_classes=8, pretrained=True).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    out_path = Path(output_checkpoint)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            images = batch["image"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / max(len(train_loader), 1)

        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                labels = batch["labels"].to(device)
                logits = model(images)
                loss = criterion(logits, labels)
                val_loss += loss.item()

        avg_val_loss = val_loss / max(len(val_loader), 1)
        print(f"Epoch [{epoch:02d}/{epochs:02d}] - Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), str(out_path))
            print(f"  [+] Saved new best ODIR model checkpoint to: {out_path}")

    print(f"\n[OK] ODIR-5K Multi-Disease Training complete! Checkpoint saved to: {out_path.resolve()}")


def main():
    parser = argparse.ArgumentParser(description="Train DenseNet-121 on ODIR-5K Dataset")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output_checkpoint", type=str, default="checkpoints/classification/best_odir_classifier.pth")
    args = parser.parse_args()

    train_multi_disease(epochs=args.epochs, batch_size=args.batch_size, limit=args.limit, output_checkpoint=args.output_checkpoint)


if __name__ == "__main__":
    main()
