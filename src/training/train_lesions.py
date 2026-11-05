"""Training script for Multi-Lesion Segmentation on IDRiD."""

import argparse
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.datasets.idrid_dataset import IDRiDLesionDataset
from src.models.attention_unet import AttentionUNet
from src.models.losses import DiceBCELoss


def train_lesions(
    train_manifest: str = "manifests/dr_lesions/idrid_lesions_train.csv",
    test_manifest: str = "manifests/dr_lesions/idrid_lesions_test.csv",
    epochs: int = 15,
    batch_size: int = 2,
    lr: float = 5e-4,
    image_size: int = 512,
    limit: int | None = None,
    output_checkpoint: str = "checkpoints/lesions/best_lesion_model.pth"
):
    print("===================================================================")
    print("[*] FUNDUS AI - Training Attention U-Net Lesion Segmentation on IDRiD")
    print(f"    Train manifest: {train_manifest} | Epochs: {epochs} | Batch size: {batch_size}")
    print("===================================================================\n")

    train_dataset = IDRiDLesionDataset(manifest_path=train_manifest, image_size=image_size, limit=limit)
    test_dataset = IDRiDLesionDataset(manifest_path=test_manifest, image_size=image_size, limit=limit)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training Device: {device} | Train samples: {len(train_dataset)} | Test samples: {len(test_dataset)}")

    model = AttentionUNet(in_channels=3, num_classes=4).to(device)
    criterion = DiceBCELoss(dice_weight=0.6, bce_weight=0.4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    out_path = Path(output_checkpoint)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_loss = train_loss / max(len(train_loader), 1)
        print(f"Epoch [{epoch:02d}/{epochs:02d}] - Train Loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), str(out_path))
            print(f"  [+] Saved new best lesion checkpoint to: {out_path}")

    print(f"\n[OK] IDRiD Lesion Training complete! Checkpoint saved to: {out_path.resolve()}")


def main():
    parser = argparse.ArgumentParser(description="Train Attention U-Net on IDRiD Lesions")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output_checkpoint", type=str, default="checkpoints/lesions/best_lesion_model.pth")
    args = parser.parse_args()

    train_lesions(epochs=args.epochs, batch_size=args.batch_size, limit=args.limit, output_checkpoint=args.output_checkpoint)


if __name__ == "__main__":
    main()
