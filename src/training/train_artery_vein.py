"""Training script for Artery/Vein 3-Class Segmentation on RITE."""

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

from src.datasets.rite_dataset import RITEManifestDataset
from src.models.unet import UNet
from src.models.losses import MultiClassDiceCrossEntropyLoss


def train_artery_vein(
    train_manifest: str = "manifests/artery_vein/rite_train.csv",
    test_manifest: str = "manifests/artery_vein/rite_test.csv",
    epochs: int = 10,
    batch_size: int = 2,
    lr: float = 1e-3,
    image_size: int = 512,
    limit: int | None = None,
    output_checkpoint: str = "checkpoints/artery_vein/best_av_model.pth"
):
    print("===================================================================")
    print("[*] FUNDUS AI - Training 3-Class Artery/Vein U-Net on RITE")
    print(f"    Train manifest: {train_manifest} | Epochs: {epochs} | Batch size: {batch_size}")
    print("===================================================================\n")

    train_dataset = RITEManifestDataset(manifest_path=train_manifest, image_size=image_size, limit=limit)
    test_dataset = RITEManifestDataset(manifest_path=test_manifest, image_size=image_size, limit=limit)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training Device: {device} | Train samples: {len(train_dataset)} | Test samples: {len(test_dataset)}")

    model = UNet(in_channels=3, num_classes=3).to(device)
    criterion = MultiClassDiceCrossEntropyLoss(num_classes=3)
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
            print(f"  [+] Saved new best model checkpoint to: {out_path}")

    print(f"\n[OK] RITE Training complete! Checkpoint saved to: {out_path.resolve()}")


def main():
    parser = argparse.ArgumentParser(description="Train 3-Class U-Net on RITE Dataset")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output_checkpoint", type=str, default="checkpoints/artery_vein/best_av_model.pth")
    args = parser.parse_args()

    train_artery_vein(epochs=args.epochs, batch_size=args.batch_size, limit=args.limit, output_checkpoint=args.output_checkpoint)


if __name__ == "__main__":
    main()
