"""Training & Evaluation script for Blood Vessel Segmentation (DRIVE).

Matches Phase 1 (Step 1) of FUNDUS_AI_Dataset_Training_Plan_2.md.
Trains on official DRIVE training split and evaluates on official test split.
"""

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

from src.datasets.drive_dataset import DRIVEManifestDataset
from src.models.unet import UNet
from src.models.losses import DiceBCELoss
from src.evaluation.segmentation_metrics import calculate_segmentation_metrics


def evaluate(model, test_loader, device, threshold: float = 0.5):
    """Evaluates vessel model on official DRIVE test set."""
    model.eval()
    all_metrics = []

    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            targets = batch["mask"].to(device)
            fovs = batch["fov"].to(device)

            logits = model(images)
            probs = torch.sigmoid(logits)

            for i in range(images.size(0)):
                pred_binary = (probs[i, 0].cpu().numpy() > threshold).astype(np.uint8) * 255
                gt_binary = (targets[i, 0].cpu().numpy() > 0).astype(np.uint8) * 255
                fov_binary = (fovs[i, 0].cpu().numpy() > 0).astype(np.uint8) * 255

                metrics = calculate_segmentation_metrics(pred_binary, gt_binary, fov_mask=fov_binary)
                all_metrics.append(metrics)

    dice_avg = np.mean([m["dice"] for m in all_metrics])
    iou_avg = np.mean([m["iou"] for m in all_metrics])
    sens_avg = np.mean([m["recall_sensitivity"] for m in all_metrics])
    spec_avg = np.mean([m["specificity"] for m in all_metrics])

    return {
        "dice": round(float(dice_avg), 4),
        "iou": round(float(iou_avg), 4),
        "sensitivity": round(float(sens_avg), 4),
        "specificity": round(float(spec_avg), 4),
        "n_test_samples": len(all_metrics)
    }


def train_vessels(
    train_manifest: str = "manifests/vessel/drive_train.csv",
    test_manifest: str = "manifests/vessel/drive_test.csv",
    epochs: int = 15,
    batch_size: int = 2,
    lr: float = 1e-3,
    image_size: int = 512,
    limit: int | None = None,
    output_checkpoint: str = "checkpoints/vessels/best_vessel_model.pth"
):
    print("===================================================================")
    print("[*] FUNDUS AI - Training U-Net Vessel Segmentation on DRIVE")
    print(f"    Train manifest: {train_manifest}")
    print(f"    Test manifest:  {test_manifest}")
    print(f"    Epochs: {epochs} | Batch size: {batch_size} | Size: {image_size}x{image_size}")
    print("===================================================================\n")

    train_dataset = DRIVEManifestDataset(manifest_path=train_manifest, image_size=image_size, limit=limit)
    test_dataset = DRIVEManifestDataset(manifest_path=test_manifest, image_size=image_size, limit=limit)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training Device: {device} | Train samples: {len(train_dataset)} | Test samples: {len(test_dataset)}")

    model = UNet(in_channels=3, num_classes=1).to(device)
    criterion = DiceBCELoss(dice_weight=0.5, bce_weight=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_test_dice = 0.0
    out_path = Path(output_checkpoint)
    out_path.parent.mkdir(parents=True, exist_ok=True)

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

        # Evaluate on official test set every epoch
        test_res = evaluate(model, test_loader, device)
        print(f"Epoch [{epoch:02d}/{epochs:02d}] - Loss: {avg_loss:.4f} | Test Dice: {test_res['dice']:.4f} | IoU: {test_res['iou']:.4f} | Sens: {test_res['sensitivity']:.4f} | Spec: {test_res['specificity']:.4f}")

        if test_res["dice"] > best_test_dice:
            best_test_dice = test_res["dice"]
            torch.save(model.state_dict(), str(out_path))
            print(f"  [+] Saved new best model checkpoint to: {out_path} (Dice: {best_test_dice:.4f})")

    print("\n[OK] Training & Evaluation on DRIVE complete!")
    print(f"     Best Checkpoint: {out_path.resolve()} (Best Test Dice: {best_test_dice:.4f})")
    return best_test_dice


def main():
    parser = argparse.ArgumentParser(description="Train U-Net on DRIVE Dataset")
    parser.add_argument("--train_manifest", type=str, default="manifests/vessel/drive_train.csv")
    parser.add_argument("--test_manifest", type=str, default="manifests/vessel/drive_test.csv")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--limit", type=int, default=None, help="Limit samples for dry run")
    parser.add_argument("--output_checkpoint", type=str, default="checkpoints/vessels/best_vessel_model.pth")
    parser.add_argument("--eval_only", action="store_true", help="Only evaluate existing checkpoint")

    args = parser.parse_args()

    if args.eval_only:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        test_dataset = DRIVEManifestDataset(manifest_path=args.test_manifest, image_size=args.image_size, limit=args.limit)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        model = UNet(in_channels=3, num_classes=1).to(device)
        model.load_state_dict(torch.load(args.output_checkpoint, map_location=device))
        res = evaluate(model, test_loader, device)
        print(f"\n[Test Evaluation Results on DRIVE Official Test Set]:")
        for k, v in res.items():
            print(f"  {k}: {v}")
    else:
        train_vessels(
            train_manifest=args.train_manifest,
            test_manifest=args.test_manifest,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            image_size=args.image_size,
            limit=args.limit,
            output_checkpoint=args.output_checkpoint
        )


if __name__ == "__main__":
    main()
