"""Training & Validation script for Optic Disc & Optic Cup Segmentation (Drishti-GS1).

Matches Phase 1 (Step 1) of FUNDUS_AI_Dataset_Training_Plan_2.md.
Trains a 2-channel U-Net on Drishti-GS1 training split and evaluates on official test split.
Outputs:
  Channel 0: Optic Disc (OD)
  Channel 1: Optic Cup (OC)
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

from src.datasets.optic_disc_cup_dataset import DrishtiDataset
from src.models.unet import UNet
from src.models.losses import MultiChannelDiceBCELoss
from src.evaluation.segmentation_metrics import calculate_segmentation_metrics
from src.measurements.optic_disc_metrics import calculate_disc_cup_metrics


def evaluate(model, test_loader, device, threshold: float = 0.5):
    """Evaluates 2-channel model on Drishti-GS1 test split."""
    model.eval()
    od_dices = []
    cup_dices = []
    cdr_errors = []

    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            targets = batch["mask"].to(device)
            cdr_refs = batch["cdr_ref"].numpy()

            logits = model(images)
            probs = torch.sigmoid(logits)

            for i in range(images.size(0)):
                # Channel 0: Optic Disc
                pred_od = (probs[i, 0].cpu().numpy() > threshold).astype(np.uint8) * 255
                gt_od = (targets[i, 0].cpu().numpy() > 0).astype(np.uint8) * 255
                m_od = calculate_segmentation_metrics(pred_od, gt_od)
                od_dices.append(m_od["dice"])

                # Channel 1: Optic Cup
                pred_cup = (probs[i, 1].cpu().numpy() > threshold).astype(np.uint8) * 255
                gt_cup = (targets[i, 1].cpu().numpy() > 0).astype(np.uint8) * 255
                m_cup = calculate_segmentation_metrics(pred_cup, gt_cup)
                cup_dices.append(m_cup["dice"])

                # Calculate vertical CDR from predicted masks
                morph_metrics = calculate_disc_cup_metrics(pred_od, pred_cup)
                pred_vcdr = morph_metrics["optic_cup"]["vertical_cdr"]

                ref_cdr = float(cdr_refs[i])
                if ref_cdr > 0:
                    cdr_errors.append(abs(pred_vcdr - ref_cdr))

    mean_od_dice = float(np.mean(od_dices)) if od_dices else 0.0
    mean_cup_dice = float(np.mean(cup_dices)) if cup_dices else 0.0
    mean_dice = (mean_od_dice + mean_cup_dice) / 2.0
    mean_cdr_mae = float(np.mean(cdr_errors)) if cdr_errors else 0.0

    return {
        "mean_dice": mean_dice,
        "od_dice": mean_od_dice,
        "cup_dice": mean_cup_dice,
        "cdr_mae": mean_cdr_mae,
        "num_evaluated": len(od_dices)
    }


def train_optic_disc_cup(
    train_manifest: str = "manifests/optic_disc_cup/drishti_train.csv",
    test_manifest: str = "manifests/optic_disc_cup/drishti_test.csv",
    epochs: int = 15,
    batch_size: int = 2,
    lr: float = 1e-3,
    image_size: int = 512,
    limit: int | None = None,
    output_checkpoint: str = "checkpoints/optic_disc_cup/best_od_oc_model.pth"
):
    print("===================================================================")
    print("[*] FUNDUS AI - Training U-Net Optic Disc & Cup Segmentation")
    print(f"    Train manifest: {train_manifest}")
    print(f"    Test manifest:  {test_manifest}")
    print(f"    Epochs: {epochs} | Batch size: {batch_size} | Size: {image_size}x{image_size}")
    print("===================================================================\n")

    train_dataset = DrishtiDataset(manifest_path=train_manifest, image_size=image_size, limit=limit)
    test_dataset = DrishtiDataset(manifest_path=test_manifest, image_size=image_size, limit=limit)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training Device: {device} | Train samples: {len(train_dataset)} | Test samples: {len(test_dataset)}")

    # 2 output classes: Channel 0 = Optic Disc, Channel 1 = Optic Cup
    model = UNet(in_channels=3, num_classes=2).to(device)
    criterion = MultiChannelDiceBCELoss(dice_weight=0.5, bce_weight=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_mean_dice = 0.0
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
        print(f"Epoch [{epoch:02d}/{epochs:02d}] - Loss: {avg_loss:.4f} | Test Mean Dice: {test_res['mean_dice']:.4f} (OD: {test_res['od_dice']:.4f}, Cup: {test_res['cup_dice']:.4f}) | CDR MAE: {test_res['cdr_mae']:.4f}")

        if test_res["mean_dice"] > best_mean_dice:
            best_mean_dice = test_res["mean_dice"]
            torch.save(model.state_dict(), str(out_path))
            print(f"  [+] Saved new best model checkpoint to: {out_path} (Mean Dice: {best_mean_dice:.4f})")

    print("\n[OK] Training & Evaluation on Drishti-GS1 complete!")
    print(f"     Best Checkpoint: {out_path} (Best Mean Dice: {best_mean_dice:.4f})")
    return best_mean_dice


def main():
    parser = argparse.ArgumentParser(description="Train U-Net on Drishti-GS1 (Optic Disc & Cup)")
    parser.add_argument("--train_manifest", type=str, default="manifests/optic_disc_cup/drishti_train.csv")
    parser.add_argument("--test_manifest", type=str, default="manifests/optic_disc_cup/drishti_test.csv")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--limit", type=int, default=None, help="Limit samples for dry run")
    parser.add_argument("--output_checkpoint", type=str, default="checkpoints/optic_disc_cup/best_od_oc_model.pth")
    parser.add_argument("--eval_only", action="store_true", help="Only evaluate existing checkpoint")

    args = parser.parse_args()

    if args.eval_only:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        test_dataset = DrishtiDataset(manifest_path=args.test_manifest, image_size=args.image_size, limit=args.limit)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        model = UNet(in_channels=3, num_classes=2).to(device)
        model.load_state_dict(torch.load(args.output_checkpoint, map_location=device))
        res = evaluate(model, test_loader, device)
        print(f"\n[Test Evaluation Results on Drishti-GS1 Official Test Set]:")
        for k, v in res.items():
            print(f"  {k}: {v}")
    else:
        train_optic_disc_cup(
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

