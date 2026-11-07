"""Training & Validation script for Blood Vessel Segmentation (DRIVE).

Features:
- Aspect-ratio preserving letterboxing
- Zero data leakage: strict separation of Train, Validation, and Test sets
- Computation of both Train Loss and Val Loss with dual loss curve generation
- Checkpoint selection based on validation performance
- CSV metrics export and best-checkpoint sample prediction rendering
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
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from src.datasets.drive_dataset import DRIVEManifestDataset
from src.datasets.transforms import JointFundusAugmentation
from src.models.unet import UNet
from src.models.losses import DiceBCELoss
from src.evaluation.segmentation_metrics import calculate_segmentation_metrics
from src.visualization.training_plots import (
    plot_segmentation_training_results,
    plot_sample_predictions,
    save_history_csv
)


def evaluate(model, data_loader, criterion, device, threshold: float = 0.5):
    """Evaluates vessel segmentation model computing both loss and anatomical metrics."""
    model.eval()
    val_loss = 0.0
    all_metrics = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="[Evaluating]", leave=False):
            images = batch["image"].to(device)
            targets = batch["mask"].to(device)
            fovs = batch["fov"].to(device)

            logits = model(images)
            loss = criterion(logits, targets)
            val_loss += loss.item()

            probs = torch.sigmoid(logits)

            for i in range(images.size(0)):
                pred_binary = (probs[i, 0].cpu().numpy() > threshold).astype(np.uint8) * 255
                gt_binary = (targets[i, 0].cpu().numpy() > 0).astype(np.uint8) * 255
                fov_binary = (fovs[i, 0].cpu().numpy() > 0).astype(np.uint8) * 255

                metrics = calculate_segmentation_metrics(pred_binary, gt_binary, fov_mask=fov_binary)
                all_metrics.append(metrics)

    avg_val_loss = val_loss / max(len(data_loader), 1)
    dice_avg = np.mean([m["dice"] for m in all_metrics]) if all_metrics else 0.0
    iou_avg = np.mean([m["iou"] for m in all_metrics]) if all_metrics else 0.0
    sens_avg = np.mean([m["recall_sensitivity"] for m in all_metrics]) if all_metrics else 0.0
    spec_avg = np.mean([m["specificity"] for m in all_metrics]) if all_metrics else 0.0

    return {
        "val_loss": avg_val_loss,
        "dice": round(float(dice_avg), 4),
        "iou": round(float(iou_avg), 4),
        "sensitivity": round(float(sens_avg), 4),
        "specificity": round(float(spec_avg), 4),
        "n_samples": len(all_metrics)
    }


def train_vessels(
    train_manifest: str = "manifests/vessel/drive_train.csv",
    test_manifest: str = "manifests/vessel/drive_test.csv",
    epochs: int = 15,
    batch_size: int = 2,
    lr: float = 3e-4,
    image_size: int = 512,
    val_split: float = 0.2,
    limit: int | None = None,
    output_checkpoint: str = "checkpoints/vessels/best_vessel_model.pth",
    plot_dir: str = "outputs/plots/vessels"
):
    print("===================================================================")
    print("[*] FUNDUS AI - Training U-Net Vessel Segmentation on DRIVE")
    print(f"    Train manifest: {train_manifest}")
    print(f"    Test manifest:  {test_manifest}")
    print(f"    Epochs: {epochs} | Batch size: {batch_size} | Size: {image_size}x{image_size} | LR: {lr}")
    print(f"    Plots output:   {plot_dir}")
    print("===================================================================\n")

    aug = JointFundusAugmentation(p_flip_h=0.5, p_flip_v=0.5, p_rotate=0.5, max_angle=20.0)
    full_dataset = DRIVEManifestDataset(manifest_path=train_manifest, image_size=image_size, transform=aug, limit=limit)
    test_dataset = DRIVEManifestDataset(manifest_path=test_manifest, image_size=image_size, limit=limit)

    # Clean Train/Val split with fixed seed to avoid test data snooping
    total_train = len(full_dataset)
    n_val = max(1, int(val_split * total_train))
    n_train = total_train - n_val
    train_dataset, val_dataset = random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training Device: {device} | Train: {n_train} | Val: {n_val} | Isolated Test: {len(test_dataset)}")

    model = UNet(in_channels=3, num_classes=1).to(device)
    criterion = DiceBCELoss(dice_weight=0.5, bce_weight=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_dice = 0.0
    out_path = Path(output_checkpoint)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # History dictionary for tracking metrics and plotting (includes val_loss!)
    history = {
        "loss": [],
        "val_loss": [],
        "test_dice": [],
        "test_iou": [],
        "sensitivity": [],
        "specificity": []
    }

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{epochs:02d} [Train]", leave=False)
        for batch in pbar:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        avg_loss = train_loss / max(len(train_loader), 1)

        # Evaluate strictly on VALIDATION split (never on test split)
        val_res = evaluate(model, val_loader, criterion, device)
        print(f"Epoch [{epoch:02d}/{epochs:02d}] - Train Loss: {avg_loss:.4f} | Val Loss: {val_res['val_loss']:.4f} | Val Dice: {val_res['dice']:.4f} | IoU: {val_res['iou']:.4f} | Sens: {val_res['sensitivity']:.4f}")

        history["loss"].append(round(avg_loss, 4))
        history["val_loss"].append(round(val_res["val_loss"], 4))
        history["test_dice"].append(val_res["dice"])
        history["test_iou"].append(val_res["iou"])
        history["sensitivity"].append(val_res["sensitivity"])
        history["specificity"].append(val_res["specificity"])

        # Checkpoint selection strictly based on validation performance
        if val_res["dice"] > best_val_dice:
            best_val_dice = val_res["dice"]
            torch.save(model.state_dict(), str(out_path))
            print(f"  [+] Saved new best model checkpoint to: {out_path} (Val Dice: {best_val_dice:.4f})")

    print("\n[OK] Training & Validation on DRIVE complete!")
    print(f"     Best Checkpoint: {out_path.resolve()} (Best Val Dice: {best_val_dice:.4f})")

    # Reload best checkpoint for plotting and external evaluation
    if out_path.is_file():
        print(f"[*] Reloading best checkpoint from {out_path} for final evaluation & visual analytics...")
        model.load_state_dict(torch.load(str(out_path), map_location=device))

    # Generate and save all evaluation plots and metrics CSV
    plots_path = Path(plot_dir)
    plots_path.mkdir(parents=True, exist_ok=True)
    save_history_csv(history, plots_path / "history.csv")
    plot_segmentation_training_results(
        history=history,
        out_dir=plots_path,
        dataset_name="DRIVE",
        task_name="Blood Vessel Segmentation"
    )
    plot_sample_predictions(
        model=model,
        test_loader=val_loader,
        device=device,
        out_dir=plots_path,
        dataset_name="DRIVE",
        task_type="vessel",
        num_samples=3
    )

    # Isolated Final Test Evaluation (Only run once at the very end!)
    print("\n[*] Running ISOLATED External Benchmark on Unseen Test Split...")
    test_res = evaluate(model, test_loader, criterion, device)
    print(f"  [DRIVE Official Test Result] Loss: {test_res['val_loss']:.4f} | Dice: {test_res['dice']:.4f} | IoU: {test_res['iou']:.4f} | Sens: {test_res['sensitivity']:.4f} | Spec: {test_res['specificity']:.4f}")

    return best_val_dice


def main():
    parser = argparse.ArgumentParser(description="Train U-Net on DRIVE Blood Vessel Segmentation")
    parser.add_argument("--train_manifest", type=str, default="manifests/vessel/drive_train.csv")
    parser.add_argument("--test_manifest", type=str, default="manifests/vessel/drive_test.csv")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--limit", type=int, default=None, help="Limit samples for dry run")
    parser.add_argument("--output_checkpoint", type=str, default="checkpoints/vessels/best_vessel_model.pth")
    parser.add_argument("--plot_dir", type=str, default="outputs/plots/vessels", help="Directory to save training curves and sample plots")
    parser.add_argument("--eval_only", action="store_true", help="Only evaluate existing checkpoint")

    args = parser.parse_args()

    if args.eval_only:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        test_dataset = DRIVEManifestDataset(manifest_path=args.test_manifest, image_size=args.image_size, limit=args.limit)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        model = UNet(in_channels=3, num_classes=1).to(device)
        model.load_state_dict(torch.load(args.output_checkpoint, map_location=device))
        criterion = DiceBCELoss()
        res = evaluate(model, test_loader, criterion, device)
        print(f"\n[Test Evaluation Results on DRIVE Test Set]:")
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
            val_split=args.val_split,
            limit=args.limit,
            output_checkpoint=args.output_checkpoint,
            plot_dir=args.plot_dir
        )


if __name__ == "__main__":
    main()
