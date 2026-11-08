"""Training & Validation script for Artery/Vein 3-Class Segmentation (RITE).

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

from src.datasets.rite_dataset import RITEManifestDataset
from src.datasets.transforms import JointFundusAugmentation
from src.models.unet import UNet
from src.models.losses import MultiClassDiceCrossEntropyLoss
from src.evaluation.segmentation_metrics import calculate_segmentation_metrics
from src.visualization.training_plots import (
    plot_segmentation_training_results,
    plot_sample_predictions,
    save_history_csv
)


def evaluate(model, data_loader, criterion, device):
    """Evaluates 3-class segmentation model computing both loss and per-class dice."""
    model.eval()
    val_loss = 0.0
    artery_dices = []
    vein_dices = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="[Evaluating]", leave=False):
            images = batch["image"].to(device)
            targets = batch["mask"].to(device)

            logits = model(images)
            loss = criterion(logits, targets)
            val_loss += loss.item()

            preds = torch.argmax(logits, dim=1).cpu().numpy()
            gts = targets.cpu().numpy()

            for i in range(images.size(0)):
                # Class 1: Artery
                pred_art = (preds[i] == 1).astype(np.uint8) * 255
                gt_art = (gts[i] == 1).astype(np.uint8) * 255
                m_art = calculate_segmentation_metrics(pred_art, gt_art)
                artery_dices.append(m_art["dice"])

                # Class 2: Vein
                pred_vein = (preds[i] == 2).astype(np.uint8) * 255
                gt_vein = (gts[i] == 2).astype(np.uint8) * 255
                m_vein = calculate_segmentation_metrics(pred_vein, gt_vein)
                vein_dices.append(m_vein["dice"])

    avg_val_loss = val_loss / max(len(data_loader), 1)
    mean_art = float(np.mean(artery_dices)) if artery_dices else 0.0
    mean_vein = float(np.mean(vein_dices)) if vein_dices else 0.0
    mean_dice = (mean_art + mean_vein) / 2.0

    return {
        "val_loss": avg_val_loss,
        "mean_dice": mean_dice,
        "artery_dice": mean_art,
        "vein_dice": mean_vein,
        "n_samples": len(artery_dices)
    }


def train_artery_vein(
    train_manifest: str = "manifests/artery_vein/rite_train.csv",
    test_manifest: str = "manifests/artery_vein/rite_test.csv",
    epochs: int = 15,
    batch_size: int = 2,
    lr: float = 3e-4,
    image_size: int = 512,
    val_split: float = 0.2,
    patience: int = 6,
    limit: int | None = None,
    output_checkpoint: str = "checkpoints/artery_vein/best_av_model.pth",
    plot_dir: str = "outputs/plots/artery_vein"
):
    print("===================================================================")
    print("[*] FUNDUS AI - Training 3-Class Artery/Vein U-Net on RITE")
    print(f"    Train manifest: {train_manifest} | Epochs: {epochs} | Batch size: {batch_size} | LR: {lr}")
    print(f"    Early stopping patience: {patience} epochs")
    print(f"    Plots output:   {plot_dir}")
    print("===================================================================\n")

    aug = JointFundusAugmentation(
        p_flip_h=0.5,
        p_flip_v=0.5,
        p_rotate=0.7,
        max_angle=45.0,
        p_scale=0.5,
        p_jitter=0.6,
        p_clahe=0.5
    )
    full_dataset = RITEManifestDataset(manifest_path=train_manifest, image_size=image_size, transform=aug, limit=limit)
    test_dataset = RITEManifestDataset(manifest_path=test_manifest, image_size=image_size, limit=limit)

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

    model = UNet(in_channels=3, num_classes=3).to(device)
    criterion = MultiClassDiceCrossEntropyLoss(num_classes=3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    out_path = Path(output_checkpoint)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    best_val_dice = 0.0
    best_epoch = 0
    patience_counter = 0

    history = {
        "loss": [],
        "val_loss": [],
        "mean_dice": [],
        "od_dice": [],   # Used by plotter to display Artery Dice
        "cup_dice": []   # Used by plotter to display Vein Dice
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

        val_res = evaluate(model, val_loader, criterion, device)
        print(f"Epoch [{epoch:02d}/{epochs:02d}] - Train Loss: {avg_loss:.4f} | Val Loss: {val_res['val_loss']:.4f} | Val Mean Dice: {val_res['mean_dice']:.4f} (Artery: {val_res['artery_dice']:.4f}, Vein: {val_res['vein_dice']:.4f})")

        history["loss"].append(round(avg_loss, 4))
        history["val_loss"].append(round(val_res["val_loss"], 4))
        history["mean_dice"].append(round(val_res["mean_dice"], 4))
        history["od_dice"].append(round(val_res["artery_dice"], 4))
        history["cup_dice"].append(round(val_res["vein_dice"], 4))

        if val_res["mean_dice"] > best_val_dice:
            best_val_dice = val_res["mean_dice"]
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), str(out_path))
            print(f"  [+] Saved new best model checkpoint to: {out_path} (Val Mean Dice: {best_val_dice:.4f})")
        else:
            patience_counter += 1
            print(f"  [-] Val Mean Dice did not improve. Early stopping counter: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print(f"\n[!] Early stopping triggered at epoch {epoch}. Best epoch was Epoch {best_epoch} with Val Mean Dice: {best_val_dice:.4f}")
                break

    print(f"\n[OK] RITE Training complete! Best Val Mean Dice: {best_val_dice:.4f} (at Epoch {best_epoch})")
    print(f"     Checkpoint saved to: {out_path.resolve()}")

    # Reload best checkpoint for plotting and evaluation
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
        dataset_name="RITE",
        task_name="Artery/Vein Segmentation"
    )
    plot_sample_predictions(
        model=model,
        test_loader=val_loader,
        device=device,
        out_dir=plots_path,
        dataset_name="RITE",
        task_type="artery_vein",
        num_samples=3
    )

    # Isolated Final Test Evaluation
    print("\n[*] Running ISOLATED External Benchmark on RITE Test Split...")
    test_res = evaluate(model, test_loader, criterion, device)
    print(f"  [RITE Official Test Result] Loss: {test_res['val_loss']:.4f} | Mean Dice: {test_res['mean_dice']:.4f} (Artery: {test_res['artery_dice']:.4f}, Vein: {test_res['vein_dice']:.4f})")

    return best_val_dice


def main():
    parser = argparse.ArgumentParser(description="Train 3-Class U-Net on RITE Artery/Vein Segmentation")
    parser.add_argument("--train_manifest", type=str, default="manifests/artery_vein/rite_train.csv")
    parser.add_argument("--test_manifest", type=str, default="manifests/artery_vein/rite_test.csv")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=6, help="Early stopping patience in epochs")
    parser.add_argument("--limit", type=int, default=None, help="Limit samples for dry run")
    parser.add_argument("--output_checkpoint", type=str, default="checkpoints/artery_vein/best_av_model.pth")
    parser.add_argument("--plot_dir", type=str, default="outputs/plots/artery_vein", help="Directory to save training curves and sample plots")
    parser.add_argument("--eval_only", action="store_true", help="Only evaluate existing checkpoint")

    args = parser.parse_args()

    if args.eval_only:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        test_dataset = RITEManifestDataset(manifest_path=args.test_manifest, image_size=args.image_size, limit=args.limit)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        model = UNet(in_channels=3, num_classes=3).to(device)
        model.load_state_dict(torch.load(args.output_checkpoint, map_location=device))
        criterion = MultiClassDiceCrossEntropyLoss(num_classes=3)
        res = evaluate(model, test_loader, criterion, device)
        print(f"\n[Test Evaluation Results on RITE Test Set]:")
        for k, v in res.items():
            print(f"  {k}: {v}")
    else:
        train_artery_vein(
            train_manifest=args.train_manifest,
            test_manifest=args.test_manifest,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            image_size=args.image_size,
            val_split=args.val_split,
            patience=args.patience,
            limit=args.limit,
            output_checkpoint=args.output_checkpoint,
            plot_dir=args.plot_dir
        )


if __name__ == "__main__":
    main()
