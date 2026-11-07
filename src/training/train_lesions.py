"""Training & Validation script for Diabetic Retinopathy Lesions (IDRiD).

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

from src.datasets.idrid_dataset import IDRiDLesionDataset
from src.datasets.transforms import JointFundusAugmentation
from src.models.attention_unet import AttentionUNet
from src.models.losses import DiceBCELoss
from src.evaluation.segmentation_metrics import calculate_segmentation_metrics
from src.training.resume_utils import load_training_state, save_training_state
from src.visualization.training_plots import (
    plot_segmentation_training_results,
    plot_sample_predictions,
    save_history_csv
)


def evaluate(model, data_loader, criterion, device, threshold: float = 0.5):
    """Evaluates 4-channel lesion segmentation model computing both loss and mean dice."""
    model.eval()
    val_loss = 0.0
    channel_dices = [[] for _ in range(4)]

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="[Evaluating]", leave=False):
            images = batch["image"].to(device)
            targets = batch["mask"].to(device)

            logits = model(images)
            loss = criterion(logits, targets)
            val_loss += loss.item()

            probs = torch.sigmoid(logits).cpu().numpy()
            gts = targets.cpu().numpy()

            for i in range(images.size(0)):
                for c in range(4):
                    pred_c = (probs[i, c] > threshold).astype(np.uint8) * 255
                    gt_c = (gts[i, c] > 0).astype(np.uint8) * 255
                    m = calculate_segmentation_metrics(pred_c, gt_c)
                    channel_dices[c].append(m["dice"])

    avg_val_loss = val_loss / max(len(data_loader), 1)
    dices_per_class = [float(np.mean(d)) if d else 0.0 for d in channel_dices]
    mean_dice = float(np.mean(dices_per_class))

    return {
        "val_loss": avg_val_loss,
        "mean_dice": round(mean_dice, 4),
        "ma_dice": round(dices_per_class[0], 4),
        "he_dice": round(dices_per_class[1], 4),
        "ex_dice": round(dices_per_class[2], 4),
        "se_dice": round(dices_per_class[3], 4),
        "n_samples": len(channel_dices[0])
    }


def train_lesions(
    train_manifest: str = "manifests/dr_lesions/idrid_lesions_train.csv",
    test_manifest: str = "manifests/dr_lesions/idrid_lesions_test.csv",
    epochs: int = 15,
    batch_size: int = 2,
    lr: float = 3e-4,
    image_size: int = 512,
    val_split: float = 0.2,
    limit: int | None = None,
    output_checkpoint: str = "checkpoints/lesions/best_lesion_model.pth",
    plot_dir: str = "outputs/plots/lesions",
    resume: bool = False
):
    print("===================================================================")
    print("[*] FUNDUS AI - Training Attention U-Net on IDRiD Retinal Lesions")
    print(f"    Train manifest: {train_manifest} | Epochs: {epochs} | Batch size: {batch_size} | LR: {lr}")
    print(f"    Plots output:   {plot_dir}")
    print("===================================================================\n")

    aug = JointFundusAugmentation(p_flip_h=0.5, p_flip_v=0.5, p_rotate=0.5, max_angle=20.0)
    full_dataset = IDRiDLesionDataset(manifest_path=train_manifest, image_size=image_size, transform=aug, limit=limit)
    test_dataset = IDRiDLesionDataset(manifest_path=test_manifest, image_size=image_size, limit=limit)

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

    model = AttentionUNet(in_channels=3, num_classes=4).to(device)
    criterion = DiceBCELoss(dice_weight=0.6, bce_weight=0.4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    out_path = Path(output_checkpoint)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    best_val_dice = 0.0

    history = {
        "loss": [],
        "val_loss": [],
        "mean_dice": []
    }

    start_epoch = 1
    if resume:
        start_epoch, saved = load_training_state(out_path, model, optimizer, scheduler, device, epochs)
        best_val_dice = saved.get("best_val_dice", best_val_dice)
        history = saved.get("history", history)

    for epoch in range(start_epoch, epochs + 1):
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
        print(f"Epoch [{epoch:02d}/{epochs:02d}] - Train Loss: {avg_loss:.4f} | Val Loss: {val_res['val_loss']:.4f} | Val Mean Dice: {val_res['mean_dice']:.4f}")

        history["loss"].append(round(avg_loss, 4))
        history["val_loss"].append(round(val_res["val_loss"], 4))
        history["mean_dice"].append(round(val_res["mean_dice"], 4))

        if val_res["mean_dice"] > best_val_dice:
            best_val_dice = val_res["mean_dice"]
            torch.save(model.state_dict(), str(out_path))
            print(f"  [+] Saved new best lesion checkpoint to: {out_path} (Val Mean Dice: {best_val_dice:.4f})")

        save_training_state(out_path, epoch, model, optimizer, scheduler,
                            best_val_dice=best_val_dice, history=history)

    print(f"\n[OK] IDRiD Lesion Training complete! Best Val Mean Dice: {best_val_dice:.4f}")
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
        dataset_name="IDRiD",
        task_name="Diabetic Retinopathy Lesions"
    )
    plot_sample_predictions(
        model=model,
        test_loader=val_loader,
        device=device,
        out_dir=plots_path,
        dataset_name="IDRiD",
        task_type="lesions",
        num_samples=3
    )

    # Isolated Final Test Evaluation
    print("\n[*] Running ISOLATED External Benchmark on IDRiD Test Split...")
    test_res = evaluate(model, test_loader, criterion, device)
    print(f"  [IDRiD Official Test Result] Loss: {test_res['val_loss']:.4f} | Mean Dice: {test_res['mean_dice']:.4f} (MA: {test_res['ma_dice']:.4f}, HE: {test_res['he_dice']:.4f}, EX: {test_res['ex_dice']:.4f}, SE: {test_res['se_dice']:.4f})")

    return best_val_dice


def main():
    parser = argparse.ArgumentParser(description="Train Attention U-Net on IDRiD Lesions")
    parser.add_argument("--train_manifest", type=str, default="manifests/dr_lesions/idrid_lesions_train.csv")
    parser.add_argument("--test_manifest", type=str, default="manifests/dr_lesions/idrid_lesions_test.csv")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output_checkpoint", type=str, default="checkpoints/lesions/best_lesion_model.pth")
    parser.add_argument("--plot_dir", type=str, default="outputs/plots/lesions")
    parser.add_argument("--resume", action="store_true", help="Continue from <checkpoint>_last.pth saved after the last finished epoch")
    args = parser.parse_args()

    train_lesions(
        train_manifest=args.train_manifest,
        test_manifest=args.test_manifest,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        image_size=args.image_size,
        val_split=args.val_split,
        limit=args.limit,
        output_checkpoint=args.output_checkpoint,
        plot_dir=args.plot_dir,
        resume=args.resume
    )


if __name__ == "__main__":
    main()
