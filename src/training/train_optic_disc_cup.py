"""Training & Validation script for Optic Disc & Optic Cup Segmentation.

Features:
- Aspect-ratio preserving letterboxing
- Zero data leakage: strict separation of Train, Validation, and Test sets
- Computation of both Train Loss and Val Loss with dual loss curve generation
- Joint Medical Data Augmentation (flips, rotation, scaling, CLAHE, jitter)
- Multi-domain training support (REFUGE2 + Drishti-GS1)
- CosineAnnealingLR scheduling
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

from src.datasets.optic_disc_cup_dataset import (
    DrishtiDataset,
    REFUGE2Dataset,
    PAPILADataset,
    MixedOpticDiscDataset
)
from src.datasets.transforms import JointFundusAugmentation
from src.models.unet import UNet
from src.models.losses import MultiChannelDiceBCELoss
from src.evaluation.segmentation_metrics import calculate_segmentation_metrics
from src.measurements.optic_disc_metrics import calculate_disc_cup_metrics
from src.visualization.training_plots import (
    plot_segmentation_training_results,
    plot_sample_predictions,
    save_history_csv
)


def evaluate(model, data_loader, criterion, device, threshold: float = 0.5):
    """Evaluates 2-channel model computing both loss and anatomical segmentation metrics."""
    model.eval()
    val_loss = 0.0
    od_dices = []
    cup_dices = []
    cdr_errors = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="[Evaluating]", leave=False):
            images = batch["image"].to(device)
            targets = batch["mask"].to(device)
            cdr_refs = batch.get("cdr_ref", torch.tensor([-1.0] * images.size(0))).numpy()

            logits = model(images)
            loss = criterion(logits, targets)
            val_loss += loss.item()

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
                if ref_cdr > 0 and pred_vcdr is not None:
                    cdr_errors.append(abs(pred_vcdr - ref_cdr))

    avg_val_loss = val_loss / max(len(data_loader), 1)
    mean_od_dice = float(np.mean(od_dices)) if od_dices else 0.0
    mean_cup_dice = float(np.mean(cup_dices)) if cup_dices else 0.0
    mean_dice = (mean_od_dice + mean_cup_dice) / 2.0
    mean_cdr_mae = float(np.mean(cdr_errors)) if cdr_errors else 0.0

    return {
        "val_loss": avg_val_loss,
        "mean_dice": mean_dice,
        "od_dice": mean_od_dice,
        "cup_dice": mean_cup_dice,
        "cdr_mae": mean_cdr_mae,
        "num_evaluated": len(od_dices)
    }


def train_optic_disc_cup(
    dataset_type: str = "refuge2",
    train_manifest: str = "manifests/optic_disc_cup/refuge2_train.csv",
    val_manifest: str = "manifests/optic_disc_cup/refuge2_val.csv",
    test_manifest: str | None = None,
    epochs: int = 15,
    batch_size: int = 4,
    lr: float = 3e-4,
    image_size: int = 512,
    limit: int | None = None,
    use_augmentation: bool = True,
    output_checkpoint: str = "checkpoints/optic_disc_cup/best_od_oc_model.pth",
    plot_dir: str = "outputs/plots/optic_disc_cup"
):
    print("===================================================================")
    print(f"[*] FUNDUS AI - Training U-Net Optic Disc & Cup on mode: {dataset_type.upper()}")
    print(f"    Train manifest: {train_manifest}")
    print(f"    Val manifest:   {val_manifest}")
    print(f"    Test manifest:  {test_manifest}")
    print(f"    Epochs: {epochs} | Batch size: {batch_size} | Size: {image_size}x{image_size} | LR: {lr}")
    print(f"    Plots output:   {plot_dir}")
    print("===================================================================\n")

    aug = JointFundusAugmentation() if use_augmentation else None

    # Construct Train and Val datasets based on dataset_type
    if dataset_type == "multi":
        # Multi-domain: Combine REFUGE2 train + Drishti train
        refuge_train = REFUGE2Dataset("manifests/optic_disc_cup/refuge2_train.csv", image_size=image_size, transform=aug, limit=limit)
        drishti_train = DrishtiDataset("manifests/optic_disc_cup/drishti_train.csv", image_size=image_size, transform=aug, limit=limit)
        train_dataset = MixedOpticDiscDataset([refuge_train, drishti_train])

        refuge_val = REFUGE2Dataset("manifests/optic_disc_cup/refuge2_val.csv", image_size=image_size, limit=limit)
        drishti_test = DrishtiDataset("manifests/optic_disc_cup/drishti_test.csv", image_size=image_size, limit=limit)
        val_dataset = MixedOpticDiscDataset([refuge_val, drishti_test])
        test_dataset = PAPILADataset("manifests/optic_disc_cup/papila_external_test.csv", image_size=image_size, limit=limit)
    elif dataset_type == "drishti":
        full_drishti = DrishtiDataset(train_manifest, image_size=image_size, limit=limit)
        n_total = len(full_drishti)
        n_val = max(1, int(0.2 * n_total))
        n_train = n_total - n_val
        train_dataset, val_dataset = random_split(full_drishti, [n_train, n_val], generator=torch.Generator().manual_seed(42))
        test_dataset = DrishtiDataset(test_manifest or "manifests/optic_disc_cup/drishti_test.csv", image_size=image_size, limit=limit)
    else:  # refuge2 default
        train_dataset = REFUGE2Dataset(train_manifest, image_size=image_size, transform=aug, limit=limit)
        val_dataset = REFUGE2Dataset(val_manifest, image_size=image_size, limit=limit)
        test_dataset = PAPILADataset("manifests/optic_disc_cup/papila_external_test.csv", image_size=image_size, limit=limit)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False) if test_dataset else None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training Device: {device} | Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")
    if test_loader:
        print(f"    Isolated External Test samples: {len(test_dataset)}")

    # 2 output classes: Channel 0 = Optic Disc, Channel 1 = Optic Cup
    model = UNet(in_channels=3, num_classes=2).to(device)
    criterion = MultiChannelDiceBCELoss(dice_weight=0.5, bce_weight=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_dice = 0.0
    out_path = Path(output_checkpoint)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # History dictionary for tracking metrics and plotting (includes val_loss!)
    history = {
        "loss": [],
        "val_loss": [],
        "mean_dice": [],
        "od_dice": [],
        "cup_dice": [],
        "cdr_mae": []
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
        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] - Train Loss: {avg_loss:.4f} | Val Loss: {val_res['val_loss']:.4f} "
            f"| Val Mean Dice: {val_res['mean_dice']:.4f} (OD: {val_res['od_dice']:.4f}, Cup: {val_res['cup_dice']:.4f}) "
            f"| CDR MAE: {val_res['cdr_mae']:.4f}"
        )

        history["loss"].append(round(avg_loss, 4))
        history["val_loss"].append(round(val_res["val_loss"], 4))
        history["mean_dice"].append(round(val_res["mean_dice"], 4))
        history["od_dice"].append(round(val_res["od_dice"], 4))
        history["cup_dice"].append(round(val_res["cup_dice"], 4))
        history["cdr_mae"].append(round(val_res["cdr_mae"], 4))

        # Checkpoint selection strictly based on validation performance
        if val_res["mean_dice"] > best_val_dice:
            best_val_dice = val_res["mean_dice"]
            torch.save(model.state_dict(), str(out_path))
            print(f"  [+] Saved new best model checkpoint to: {out_path} (Val Mean Dice: {best_val_dice:.4f})")

    print(f"\n[OK] Training & Validation complete!")
    print(f"     Best Checkpoint saved to: {out_path} (Best Val Mean Dice: {best_val_dice:.4f})")

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
        dataset_name=dataset_type.upper(),
        task_name="Optic Disc & Cup Segmentation"
    )
    plot_sample_predictions(
        model=model,
        test_loader=val_loader,
        device=device,
        out_dir=plots_path,
        dataset_name=dataset_type.upper(),
        task_type="optic_disc_cup",
        num_samples=3
    )

    # Isolated Final Test Evaluation (Only run once at the very end!)
    if test_loader is not None:
        print("\n[*] Running ISOLATED External Benchmark on Unseen Test Split...")
        test_res = evaluate(model, test_loader, criterion, device)
        print(f"  [Benchmark Result] Test Loss: {test_res['val_loss']:.4f} | Mean Dice: {test_res['mean_dice']:.4f} (OD: {test_res['od_dice']:.4f}, Cup: {test_res['cup_dice']:.4f}) | CDR MAE: {test_res['cdr_mae']:.4f}")

    return best_val_dice


def main():
    parser = argparse.ArgumentParser(description="Train U-Net on Optic Disc & Cup (REFUGE2 / Drishti-GS1 / Multi-Domain)")
    parser.add_argument("--dataset", type=str, choices=["refuge2", "drishti", "multi"], default="refuge2", help="Target dataset mode")
    parser.add_argument("--train_manifest", type=str, default="manifests/optic_disc_cup/refuge2_train.csv")
    parser.add_argument("--val_manifest", type=str, default="manifests/optic_disc_cup/refuge2_val.csv")
    parser.add_argument("--test_manifest", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--no_aug", action="store_true", help="Disable data augmentation")
    parser.add_argument("--limit", type=int, default=None, help="Limit samples for dry run")
    parser.add_argument("--output_checkpoint", type=str, default="checkpoints/optic_disc_cup/best_od_oc_model.pth")
    parser.add_argument("--plot_dir", type=str, default="outputs/plots/optic_disc_cup", help="Directory to save training curves and sample plots")
    parser.add_argument("--eval_only", action="store_true", help="Only evaluate existing checkpoint")

    args = parser.parse_args()

    if args.dataset == "drishti":
        args.train_manifest = "manifests/optic_disc_cup/drishti_train.csv"
        args.test_manifest = "manifests/optic_disc_cup/drishti_test.csv"

    if args.eval_only:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        eval_manifest = args.test_manifest or args.val_manifest
        test_dataset = REFUGE2Dataset(eval_manifest, image_size=args.image_size, limit=args.limit) if "refuge" in eval_manifest else DrishtiDataset(eval_manifest, image_size=args.image_size, limit=args.limit)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        model = UNet(in_channels=3, num_classes=2).to(device)
        model.load_state_dict(torch.load(args.output_checkpoint, map_location=device))
        criterion = MultiChannelDiceBCELoss()
        res = evaluate(model, test_loader, criterion, device)
        print(f"\n[Test Evaluation Results on {eval_manifest}]:")
        for k, v in res.items():
            print(f"  {k}: {v}")
    else:
        train_optic_disc_cup(
            dataset_type=args.dataset,
            train_manifest=args.train_manifest,
            val_manifest=args.val_manifest,
            test_manifest=args.test_manifest,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            image_size=args.image_size,
            limit=args.limit,
            use_augmentation=not args.no_aug,
            output_checkpoint=args.output_checkpoint,
            plot_dir=args.plot_dir
        )


if __name__ == "__main__":
    main()
