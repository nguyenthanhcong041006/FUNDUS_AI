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
from tqdm import tqdm
from sklearn.metrics import f1_score

from src.datasets.odir_dataset import ODIRDataset, ODIR_DISEASE_COLUMNS
from src.datasets.transforms import ClassificationFundusAugmentation
from src.models.densenet import FundusMultiLabelDenseNet
from src.visualization.training_plots import (
    plot_classification_training_results,
    save_history_csv
)


def evaluate(model, val_loader, criterion, device):
    """Evaluates multi-label model on validation split."""
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="[ODIR Val]", leave=False):
            images = batch["image"].to(device)
            labels = batch["labels"].to(device)

            logits = model(images)
            loss = criterion(logits, labels)
            val_loss += loss.item()

            probs = torch.sigmoid(logits).cpu().numpy()
            targets = labels.cpu().numpy()

            all_preds.extend(probs)
            all_targets.extend(targets)

    avg_loss = val_loss / max(len(val_loader), 1)
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Multi-label Macro F1 Score
    binary_preds = (all_preds > 0.5).astype(np.float32)
    macro_f1 = f1_score(all_targets, binary_preds, average="macro", zero_division=0)

    return {
        "val_loss": avg_loss,
        "f1": float(macro_f1)
    }


def train_multi_disease(
    train_manifest: str = "manifests/multi_disease/odir_train.csv",
    val_manifest: str = "manifests/multi_disease/odir_val.csv",
    epochs: int = 15,
    batch_size: int = 8,
    lr: float = 3e-4,
    image_size: int = 512,
    patience: int = 6,
    use_augmentation: bool = True,
    limit: int | None = None,
    output_checkpoint: str = "checkpoints/classification/best_odir_classifier.pth",
    plot_dir: str = "outputs/plots/multi_disease"
):
    print("===================================================================")
    print("[*] FUNDUS AI - Training DenseNet-121 Multi-Disease on ODIR-5K")
    print(f"    Train manifest: {train_manifest} | Val manifest: {val_manifest}")
    print(f"    Epochs: {epochs} | Batch size: {batch_size} | LR: {lr}")
    print(f"    Early stopping patience: {patience} epochs | Augmentation: {use_augmentation}")
    print(f"    Target classes (8): {ODIR_DISEASE_COLUMNS}")
    print(f"    Plots output:   {plot_dir}")
    print("===================================================================\n")

    train_aug = ClassificationFundusAugmentation() if use_augmentation else None
    train_dataset = ODIRDataset(manifest_path=train_manifest, image_size=image_size, transform=train_aug, limit=limit)
    val_dataset = ODIRDataset(manifest_path=val_manifest, image_size=image_size, transform=None, limit=limit // 4 if limit else None)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training Device: {device} | Train eyes: {len(train_dataset)} | Val eyes: {len(val_dataset)}")

    model = FundusMultiLabelDenseNet(num_classes=8, pretrained=True).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    out_path = Path(output_checkpoint)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    best_f1 = -1.0
    best_epoch = 0
    patience_counter = 0

    history = {
        "loss": [],
        "val_loss": [],
        "f1": []
    }

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{epochs:02d} [Train]", leave=False)
        for batch in pbar:
            images = batch["image"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        avg_train_loss = train_loss / max(len(train_loader), 1)

        val_res = evaluate(model, val_loader, criterion, device)
        avg_val_loss = val_res["val_loss"]
        val_f1 = val_res["f1"]
        print(f"Epoch [{epoch:02d}/{epochs:02d}] - Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Macro F1: {val_f1:.4f}")

        history["loss"].append(round(avg_train_loss, 4))
        history["val_loss"].append(round(avg_val_loss, 4))
        history["f1"].append(round(val_f1, 4))

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), str(out_path))
            print(f"  [+] Saved new best ODIR model checkpoint to: {out_path} (Macro F1: {best_f1:.4f})")
        else:
            patience_counter += 1
            print(f"  [-] Macro F1 did not improve. Early stopping counter: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print(f"\n[!] Early stopping triggered at epoch {epoch}. Best epoch was Epoch {best_epoch} with Macro F1: {best_f1:.4f}")
                break

    print(f"\n[OK] ODIR-5K Multi-Disease Training complete! Best Macro F1: {best_f1:.4f} (at Epoch {best_epoch})")
    print(f"     Checkpoint saved to: {out_path.resolve()}")

    # Reload best checkpoint for final evaluation & plotting
    if out_path.is_file():
        print(f"[*] Reloading best checkpoint from {out_path} for final evaluation & visual analytics...")
        model.load_state_dict(torch.load(str(out_path), map_location=device))

    # Generate and save all evaluation plots and metrics CSV
    print("\n[*] Generating comprehensive training plots & learning curves...")
    plots_path = Path(plot_dir)
    plots_path.mkdir(parents=True, exist_ok=True)
    save_history_csv(history, plots_path / "history.csv")
    plot_classification_training_results(
        history=history,
        out_dir=plots_path,
        dataset_name="ODIR_5K",
        task_name="Multi-Disease Multi-Label Classification"
    )

    return best_f1


def main():
    parser = argparse.ArgumentParser(description="Train DenseNet-121 on ODIR-5K Dataset")
    parser.add_argument("--train_manifest", type=str, default="manifests/multi_disease/odir_train.csv")
    parser.add_argument("--val_manifest", type=str, default="manifests/multi_disease/odir_val.csv")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--no_aug", action="store_true", help="Disable data augmentation")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output_checkpoint", type=str, default="checkpoints/classification/best_odir_classifier.pth")
    parser.add_argument("--plot_dir", type=str, default="outputs/plots/multi_disease")
    args = parser.parse_args()

    train_multi_disease(
        train_manifest=args.train_manifest,
        val_manifest=args.val_manifest,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        image_size=args.image_size,
        patience=args.patience,
        use_augmentation=not args.no_aug,
        limit=args.limit,
        output_checkpoint=args.output_checkpoint,
        plot_dir=args.plot_dir
    )


if __name__ == "__main__":
    main()
