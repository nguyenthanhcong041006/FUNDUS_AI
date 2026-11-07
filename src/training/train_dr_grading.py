"""Training script for 5-Class DR Severity Classification on APTOS 2019.

Features:
- Aspect-ratio preserving letterboxing
- Medical data augmentation (flips, rotations, scaling, CLAHE, jitter)
- CosineAnnealingLR scheduling
- Early stopping based on validation QWK and validation loss to prevent overfitting
- Reloading best checkpoint before generating confusion matrix and metrics CSV
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
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from sklearn.metrics import cohen_kappa_score, confusion_matrix

from src.datasets.aptos_dataset import APTOSDataset
from src.datasets.transforms import ClassificationFundusAugmentation
from src.models.efficientnet import FundusClassifier
from src.training.resume_utils import load_training_state, save_training_state
from src.visualization.training_plots import (
    plot_classification_training_results,
    save_history_csv
)

DR_CLASS_NAMES = ["0: No DR", "1: Mild", "2: Moderate", "3: Severe", "4: Proliferative"]


def evaluate(model, val_loader, criterion, device):
    """Evaluates DR classifier on validation split."""
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="[APTOS Val]", leave=False):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            logits = model(images)
            loss = criterion(logits, labels)
            val_loss += loss.item()

            preds = torch.argmax(logits, dim=1).cpu().numpy()
            targets = labels.cpu().numpy()

            all_preds.extend(preds)
            all_targets.extend(targets)

    avg_loss = val_loss / max(len(val_loader), 1)
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    if len(all_targets) == 0:
        return {
            "val_loss": avg_loss,
            "val_acc": 0.0,
            "qwk": 0.0,
            "confusion_matrix": np.zeros((5, 5), dtype=int)
        }

    acc = (np.sum(all_preds == all_targets) / len(all_targets)) * 100.0
    qwk = cohen_kappa_score(all_targets, all_preds, weights="quadratic")
    cm = confusion_matrix(all_targets, all_preds, labels=[0, 1, 2, 3, 4])

    return {
        "val_loss": avg_loss,
        "val_acc": acc,
        "qwk": float(qwk),
        "confusion_matrix": cm
    }


def train_dr_grading(
    manifest_path: str = "manifests/dr_grading/aptos_train.csv",
    epochs: int = 15,
    batch_size: int = 8,
    lr: float = 3e-4,
    image_size: int = 512,
    val_split: float = 0.2,
    patience: int = 6,
    use_augmentation: bool = True,
    limit: int | None = None,
    output_checkpoint: str = "checkpoints/classification/best_dr_classifier.pth",
    plot_dir: str = "outputs/plots/dr_grading",
    resume: bool = False
):
    print("===================================================================")
    print("[*] FUNDUS AI - Training EfficientNet-B0 DR Grading on APTOS 2019")
    print(f"    Manifest: {manifest_path} | Epochs: {epochs} | Batch size: {batch_size} | LR: {lr}")
    print(f"    Early stopping patience: {patience} epochs | Augmentation: {use_augmentation}")
    print(f"    Plots output: {plot_dir}")
    print("===================================================================\n")

    aug = ClassificationFundusAugmentation() if use_augmentation else None

    # Load separate dataset instances for train (with augmentation) and val (clean)
    train_source = APTOSDataset(manifest_path=manifest_path, image_size=image_size, transform=aug, limit=limit)
    val_source = APTOSDataset(manifest_path=manifest_path, image_size=image_size, transform=None, limit=limit)

    total_samples = len(train_source)
    if total_samples > 1:
        val_size = max(1, int(val_split * total_samples))
        train_size = total_samples - val_size
    else:
        train_size = total_samples
        val_size = 0

    indices = torch.randperm(total_samples, generator=torch.Generator().manual_seed(42)).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_dataset = Subset(train_source, train_indices)
    val_dataset = Subset(val_source, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training Device: {device} | Train samples: {train_size} | Val samples: {val_size}")

    model = FundusClassifier(backbone="efficientnet_b0", num_classes=5, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    out_path = Path(output_checkpoint)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    best_qwk = -1.0
    best_epoch = 0
    patience_counter = 0

    history = {
        "loss": [],
        "val_loss": [],
        "accuracy": [],
        "val_acc": [],
        "qwk": []
    }
    last_cm = None

    start_epoch = 1
    if resume:
        start_epoch, saved = load_training_state(out_path, model, optimizer, scheduler, device, epochs)
        best_qwk = saved.get("best_qwk", best_qwk)
        best_epoch = saved.get("best_epoch", best_epoch)
        patience_counter = saved.get("patience_counter", patience_counter)
        history = saved.get("history", history)

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{epochs:02d} [Train]", leave=False)
        for batch in pbar:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            running_acc = (correct / max(total, 1)) * 100.0
            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{running_acc:.1f}%")

        scheduler.step()
        avg_loss = train_loss / max(len(train_loader), 1)
        train_acc = (correct / max(total, 1)) * 100.0

        val_res = evaluate(model, val_loader, criterion, device)
        last_cm = val_res["confusion_matrix"]
        print(f"Epoch [{epoch:02d}/{epochs:02d}] - Train Loss: {avg_loss:.4f} | Val Loss: {val_res['val_loss']:.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_res['val_acc']:.2f}% | QWK: {val_res['qwk']:.4f}")

        history["loss"].append(round(avg_loss, 4))
        history["val_loss"].append(round(val_res["val_loss"], 4))
        history["accuracy"].append(round(train_acc, 2))
        history["val_acc"].append(round(val_res["val_acc"], 2))
        history["qwk"].append(round(val_res["qwk"], 4))

        # Checkpoint selection & Early Stopping tracking
        if val_res["qwk"] > best_qwk:
            best_qwk = val_res["qwk"]
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), str(out_path))
            print(f"  [+] Saved new best DR classifier checkpoint to: {out_path} (Best QWK: {best_qwk:.4f})")
        else:
            patience_counter += 1
            print(f"  [-] QWK did not improve. Early stopping counter: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print(f"\n[!] Early stopping triggered at epoch {epoch}. Best epoch was Epoch {best_epoch} with QWK: {best_qwk:.4f}")
                save_training_state(out_path, epoch, model, optimizer, scheduler, finished=True,
                                    best_qwk=best_qwk, best_epoch=best_epoch, patience_counter=patience_counter, history=history)
                break

        save_training_state(out_path, epoch, model, optimizer, scheduler,
                            best_qwk=best_qwk, best_epoch=best_epoch, patience_counter=patience_counter, history=history)

    print(f"\n[OK] APTOS Training complete! Best QWK Kappa: {best_qwk:.4f} (at Epoch {best_epoch})")
    print(f"     Checkpoint saved to: {out_path.resolve()}")

    # Reload best checkpoint for plotting and evaluation
    if out_path.is_file():
        print(f"[*] Reloading best checkpoint from {out_path} for final evaluation & visual analytics...")
        model.load_state_dict(torch.load(str(out_path), map_location=device))
        best_val = evaluate(model, val_loader, criterion, device)
        last_cm = best_val["confusion_matrix"]

    # Generate and save all evaluation plots and metrics CSV
    print("\n[*] Generating comprehensive training plots & confusion matrix heatmap...")
    plots_path = Path(plot_dir)
    plots_path.mkdir(parents=True, exist_ok=True)
    save_history_csv(history, plots_path / "history.csv")
    plot_classification_training_results(
        history=history,
        out_dir=plots_path,
        dataset_name="APTOS_2019",
        task_name="Diabetic Retinopathy 5-Class Severity",
        confusion_mat=last_cm,
        class_names=DR_CLASS_NAMES
    )

    return best_qwk


def main():
    parser = argparse.ArgumentParser(description="Train EfficientNet-B0 on APTOS 2019")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=6, help="Early stopping patience in epochs")
    parser.add_argument("--no_aug", action="store_true", help="Disable data augmentation")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output_checkpoint", type=str, default="checkpoints/classification/best_dr_classifier.pth")
    parser.add_argument("--plot_dir", type=str, default="outputs/plots/dr_grading")
    parser.add_argument("--resume", action="store_true", help="Continue from <checkpoint>_last.pth saved after the last finished epoch")
    args = parser.parse_args()

    train_dr_grading(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        image_size=args.image_size,
        val_split=args.val_split,
        patience=args.patience,
        use_augmentation=not args.no_aug,
        limit=args.limit,
        output_checkpoint=args.output_checkpoint,
        plot_dir=args.plot_dir,
        resume=args.resume
    )


if __name__ == "__main__":
    main()
