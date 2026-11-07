"""Automated Training Visualization & Analytics Module for FUNDUS AI.

Generates:
  1. Multi-panel Learning Curves (Loss, Dice, IoU, Accuracy, CDR MAE, etc.)
  2. Confusion Matrix Heatmaps for Classification Tasks
  3. Visual Sample Predictions (Original vs Ground Truth vs Model Prediction)
  4. Tabular Training History CSV (history.csv)
"""

from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
import torch

# Use non-interactive backend for server/CLI compatibility
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_history_csv(history: Dict[str, List[Any]], out_path: Path):
    """Saves training epoch history dictionary to CSV."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(history)
    df.to_csv(out_path, index=False)
    print(f"  [+] Saved training history metrics to: {out_path}")


def plot_segmentation_training_results(
    history: Dict[str, List[float]],
    out_dir: Path,
    dataset_name: str,
    task_name: str
) -> Path:
    """Plots comprehensive segmentation learning curves."""
    out_dir.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history["loss"]) + 1)

    # Determine panels needed
    has_cdr = "cdr_mae" in history
    has_class_dices = "od_dice" in history and "cup_dice" in history
    num_plots = 3 if (has_cdr or has_class_dices) else 2

    fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 5))
    if num_plots == 1:
        axes = [axes]

    # Panel 1: Loss Curve
    axes[0].plot(epochs, history["loss"], "b-o", linewidth=2, label="Train Loss")
    if "val_loss" in history:
        axes[0].plot(epochs, history["val_loss"], "r--s", linewidth=2, label="Val Loss")
    axes[0].set_title(f"Loss Curve ({dataset_name})", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Epoch", fontsize=11)
    axes[0].set_ylabel("Loss", fontsize=11)
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend(fontsize=11)

    # Panel 2: Dice & IoU Curves
    dice_key = "test_dice" if "test_dice" in history else ("mean_dice" if "mean_dice" in history else "dice")
    if dice_key in history:
        axes[1].plot(epochs, history[dice_key], "g-o", linewidth=2, label=f"Dice ({max(history[dice_key]):.4f})")
    if "iou" in history or "test_iou" in history:
        iou_key = "test_iou" if "test_iou" in history else "iou"
        axes[1].plot(epochs, history[iou_key], "m--^", linewidth=2, label=f"IoU ({max(history[iou_key]):.4f})")
    if "sensitivity" in history:
        axes[1].plot(epochs, history["sensitivity"], "c:", linewidth=1.5, label="Sensitivity")
    axes[1].set_title(f"Segmentation Performance ({task_name})", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Epoch", fontsize=11)
    axes[1].set_ylabel("Score", fontsize=11)
    axes[1].set_ylim([0, 1.05])
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend(fontsize=11)

    # Panel 3: Domain-specific metrics (CDR MAE for Drishti, OD/Cup separate, etc.)
    if num_plots == 3:
        if has_cdr:
            axes[2].plot(epochs, history["cdr_mae"], "r-o", linewidth=2, label="vCDR MAE Error")
            if "od_dice" in history and "cup_dice" in history:
                axes[2].plot(epochs, history["od_dice"], "g--", label="OD Dice")
                axes[2].plot(epochs, history["cup_dice"], "m--", label="Cup Dice")
            axes[2].set_title("Optic Disc & Cup / vCDR Error", fontsize=13, fontweight="bold")
            axes[2].set_xlabel("Epoch", fontsize=11)
            axes[2].set_ylabel("Metric Value", fontsize=11)
            axes[2].grid(True, linestyle="--", alpha=0.6)
            axes[2].legend(fontsize=11)
        elif has_class_dices:
            axes[2].plot(epochs, history["od_dice"], "g-o", linewidth=2, label="Optic Disc Dice")
            axes[2].plot(epochs, history["cup_dice"], "m-s", linewidth=2, label="Optic Cup Dice")
            axes[2].set_title("Individual Structure Dice", fontsize=13, fontweight="bold")
            axes[2].set_xlabel("Epoch", fontsize=11)
            axes[2].set_ylabel("Dice Score", fontsize=11)
            axes[2].set_ylim([0, 1.05])
            axes[2].grid(True, linestyle="--", alpha=0.6)
            axes[2].legend(fontsize=11)

    plt.tight_layout()
    plot_path = out_dir / f"{dataset_name.lower()}_training_curves.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"  [+] Saved learning curves plot to: {plot_path}")

    # Also save CSV
    save_history_csv(history, out_dir / f"{dataset_name.lower()}_training_history.csv")
    return plot_path


def plot_classification_training_results(
    history: Dict[str, List[float]],
    out_dir: Path,
    dataset_name: str,
    task_name: str,
    confusion_mat: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None
) -> Path:
    """Plots classification learning curves and confusion matrix."""
    out_dir.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history["loss"]) + 1)
    num_plots = 3 if confusion_mat is not None else 2

    fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 5))
    if num_plots == 1:
        axes = [axes]

    # Panel 1: Loss
    axes[0].plot(epochs, history["loss"], "b-o", linewidth=2, label="Train Loss")
    if "val_loss" in history:
        axes[0].plot(epochs, history["val_loss"], "r--s", linewidth=2, label="Val Loss")
    axes[0].set_title(f"Loss Curve ({dataset_name})", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Epoch", fontsize=11)
    axes[0].set_ylabel("Loss", fontsize=11)
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend(fontsize=11)

    # Panel 2: Accuracy or Metric
    acc_key = "accuracy" if "accuracy" in history else ("acc" if "acc" in history else None)
    if acc_key in history and len(history[acc_key]) > 0:
        axes[1].plot(epochs, history[acc_key], "g-o", linewidth=2, label=f"Train Acc ({max(history[acc_key]):.2f}%)")
    if "val_acc" in history and len(history["val_acc"]) > 0:
        axes[1].plot(epochs, history["val_acc"], "y--s", linewidth=2, label=f"Val Acc ({max(history['val_acc']):.2f}%)")
    if "qwk" in history and len(history["qwk"]) > 0:
        axes[1].plot(epochs, history["qwk"], "m--^", linewidth=2, label=f"QWK Kappa ({max(history['qwk']):.4f})")
    if "f1" in history and len(history["f1"]) > 0:
        axes[1].plot(epochs, history["f1"], "c-d", linewidth=2, label=f"F1 Score ({max(history['f1']):.4f})")
    axes[1].set_title(f"Performance ({task_name})", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Epoch", fontsize=11)
    axes[1].set_ylabel("Metric Value", fontsize=11)
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend(fontsize=11)

    # Panel 3: Confusion Matrix
    if confusion_mat is not None and num_plots == 3:
        import sklearn.metrics
        im = axes[2].imshow(confusion_mat, interpolation="nearest", cmap="Blues")
        axes[2].set_title("Confusion Matrix (Final)", fontsize=13, fontweight="bold")
        fig.colorbar(im, ax=axes[2])
        if class_names is not None:
            tick_marks = np.arange(len(class_names))
            axes[2].set_xticks(tick_marks)
            axes[2].set_xticklabels(class_names, rotation=45, ha="right", fontsize=9)
            axes[2].set_yticks(tick_marks)
            axes[2].set_yticklabels(class_names, fontsize=9)

        # Print numbers inside cells
        thresh = confusion_mat.max() / 2.0
        for i in range(confusion_mat.shape[0]):
            for j in range(confusion_mat.shape[1]):
                axes[2].text(j, i, format(confusion_mat[i, j], "d"),
                             ha="center", va="center",
                             color="white" if confusion_mat[i, j] > thresh else "black")
        axes[2].set_ylabel("True Label", fontsize=11)
        axes[2].set_xlabel("Predicted Label", fontsize=11)

    plt.tight_layout()
    plot_path = out_dir / f"{dataset_name.lower()}_training_curves.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"  [+] Saved classification learning curves to: {plot_path}")

    save_history_csv(history, out_dir / f"{dataset_name.lower()}_training_history.csv")
    return plot_path


def plot_sample_predictions(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    out_dir: Path,
    dataset_name: str,
    task_type: str = "vessel",
    num_samples: int = 3
) -> Path:
    """Visualizes model predictions against Ground Truth on sample test images.
    
    Supported task_type: 'vessel', 'optic_disc_cup', 'artery_vein', 'lesions'.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()

    samples_collected = []
    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            targets = batch["mask"].to(device)
            logits = model(images)

            if task_type == "artery_vein":
                # Multi-class: 0=BG, 1=Artery, 2=Vein
                preds = torch.argmax(logits, dim=1).cpu().numpy() # (B, H, W)
                targets_np = targets.cpu().numpy() # (B, H, W)
            else:
                probs = torch.sigmoid(logits).cpu().numpy()
                preds = (probs > 0.5).astype(np.float32)
                targets_np = targets.cpu().numpy()

            for i in range(images.size(0)):
                img_rgb = images[i].cpu().numpy().transpose(1, 2, 0)
                gt = targets_np[i]
                pred = preds[i]
                img_id = batch["image_id"][i] if "image_id" in batch else f"sample_{len(samples_collected)}"
                cdr_ref = batch.get("cdr_ref", [None])[i] if "cdr_ref" in batch else None

                samples_collected.append({
                    "image": np.clip(img_rgb, 0, 1),
                    "gt": gt,
                    "pred": pred,
                    "id": img_id,
                    "cdr_ref": float(cdr_ref) if cdr_ref is not None else None
                })
                if len(samples_collected) >= num_samples:
                    break
            if len(samples_collected) >= num_samples:
                break

    if not samples_collected:
        return out_dir

    num_rows = len(samples_collected)
    fig, axes = plt.subplots(num_rows, 3, figsize=(12, 4 * num_rows))
    if num_rows == 1:
        axes = np.expand_dims(axes, 0)

    for r, s in enumerate(samples_collected):
        # Col 0: Original Image
        axes[r, 0].imshow(s["image"])
        axes[r, 0].set_title(f"Image: {s['id']}", fontsize=11, fontweight="bold")
        axes[r, 0].axis("off")

        # Col 1: Ground Truth
        if task_type == "vessel":
            gt_disp = s["gt"][0] if s["gt"].ndim == 3 else s["gt"]
            axes[r, 1].imshow(gt_disp, cmap="gray")
            axes[r, 1].set_title("Ground Truth Vessel", fontsize=11, fontweight="bold")
        elif task_type == "optic_disc_cup":
            h, w = s["gt"].shape[-2:]
            rgb_gt = np.zeros((h, w, 3), dtype=np.float32)
            rgb_gt[:, :, 1] = s["gt"][0]  # Green: Optic Disc
            rgb_gt[:, :, 2] = s["gt"][1]  # Blue: Optic Cup
            axes[r, 1].imshow(rgb_gt)
            cdr_txt = f" (Ref CDR: {s['cdr_ref']:.2f})" if s["cdr_ref"] and s["cdr_ref"] > 0 else ""
            axes[r, 1].set_title(f"GT: OD (Green) + Cup (Blue){cdr_txt}", fontsize=10, fontweight="bold")
        elif task_type == "artery_vein":
            gt_mask = s["gt"]
            h, w = gt_mask.shape[-2:]
            rgb_gt = np.zeros((h, w, 3), dtype=np.float32)
            rgb_gt[gt_mask == 1, 0] = 1.0  # Red: Artery
            rgb_gt[gt_mask == 2, 2] = 1.0  # Blue: Vein
            axes[r, 1].imshow(rgb_gt)
            axes[r, 1].set_title("GT: Artery (Red) / Vein (Blue)", fontsize=10, fontweight="bold")
        elif task_type == "lesions":
            # 4 channels: 0:MA, 1:HE, 2:EX, 3:SE
            h, w = s["gt"].shape[-2:]
            rgb_gt = np.zeros((h, w, 3), dtype=np.float32)
            rgb_gt[:, :, 0] = np.clip(s["gt"][0] + s["gt"][1], 0, 1)  # Red: MA + HE
            rgb_gt[:, :, 1] = np.clip(s["gt"][2] + s["gt"][3], 0, 1)  # Green: EX + SE
            rgb_gt[:, :, 2] = np.clip(s["gt"][3], 0, 1)               # Blue: SE (making SE cyan)
            axes[r, 1].imshow(rgb_gt)
            axes[r, 1].set_title("GT: Red=HE/MA, Yellow=EX, Cyan=SE", fontsize=10, fontweight="bold")
        else:
            axes[r, 1].imshow(s["gt"][0] if s["gt"].ndim == 3 else s["gt"], cmap="gray")
            axes[r, 1].set_title("Ground Truth", fontsize=11, fontweight="bold")
        axes[r, 1].axis("off")

        # Col 2: Model Prediction
        if task_type == "vessel":
            pred_disp = s["pred"][0] if s["pred"].ndim == 3 else s["pred"]
            axes[r, 2].imshow(pred_disp, cmap="gray")
            axes[r, 2].set_title("Predicted Vessel Mask", fontsize=11, fontweight="bold")
        elif task_type == "optic_disc_cup":
            h, w = s["pred"].shape[-2:]
            rgb_pred = np.zeros((h, w, 3), dtype=np.float32)
            rgb_pred[:, :, 1] = s["pred"][0]  # Green: Optic Disc
            rgb_pred[:, :, 2] = s["pred"][1]  # Blue: Optic Cup
            axes[r, 2].imshow(rgb_pred)
            axes[r, 2].set_title("Predicted: OD (Green) + Cup (Blue)", fontsize=10, fontweight="bold")
        elif task_type == "artery_vein":
            pred_mask = s["pred"]
            h, w = pred_mask.shape[-2:]
            rgb_pred = np.zeros((h, w, 3), dtype=np.float32)
            rgb_pred[pred_mask == 1, 0] = 1.0  # Red: Artery
            rgb_pred[pred_mask == 2, 2] = 1.0  # Blue: Vein
            axes[r, 2].imshow(rgb_pred)
            axes[r, 2].set_title("Pred: Artery (Red) / Vein (Blue)", fontsize=10, fontweight="bold")
        elif task_type == "lesions":
            h, w = s["pred"].shape[-2:]
            rgb_pred = np.zeros((h, w, 3), dtype=np.float32)
            rgb_pred[:, :, 0] = np.clip(s["pred"][0] + s["pred"][1], 0, 1)
            rgb_pred[:, :, 1] = np.clip(s["pred"][2] + s["pred"][3], 0, 1)
            rgb_pred[:, :, 2] = np.clip(s["pred"][3], 0, 1)
            axes[r, 2].imshow(rgb_pred)
            axes[r, 2].set_title("Pred: Red=HE/MA, Yellow=EX, Cyan=SE", fontsize=10, fontweight="bold")
        else:
            axes[r, 2].imshow(s["pred"][0] if s["pred"].ndim == 3 else s["pred"], cmap="gray")
            axes[r, 2].set_title("Predicted Mask", fontsize=11, fontweight="bold")
        axes[r, 2].axis("off")

    plt.tight_layout()
    plot_path = out_dir / f"{dataset_name.lower()}_sample_predictions.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"  [+] Saved visual sample predictions to: {plot_path}")
    return plot_path
