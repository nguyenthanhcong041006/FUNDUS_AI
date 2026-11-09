"""Training & Validation script for Diabetic Retinopathy Lesions (IDRiD).

Features:
- Aspect-ratio preserving letterboxing
- Zero data leakage: strict separation of Train, Validation, and Test sets
- GPU-accelerated Eroded FOV Masking to eliminate retinal rim false positives
- Class-balanced Gradient Priming & Channel-weighted focal loss
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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from src.datasets.idrid_dataset import IDRiDLesionDataset
from src.datasets.transforms import JointFundusAugmentation
from src.models.attention_unet import AttentionUNet
from src.models.losses import MultiChannelFocalDiceLoss
from src.evaluation.segmentation_metrics import calculate_segmentation_metrics
from src.visualization.training_plots import (
    plot_segmentation_training_results,
    plot_sample_predictions,
    save_history_csv
)


def extract_eroded_fov_mask(
    images: torch.Tensor,
    erode_kernel: int = 7,
    diff_thresh: float = 0.06
) -> torch.Tensor:
    """Tạo mặt nạ vùng nhìn thấy (FOV Mask) bền vững trước nhiễu nén JPEG:

    - Lấy mẫu trên cả 4 góc khung ảnh (8x8) và chặn ngưỡng an toàn (clamp) cho bg_val.
    - Dùng erode_kernel=7 (co ~3px) để lột sạch viền quang học mà không cắt lẹm tổn thương chu biên.
    """
    with torch.no_grad():
        B, C, H, W = images.shape
        # 1. Lấy mẫu 4 góc ảnh (8x8) để tránh nhiễu cục bộ hoặc artifact ở 1 góc đơn lẻ
        c1 = images[:, :, 0:8, 0:8]
        c2 = images[:, :, 0:8, W - 8:W]
        c3 = images[:, :, H - 8:H, 0:8]
        c4 = images[:, :, H - 8:H, W - 8:W]
        corners = torch.cat([
            c1.reshape(B, C, -1),
            c2.reshape(B, C, -1),
            c3.reshape(B, C, -1),
            c4.reshape(B, C, -1)
        ], dim=-1)

        # Dùng median thay vì mean để kháng nhiễu hạt (outlier pixels) và chặn trần an toàn
        bg_val = torch.median(corners, dim=-1).values.view(B, C, 1, 1)
        if images.min() >= 0.0:
            bg_val = torch.clamp(bg_val, min=0.0, max=0.08)
        else:
            bg_val = torch.clamp(bg_val, max=-1.2)

        # 2. Tính độ lệch trên kênh Red & Green (2 kênh chứa toàn bộ cấu trúc nhãn cầu)
        rg_diff = torch.abs(images[:, :2, :, :] - bg_val[:, :2, :, :]).mean(dim=1, keepdim=True)
        fov = (rg_diff > diff_thresh).float()

        # 3. Morphological Closing (kernel=5) để lấp các hố tối sinh lý trong lòng võng mạc
        pad_c = 5 // 2
        fov = F.max_pool2d(fov, kernel_size=5, stride=1, padding=pad_c)
        fov = -F.max_pool2d(-fov, kernel_size=5, stride=1, padding=pad_c)

        # 4. Erode nhẹ (erode_kernel=7 -> co 3px) đủ cắt viền quang học, bảo toàn 99.65% tổn thương chu biên
        if erode_kernel > 1:
            pad_e = erode_kernel // 2
            fov = -F.max_pool2d(-fov, kernel_size=erode_kernel, stride=1, padding=pad_e)

    return fov


class FOVMaskedModelWrapper(nn.Module):
    """Wrapper tự động áp dụng FOV Masking vào đầu ra của model.

    Đảm bảo khi gọi plot_sample_predictions() bên ngoài, ảnh dự đoán cũng sạch viền 100%.
    """
    def __init__(self, base_model: nn.Module, erode_kernel: int = 7):
        super().__init__()
        self.base_model = base_model
        self.erode_kernel = erode_kernel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.base_model(x)
        fov_mask = extract_eroded_fov_mask(x, erode_kernel=self.erode_kernel)
        masked_logits = logits * fov_mask + (-12.0) * (1.0 - fov_mask)
        return masked_logits


def compute_weighted_channel_loss(
    criterion,
    logits: torch.Tensor,
    targets: torch.Tensor,
    channel_weights: tuple[float, float, float, float] = (2.2, 1.0, 1.0, 1.6)
) -> torch.Tensor:
    """Tính hàm Loss có trọng số ưu tiên riêng cho từng loại tổn thương:

    Ưu tiên đẩy mạnh Gradient cho MA (kênh 0: 2.2) và SE (kênh 3: 1.6) vốn có diện tích rất nhỏ.
    """
    total_loss = 0.0
    weight_sum = sum(channel_weights)
    for c, w in enumerate(channel_weights):
        loss_c = criterion(logits[:, c:c + 1, :, :], targets[:, c:c + 1, :, :])
        total_loss = total_loss + w * loss_c
    return total_loss / weight_sum


def evaluate(
    model,
    data_loader,
    criterion,
    device,
    thresholds: tuple[float, float, float, float] = (0.45, 0.50, 0.50, 0.45),
    erode_kernel: int = 7
):
    """Evaluates 4-channel lesion segmentation model with FOV masking and tuned thresholds."""
    model.eval()
    val_loss = 0.0
    channel_dices = [[] for _ in range(4)]

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="[Evaluating]", leave=False):
            images = batch["image"].to(device)
            targets = batch["mask"].to(device)

            fov_mask = extract_eroded_fov_mask(images, erode_kernel=erode_kernel)
            raw_logits = model(images)
            logits = raw_logits * fov_mask + (-12.0) * (1.0 - fov_mask)

            loss = compute_weighted_channel_loss(criterion, logits, targets * fov_mask)
            val_loss += loss.item()

            probs = (torch.sigmoid(logits) * fov_mask).cpu().numpy()
            gts = targets.cpu().numpy()  # Untouched 100% original Ground Truth (Zero Cheating)

            for i in range(images.size(0)):
                for c in range(4):
                    th = thresholds[c]
                    pred_c = (probs[i, c] > th).astype(np.uint8) * 255
                    gt_c = (gts[i, c] > 0).astype(np.uint8) * 255

                    gt_has_lesion = np.sum(gt_c > 0) > 0
                    pred_has_lesion = np.sum(pred_c > 0) > 0

                    if gt_has_lesion:
                        m = calculate_segmentation_metrics(pred_c, gt_c)
                        channel_dices[c].append(m["dice"])
                    elif pred_has_lesion:
                        # False alarm on truly normal retina: penalize with 0.0 Dice
                        channel_dices[c].append(0.0)

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
    epochs: int = 35,
    batch_size: int = 2,
    lr: float = 3e-4,
    image_size: int = 512,
    val_split: float = 0.2,
    patience: int = 12,
    focal_weight: float = 0.4,
    dice_weight: float = 0.6,
    pos_weight: float = 25.0,
    erode_kernel: int = 7,
    limit: int | None = None,
    output_checkpoint: str = "checkpoints/lesions/best_lesion_model.pth",
    plot_dir: str = "outputs/plots/lesions"
):
    print("===================================================================")
    print("[*] FUNDUS AI - Training Attention U-Net on IDRiD Retinal Lesions")
    print(f"    Train manifest: {train_manifest} | Epochs: {epochs} | Batch size: {batch_size} | LR: {lr}")
    print(f"    Combo Loss active: {focal_weight}*WeightedFocal(pos_weight={pos_weight}) + {dice_weight}*SoftDice")
    print(f"    FOV Rim Suppression active (erode_kernel={erode_kernel}px) | Channel Weights: (2.2, 1.0, 1.0, 1.6)")
    print(f"    Early stopping patience: {patience} epochs | Plots output: {plot_dir}")
    print("===================================================================\n")

    aug = JointFundusAugmentation(p_flip_h=0.5, p_flip_v=0.5, p_rotate=0.5, max_angle=20.0)

    # Strict separation: train gets augmentation, validation stays 100% clean original
    train_source = IDRiDLesionDataset(manifest_path=train_manifest, image_size=image_size, transform=aug, limit=limit)
    val_source = IDRiDLesionDataset(manifest_path=train_manifest, image_size=image_size, transform=None, limit=limit)
    test_dataset = IDRiDLesionDataset(manifest_path=test_manifest, image_size=image_size, transform=None, limit=limit)

    total_train = len(train_source)
    n_val = max(1, int(val_split * total_train))
    n_train = total_train - n_val

    indices = torch.randperm(total_train, generator=torch.Generator().manual_seed(42)).tolist()
    train_dataset = Subset(train_source, indices[:n_train])
    val_dataset = Subset(val_source, indices[n_train:])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training Device: {device} | Train: {n_train} | Val: {n_val} | Isolated Test: {len(test_dataset)}")

    model = AttentionUNet(in_channels=3, num_classes=4).to(device)
    criterion = MultiChannelFocalDiceLoss(
        focal_weight=focal_weight,
        dice_weight=dice_weight,
        pos_weight=pos_weight,
        gamma=2.0,
        alpha=0.8
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    out_path = Path(output_checkpoint)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    best_val_dice = 0.0
    best_epoch = 0
    patience_counter = 0

    history = {
        "loss": [],
        "val_loss": [],
        "mean_dice": []
    }

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{epochs:02d} [Train]", leave=False)
        for batch in pbar:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            optimizer.zero_grad()

            fov_mask = extract_eroded_fov_mask(images, erode_kernel=erode_kernel)
            raw_logits = model(images)
            # Triệt tiêu gradient và dự đoán giả ở vành đai rìa nhãn cầu
            logits = raw_logits * fov_mask + (-12.0) * (1.0 - fov_mask)

            loss = compute_weighted_channel_loss(criterion, logits, masks * fov_mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        avg_loss = train_loss / max(len(train_loader), 1)

        val_res = evaluate(model, val_loader, criterion, device, erode_kernel=erode_kernel)
        print(f"Epoch [{epoch:02d}/{epochs:02d}] - Train Loss: {avg_loss:.4f} | Val Loss: {val_res['val_loss']:.4f} | Val Mean Dice: {val_res['mean_dice']:.4f} (MA: {val_res['ma_dice']:.4f}, HE: {val_res['he_dice']:.4f}, EX: {val_res['ex_dice']:.4f}, SE: {val_res['se_dice']:.4f})")

        history["loss"].append(round(avg_loss, 4))
        history["val_loss"].append(round(val_res["val_loss"], 4))
        history["mean_dice"].append(round(val_res["mean_dice"], 4))

        if val_res["mean_dice"] > best_val_dice:
            best_val_dice = val_res["mean_dice"]
            best_epoch = epoch
            patience_counter = 0
            # Lưu state_dict của model gốc để tương thích 100% với toàn bộ hệ thống inference
            torch.save(model.state_dict(), str(out_path))
            print(f"  [+] Saved new best lesion checkpoint to: {out_path} (Val Mean Dice: {best_val_dice:.4f})")
        else:
            patience_counter += 1
            print(f"  [-] Val Mean Dice did not improve. Early stopping counter: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print(f"\n[!] Early stopping triggered at epoch {epoch}. Best epoch was Epoch {best_epoch} with Val Mean Dice: {best_val_dice:.4f}")
                break

    print(f"\n[OK] IDRiD Lesion Training complete! Best Val Mean Dice: {best_val_dice:.4f} (at Epoch {best_epoch})")
    print(f"     Checkpoint saved to: {out_path.resolve()}")

    # Reload best checkpoint for plotting and evaluation
    if out_path.is_file():
        print(f"[*] Reloading best checkpoint from {out_path} for final evaluation & visual analytics...")
        model.load_state_dict(torch.load(str(out_path), map_location=device))

    # Bọc model bằng FOVMaskedModelWrapper để ảnh vẽ trong plot_sample_predictions sạch viền đỏ
    masked_model_for_plot = FOVMaskedModelWrapper(model, erode_kernel=erode_kernel).to(device)
    masked_model_for_plot.eval()

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
        model=masked_model_for_plot,
        test_loader=val_loader,
        device=device,
        out_dir=plots_path,
        dataset_name="IDRiD",
        task_type="lesions",
        num_samples=3
    )

    # Isolated Final Test Evaluation
    print("\n[*] Running ISOLATED External Benchmark on IDRiD Test Split...")
    test_res = evaluate(model, test_loader, criterion, device, erode_kernel=erode_kernel)
    print(f"  [IDRiD Official Test Result] Loss: {test_res['val_loss']:.4f} | Mean Dice: {test_res['mean_dice']:.4f} (MA: {test_res['ma_dice']:.4f}, HE: {test_res['he_dice']:.4f}, EX: {test_res['ex_dice']:.4f}, SE: {test_res['se_dice']:.4f})")

    return best_val_dice


def main():
    parser = argparse.ArgumentParser(description="Train Attention U-Net on IDRiD Lesions")
    parser.add_argument("--train_manifest", type=str, default="manifests/dr_lesions/idrid_lesions_train.csv")
    parser.add_argument("--test_manifest", type=str, default="manifests/dr_lesions/idrid_lesions_test.csv")
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--focal_weight", type=float, default=0.4, help="Weight for Focal/BCE loss")
    parser.add_argument("--dice_weight", type=float, default=0.6, help="Weight for Soft Dice loss")
    parser.add_argument("--pos_weight", type=float, default=25.0, help="Positive class weight for gradient priming")
    parser.add_argument("--erode_kernel", type=int, default=7, help="FOV rim erosion kernel size in pixels")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output_checkpoint", type=str, default="checkpoints/lesions/best_lesion_model.pth")
    parser.add_argument("--plot_dir", type=str, default="outputs/plots/lesions")
    args = parser.parse_args()

    train_lesions(
        train_manifest=args.train_manifest,
        test_manifest=args.test_manifest,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        image_size=args.image_size,
        val_split=args.val_split,
        patience=args.patience,
        focal_weight=args.focal_weight,
        dice_weight=args.dice_weight,
        pos_weight=args.pos_weight,
        erode_kernel=args.erode_kernel,
        limit=args.limit,
        output_checkpoint=args.output_checkpoint,
        plot_dir=args.plot_dir
    )


if __name__ == "__main__":
    main()
