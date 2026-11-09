"""Sliding Window Inference & Gaussian Blending Stitcher (SOTA Phase 3).

Enables dense, high-resolution lesion segmentation over full fundus images
by evaluating overlapping patches and blending them with a 2D Gaussian weight window.
Eliminates boundary discontinuities and grid artifacts.
"""

from typing import Tuple, Dict, Any, List, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
import cv2

from src.datasets.utils import resolve_path
from src.preprocessing.normalization import letterbox_rgb, letterbox_mask
from src.preprocessing.enhancement import ben_graham_equalize
from src.evaluation.segmentation_metrics import calculate_segmentation_metrics


def create_gaussian_weight_window(patch_size: int = 512, sigma_scale: float = 0.25) -> torch.Tensor:
    """Creates a 2D Gaussian weight window of shape (1, 1, patch_size, patch_size).

    Center has weight 1.0, tapering off towards the edges to minimize boundary artifacts.
    """
    sigma = patch_size * sigma_scale
    y = torch.linspace(-patch_size / 2.0, patch_size / 2.0, patch_size)
    x = torch.linspace(-patch_size / 2.0, patch_size / 2.0, patch_size)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    w = torch.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    w = w / w.max()
    return w.float().unsqueeze(0).unsqueeze(0)


def sliding_window_inference(
    model: nn.Module,
    image_tensor: torch.Tensor,
    patch_size: int = 512,
    stride: int = 256,
    batch_size: int = 8,
    device: torch.device = torch.device("cpu")
) -> torch.Tensor:
    """Runs dense sliding window prediction with 2D Gaussian blending over a full image tensor.

    Args:
        model: Evaluated segmentation model.
        image_tensor: Tensor of shape (1, 3, H, W) normalized to [0, 1].
        patch_size: Square patch dimension (e.g. 512).
        stride: Step size between patches (e.g. 256 for 50% overlap).
        batch_size: Mini-batch size for GPU inference.
        device: PyTorch device.

    Returns:
        stitched_probs: Probability map of shape (1, num_classes, H, W).
    """
    model.eval()
    B, C, H, W = image_tensor.shape
    assert B == 1, "Sliding window inference currently processes one full image at a time."

    weight_window = create_gaussian_weight_window(patch_size).to(device)

    # Grid steps
    y_steps = list(range(0, H - patch_size + 1, stride))
    if y_steps[-1] != H - patch_size:
        y_steps.append(H - patch_size)
    x_steps = list(range(0, W - patch_size + 1, stride))
    if x_steps[-1] != W - patch_size:
        x_steps.append(W - patch_size)

    # Collect coordinates
    coords = [(ys, xs) for ys in y_steps for xs in x_steps]

    # Pre-allocate accumulation canvases
    num_classes = 4
    accum_probs = torch.zeros((1, num_classes, H, W), device=device)
    accum_weights = torch.zeros((1, 1, H, W), device=device)

    with torch.no_grad():
        for i in range(0, len(coords), batch_size):
            batch_coords = coords[i:i + batch_size]
            patches = []
            for ys, xs in batch_coords:
                patch = image_tensor[:, :, ys:ys + patch_size, xs:xs + patch_size]
                patches.append(patch)

            batch_patches = torch.cat(patches, dim=0).to(device)  # (N, 3, patch_size, patch_size)
            logits = model(batch_patches)
            probs = torch.sigmoid(logits)

            for j, (ys, xs) in enumerate(batch_coords):
                p_j = probs[j:j+1]  # (1, num_classes, patch_size, patch_size)
                accum_probs[:, :, ys:ys + patch_size, xs:xs + patch_size] += p_j * weight_window
                accum_weights[:, :, ys:ys + patch_size, xs:xs + patch_size] += weight_window

    stitched_probs = accum_probs / torch.clamp(accum_weights, min=1e-6)
    return stitched_probs


def preload_validation_records(
    records: List[Dict[str, Any]],
    base_size: int = 2048,
    equalize_illumination: bool = True
) -> List[Dict[str, Any]]:
    """Pre-processes and caches validation images and masks in RAM to avoid redundant I/O."""
    cached = []
    for rec in tqdm(records, desc="[*] Pre-caching validation images in RAM", leave=False):
        img_path = resolve_path(rec["image_path"])
        if not img_path.is_file():
            continue
        img_orig = Image.open(img_path).convert("RGB")
        img_np_orig = np.array(img_orig, dtype=np.uint8)
        orig_h, orig_w = img_np_orig.shape[:2]

        img_lb, pad_meta = letterbox_rgb(img_np_orig, target_size=base_size)
        gray = cv2.cvtColor(img_lb, cv2.COLOR_RGB2GRAY)
        fov_mask = (gray > 10).astype(np.uint8)

        if equalize_illumination:
            img_processed = ben_graham_equalize(img_lb, sigma=30.0, fov_mask=fov_mask)
        else:
            img_processed = img_lb

        img_tensor = torch.from_numpy(img_processed.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0

        def _load_single_mask(mask_path):
            if not mask_path or pd.isna(mask_path) or str(mask_path).strip() == "":
                return np.zeros((orig_h, orig_w), dtype=np.uint8)
            p = resolve_path(mask_path)
            if not p.is_file():
                return np.zeros((orig_h, orig_w), dtype=np.uint8)
            m = Image.open(p).convert("L")
            return (np.array(m, dtype=np.uint8) > 0).astype(np.uint8)

        ma_raw = _load_single_mask(rec.get("ma_mask_path", ""))
        he_raw = _load_single_mask(rec.get("he_mask_path", ""))
        ex_raw = _load_single_mask(rec.get("ex_mask_path", ""))
        se_raw = _load_single_mask(rec.get("se_mask_path", ""))

        ma_lb = letterbox_mask(ma_raw, target_size=base_size, pad_meta=pad_meta)
        he_lb = letterbox_mask(he_raw, target_size=base_size, pad_meta=pad_meta)
        ex_lb = letterbox_mask(ex_raw, target_size=base_size, pad_meta=pad_meta)
        se_lb = letterbox_mask(se_raw, target_size=base_size, pad_meta=pad_meta)

        cached.append({
            "img_tensor": img_tensor,
            "fov_mask": fov_mask,
            "gts": [ma_lb, he_lb, ex_lb, se_lb],
            "image_id": rec.get("image_id", "")
        })
    return cached


def evaluate_patch_model_full_images(
    model: nn.Module,
    records: List[Dict[str, Any]],
    device: torch.device,
    thresholds: Tuple[float, float, float, float] = (0.60, 0.65, 0.55, 0.55),
    base_size: int = 2048,
    patch_size: int = 512,
    stride: int = 256,
    batch_size: int = 8,
    equalize_illumination: bool = True
) -> Dict[str, float]:
    """Evaluates patch-trained lesion model on full images via dense sliding-window stitching."""
    model.eval()
    channel_dices = [[] for _ in range(4)]

    def _load_single_mask(mask_path, orig_h, orig_w):
        if not mask_path or pd.isna(mask_path) or str(mask_path).strip() == "":
            return np.zeros((orig_h, orig_w), dtype=np.uint8)
        p = resolve_path(mask_path)
        if not p.is_file():
            return np.zeros((orig_h, orig_w), dtype=np.uint8)
        m = Image.open(p).convert("L")
        return (np.array(m, dtype=np.uint8) > 0).astype(np.uint8)

    for rec in tqdm(records, desc="[Sliding Window Eval]", leave=False):
        if "img_tensor" in rec and "fov_mask" in rec and "gts" in rec:
            img_tensor = rec["img_tensor"]
            if img_tensor.device != device:
                img_tensor = img_tensor.to(device)
            fov_mask = rec["fov_mask"]
            gts = rec["gts"]
        else:
            img_path = resolve_path(rec["image_path"])
            if not img_path.is_file():
                continue

            img_orig = Image.open(img_path).convert("RGB")
            img_np_orig = np.array(img_orig, dtype=np.uint8)
            orig_h, orig_w = img_np_orig.shape[:2]

            img_lb, pad_meta = letterbox_rgb(img_np_orig, target_size=base_size)
            gray = cv2.cvtColor(img_lb, cv2.COLOR_RGB2GRAY)
            fov_mask = (gray > 10).astype(np.uint8)

            if equalize_illumination:
                img_processed = ben_graham_equalize(img_lb, sigma=30.0, fov_mask=fov_mask)
            else:
                img_processed = img_lb

            # Form tensor
            img_tensor = torch.from_numpy(img_processed.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
            img_tensor = img_tensor.to(device)

            # Ground truths
            ma_raw = _load_single_mask(rec.get("ma_mask_path", ""), orig_h, orig_w)
            he_raw = _load_single_mask(rec.get("he_mask_path", ""), orig_h, orig_w)
            ex_raw = _load_single_mask(rec.get("ex_mask_path", ""), orig_h, orig_w)
            se_raw = _load_single_mask(rec.get("se_mask_path", ""), orig_h, orig_w)

            ma_lb = letterbox_mask(ma_raw, target_size=base_size, pad_meta=pad_meta)
            he_lb = letterbox_mask(he_raw, target_size=base_size, pad_meta=pad_meta)
            ex_lb = letterbox_mask(ex_raw, target_size=base_size, pad_meta=pad_meta)
            se_lb = letterbox_mask(se_raw, target_size=base_size, pad_meta=pad_meta)
            gts = [ma_lb, he_lb, ex_lb, se_lb]

        # Dense prediction
        probs = sliding_window_inference(
            model=model,
            image_tensor=img_tensor,
            patch_size=patch_size,
            stride=stride,
            batch_size=batch_size,
            device=device
        )
        probs_np = probs[0].cpu().numpy()  # (4, base_size, base_size)

        # Mask outside FOV
        for c in range(4):
            probs_np[c] = probs_np[c] * fov_mask

        # Ground truths
        ma_raw = _load_single_mask(rec.get("ma_mask_path", ""), orig_h, orig_w)
        he_raw = _load_single_mask(rec.get("he_mask_path", ""), orig_h, orig_w)
        ex_raw = _load_single_mask(rec.get("ex_mask_path", ""), orig_h, orig_w)
        se_raw = _load_single_mask(rec.get("se_mask_path", ""), orig_h, orig_w)

        ma_lb = letterbox_mask(ma_raw, target_size=base_size, pad_meta=pad_meta)
        he_lb = letterbox_mask(he_raw, target_size=base_size, pad_meta=pad_meta)
        ex_lb = letterbox_mask(ex_raw, target_size=base_size, pad_meta=pad_meta)
        se_lb = letterbox_mask(se_raw, target_size=base_size, pad_meta=pad_meta)
        gts = [ma_lb, he_lb, ex_lb, se_lb]

        for c in range(4):
            th = thresholds[c]
            pred_c = (probs_np[c] > th).astype(np.uint8) * 255
            gt_c = (gts[c] > 0).astype(np.uint8) * 255

            gt_has_lesion = np.sum(gt_c > 0) > 0
            pred_has_lesion = np.sum(pred_c > 0) > 0

            if gt_has_lesion:
                m = calculate_segmentation_metrics(pred_c, gt_c)
                channel_dices[c].append(m["dice"])
            elif pred_has_lesion:
                channel_dices[c].append(0.0)

    dices_per_class = [float(np.mean(d)) if d else 0.0 for d in channel_dices]
    mean_dice = float(np.mean(dices_per_class))

    return {
        "mean_dice": round(mean_dice, 4),
        "ma_dice": round(dices_per_class[0], 4),
        "he_dice": round(dices_per_class[1], 4),
        "ex_dice": round(dices_per_class[2], 4),
        "se_dice": round(dices_per_class[3], 4),
        "n_samples": len(records)
    }

