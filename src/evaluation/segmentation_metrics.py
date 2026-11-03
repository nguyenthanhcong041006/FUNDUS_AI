"""Segmentation evaluation metrics (Dice, IoU, Precision, Recall, Specificity)."""

from typing import Dict, Any, Optional
import numpy as np


def calculate_segmentation_metrics(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    fov_mask: Optional[np.ndarray] = None,
    smooth: float = 1e-6
) -> Dict[str, float]:
    """Calculates pixel-level overlap and diagnostic metrics.

    Matches Section 16.1 of FUNDUS AI Implementation Plan.

    Args:
        pred_mask: Predicted binary mask (0 or 1/255).
        gt_mask: Ground truth reference binary mask.
        fov_mask: Optional retinal FOV mask to evaluate only within valid retinal area.
        smooth: Epsilon for numerical stability.

    Returns:
        Dictionary of Dice, IoU, Precision, Recall, and Specificity.
    """
    pred_bool = pred_mask > 0
    gt_bool = gt_mask > 0

    if fov_mask is not None:
        valid = fov_mask > 0
        pred_bool = pred_bool & valid
        gt_bool = gt_bool & valid

    tp = np.sum(pred_bool & gt_bool)
    fp = np.sum(pred_bool & (~gt_bool))
    fn = np.sum((~pred_bool) & gt_bool)
    tn = np.sum((~pred_bool) & (~gt_bool))

    dice = (2.0 * tp + smooth) / (2.0 * tp + fp + fn + smooth)
    iou = (tp + smooth) / (tp + fp + fn + smooth)
    precision = (tp + smooth) / (tp + fp + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)
    specificity = (tn + smooth) / (tn + fp + smooth)

    return {
        "dice": round(float(dice), 4),
        "iou": round(float(iou), 4),
        "precision": round(float(precision), 4),
        "recall_sensitivity": round(float(recall), 4),
        "specificity": round(float(specificity), 4),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }

