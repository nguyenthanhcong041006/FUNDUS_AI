"""Measurement validation: Mean Absolute Error and Bland-Altman analysis."""

from typing import Dict, Any, List
import numpy as np


def calculate_mae(
    predicted: List[float] | np.ndarray,
    reference: List[float] | np.ndarray
) -> float:
    """Computes Mean Absolute Error between predicted and reference measurements."""
    p = np.array(predicted, dtype=np.float64)
    r = np.array(reference, dtype=np.float64)
    return float(np.mean(np.abs(p - r)))


def bland_altman_analysis(
    predicted: List[float] | np.ndarray,
    reference: List[float] | np.ndarray
) -> Dict[str, float]:
    """Calculates Bland-Altman agreement statistics for clinical validation."""
    p = np.array(predicted, dtype=np.float64)
    r = np.array(reference, dtype=np.float64)

    diff = p - r
    mean_diff = float(np.mean(diff))
    std_diff = float(np.std(diff, ddof=1)) if len(diff) > 1 else 0.0

    upper_loa = mean_diff + 1.96 * std_diff
    lower_loa = mean_diff - 1.96 * std_diff

    return {
        "mean_difference_bias": round(mean_diff, 4),
        "std_difference": round(std_diff, 4),
        "upper_limit_of_agreement": round(upper_loa, 4),
        "lower_limit_of_agreement": round(lower_loa, 4),
        "n_samples": len(p)
    }

