"""Quantitative metrics for retinal lesions."""

from typing import Dict, Any, List, Optional
import cv2
import numpy as np


def calculate_lesion_metrics(
    lesion_mask: np.ndarray,
    lesion_type: str,
    fov_mask: Optional[np.ndarray] = None,
    fovea_coords: Optional[List[int]] = None
) -> Dict[str, Any]:
    """Calculates quantitative metrics for a specific lesion type.

    Args:
        lesion_mask: Binary uint8 mask of the lesion.
        lesion_type: Name/label of the lesion (e.g., 'hard_exudate', 'hemorrhage').
        fov_mask: Retinal FOV mask.
        fovea_coords: [x, y] coordinates of the fovea.

    Returns:
        Dictionary with count, total area, area ratio, and list of individual lesions.
    """
    h, w = lesion_mask.shape[:2]
    valid_fov = int(np.sum(fov_mask > 0)) if fov_mask is not None else (h * w)

    contours, _ = cv2.findContours(lesion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    individual_lesions = []

    for c in contours:
        area = cv2.contourArea(c)
        if area > 1:
            M = cv2.moments(c)
            cx = int(M["m10"] / (M["m00"] + 1e-6))
            cy = int(M["m01"] / (M["m00"] + 1e-6))

            item = {
                "centroid": [cx, cy],
                "area_px2": float(area),
            }
            if fovea_coords:
                fx, fy = fovea_coords
                dist = float(np.sqrt((cx - fx)**2 + (cy - fy)**2))
                item["distance_to_fovea_px"] = round(dist, 1)

            individual_lesions.append(item)

    total_area_px2 = float(np.sum(lesion_mask > 0))
    area_ratio = (total_area_px2 / max(valid_fov, 1)) * 100.0

    return {
        "type": lesion_type,
        "count": len(individual_lesions),
        "total_area_px2": total_area_px2,
        "area_ratio_percent": round(float(area_ratio), 4),
        "individual_lesions": individual_lesions
    }

