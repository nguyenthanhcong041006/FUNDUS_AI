"""Quantitative metrics for retinal vascular tree."""

from typing import Dict, Any, Optional
import cv2
import numpy as np


def _morphological_skeleton(binary_mask: np.ndarray) -> np.ndarray:
    """Computes the topological skeleton using morphological thinning."""
    skel = np.zeros(binary_mask.shape, dtype=np.uint8)
    img = (binary_mask > 0).astype(np.uint8) * 255
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    done = False
    while not done:
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()

        zeros = img.size - cv2.countNonZero(img)
        if zeros == img.size:
            done = True

    return skel


def calculate_vessel_metrics(
    vessel_mask: np.ndarray,
    fov_mask: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """Calculates density, total length, branch points, and estimated mean width.

    Args:
        vessel_mask: Binary vessel mask (0 or 255).
        fov_mask: Retinal FOV mask.

    Returns:
        Dictionary of vascular metrics.
    """
    h, w = vessel_mask.shape[:2]

    # Exclude pixels outside FOV
    if fov_mask is not None:
        effective_vessels = cv2.bitwise_and(vessel_mask, fov_mask)
        valid_fov_pixels = int(np.sum(fov_mask > 0))
    else:
        effective_vessels = vessel_mask
        valid_fov_pixels = h * w

    vessel_pixel_count = int(np.sum(effective_vessels > 0))
    density_percent = (vessel_pixel_count / max(valid_fov_pixels, 1)) * 100.0

    # Skeletonize to determine centerline length and branch points
    skeleton = _morphological_skeleton(effective_vessels)
    total_length_px = int(np.sum(skeleton > 0))

    # Mean vessel width = Area / Length
    mean_width_px = round(float(vessel_pixel_count / max(total_length_px, 1)), 2)

    # Branch point estimation: points on skeleton with 3 or more neighboring skeleton pixels
    skel_binary = (skeleton > 0).astype(np.uint8)
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    neighbor_count = cv2.filter2D(skel_binary, -1, kernel)
    branch_points = int(np.sum((skel_binary == 1) & (neighbor_count >= 3)))

    return {
        "vessel_pixel_count": vessel_pixel_count,
        "valid_fov_pixels": valid_fov_pixels,
        "density_percent": round(float(density_percent), 2),
        "total_length_px": total_length_px,
        "estimated_mean_width_px": mean_width_px,
        "branch_point_count": branch_points
    }

