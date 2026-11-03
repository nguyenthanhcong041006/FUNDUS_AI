"""Quantitative metrics for Optic Disc and Optic Cup."""

from typing import Dict, Any
import cv2
import numpy as np


def calculate_disc_cup_metrics(
    od_mask: np.ndarray,
    oc_mask: np.ndarray
) -> Dict[str, Any]:
    """Calculates anatomical measurements and CDR ratios.

    Args:
        od_mask: Binary mask of optic disc (0 or 255).
        oc_mask: Binary mask of optic cup (0 or 255).

    Returns:
        Dictionary of quantitative measurements.
    """
    disc_area = float(np.sum(od_mask > 0))
    cup_area = float(np.sum(oc_mask > 0))

    disc_vert_d, disc_horiz_d = 0.0, 0.0
    cup_vert_d, cup_horiz_d = 0.0, 0.0

    # Disc contours
    od_contours, _ = cv2.findContours(od_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if od_contours:
        c = max(od_contours, key=cv2.contourArea)
        _, _, bw, bh = cv2.boundingRect(c)
        disc_horiz_d = float(bw)
        disc_vert_d = float(bh)

    # Cup contours
    oc_contours, _ = cv2.findContours(oc_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if oc_contours:
        c = max(oc_contours, key=cv2.contourArea)
        _, _, bw, bh = cv2.boundingRect(c)
        cup_horiz_d = float(bw)
        cup_vert_d = float(bh)

    vcdr = round(cup_vert_d / max(disc_vert_d, 1.0), 3) if disc_vert_d > 0 else 0.0
    hcdr = round(cup_horiz_d / max(disc_horiz_d, 1.0), 3) if disc_horiz_d > 0 else 0.0
    area_ratio = round(cup_area / max(disc_area, 1.0), 3) if disc_area > 0 else 0.0

    return {
        "optic_disc": {
            "area_px2": disc_area,
            "horizontal_diameter_px": disc_horiz_d,
            "vertical_diameter_px": disc_vert_d,
        },
        "optic_cup": {
            "area_px2": cup_area,
            "horizontal_diameter_px": cup_horiz_d,
            "vertical_diameter_px": cup_vert_d,
            "vertical_cdr": min(vcdr, 1.0),
            "horizontal_cdr": min(hcdr, 1.0),
            "cup_disc_area_ratio": min(area_ratio, 1.0),
        },
        "clinical_disclaimer": "Cup-to-disc ratio is an image measurement and must not be used alone to diagnose glaucoma."
    }

