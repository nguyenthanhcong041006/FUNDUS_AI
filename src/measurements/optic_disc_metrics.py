"""Quantitative metrics for Optic Disc and Optic Cup with anatomical constraints."""

from typing import Dict, Any, Optional
import cv2
import numpy as np


def calculate_disc_cup_metrics(
    od_mask: np.ndarray,
    oc_mask: np.ndarray
) -> Dict[str, Any]:
    """Calculates anatomical measurements and CDR ratios with clinical fail-safes.

    Args:
        od_mask: Binary mask of optic disc (0 or 255).
        oc_mask: Binary mask of optic cup (0 or 255).

    Returns:
        Dictionary of quantitative measurements. If optic disc is not detected,
        vcdr and related ratios are None (never defaulted to 0.0).
    """
    disc_area = float(np.sum(od_mask > 0))
    cup_area = float(np.sum(oc_mask > 0))

    disc_vert_d, disc_horiz_d = 0.0, 0.0
    cup_vert_d, cup_horiz_d = 0.0, 0.0

    # Disc contours
    od_contours, _ = cv2.findContours(od_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    disc_detected = False
    if od_contours:
        c = max(od_contours, key=cv2.contourArea)
        if cv2.contourArea(c) > 50:  # Minimum valid disc area threshold
            _, _, bw, bh = cv2.boundingRect(c)
            disc_horiz_d = float(bw)
            disc_vert_d = float(bh)
            disc_detected = True

    # Cup contours
    oc_contours, _ = cv2.findContours(oc_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cup_detected = False
    if oc_contours:
        c = max(oc_contours, key=cv2.contourArea)
        if cv2.contourArea(c) > 10:
            _, _, bw, bh = cv2.boundingRect(c)
            cup_horiz_d = float(bw)
            cup_vert_d = float(bh)
            cup_detected = True

    # Anatomical Constraint Check: Cup must reside inside Disc
    anatomical_violation = False
    if cup_detected and disc_detected:
        outside_pixels = float(np.sum((oc_mask > 0) & (od_mask == 0)))
        outside_ratio = outside_pixels / max(cup_area, 1.0)
        if outside_ratio > 0.15 or cup_area > disc_area or cup_vert_d > disc_vert_d:
            anatomical_violation = True

    # Calculate ratios only if disc is reliably detected
    if disc_detected and disc_vert_d > 0 and disc_horiz_d > 0:
        if cup_detected:
            vcdr = min(round(cup_vert_d / disc_vert_d, 3), 1.0)
            hcdr = min(round(cup_horiz_d / disc_horiz_d, 3), 1.0)
            area_ratio = min(round(cup_area / disc_area, 3), 1.0)
        else:
            vcdr = 0.0
            hcdr = 0.0
            area_ratio = 0.0
    else:
        vcdr = None
        hcdr = None
        area_ratio = None

    return {
        "optic_disc": {
            "detected": disc_detected,
            "area_px2": disc_area,
            "horizontal_diameter_px": disc_horiz_d,
            "vertical_diameter_px": disc_vert_d,
        },
        "optic_cup": {
            "detected": cup_detected,
            "area_px2": cup_area,
            "horizontal_diameter_px": cup_horiz_d,
            "vertical_diameter_px": cup_vert_d,
            "vertical_cdr": vcdr,
            "horizontal_cdr": hcdr,
            "cup_disc_area_ratio": area_ratio,
            "anatomical_violation": anatomical_violation,
        },
        "clinical_disclaimer": "Cup-to-disc ratio is an image measurement and must not be used alone to diagnose glaucoma."
    }
