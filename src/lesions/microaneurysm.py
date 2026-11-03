"""Microaneurysm (MA) detection and candidate analysis."""

from typing import Dict, Any, Tuple, List, Optional
import cv2
import numpy as np


class MicroaneurysmDetector:
    """Detects punctate red lesions (microaneurysms)."""

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path

    def detect(
        self,
        image_bgr: np.ndarray,
        fov_mask: Optional[np.ndarray] = None,
        vessel_mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, List[Dict[str, Any]], Dict[str, Any]]:
        """Extracts microaneurysm candidates and pixel mask.

        Args:
            image_bgr: Fundus image.
            fov_mask: Retinal FOV mask.
            vessel_mask: Vessel mask to exclude thick vessel branches.

        Returns:
            ma_mask: Binary uint8 mask.
            lesions_list: List of individual lesion properties.
            summary: Count, total area, area ratio.
        """
        h, w = image_bgr.shape[:2]
        ma_mask = np.zeros((h, w), dtype=np.uint8)

        # Classical candidate extraction: Green channel bottom-hat for small dark spots
        green = image_bgr[:, :, 1]
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        bothat = cv2.morphologyEx(green, cv2.MORPH_BLACKHAT, kernel)

        # Threshold small dark spots
        _, thresh = cv2.threshold(bothat, 25, 255, cv2.THRESH_BINARY)

        # Exclude major vessels
        if vessel_mask is not None:
            dilated_vessels = cv2.dilate(vessel_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
            thresh = cv2.bitwise_and(thresh, cv2.bitwise_not(dilated_vessels))

        if fov_mask is not None:
            thresh = cv2.bitwise_and(thresh, fov_mask)

        # Filter components by size (MA is typically 2-50 px²)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        lesions_list = []

        for c in contours:
            area = cv2.contourArea(c)
            if 2 <= area <= 60:
                cv2.drawContours(ma_mask, [c], -1, 255, -1)
                M = cv2.moments(c)
                cx = int(M["m10"] / (M["m00"] + 1e-6))
                cy = int(M["m01"] / (M["m00"] + 1e-6))
                lesions_list.append({
                    "type": "microaneurysm",
                    "centroid": [cx, cy],
                    "area_px2": float(area),
                    "confidence": 0.85
                })

        total_area = float(np.sum(ma_mask > 0))
        valid_fov = int(np.sum(fov_mask > 0)) if fov_mask is not None else (h * w)

        summary = {
            "type": "microaneurysm",
            "count": len(lesions_list),
            "total_area_px2": total_area,
            "area_ratio_percent": round((total_area / max(valid_fov, 1)) * 100.0, 4)
        }

        return ma_mask, lesions_list, summary

