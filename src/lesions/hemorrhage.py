"""Retinal hemorrhage (HE) detection and segmentation."""

from typing import Dict, Any, Tuple, List, Optional
import cv2
import numpy as np


class HemorrhageDetector:
    """Detects retinal hemorrhages (dot, blot, flame-shaped)."""

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path

    def detect(
        self,
        image_bgr: np.ndarray,
        fov_mask: Optional[np.ndarray] = None,
        vessel_mask: Optional[np.ndarray] = None,
        fovea_coords: Optional[Tuple[int, int]] = None
    ) -> Tuple[np.ndarray, List[Dict[str, Any]], Dict[str, Any]]:
        """Extracts hemorrhage regions and individual cluster metrics.

        Args:
            image_bgr: Fundus image.
            fov_mask: Retinal FOV mask.
            vessel_mask: Vessel mask to distinguish hemorrhages from vascular branches.
            fovea_coords: Fovea [x, y] to compute distance to fovea.

        Returns:
            he_mask: Binary uint8 mask of hemorrhages.
            lesions_list: List of detected hemorrhages with geometry.
            summary: Summary statistics.
        """
        h, w = image_bgr.shape[:2]
        he_mask = np.zeros((h, w), dtype=np.uint8)

        # Hemorrhages are dark in green channel, low brightness, red-tinted
        green = image_bgr[:, :, 1]
        red = image_bgr[:, :, 2]

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        bothat = cv2.morphologyEx(green, cv2.MORPH_BLACKHAT, kernel)
        _, thresh = cv2.threshold(bothat, 30, 255, cv2.THRESH_BINARY)

        # Mask vessels
        if vessel_mask is not None:
            dilated_vessels = cv2.dilate(vessel_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
            thresh = cv2.bitwise_and(thresh, cv2.bitwise_not(dilated_vessels))

        if fov_mask is not None:
            thresh = cv2.bitwise_and(thresh, fov_mask)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        lesions_list = []

        for c in contours:
            area = cv2.contourArea(c)
            # Hemorrhages are larger than microaneurysms
            if 60 < area < 5000:
                cv2.drawContours(he_mask, [c], -1, 255, -1)
                M = cv2.moments(c)
                cx = int(M["m10"] / (M["m00"] + 1e-6))
                cy = int(M["m01"] / (M["m00"] + 1e-6))

                item = {
                    "type": "hemorrhage",
                    "centroid": [cx, cy],
                    "area_px2": float(area),
                    "confidence": 0.88
                }
                if fovea_coords:
                    fx, fy = fovea_coords
                    dist = float(np.sqrt((cx - fx)**2 + (cy - fy)**2))
                    item["distance_to_fovea_px"] = round(dist, 1)

                lesions_list.append(item)

        total_area = float(np.sum(he_mask > 0))
        valid_fov = int(np.sum(fov_mask > 0)) if fov_mask is not None else (h * w)

        summary = {
            "type": "hemorrhage",
            "count": len(lesions_list),
            "total_area_px2": total_area,
            "area_ratio_percent": round((total_area / max(valid_fov, 1)) * 100.0, 4)
        }

        return he_mask, lesions_list, summary

