"""Cotton wool spot (CWS / soft exudate) detection."""

from typing import Dict, Any, Tuple, List, Optional
import cv2
import numpy as np


class CottonWoolSpotDetector:
    """Detects soft exudates / cotton wool spots (nerve fiber layer infarcts)."""

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path

    def detect(
        self,
        image_bgr: np.ndarray,
        fov_mask: Optional[np.ndarray] = None,
        od_mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, List[Dict[str, Any]], Dict[str, Any]]:
        """Extracts cotton wool spot candidates.

        Args:
            image_bgr: Fundus image.
            fov_mask: Retinal FOV mask.
            od_mask: Optic disc mask to exclude false positives.

        Returns:
            cws_mask: Binary uint8 mask.
            lesions_list: List of individual CWS spots.
            summary: Summary statistics.
        """
        h, w = image_bgr.shape[:2]
        cws_mask = np.zeros((h, w), dtype=np.uint8)

        # CWS are grayish-white fluffy patches with ill-defined borders
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)

        _, thresh = cv2.threshold(tophat, 40, 255, cv2.THRESH_BINARY)

        if od_mask is not None:
            dilated_od = cv2.dilate(od_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)))
            thresh = cv2.bitwise_and(thresh, cv2.bitwise_not(dilated_od))

        if fov_mask is not None:
            thresh = cv2.bitwise_and(thresh, fov_mask)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        lesions_list = []

        for c in contours:
            area = cv2.contourArea(c)
            if 100 < area < 4000:
                cv2.drawContours(cws_mask, [c], -1, 255, -1)
                M = cv2.moments(c)
                cx = int(M["m10"] / (M["m00"] + 1e-6))
                cy = int(M["m01"] / (M["m00"] + 1e-6))
                lesions_list.append({
                    "type": "cotton_wool_spot",
                    "centroid": [cx, cy],
                    "area_px2": float(area),
                    "confidence": 0.82
                })

        total_area = float(np.sum(cws_mask > 0))
        valid_fov = int(np.sum(fov_mask > 0)) if fov_mask is not None else (h * w)

        summary = {
            "type": "cotton_wool_spot",
            "count": len(lesions_list),
            "total_area_px2": total_area,
            "area_ratio_percent": round((total_area / max(valid_fov, 1)) * 100.0, 4)
        }

        return cws_mask, lesions_list, summary

