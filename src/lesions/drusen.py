"""Drusen detection for Age-related Macular Degeneration (AMD) assessment."""

from typing import Dict, Any, Tuple, List, Optional
import cv2
import numpy as np


class DrusenDetector:
    """Detects sub-retinal drusen deposits in the macula."""

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path

    def detect(
        self,
        image_bgr: np.ndarray,
        macula_mask: Optional[np.ndarray] = None,
        od_mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, List[Dict[str, Any]], Dict[str, Any]]:
        """Extracts drusen deposits located in or around the macular region."""
        h, w = image_bgr.shape[:2]
        drusen_mask = np.zeros((h, w), dtype=np.uint8)

        red = image_bgr[:, :, 2]
        green = image_bgr[:, :, 1]
        yellow_contrast = cv2.subtract(red, green)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        tophat = cv2.morphologyEx(yellow_contrast, cv2.MORPH_TOPHAT, kernel)
        _, thresh = cv2.threshold(tophat, 20, 255, cv2.THRESH_BINARY)

        if macula_mask is not None:
            thresh = cv2.bitwise_and(thresh, macula_mask)
        if od_mask is not None:
            dilated_od = cv2.dilate(od_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)))
            thresh = cv2.bitwise_and(thresh, cv2.bitwise_not(dilated_od))

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        lesions_list = []

        for c in contours:
            area = cv2.contourArea(c)
            if 10 < area < 800:
                cv2.drawContours(drusen_mask, [c], -1, 255, -1)
                M = cv2.moments(c)
                cx = int(M["m10"] / (M["m00"] + 1e-6))
                cy = int(M["m01"] / (M["m00"] + 1e-6))
                lesions_list.append({
                    "type": "drusen",
                    "centroid": [cx, cy],
                    "area_px2": float(area),
                    "confidence": 0.80
                })

        summary = {
            "type": "drusen",
            "count": len(lesions_list),
            "total_area_px2": float(np.sum(drusen_mask > 0))
        }

        return drusen_mask, lesions_list, summary

