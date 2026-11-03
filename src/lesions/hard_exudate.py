"""Hard exudate (EX) detection and segmentation."""

from typing import Dict, Any, Tuple, List, Optional
import cv2
import numpy as np


class HardExudateDetector:
    """Detects lipid deposits and hard exudates in the retinal parenchyma."""

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path

    def detect(
        self,
        image_bgr: np.ndarray,
        fov_mask: Optional[np.ndarray] = None,
        od_mask: Optional[np.ndarray] = None,
        fovea_coords: Optional[Tuple[int, int]] = None
    ) -> Tuple[np.ndarray, List[Dict[str, Any]], Dict[str, Any]]:
        """Segments hard exudates and computes cluster statistics.

        Args:
            image_bgr: Fundus image.
            fov_mask: Retinal FOV mask.
            od_mask: Optic disc mask (to eliminate false positives on bright disc).
            fovea_coords: Fovea location.

        Returns:
            ex_mask: Binary uint8 mask for hard exudates.
            lesions_list: List of detected exudate clusters.
            summary: Summary stats.
        """
        h, w = image_bgr.shape[:2]
        ex_mask = np.zeros((h, w), dtype=np.uint8)

        # Hard exudates have high intensity in both green and red, distinct yellow color
        # In LAB color space, high L and positive b (yellowish)
        lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
        l, a_ch, b_ch = cv2.split(lab)

        # Top-hat to highlight small bright structures
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        tophat = cv2.morphologyEx(l, cv2.MORPH_TOPHAT, kernel)

        # Threshold bright yellowish lesions
        _, thresh = cv2.threshold(tophat, 35, 255, cv2.THRESH_BINARY)
        yellow_condition = (b_ch > 135)
        thresh = np.where(yellow_condition, thresh, 0).astype(np.uint8)

        # Exclude optic disc
        if od_mask is not None:
            dilated_od = cv2.dilate(od_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)))
            thresh = cv2.bitwise_and(thresh, cv2.bitwise_not(dilated_od))

        if fov_mask is not None:
            thresh = cv2.bitwise_and(thresh, fov_mask)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        lesions_list = []

        for c in contours:
            area = cv2.contourArea(c)
            if 5 < area < 3000:
                cv2.drawContours(ex_mask, [c], -1, 255, -1)
                M = cv2.moments(c)
                cx = int(M["m10"] / (M["m00"] + 1e-6))
                cy = int(M["m01"] / (M["m00"] + 1e-6))

                item = {
                    "type": "hard_exudate",
                    "centroid": [cx, cy],
                    "area_px2": float(area),
                    "confidence": 0.92
                }
                if fovea_coords:
                    fx, fy = fovea_coords
                    dist = float(np.sqrt((cx - fx)**2 + (cy - fy)**2))
                    item["distance_to_fovea_px"] = round(dist, 1)

                lesions_list.append(item)

        total_area = float(np.sum(ex_mask > 0))
        valid_fov = int(np.sum(fov_mask > 0)) if fov_mask is not None else (h * w)

        summary = {
            "type": "hard_exudate",
            "count": len(lesions_list),
            "total_area_px2": total_area,
            "area_ratio_percent": round((total_area / max(valid_fov, 1)) * 100.0, 4)
        }

        return ex_mask, lesions_list, summary

