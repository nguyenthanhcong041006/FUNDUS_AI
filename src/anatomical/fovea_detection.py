"""Macula and Fovea landmark localization."""

from typing import Dict, Any, Tuple, Optional
import math
import cv2
import numpy as np


class FoveaDetector:
    """Localizes the fovea centralis and estimates macular region."""

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path

    def detect(
        self,
        image_bgr: np.ndarray,
        od_metrics: Optional[Dict[str, Any]] = None,
        fov_mask: Optional[np.ndarray] = None
    ) -> Tuple[Dict[str, Any], np.ndarray]:
        """Detects fovea landmark coordinates and generates macular ROI mask.

        Args:
            image_bgr: Preprocessed fundus image.
            od_metrics: Metrics containing optic disc centroid [cx, cy].
            fov_mask: Retinal FOV mask.

        Returns:
            fovea_data: Dict with (x, y) coordinates, disc_fovea_distance_px, confidence.
            macula_mask: Binary uint8 mask for the macular region.
        """
        h, w = image_bgr.shape[:2]
        macula_mask = np.zeros((h, w), dtype=np.uint8)

        # Optic disc position
        disc_is_detected = od_metrics.get("detected", False) if od_metrics else False
        if not disc_is_detected and od_metrics and od_metrics.get("area_px2", 0) > 50:
            disc_is_detected = True

        od_cx, od_cy = (w // 3, h // 2)
        if disc_is_detected and od_metrics:
            if "centroid" in od_metrics:
                od_cx, od_cy = od_metrics["centroid"]
            elif "center" in od_metrics:
                od_cx, od_cy = od_metrics["center"]

        # If OD is on the left half of image, fovea is temporally located to the right
        # If OD is on the right half, fovea is temporally located to the left
        expected_dir = 1 if od_cx < (w // 2) else -1
        disc_diam = od_metrics.get("horizontal_diameter_px", 80.0) if od_metrics else 80.0
        if disc_diam <= 0:
            disc_diam = 80.0

        # Estimated anatomical distance: ~2.5 disc diameters
        est_dist = int(disc_diam * 2.5)
        est_fx = int(np.clip(od_cx + (expected_dir * est_dist), 20, w - 20))
        est_fy = int(np.clip(od_cy, 20, h - 20))

        # Macula is relatively darker and avascular in the green channel
        green = image_bgr[:, :, 1]
        roi_half = int(disc_diam * 0.8)
        y1, y2 = max(0, est_fy - roi_half), min(h, est_fy + roi_half)
        x1, x2 = max(0, est_fx - roi_half), min(w, est_fx + roi_half)

        roi = green[y1:y2, x1:x2]
        if roi.size > 0:
            blurred_roi = cv2.GaussianBlur(roi, (11, 11), 0)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(blurred_roi)
            fovea_x = x1 + min_loc[0]
            fovea_y = y1 + min_loc[1]
        else:
            fovea_x, fovea_y = est_fx, est_fy

        # Distance to optic disc (valid only if optic disc was detected)
        if disc_is_detected:
            df_distance = round(float(math.sqrt((fovea_x - od_cx)**2 + (fovea_y - od_cy)**2)), 1)
            od_coords = [int(od_cx), int(od_cy)]
        else:
            df_distance = None
            od_coords = None

        # Macular zone mask (approx 1.5 disc diameters radius around fovea)
        macula_radius = int(disc_diam * 0.75)
        cv2.circle(macula_mask, (fovea_x, fovea_y), macula_radius, 255, -1)
        if fov_mask is not None:
            macula_mask = cv2.bitwise_and(macula_mask, fov_mask)

        fovea_data = {
            "fovea_coordinates": [int(fovea_x), int(fovea_y)],
            "optic_disc_coordinates": od_coords,
            "disc_fovea_distance_px": df_distance,
            "macula_radius_px": macula_radius,
            "detected": True
        }

        return fovea_data, macula_mask

