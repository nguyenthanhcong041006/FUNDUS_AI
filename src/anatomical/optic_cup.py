"""Optic Cup (OC) segmentation and Cup-to-Disc Ratio (CDR) calculation."""

from typing import Dict, Any, Tuple, Optional
import cv2
import numpy as np


class OpticCupSegmenter:
    """Segments Optic Cup within the Optic Disc region."""

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.model = None

    def segment(
        self,
        image_bgr: np.ndarray,
        od_mask: np.ndarray,
        od_metrics: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Segments the optic cup inside the optic disc mask.

        Args:
            image_bgr: Preprocessed fundus image.
            od_mask: Segmented binary mask of the optic disc.
            od_metrics: Geometric metrics of the optic disc.

        Returns:
            oc_mask: Binary uint8 mask (0 or 255) for Optic Cup.
            metrics: CDR ratios (VCDR, HCDR, area ratio) and cup measurements.
        """
        h, w = od_mask.shape[:2]
        oc_mask = np.zeros((h, w), dtype=np.uint8)

        if self.model is not None:
            pass
        else:
            # Baseline morphological cup estimation inside disc
            if np.sum(od_mask > 0) > 0:
                # Erode OD to estimate central cup region
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
                oc_mask = cv2.erode(od_mask, kernel, iterations=2)

        contours, _ = cv2.findContours(oc_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        metrics: Dict[str, Any] = {
            "area_px2": 0.0,
            "horizontal_diameter_px": 0.0,
            "vertical_diameter_px": 0.0,
            "vertical_cdr": 0.0,
            "horizontal_cdr": 0.0,
            "cup_disc_area_ratio": 0.0,
            "detected": False
        }

        if contours:
            c = max(contours, key=cv2.contourArea)
            area = float(cv2.contourArea(c))
            x, y, bw, bh = cv2.boundingRect(c)

            od_vert_d = od_metrics.get("vertical_diameter_px", 1.0) if od_metrics else 1.0
            od_horiz_d = od_metrics.get("horizontal_diameter_px", 1.0) if od_metrics else 1.0
            od_area = od_metrics.get("area_px2", 1.0) if od_metrics else 1.0

            vcdr = float(bh / max(od_vert_d, 1.0))
            hcdr = float(bw / max(od_horiz_d, 1.0))
            area_ratio = float(area / max(od_area, 1.0))

            metrics = {
                "area_px2": area,
                "horizontal_diameter_px": float(bw),
                "vertical_diameter_px": float(bh),
                "vertical_cdr": round(min(vcdr, 1.0), 3),
                "horizontal_cdr": round(min(hcdr, 1.0), 3),
                "cup_disc_area_ratio": round(min(area_ratio, 1.0), 3),
                "detected": True
            }

        return oc_mask, metrics

