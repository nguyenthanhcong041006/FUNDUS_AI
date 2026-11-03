"""Optic Disc (OD) segmentation and localization."""

from typing import Dict, Any, Tuple, Optional
import cv2
import numpy as np


class OpticDiscSegmenter:
    """Segments the Optic Disc from fundus images."""

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.model = None
        if model_path:
            self._load_model(model_path)

    def _load_model(self, path: str):
        # Placeholder for loading trained PyTorch / ONNX checkpoint
        self.model = None

    def segment(
        self,
        image_bgr: np.ndarray,
        fov_mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Segments the optic disc, returning a binary mask and geometry metrics.

        Args:
            image_bgr: Preprocessed fundus image.
            fov_mask: Retinal FOV binary mask.

        Returns:
            od_mask: Binary uint8 mask (0 or 255) for Optic Disc.
            metrics: Centroid, bounding box, area, vertical/horizontal diameter.
        """
        h, w = image_bgr.shape[:2]
        od_mask = np.zeros((h, w), dtype=np.uint8)

        if self.model is not None:
            # Model inference logic when checkpoint is provided
            pass
        else:
            # Robust candidate heuristic fallback for unweighted pipeline testing:
            # Optic disc is typically the brightest circular region in the red/gray channel
            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
            red = image_bgr[:, :, 2]
            bright = cv2.addWeighted(red, 0.7, gray, 0.3, 0)

            if fov_mask is not None:
                # Erode FOV slightly to avoid rim artifacts
                kernel_fov = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
                inner_fov = cv2.erode(fov_mask, kernel_fov)
                bright = np.where(inner_fov > 0, bright, 0)

            blurred = cv2.GaussianBlur(bright, (15, 15), 0)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(blurred)

            # Extract local circle around the peak
            cx, cy = max_loc
            approx_radius = int(min(h, w) * 0.08)
            cv2.circle(od_mask, (cx, cy), approx_radius, 255, -1)

        # Compute contours and morphological metrics
        contours, _ = cv2.findContours(od_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        metrics: Dict[str, Any] = {
            "centroid": [0, 0],
            "area_px2": 0,
            "horizontal_diameter_px": 0.0,
            "vertical_diameter_px": 0.0,
            "bbox_xywh": [0, 0, 0, 0],
            "detected": False
        }

        if contours:
            c = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(c)
            if area > 100:
                M = cv2.moments(c)
                cx = int(M["m10"] / (M["m00"] + 1e-6))
                cy = int(M["m01"] / (M["m00"] + 1e-6))
                x, y, bw, bh = cv2.boundingRect(c)

                metrics = {
                    "centroid": [cx, cy],
                    "area_px2": float(area),
                    "horizontal_diameter_px": float(bw),
                    "vertical_diameter_px": float(bh),
                    "bbox_xywh": [int(x), int(y), int(bw), int(bh)],
                    "detected": True
                }

        return od_mask, metrics

