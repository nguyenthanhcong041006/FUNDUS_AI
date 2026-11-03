"""Artery and Vein (A/V) analysis and ratio estimation."""

from typing import Dict, Any, Tuple, Optional
import cv2
import numpy as np


class ArteryVeinAnalyzer:
    """Classifies segmented vessels into arterioles and venules."""

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path

    def analyze(
        self,
        image_bgr: np.ndarray,
        vessel_mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Separates vessel pixels into artery and vein masks.

        Args:
            image_bgr: Input fundus image.
            vessel_mask: Binary blood vessel mask.

        Returns:
            artery_mask: Binary mask of arteries.
            vein_mask: Binary mask of veins.
            metrics: Arteriolar density, venular density, A/V ratio.
        """
        h, w = vessel_mask.shape[:2]

        # Arteries generally appear slightly brighter and redder than darker/broader veins
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        val = hsv[:, :, 2]

        vessel_bool = vessel_mask > 0
        if np.sum(vessel_bool) > 0:
            median_val = np.median(val[vessel_bool])
            artery_mask = np.where((val > median_val) & vessel_bool, 255, 0).astype(np.uint8)
            vein_mask = np.where((val <= median_val) & vessel_bool, 255, 0).astype(np.uint8)
        else:
            artery_mask = np.zeros((h, w), dtype=np.uint8)
            vein_mask = np.zeros((h, w), dtype=np.uint8)

        artery_pixels = int(np.sum(artery_mask > 0))
        vein_pixels = int(np.sum(vein_mask > 0))

        av_ratio = float(artery_pixels / max(vein_pixels, 1))

        metrics = {
            "artery_pixels": artery_pixels,
            "vein_pixels": vein_pixels,
            "av_pixel_ratio": round(av_ratio, 3),
            "status": "screening_estimate"
        }

        return artery_mask, vein_mask, metrics

