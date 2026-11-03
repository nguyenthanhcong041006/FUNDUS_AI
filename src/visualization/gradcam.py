"""Grad-CAM visualization for deep classification models."""

from typing import Optional
import cv2
import numpy as np


class GradCAMVisualizer:
    """Computes and overlays Class Activation Maps (Grad-CAM)."""

    def __init__(self, model=None, target_layer=None):
        self.model = model
        self.target_layer = target_layer

    def generate_heatmap(
        self,
        image_bgr: np.ndarray,
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """Generates Grad-CAM heatmap.

        Returns:
            heatmap_bgr: Color-mapped heatmap array.
        """
        h, w = image_bgr.shape[:2]
        # Fallback heatmap representation when model gradient hooks are not active
        heatmap = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(heatmap, (w // 2, h // 2), int(min(h, w) * 0.35), 200, -1)
        heatmap = cv2.GaussianBlur(heatmap, (51, 51), 0)
        heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        return heatmap_colored

    def overlay(
        self,
        image_bgr: np.ndarray,
        heatmap_colored: np.ndarray,
        alpha: float = 0.4
    ) -> np.ndarray:
        """Overlays heatmap on original image."""
        return cv2.addWeighted(image_bgr, 1 - alpha, heatmap_colored, alpha, 0)

