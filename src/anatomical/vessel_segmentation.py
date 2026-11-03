"""Blood vessel segmentation and vessel tree extraction."""

from typing import Dict, Any, Tuple, Optional
import cv2
import numpy as np


class VesselSegmenter:
    """Segments retinal blood vessels using deep learning or adaptive morphological filters."""

    def __init__(self, model_path: Optional[str] = None, threshold: float = 0.5):
        self.model_path = model_path
        self.threshold = threshold
        self.model = None

    def segment(
        self,
        image_bgr: np.ndarray,
        fov_mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Segments blood vessels and calculates basic vessel metrics.

        Args:
            image_bgr: Preprocessed fundus image.
            fov_mask: Retinal field of view mask.

        Returns:
            vessel_mask: Binary uint8 mask (0 or 255) of retinal vessels.
            metrics: Vessel density, pixel count.
        """
        h, w = image_bgr.shape[:2]

        if self.model is not None:
            # Model inference placeholder
            vessel_mask = np.zeros((h, w), dtype=np.uint8)
        else:
            # High-performance classical vessel enhancement fallback:
            # Green channel + CLAHE + Top-hat / Morphological opening
            green = image_bgr[:, :, 1]
            clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
            enhanced = clahe.apply(green)

            # Invert: vessels are dark in green channel, so invert makes them bright
            inv = 255 - enhanced

            # Top-hat morphological operation with disc elements
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            tophat = cv2.morphologyEx(inv, cv2.MORPH_TOPHAT, kernel)

            # Adaptive thresholding
            thresh = cv2.adaptiveThreshold(
                tophat, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 15, -2
            )

            # Clean small isolated noise dots
            clean_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, clean_kernel)

            vessel_mask = cleaned

        # Mask out anything outside the FOV
        if fov_mask is not None:
            vessel_mask = cv2.bitwise_and(vessel_mask, fov_mask)

        vessel_pixels = int(np.sum(vessel_mask > 0))
        valid_fov_pixels = int(np.sum(fov_mask > 0)) if fov_mask is not None else (h * w)

        vessel_density = float((vessel_pixels / max(valid_fov_pixels, 1)) * 100.0)

        metrics = {
            "vessel_pixel_count": vessel_pixels,
            "vessel_density_percent": round(vessel_density, 2),
            "valid_fov_pixels": valid_fov_pixels
        }

        return vessel_mask, metrics

