"""Retinal Field of View (FOV) detection module."""

from typing import Dict, Tuple, Any, Optional
import cv2
import numpy as np


class FOVDetector:
    """Detects and segments the valid retinal aperture (Field of View)."""

    def __init__(
        self,
        threshold_value: int = 15,
        morph_kernel_size: int = 11,
        min_area_ratio: float = 0.15,
    ):
        self.threshold_value = threshold_value
        self.morph_kernel_size = morph_kernel_size
        self.min_area_ratio = min_area_ratio

    def detect(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Extracts the retinal mask and spatial metadata.

        Args:
            image: Input image (BGR or grayscale).

        Returns:
            fov_mask: Binary uint8 mask (0 or 255) with shape (H, W).
            metadata: Dict with bbox, pixel count, area ratio, etc.
        """
        if len(image.shape) == 3:
            # Green channel or grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            green = image[:, :, 1]
            ref_channel = np.maximum(gray, green)
        else:
            ref_channel = image.copy()

        h, w = ref_channel.shape
        total_pixels = h * w

        # Thresholding
        _, binary = cv2.threshold(
            ref_channel, self.threshold_value, 255, cv2.THRESH_BINARY
        )

        # Morphological closing to fill holes inside retina
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self.morph_kernel_size, self.morph_kernel_size)
        )
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)

        # Find largest connected component / contour
        contours, _ = cv2.findContours(
            opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        fov_mask = np.zeros((h, w), dtype=np.uint8)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            contour_area = cv2.contourArea(largest_contour)

            if contour_area >= (total_pixels * self.min_area_ratio):
                cv2.drawContours(fov_mask, [largest_contour], -1, 255, thickness=-1)
                x, y, bw, bh = cv2.boundingRect(largest_contour)
            else:
                # Fallback to whole image if no large component found
                fov_mask[:] = 255
                x, y, bw, bh = 0, 0, w, h
        else:
            fov_mask[:] = 255
            x, y, bw, bh = 0, 0, w, h

        valid_pixels = int(np.sum(fov_mask > 0))
        valid_ratio = float(valid_pixels / total_pixels)

        metadata = {
            "bbox_xywh": [int(x), int(y), int(bw), int(bh)],
            "retinal_pixel_count": valid_pixels,
            "total_pixels": total_pixels,
            "valid_pixel_ratio": round(valid_ratio, 4),
        }

        return fov_mask, metadata


def detect_retinal_fov(
    image: np.ndarray,
    threshold_value: int = 15
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Convenience functional wrapper for FOVDetector."""
    detector = FOVDetector(threshold_value=threshold_value)
    return detector.detect(image)

