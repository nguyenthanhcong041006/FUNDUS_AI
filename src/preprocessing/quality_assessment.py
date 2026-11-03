"""Image Quality Assessment (IQA) module for fundus photography."""

from typing import Dict, Any, List, Optional
import cv2
import numpy as np


class QualityAssessor:
    """Evaluates technical gradability of retinal fundus images."""

    def __init__(
        self,
        blur_threshold: float = 80.0,
        min_brightness: float = 25.0,
        max_brightness: float = 230.0,
        min_contrast: float = 18.0,
        min_fov_ratio: float = 0.20,
    ):
        self.blur_threshold = blur_threshold
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness
        self.min_contrast = min_contrast
        self.min_fov_ratio = min_fov_ratio

    def assess(
        self,
        image: np.ndarray,
        fov_mask: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Calculates sharpness, brightness, contrast, and gradability status.

        Args:
            image: BGR or RGB fundus image.
            fov_mask: Optional binary mask (0 or 255) of the retinal FOV.

        Returns:
            Dictionary with quality scores, flags, and overall gradability status.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        h, w = gray.shape
        total_pixels = h * w

        # Retinal ROI region for illumination metrics
        if fov_mask is not None and np.sum(fov_mask > 0) > 0:
            roi_pixels = gray[fov_mask > 0]
            fov_pixel_count = int(np.sum(fov_mask > 0))
            fov_ratio = fov_pixel_count / total_pixels
        else:
            roi_pixels = gray
            fov_pixel_count = total_pixels
            fov_ratio = 1.0

        # Blur estimation using Variance of Laplacian
        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        # Brightness and contrast within retinal area
        mean_brightness = float(np.mean(roi_pixels))
        contrast_std = float(np.std(roi_pixels))

        # Under-exposure and over-exposure ratios
        under_exposed_ratio = float(np.sum(roi_pixels < 10) / len(roi_pixels))
        over_exposed_ratio = float(np.sum(roi_pixels > 245) / len(roi_pixels))

        warnings: List[str] = []
        is_ungradable = False

        if blur_score < (self.blur_threshold * 0.5):
            warnings.append("Severe blur detected - detail lost")
            is_ungradable = True
        elif blur_score < self.blur_threshold:
            warnings.append("Moderate blur detected")

        if mean_brightness < self.min_brightness:
            warnings.append("Severe underexposure / dark image")
            is_ungradable = True
        elif mean_brightness > self.max_brightness:
            warnings.append("Severe overexposure / washed out image")
            is_ungradable = True

        if contrast_std < self.min_contrast:
            warnings.append("Low image contrast")

        if fov_ratio < self.min_fov_ratio:
            warnings.append("Retinal FOV coverage too low")
            is_ungradable = True

        if is_ungradable:
            status = "ungradable"
        elif len(warnings) > 0:
            status = "review_required"
        else:
            status = "gradable"

        return {
            "status": status,
            "blur_score": round(blur_score, 2),
            "mean_brightness": round(mean_brightness, 2),
            "contrast_std": round(contrast_std, 2),
            "fov_coverage_ratio": round(fov_ratio, 4),
            "underexposed_ratio": round(under_exposed_ratio, 4),
            "overexposed_ratio": round(over_exposed_ratio, 4),
            "warnings": warnings,
            "is_gradable": status == "gradable"
        }


def assess_image_quality(
    image: np.ndarray,
    fov_mask: Optional[np.ndarray] = None,
    blur_threshold: float = 80.0
) -> Dict[str, Any]:
    """Convenience functional wrapper for QualityAssessor."""
    assessor = QualityAssessor(blur_threshold=blur_threshold)
    return assessor.assess(image, fov_mask=fov_mask)

