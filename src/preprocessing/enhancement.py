"""Fundus enhancement using OpenCV and color-space representations."""

from typing import Dict, Any, Optional
import cv2
import numpy as np


class FundusEnhancer:
    """Enhances fundus image channels for anatomical and lesion segmentation."""

    def __init__(
        self,
        clahe_clip_limit: float = 2.0,
        clahe_tile_grid: tuple = (8, 8),
        blur_kernel_size: tuple = (3, 3),
        blur_sigma: float = 0.0,
    ):
        self.clahe = cv2.createCLAHE(
            clipLimit=clahe_clip_limit,
            tileGridSize=clahe_tile_grid
        )
        self.blur_kernel_size = blur_kernel_size
        self.blur_sigma = blur_sigma

    def process(
        self,
        image_bgr: np.ndarray,
        fov_mask: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """Generates green channel, CLAHE-enhanced, LAB, and HSV representations.

        Args:
            image_bgr: Input BGR image.
            fov_mask: Optional binary FOV mask.

        Returns:
            Dictionary containing:
                - 'original_bgr': Raw BGR image
                - 'rgb': Standard RGB
                - 'green': Monochromatic green channel
                - 'enhanced_green': CLAHE-applied green channel (noise reduced)
                - 'lab': LAB color space image
                - 'clahe_lab_rgb': LAB L-channel CLAHE enhanced RGB
                - 'hsv': HSV color space image
        """
        # Original channels
        b, g, r = cv2.split(image_bgr)
        image_rgb = cv2.merge([r, g, b])

        # CLAHE on green channel
        clahe_green = self.clahe.apply(g)
        enhanced_green = cv2.GaussianBlur(
            clahe_green, self.blur_kernel_size, self.blur_sigma
        )

        # LAB color space with L-channel enhancement
        lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
        l, a_ch, b_ch = cv2.split(lab)
        enhanced_l = self.clahe.apply(l)
        enhanced_lab = cv2.merge([enhanced_l, a_ch, b_ch])
        clahe_lab_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        clahe_lab_rgb = cv2.cvtColor(clahe_lab_bgr, cv2.COLOR_BGR2RGB)

        # HSV color space
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

        # Apply FOV mask if provided
        if fov_mask is not None:
            mask_bool = fov_mask > 0
            # Zero out background in auxiliary channels
            enhanced_green = np.where(mask_bool, enhanced_green, 0).astype(np.uint8)

        return {
            "original_bgr": image_bgr,
            "rgb": image_rgb,
            "green": g,
            "enhanced_green": enhanced_green,
            "lab": lab,
            "clahe_lab_rgb": clahe_lab_rgb,
            "hsv": hsv,
        }


def enhance_fundus_image(
    image_bgr: np.ndarray,
    fov_mask: Optional[np.ndarray] = None,
    clahe_clip_limit: float = 2.0
) -> Dict[str, np.ndarray]:
    """Convenience functional wrapper for FundusEnhancer."""
    enhancer = FundusEnhancer(clahe_clip_limit=clahe_clip_limit)
    return enhancer.process(image_bgr, fov_mask=fov_mask)


def ben_graham_equalize(
    image_rgb: np.ndarray,
    sigma: float = 30.0,
    fov_mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """Applies Ben Graham's retinal illumination equalization.

    Formula: I_norm = clip(4 * I - 4 * GaussianBlur(I, sigma) + 128, 0, 255)
    Neutralizes illumination vignetting and peripheral shadows while boosting
    high-frequency vascular and lesion contrast.

    Args:
        image_rgb: uint8 RGB image of shape (H, W, 3).
        sigma: Standard deviation for Gaussian kernel. Default 30.0 for fundus images.
        fov_mask: Optional binary mask (H, W) indicating retinal area.

    Returns:
        uint8 RGB image normalized by local background subtraction.
    """
    if fov_mask is None:
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        fov_mask = (gray > 10).astype(np.uint8)

    blurred = cv2.GaussianBlur(image_rgb, (0, 0), sigma)
    enhanced = cv2.addWeighted(image_rgb, 4.0, blurred, -4.0, 128)

    # Preserve dark background outside the retinal field of view
    enhanced[fov_mask == 0] = 0
    return enhanced


