"""Medical-grade spatial and photometric data augmentations for retinal fundus imaging.

Ensures spatial transforms (flips, rotations, affine scaling) are applied
identically to both RGB fundus images and multi-class binary masks, while
photometric transforms (CLAHE, Color Jitter) are applied strictly to the image.
"""

from typing import Tuple, Dict, Any, Optional
import random
import cv2
import numpy as np
import torch


class JointFundusAugmentation:
    """Joint augmentor for fundus image and segmentation mask."""

    def __init__(
        self,
        p_flip_h: float = 0.5,
        p_flip_v: float = 0.5,
        p_rotate: float = 0.5,
        max_angle: float = 25.0,
        p_scale: float = 0.5,
        scale_range: Tuple[float, float] = (0.85, 1.15),
        p_jitter: float = 0.5,
        p_clahe: float = 0.3
    ):
        self.p_flip_h = p_flip_h
        self.p_flip_v = p_flip_v
        self.p_rotate = p_rotate
        self.max_angle = max_angle
        self.p_scale = p_scale
        self.scale_range = scale_range
        self.p_jitter = p_jitter
        self.p_clahe = p_clahe

    def __call__(
        self,
        image: np.ndarray,
        mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Applies synchronized augmentations.
        
        Args:
            image: Float or uint8 RGB image of shape (H, W, 3).
            mask: Float or uint8 binary/multi-channel mask of shape (H, W) or (C, H, W).
            
        Returns:
            augmented_image, augmented_mask
        """
        # Ensure mask is channel-last or 2D during spatial cv2 warps
        mask_is_ch_first = False
        if mask.ndim == 3 and mask.shape[0] < mask.shape[2]:
            # (C, H, W) -> transpose to (H, W, C)
            mask_is_ch_first = True
            mask = np.transpose(mask, (1, 2, 0))

        h, w = image.shape[:2]

        # 1. Random Horizontal Flip
        if random.random() < self.p_flip_h:
            image = np.fliplr(image).copy()
            mask = np.fliplr(mask).copy()

        # 2. Random Vertical Flip
        if random.random() < self.p_flip_v:
            image = np.flipud(image).copy()
            mask = np.flipud(mask).copy()

        # 3. Random Rotation & Scale
        do_rot = random.random() < self.p_rotate
        do_scale = random.random() < self.p_scale

        if do_rot or do_scale:
            angle = random.uniform(-self.max_angle, self.max_angle) if do_rot else 0.0
            scale = random.uniform(self.scale_range[0], self.scale_range[1]) if do_scale else 1.0

            center = (w / 2.0, h / 2.0)
            M = cv2.getRotationMatrix2D(center, angle, scale)

            image = cv2.warpAffine(
                image, M, (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0)
            )
            mask = cv2.warpAffine(
                mask, M, (w, h),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )

        # 4. Photometric Jitter (Image only)
        if random.random() < self.p_jitter:
            alpha = random.uniform(0.85, 1.15)
            beta = random.uniform(-15, 15)
            if image.dtype == np.float32 or image.dtype == np.float64:
                image = np.clip(image * alpha + (beta / 255.0), 0.0, 1.0)
            else:
                image = np.clip(image.astype(np.float32) * alpha + beta, 0, 255).astype(image.dtype)

        # 5. Random CLAHE (Image only)
        if random.random() < self.p_clahe:
            is_float = (image.dtype == np.float32 or image.dtype == np.float64)
            img_u8 = (image * 255).astype(np.uint8) if is_float else image
            lab = cv2.cvtColor(img_u8, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=random.uniform(1.2, 2.5), tileGridSize=(8, 8))
            l_clahe = clahe.apply(l)
            lab_clahe = cv2.merge([l_clahe, a, b])
            aug_rgb = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)
            image = aug_rgb.astype(np.float32) / 255.0 if is_float else aug_rgb

        if mask_is_ch_first:
            mask = np.transpose(mask, (2, 0, 1))

        return image, mask


class ClassificationFundusAugmentation:
    """Image-only augmentor for fundus disease classification (APTOS, ODIR)."""

    def __init__(
        self,
        p_flip_h: float = 0.5,
        p_flip_v: float = 0.5,
        p_rotate: float = 0.5,
        max_angle: float = 20.0,
        p_scale: float = 0.5,
        scale_range: Tuple[float, float] = (0.9, 1.1),
        p_jitter: float = 0.5,
        p_clahe: float = 0.3
    ):
        self.p_flip_h = p_flip_h
        self.p_flip_v = p_flip_v
        self.p_rotate = p_rotate
        self.max_angle = max_angle
        self.p_scale = p_scale
        self.scale_range = scale_range
        self.p_jitter = p_jitter
        self.p_clahe = p_clahe

    def __call__(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]

        if random.random() < self.p_flip_h:
            image = np.fliplr(image).copy()
        if random.random() < self.p_flip_v:
            image = np.flipud(image).copy()

        do_rot = random.random() < self.p_rotate
        do_scale = random.random() < self.p_scale
        if do_rot or do_scale:
            angle = random.uniform(-self.max_angle, self.max_angle) if do_rot else 0.0
            scale = random.uniform(self.scale_range[0], self.scale_range[1]) if do_scale else 1.0
            M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, scale)
            image = cv2.warpAffine(
                image, M, (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0)
            )

        if random.random() < self.p_jitter:
            alpha = random.uniform(0.85, 1.15)
            beta = random.uniform(-15, 15)
            if image.dtype in (np.float32, np.float64):
                image = np.clip(image * alpha + (beta / 255.0), 0.0, 1.0)
            else:
                image = np.clip(image.astype(np.float32) * alpha + beta, 0, 255).astype(image.dtype)

        if random.random() < self.p_clahe:
            is_float = image.dtype in (np.float32, np.float64)
            img_u8 = (image * 255).astype(np.uint8) if is_float else image
            lab = cv2.cvtColor(img_u8, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=random.uniform(1.2, 2.0), tileGridSize=(8, 8))
            l_clahe = clahe.apply(l)
            lab_clahe = cv2.merge([l_clahe, a, b])
            aug_rgb = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)
            image = aug_rgb.astype(np.float32) / 255.0 if is_float else aug_rgb

        return image
