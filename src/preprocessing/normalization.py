"""Image resizing, aspect-ratio letterboxing, and tensor normalization."""

from typing import Tuple, Dict, Any, Optional
import cv2
import numpy as np


def resize_and_pad(
    image: np.ndarray,
    target_size: int = 512,
    interpolation: int = cv2.INTER_LINEAR,
    preserve_aspect_ratio: bool = True
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Resizes image to (target_size, target_size).

    If preserve_aspect_ratio is True, adds letterbox padding to maintain
    geometric proportions without distortion.

    Args:
        image: Numpy array of shape (H, W) or (H, W, C).
        target_size: Target square dimension.
        interpolation: OpenCV interpolation mode.
        preserve_aspect_ratio: Whether to preserve aspect ratio with padding.

    Returns:
        padded_image: Resized image with shape (target_size, target_size, ...).
        transform_meta: Transformation params (scale, pad_x, pad_y, original_h, original_w).
    """
    orig_h, orig_w = image.shape[:2]

    if not preserve_aspect_ratio:
        resized = cv2.resize(image, (target_size, target_size), interpolation=interpolation)
        transform_meta = {
            "scale": target_size / max(orig_h, orig_w),
            "pad_x": 0,
            "pad_y": 0,
            "orig_h": orig_h,
            "orig_w": orig_w,
            "target_size": target_size,
            "preserve_aspect_ratio": False
        }
        return resized, transform_meta

    scale = min(target_size / orig_w, target_size / orig_h)
    new_w = int(round(orig_w * scale))
    new_h = int(round(orig_h * scale))

    resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)

    pad_w = target_size - new_w
    pad_h = target_size - new_h
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    if len(image.shape) == 3:
        padded = cv2.copyMakeBorder(
            resized, pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT, value=[0, 0, 0]
        )
    else:
        padded = cv2.copyMakeBorder(
            resized, pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT, value=0
        )

    transform_meta = {
        "scale": scale,
        "pad_top": pad_top,
        "pad_bottom": pad_bottom,
        "pad_left": pad_left,
        "pad_right": pad_right,
        "orig_h": orig_h,
        "orig_w": orig_w,
        "target_size": target_size,
        "preserve_aspect_ratio": True
    }

    return padded, transform_meta


def letterbox_rgb(
    image: np.ndarray,
    target_size: int = 512,
    interpolation: int = cv2.INTER_LINEAR
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Convenience letterbox function for RGB/BGR fundus images."""
    return resize_and_pad(
        image=image,
        target_size=target_size,
        interpolation=interpolation,
        preserve_aspect_ratio=True
    )


def letterbox_mask(
    mask: np.ndarray,
    target_size: int = 512,
    pad_meta: Optional[Dict[str, Any]] = None
) -> np.ndarray:
    """Applies exact aspect-ratio preserving letterbox padding to binary or multi-class masks.
    
    Supports 2D (H, W), 3D channel-first (C, H, W), or 3D channel-last (H, W, C).
    Guarantees NEAREST neighbor interpolation to prevent class bleeding.
    """
    if pad_meta is None:
        if mask.ndim == 2:
            _, pad_meta = resize_and_pad(mask, target_size=target_size, preserve_aspect_ratio=True)
        elif mask.ndim == 3 and mask.shape[0] < mask.shape[2]:  # (C, H, W)
            dummy = np.zeros((mask.shape[1], mask.shape[2]), dtype=np.uint8)
            _, pad_meta = resize_and_pad(dummy, target_size=target_size, preserve_aspect_ratio=True)
        else:
            _, pad_meta = resize_and_pad(mask[:, :, 0], target_size=target_size, preserve_aspect_ratio=True)

    pad_top = pad_meta["pad_top"]
    pad_bottom = pad_meta["pad_bottom"]
    pad_left = pad_meta["pad_left"]
    pad_right = pad_meta["pad_right"]
    new_w = target_size - (pad_left + pad_right)
    new_h = target_size - (pad_top + pad_bottom)

    if mask.ndim == 2:
        resized = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        padded = cv2.copyMakeBorder(
            resized, pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT, value=0
        )
        return padded
    elif mask.ndim == 3 and mask.shape[0] < mask.shape[2]:  # (C, H, W)
        channels = []
        for c in range(mask.shape[0]):
            ch = mask[c]
            r = cv2.resize(ch, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            p = cv2.copyMakeBorder(
                r, pad_top, pad_bottom, pad_left, pad_right,
                cv2.BORDER_CONSTANT, value=0
            )
            channels.append(p)
        return np.stack(channels, axis=0)
    else:  # (H, W, C)
        channels = []
        for c in range(mask.shape[2]):
            ch = mask[:, :, c]
            r = cv2.resize(ch, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            p = cv2.copyMakeBorder(
                r, pad_top, pad_bottom, pad_left, pad_right,
                cv2.BORDER_CONSTANT, value=0
            )
            channels.append(p)
        return np.stack(channels, axis=2)


def unpad_mask(
    padded_mask: np.ndarray,
    pad_meta: Dict[str, Any]
) -> np.ndarray:
    """Restores a padded mask back to its original image dimensions."""
    if padded_mask.ndim == 3 and padded_mask.shape[0] < padded_mask.shape[2]:
        # (C, H, W) -> unpad channel by channel
        unpadded = []
        for c in range(padded_mask.shape[0]):
            unpadded.append(unpad_and_restore(padded_mask[c], pad_meta, interpolation=cv2.INTER_NEAREST))
        return np.stack(unpadded, axis=0)
    return unpad_and_restore(padded_mask, pad_meta, interpolation=cv2.INTER_NEAREST)


def unpad_and_restore(
    mask_or_image: np.ndarray,
    transform_meta: Dict[str, Any],
    interpolation: int = cv2.INTER_NEAREST
) -> np.ndarray:
    """Restores a padded mask/image back to its original dimensions."""
    if not transform_meta.get("preserve_aspect_ratio", True):
        return cv2.resize(
            mask_or_image,
            (transform_meta["orig_w"], transform_meta["orig_h"]),
            interpolation=interpolation
        )

    pad_top = transform_meta["pad_top"]
    target_size = transform_meta["target_size"]
    pad_bottom = transform_meta["pad_bottom"]
    pad_left = transform_meta["pad_left"]
    pad_right = transform_meta["pad_right"]

    h_end = target_size - pad_bottom
    w_end = target_size - pad_right

    cropped = mask_or_image[pad_top:h_end, pad_left:w_end]

    restored = cv2.resize(
        cropped,
        (transform_meta["orig_w"], transform_meta["orig_h"]),
        interpolation=interpolation
    )
    return restored


def normalize_tensor(
    image: np.ndarray,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
) -> np.ndarray:
    """Normalizes image array to float32 with standard mean/std."""
    img_float = image.astype(np.float32) / 255.0
    mean_arr = np.array(mean, dtype=np.float32)
    std_arr = np.array(std, dtype=np.float32)

    if len(img_float.shape) == 3:
        normalized = (img_float - mean_arr) / std_arr
    else:
        normalized = (img_float - mean_arr[0]) / std_arr[0]

    return normalized

