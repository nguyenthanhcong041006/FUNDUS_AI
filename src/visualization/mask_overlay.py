"""Mask overlay utilities with standardized color conventions."""

from typing import Dict, Tuple, Optional
import cv2
import numpy as np


# Standard BGR color mapping from Phase 13 of the plan
DEFAULT_COLORS: Dict[str, Tuple[int, int, int]] = {
    "optic_disc": (255, 0, 0),       # Blue
    "optic_cup": (0, 255, 0),        # Green
    "vessels": (0, 0, 255),          # Red
    "macula_region": (255, 255, 0),  # Cyan
    "fovea": (0, 255, 255),          # Yellow
    "microaneurysm": (0, 0, 255),    # Red
    "hemorrhage": (0, 165, 255),     # Orange
    "hard_exudate": (0, 255, 255),   # Yellow
    "cotton_wool": (255, 255, 255),  # White
    "drusen": (144, 238, 144),       # Light green
}


def overlay_mask(
    image: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int],
    alpha: float = 0.4
) -> np.ndarray:
    """Blends a binary mask onto a fundus image with specified color and opacity.

    Matches Section 17.3 of FUNDUS AI Implementation Plan.

    Args:
        image: Base BGR image.
        mask: Single-channel uint8 mask (shape H, W).
        color: BGR tuple (B, G, R).
        alpha: Blending weight for mask (0.0 to 1.0).

    Returns:
        Blended BGR image.
    """
    output = image.copy()
    colored_mask = np.zeros_like(image)
    colored_mask[mask > 0] = color

    # Blend where mask is non-zero
    mask_indices = mask > 0
    if np.any(mask_indices):
        blended = cv2.addWeighted(image, 1.0 - alpha, colored_mask, alpha, 0)
        output[mask_indices] = blended[mask_indices]

    return output


def create_multi_layer_overlay(
    image: np.ndarray,
    masks_dict: Dict[str, np.ndarray],
    alpha: float = 0.4,
    custom_colors: Optional[Dict[str, Tuple[int, int, int]]] = None
) -> np.ndarray:
    """Combines multiple masks onto a single fundus image.

    Args:
        image: Original BGR fundus image.
        masks_dict: Dict mapping layer names to binary masks.
        alpha: Opacity factor.
        custom_colors: Optional color override dictionary.

    Returns:
        Multi-layer annotated BGR image.
    """
    color_map = DEFAULT_COLORS.copy()
    if custom_colors:
        color_map.update(custom_colors)

    annotated = image.copy()

    # Recommended rendering order: FOV -> Vessels -> Exudates -> Hemorrhages -> Disc/Cup
    layer_order = [
        "vessels",
        "drusen",
        "hard_exudate",
        "cotton_wool",
        "hemorrhage",
        "microaneurysm",
        "optic_disc",
        "optic_cup",
        "macula_region"
    ]

    for layer in layer_order:
        if layer in masks_dict and masks_dict[layer] is not None:
            mask = masks_dict[layer]
            color = color_map.get(layer, (255, 255, 255))
            annotated = overlay_mask(annotated, mask, color, alpha=alpha)

    # Process any remaining unlisted masks
    for layer, mask in masks_dict.items():
        if layer not in layer_order and mask is not None:
            color = color_map.get(layer, (200, 200, 200))
            annotated = overlay_mask(annotated, mask, color, alpha=alpha)

    return annotated

