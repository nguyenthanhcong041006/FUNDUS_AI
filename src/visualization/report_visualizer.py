"""Generates composite visual inspection panels and saves annotated figures."""

from pathlib import Path
from typing import Dict, Any, Optional
import cv2
import numpy as np


def generate_visual_report(
    original_bgr: np.ndarray,
    enhanced_green: np.ndarray,
    fov_mask: np.ndarray,
    od_oc_overlay: np.ndarray,
    vessel_mask: np.ndarray,
    annotated_composite: np.ndarray,
    output_path: str | Path,
    title: str = "FUNDUS AI - Multi-Stage Retinal Analysis"
) -> str:
    """Combines key pipeline representations into a single comparison canvas.

    Panel layout:
    [Original Fundus]       [Enhanced Green (CLAHE)] [Retinal FOV Mask]
    [Optic Disc / Cup]      [Vessel Segmentation]    [Composite Overlay]

    Args:
        original_bgr: Raw input image.
        enhanced_green: Single-channel or 3-channel enhanced green.
        fov_mask: Binary retinal FOV mask.
        od_oc_overlay: Optic Disc and Cup visualization.
        vessel_mask: Segmented blood vessels.
        annotated_composite: Final blended overlay.
        output_path: Target path to save the composite image.
        title: Title banner text.

    Returns:
        Path to the saved report image file.
    """
    target_size = (384, 384)

    def prepare_tile(img: np.ndarray, label: str) -> np.ndarray:
        if len(img.shape) == 2:
            tile = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            tile = img.copy()

        tile = cv2.resize(tile, target_size)
        # Add label banner at top
        cv2.rectangle(tile, (0, 0), (target_size[0], 28), (30, 30, 30), -1)
        cv2.putText(
            tile, label, (8, 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1, cv2.LINE_AA
        )
        return tile

    t1 = prepare_tile(original_bgr, "1. Original Fundus")
    t2 = prepare_tile(enhanced_green, "2. CLAHE Green Channel")
    t3 = prepare_tile(fov_mask, "3. Retinal FOV Mask")
    t4 = prepare_tile(od_oc_overlay, "4. Optic Disc & Cup")
    t5 = prepare_tile(vessel_mask, "5. Blood Vessels")
    t6 = prepare_tile(annotated_composite, "6. Multi-Layer Diagnostic Overlay")

    row1 = np.hstack([t1, t2, t3])
    row2 = np.hstack([t4, t5, t6])
    canvas = np.vstack([row1, row2])

    # Top header bar
    header = np.zeros((40, canvas.shape[1], 3), dtype=np.uint8)
    header[:] = (45, 45, 45)
    cv2.putText(
        header, title, (15, 27),
        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA
    )

    full_figure = np.vstack([header, canvas])

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), full_figure)

    return str(out.resolve())

