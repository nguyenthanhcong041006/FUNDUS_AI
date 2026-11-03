"""Draws bounding boxes, contours, and anatomical landmarks."""

from typing import List, Tuple, Optional, Dict, Any
import cv2
import numpy as np


def draw_landmarks_and_boxes(
    image: np.ndarray,
    fovea_coords: Optional[List[int]] = None,
    od_coords: Optional[List[int]] = None,
    od_bbox: Optional[List[int]] = None,
    fov_bbox: Optional[List[int]] = None
) -> np.ndarray:
    """Annotates landmarks and bounding boxes on the fundus image."""
    annotated = image.copy()

    # Draw Optic Disc Bounding Box and Center
    if od_bbox:
        x, y, w, h = od_bbox
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(
            annotated, "Optic Disc", (x, max(15, y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA
        )

    if od_coords:
        ox, oy = od_coords
        cv2.circle(annotated, (ox, oy), 4, (255, 0, 0), -1)

    # Draw Fovea Crosshair
    if fovea_coords:
        fx, fy = fovea_coords
        arm = 12
        # Crosshair lines in yellow (0, 255, 255)
        cv2.line(annotated, (fx - arm, fy), (fx + arm, fy), (0, 255, 255), 2)
        cv2.line(annotated, (fx, fy - arm), (fx, fy + arm), (0, 255, 255), 2)
        cv2.circle(annotated, (fx, fy), 3, (0, 255, 255), -1)
        cv2.putText(
            annotated, "Fovea", (fx + 8, fy - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA
        )

    # Disc-to-Fovea connecting line
    if fovea_coords and od_coords:
        fx, fy = fovea_coords
        ox, oy = od_coords
        cv2.line(annotated, (ox, oy), (fx, fy), (0, 255, 255), 1, cv2.LINE_AA)

    return annotated

