"""Spatial distribution and geometric relationship metrics."""

from typing import Dict, Any, List, Optional
import math
import numpy as np


def calculate_spatial_distribution(
    lesion_masks: Dict[str, np.ndarray],
    fovea_coords: List[int],
    od_coords: List[int],
    macula_radius_px: int = 100
) -> Dict[str, Any]:
    """Calculates spatial distribution across quadrants and macular involvement.

    Args:
        lesion_masks: Dictionary of lesion name to binary mask.
        fovea_coords: [x, y] coordinates of fovea.
        od_coords: [x, y] coordinates of optic disc.
        macula_radius_px: Radius defining the central macular zone.

    Returns:
        Dictionary with quadrant distribution and macular involvement flags.
    """
    fx, fy = fovea_coords
    od_x, od_y = od_coords

    disc_fovea_dist = float(math.sqrt((fx - od_x)**2 + (fy - od_y)**2))

    quadrant_counts = {
        "superior_temporal": 0,
        "superior_nasal": 0,
        "inferior_temporal": 0,
        "inferior_nasal": 0,
    }

    macular_involvement = False
    temporal_is_right = fx > od_x

    for l_name, mask in lesion_masks.items():
        coords = np.argwhere(mask > 0)  # [y, x]
        for y, x in coords:
            # Check macular involvement
            d_to_fovea = math.sqrt((x - fx)**2 + (y - fy)**2)
            if d_to_fovea <= macula_radius_px:
                macular_involvement = True

            # Determine quadrant relative to fovea
            is_superior = y < fy
            is_right = x > fx
            is_temporal = is_right if temporal_is_right else not is_right

            if is_superior and is_temporal:
                quadrant_counts["superior_temporal"] += 1
            elif is_superior and not is_temporal:
                quadrant_counts["superior_nasal"] += 1
            elif not is_superior and is_temporal:
                quadrant_counts["inferior_temporal"] += 1
            else:
                quadrant_counts["inferior_nasal"] += 1

    return {
        "disc_fovea_distance_px": round(disc_fovea_dist, 1),
        "macular_involvement": macular_involvement,
        "quadrant_lesion_pixels": quadrant_counts,
        "coordinate_origin": "fovea"
    }

