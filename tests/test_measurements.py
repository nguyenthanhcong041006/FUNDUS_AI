"""Unit tests for quantitative measurement engine."""

import numpy as np
import cv2

from src.measurements.optic_disc_metrics import calculate_disc_cup_metrics
from src.measurements.vessel_metrics import calculate_vessel_metrics
from src.measurements.lesion_metrics import calculate_lesion_metrics
from src.measurements.spatial_metrics import calculate_spatial_distribution


def test_optic_disc_metrics():
    od = np.zeros((100, 100), dtype=np.uint8)
    oc = np.zeros((100, 100), dtype=np.uint8)

    cv2.circle(od, (50, 50), 20, 255, -1)
    cv2.circle(oc, (50, 50), 10, 255, -1)

    metrics = calculate_disc_cup_metrics(od, oc)
    assert metrics["optic_disc"]["area_px2"] > 0
    assert metrics["optic_cup"]["vertical_cdr"] > 0
    assert metrics["optic_cup"]["vertical_cdr"] <= 1.0


def test_vessel_metrics():
    vessel = np.zeros((100, 100), dtype=np.uint8)
    cv2.line(vessel, (10, 50), (90, 50), 255, 2)

    metrics = calculate_vessel_metrics(vessel)
    assert metrics["vessel_pixel_count"] > 0
    assert metrics["density_percent"] > 0
    assert metrics["total_length_px"] > 0


def test_lesion_metrics():
    lesion = np.zeros((100, 100), dtype=np.uint8)
    cv2.circle(lesion, (30, 30), 3, 255, -1)
    cv2.circle(lesion, (70, 70), 4, 255, -1)

    metrics = calculate_lesion_metrics(lesion, "hard_exudate")
    assert metrics["count"] == 2
    assert metrics["total_area_px2"] > 0

