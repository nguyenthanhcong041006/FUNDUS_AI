"""Unit tests for quantitative measurement engine with anatomical fail-safes."""

import numpy as np
import cv2

from src.measurements.optic_disc_metrics import calculate_disc_cup_metrics
from src.measurements.vessel_metrics import calculate_vessel_metrics
from src.measurements.lesion_metrics import calculate_lesion_metrics
from src.measurements.spatial_metrics import calculate_spatial_distribution
from src.classification.glaucoma import GlaucomaScreener
from src.classification.disease_fusion import DiseaseScreeningAggregator


def test_optic_disc_metrics_valid():
    od = np.zeros((100, 100), dtype=np.uint8)
    oc = np.zeros((100, 100), dtype=np.uint8)

    cv2.circle(od, (50, 50), 20, 255, -1)
    cv2.circle(oc, (50, 50), 10, 255, -1)

    metrics = calculate_disc_cup_metrics(od, oc)
    assert metrics["optic_disc"]["detected"] is True
    assert metrics["optic_disc"]["area_px2"] > 0
    assert metrics["optic_cup"]["detected"] is True
    assert metrics["optic_cup"]["vertical_cdr"] is not None
    assert 0.0 < metrics["optic_cup"]["vertical_cdr"] <= 1.0
    assert metrics["optic_cup"]["anatomical_violation"] is False


def test_optic_disc_metrics_missing_disc():
    """Fail-safe test: Missing optic disc must yield None for vCDR, never 0.0."""
    od = np.zeros((100, 100), dtype=np.uint8)
    oc = np.zeros((100, 100), dtype=np.uint8)

    metrics = calculate_disc_cup_metrics(od, oc)
    assert metrics["optic_disc"]["detected"] is False
    assert metrics["optic_cup"]["vertical_cdr"] is None
    assert metrics["optic_cup"]["horizontal_cdr"] is None
    assert metrics["optic_cup"]["cup_disc_area_ratio"] is None


def test_optic_disc_anatomical_violation():
    """Anatomical violation test: Cup contour outside disc must flag violation."""
    od = np.zeros((100, 100), dtype=np.uint8)
    oc = np.zeros((100, 100), dtype=np.uint8)

    cv2.circle(od, (30, 30), 15, 255, -1)
    cv2.circle(oc, (70, 70), 10, 255, -1)  # Cup entirely outside disc!

    metrics = calculate_disc_cup_metrics(od, oc)
    assert metrics["optic_cup"]["anatomical_violation"] is True


def test_glaucoma_screener_fail_safe():
    """Glaucoma screener must not assign low risk when disc is not detected."""
    screener = GlaucomaScreener()

    # Case 1: Missing disc metrics
    missing_disc_metrics = {
        "optic_disc": {"detected": False, "area_px2": 0.0},
        "optic_cup": {"vertical_cdr": None, "cup_disc_area_ratio": None}
    }
    res = screener.evaluate(missing_disc_metrics)
    assert res["risk_score"] is None
    assert res["risk_level"] == "requires_clinical_review"
    assert res["status"] == "disc_not_evaluable"

    # Case 2: Normal evaluable disc
    normal_metrics = {
        "optic_disc": {"detected": True, "area_px2": 5000.0, "vertical_diameter_px": 80.0},
        "optic_cup": {"detected": True, "vertical_cdr": 0.35, "cup_disc_area_ratio": 0.2, "anatomical_violation": False}
    }
    res_normal = screener.evaluate(normal_metrics)
    assert res_normal["risk_score"] == 0.12
    assert res_normal["risk_level"] == "low_suspicion"


def test_disease_screening_aggregator_fail_safe():
    """Aggregator must never declare normal when image is ungradable or segmentation failed."""
    aggregator = DiseaseScreeningAggregator()

    dr_normal = {"predicted_grade": 0, "grade_label": "No Apparent DR", "confidence": 0.9}
    hypertension_normal = {"risk_score": 0.0, "risk_level": "normal_limits"}
    amd_normal = {"risk_score": 0.0, "risk_level": "low_suspicion"}

    # Case 1: Image is ungradable
    res_ungradable = aggregator.aggregate(
        dr_results=dr_normal,
        glaucoma_results={"risk_score": 0.12, "risk_level": "low_suspicion"},
        hypertension_results=hypertension_normal,
        amd_results=amd_normal,
        quality_status="ungradable"
    )
    assert res_ungradable["screening_status"] == "ungradable_repeat_image"

    # Case 2: Image needs review
    res_review = aggregator.aggregate(
        dr_results=dr_normal,
        glaucoma_results={"risk_score": 0.12, "risk_level": "low_suspicion"},
        hypertension_results=hypertension_normal,
        amd_results=amd_normal,
        quality_status="review_required"
    )
    assert res_review["screening_status"] == "requires_clinical_review"

    # Case 3: Glaucoma segmentation failed (disc_not_evaluable)
    res_failed_disc = aggregator.aggregate(
        dr_results=dr_normal,
        glaucoma_results={"risk_score": None, "risk_level": "requires_clinical_review", "status": "disc_not_evaluable"},
        hypertension_results=hypertension_normal,
        amd_results=amd_normal,
        quality_status="gradable"
    )
    assert res_failed_disc["screening_status"] == "requires_clinical_review"


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
