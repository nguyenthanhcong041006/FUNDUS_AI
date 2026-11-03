"""Unit tests for preprocessing modules."""

import numpy as np
import cv2
import pytest

from src.preprocessing.quality_assessment import QualityAssessor
from src.preprocessing.fov_detection import FOVDetector
from src.preprocessing.enhancement import FundusEnhancer
from src.preprocessing.normalization import resize_and_pad, unpad_and_restore, normalize_tensor


def test_quality_assessment():
    # Sharp image
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.circle(img, (50, 50), 30, (200, 200, 200), -1)

    assessor = QualityAssessor()
    res = assessor.assess(img)

    assert "status" in res
    assert "blur_score" in res
    assert "mean_brightness" in res
    assert "contrast_std" in res


def test_fov_detection():
    # Create circular retina on black background
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.circle(img, (100, 100), 80, (50, 120, 200), -1)

    detector = FOVDetector()
    mask, meta = detector.detect(img)

    assert mask.shape == (200, 200)
    assert meta["valid_pixel_ratio"] > 0.3
    assert np.sum(mask > 0) > 0


def test_enhancement():
    img = np.full((128, 128, 3), 120, dtype=np.uint8)
    enhancer = FundusEnhancer()
    out = enhancer.process(img)

    assert "green" in out
    assert "enhanced_green" in out
    assert "rgb" in out
    assert out["enhanced_green"].shape == (128, 128)


def test_normalization():
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    padded, meta = resize_and_pad(img, target_size=256)

    assert padded.shape == (256, 256, 3)
    restored = unpad_and_restore(padded, meta)
    assert restored.shape == (100, 200, 3)

    norm = normalize_tensor(padded)
    assert norm.dtype == np.float32

