"""End-to-end integration test for FUNDUS AI pipeline."""

from pathlib import Path
import json
from main import run_pipeline


def test_pipeline_execution(tmp_path):
    output_dir = tmp_path / "test_run"
    report = run_pipeline(output_dir=str(output_dir), use_synthetic_demo=True)

    assert report is not None
    assert "quality" in report
    assert "anatomical_structures" in report
    assert "screening_status" in report

    # Verify generated files
    assert (output_dir / "original.png").exists()
    assert (output_dir / "green_channel.png").exists()
    assert (output_dir / "enhanced_green.png").exists()
    assert (output_dir / "fov_mask.png").exists()
    assert (output_dir / "annotated.png").exists()
    assert (output_dir / "report_panel.png").exists()
    assert (output_dir / "quality_report.json").exists()
    assert (output_dir / "report.json").exists()
    assert (output_dir / "measurements.csv").exists()

