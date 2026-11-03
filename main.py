"""FUNDUS AI - Complete Pipeline Execution & Reporting Script.

Matches Milestone 1 and Phases 1-14 of FUNDUS_AI_Complete_Pipeline_Implementation_Plan.md.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, Optional
import cv2
import numpy as np
import pandas as pd

from src.preprocessing.loader import load_and_validate_image
from src.preprocessing.quality_assessment import QualityAssessor
from src.preprocessing.fov_detection import FOVDetector
from src.preprocessing.enhancement import FundusEnhancer
from src.preprocessing.normalization import resize_and_pad

from src.anatomical.optic_disc import OpticDiscSegmenter
from src.anatomical.optic_cup import OpticCupSegmenter
from src.anatomical.vessel_segmentation import VesselSegmenter
from src.anatomical.fovea_detection import FoveaDetector

from src.lesions.microaneurysm import MicroaneurysmDetector
from src.lesions.hemorrhage import HemorrhageDetector
from src.lesions.hard_exudate import HardExudateDetector
from src.lesions.cotton_wool import CottonWoolSpotDetector

from src.measurements.optic_disc_metrics import calculate_disc_cup_metrics
from src.measurements.vessel_metrics import calculate_vessel_metrics
from src.measurements.lesion_metrics import calculate_lesion_metrics
from src.measurements.spatial_metrics import calculate_spatial_distribution

from src.classification.diabetic_retinopathy import DRClassifier
from src.classification.glaucoma import GlaucomaScreener
from src.classification.hypertension import HypertensionScreener
from src.classification.amd import AMDScreener
from src.classification.disease_fusion import DiseaseScreeningAggregator

from src.visualization.mask_overlay import create_multi_layer_overlay
from src.visualization.bounding_boxes import draw_landmarks_and_boxes
from src.visualization.report_visualizer import generate_visual_report


def generate_synthetic_fundus_demo() -> np.ndarray:
    """Generates a realistic synthetic fundus image for pipeline testing."""
    h, w = 512, 512
    img = np.zeros((h, w, 3), dtype=np.uint8)

    # Circular retinal FOV with orange-red background
    cv2.circle(img, (256, 256), 230, (20, 60, 180), -1)

    # Optic Disc (bright yellow-orange disc at nasal side)
    od_center = (140, 256)
    cv2.circle(img, od_center, 38, (80, 190, 240), -1)
    # Optic Cup
    cv2.circle(img, od_center, 18, (120, 220, 255), -1)

    # Macula (darker avascular zone at temporal side)
    fovea_center = (330, 256)
    cv2.circle(img, fovea_center, 40, (15, 45, 140), -1)
    cv2.circle(img, fovea_center, 8, (10, 30, 110), -1)

    # Retinal vessel branches emerging from optic disc
    for dy in [-120, -70, 0, 70, 120]:
        pts = np.array([
            od_center,
            (od_center[0] + 80, od_center[1] + dy // 2),
            (256, 256 + dy),
            (380, 256 + int(dy * 1.3))
        ], dtype=np.int32)
        cv2.polylines(img, [pts], False, (10, 20, 100), thickness=3, lineType=cv2.LINE_AA)

    # Add a few mock lesions (Hard Exudates, Hemorrhages, MA)
    cv2.circle(img, (310, 220), 4, (0, 220, 240), -1)  # Hard exudate
    cv2.circle(img, (315, 225), 3, (0, 210, 230), -1)
    cv2.circle(img, (280, 300), 7, (5, 10, 90), -1)    # Hemorrhage
    cv2.circle(img, (350, 190), 2, (5, 5, 80), -1)     # Microaneurysm

    # Add subtle Gaussian blur to simulate camera optics
    img = cv2.GaussianBlur(img, (3, 3), 0.5)
    return img


def run_pipeline(
    image_path: Optional[str] = None,
    output_dir: str = "outputs/sample_001",
    target_size: int = 512,
    use_synthetic_demo: bool = False
) -> Dict[str, Any]:
    """Executes the complete retinal fundus analysis pipeline."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    masks_dir = out_dir / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)

    print(f"[*] Starting FUNDUS AI Pipeline...")

    # 1. Image Loading & Validation
    if use_synthetic_demo or image_path is None:
        print("  -> Generating synthetic fundus image for demonstration/testing...")
        image_bgr = generate_synthetic_fundus_demo()
        metadata = {
            "file_name": "synthetic_fundus_demo.png",
            "file_path": "memory://synthetic_fundus_demo.png",
            "original_height": 512,
            "original_width": 512,
            "channels": 3
        }
    else:
        print(f"  -> Loading image: {image_path}")
        image_bgr, metadata = load_and_validate_image(image_path, target_color="BGR")

    # Resize to base pipeline resolution if needed
    image_bgr, pad_meta = resize_and_pad(image_bgr, target_size=target_size)

    # 2. Retinal FOV Detection
    print("  -> Detecting Retinal Field of View (FOV)...")
    fov_detector = FOVDetector()
    fov_mask, fov_meta = fov_detector.detect(image_bgr)

    # 3. Image Quality Assessment
    print("  -> Assessing Image Quality & Gradability...")
    assessor = QualityAssessor()
    quality_report = assessor.assess(image_bgr, fov_mask=fov_mask)
    print(f"     Quality status: {quality_report['status']} (blur: {quality_report['blur_score']:.1f})")

    # 4. Enhancement
    print("  -> Applying Contrast Enhancement (CLAHE)...")
    enhancer = FundusEnhancer()
    enhancement_outputs = enhancer.process(image_bgr, fov_mask=fov_mask)
    green_ch = enhancement_outputs["green"]
    enhanced_green = enhancement_outputs["enhanced_green"]

    # 5. Anatomical Segmentation
    print("  -> Segmenting Optic Disc and Optic Cup...")
    od_segmenter = OpticDiscSegmenter()
    od_mask, od_metrics = od_segmenter.segment(image_bgr, fov_mask=fov_mask)

    oc_segmenter = OpticCupSegmenter()
    oc_mask, oc_metrics = oc_segmenter.segment(image_bgr, od_mask, od_metrics=od_metrics)

    print("  -> Segmenting Blood Vessels...")
    vessel_segmenter = VesselSegmenter()
    vessel_mask, vessel_metrics = vessel_segmenter.segment(image_bgr, fov_mask=fov_mask)

    print("  -> Localizing Fovea and Macular Zone...")
    fovea_detector = FoveaDetector()
    fovea_data, macula_mask = fovea_detector.detect(image_bgr, od_metrics=od_metrics, fov_mask=fov_mask)

    # 6. Lesion Detection
    print("  -> Detecting Retinal Lesions (MA, HE, EX, CWS)...")
    ma_det = MicroaneurysmDetector()
    ma_mask, ma_list, ma_summary = ma_det.detect(image_bgr, fov_mask=fov_mask, vessel_mask=vessel_mask)

    he_det = HemorrhageDetector()
    he_mask, he_list, he_summary = he_det.detect(
        image_bgr, fov_mask=fov_mask, vessel_mask=vessel_mask,
        fovea_coords=tuple(fovea_data["fovea_coordinates"])
    )

    ex_det = HardExudateDetector()
    ex_mask, ex_list, ex_summary = ex_det.detect(
        image_bgr, fov_mask=fov_mask, od_mask=od_mask,
        fovea_coords=tuple(fovea_data["fovea_coordinates"])
    )

    cws_det = CottonWoolSpotDetector()
    cws_mask, cws_list, cws_summary = cws_det.detect(image_bgr, fov_mask=fov_mask, od_mask=od_mask)

    # 7. Quantitative Measurements Engine
    print("  -> Computing Quantitative Measurements...")
    od_oc_calculated = calculate_disc_cup_metrics(od_mask, oc_mask)
    vessels_calculated = calculate_vessel_metrics(vessel_mask, fov_mask=fov_mask)

    all_lesions_list = [ma_summary, he_summary, ex_summary, cws_summary]
    spatial_distribution = calculate_spatial_distribution(
        lesion_masks={"ma": ma_mask, "he": he_mask, "ex": ex_mask, "cws": cws_mask},
        fovea_coords=fovea_data["fovea_coordinates"],
        od_coords=fovea_data["optic_disc_coordinates"],
        macula_radius_px=fovea_data["macula_radius_px"]
    )

    # 8. Disease Screening Logic
    print("  -> Running Disease Screening Logic...")
    dr_clf = DRClassifier()
    dr_pred = dr_clf.predict(image_bgr, lesion_summary={"lesions": all_lesions_list})

    glaucoma_scr = GlaucomaScreener()
    glaucoma_pred = glaucoma_scr.evaluate(od_oc_calculated)

    hypertension_scr = HypertensionScreener()
    hypertension_pred = hypertension_scr.evaluate()

    amd_scr = AMDScreener()
    amd_pred = amd_scr.evaluate(spatial_summary=spatial_distribution)

    aggregator = DiseaseScreeningAggregator()
    screening_result = aggregator.aggregate(
        dr_results=dr_pred,
        glaucoma_results=glaucoma_pred,
        hypertension_results=hypertension_pred,
        amd_results=amd_pred,
        quality_status=quality_report["status"]
    )

    # 9. Visualization & Overlay Construction
    print("  -> Building Visualization & Diagnostic Overlays...")
    layers = {
        "vessels": vessel_mask,
        "optic_disc": od_mask,
        "optic_cup": oc_mask,
        "macula_region": macula_mask,
        "microaneurysm": ma_mask,
        "hemorrhage": he_mask,
        "hard_exudate": ex_mask,
        "cotton_wool": cws_mask,
    }
    composite_overlay = create_multi_layer_overlay(image_bgr, layers, alpha=0.45)
    composite_annotated = draw_landmarks_and_boxes(
        composite_overlay,
        fovea_coords=fovea_data["fovea_coordinates"],
        od_coords=fovea_data["optic_disc_coordinates"],
        od_bbox=od_metrics.get("bbox_xywh", None)
    )

    od_oc_visual = create_multi_layer_overlay(
        image_bgr, {"optic_disc": od_mask, "optic_cup": oc_mask}, alpha=0.5
    )

    # 10. Save Outputs
    print(f"  -> Writing files to {out_dir}...")

    # Milestone 1 acceptance criteria files:
    cv2.imwrite(str(out_dir / "original.png"), image_bgr)
    cv2.imwrite(str(out_dir / "green_channel.png"), green_ch)
    cv2.imwrite(str(out_dir / "enhanced_green.png"), enhanced_green)
    cv2.imwrite(str(out_dir / "fov_mask.png"), fov_mask)
    cv2.imwrite(str(out_dir / "annotated.png"), composite_annotated)

    # Save individual masks
    cv2.imwrite(str(masks_dir / "optic_disc.png"), od_mask)
    cv2.imwrite(str(masks_dir / "optic_cup.png"), oc_mask)
    cv2.imwrite(str(masks_dir / "vessels.png"), vessel_mask)
    cv2.imwrite(str(masks_dir / "hard_exudates.png"), ex_mask)
    cv2.imwrite(str(masks_dir / "hemorrhages.png"), he_mask)
    cv2.imwrite(str(masks_dir / "microaneurysms.png"), ma_mask)
    cv2.imwrite(str(masks_dir / "cotton_wool.png"), cws_mask)

    # Visual report panel
    report_panel_path = out_dir / "report_panel.png"
    generate_visual_report(
        original_bgr=image_bgr,
        enhanced_green=enhanced_green,
        fov_mask=fov_mask,
        od_oc_overlay=od_oc_visual,
        vessel_mask=vessel_mask,
        annotated_composite=composite_annotated,
        output_path=report_panel_path
    )

    # Final Structured JSON Report (Section 18 format)
    full_report = {
        "image_id": metadata["file_name"],
        "quality": quality_report,
        "anatomical_structures": {
            "optic_disc": od_oc_calculated["optic_disc"],
            "optic_cup": od_oc_calculated["optic_cup"],
            "fovea": {
                "x": fovea_data["fovea_coordinates"][0],
                "y": fovea_data["fovea_coordinates"][1],
                "disc_fovea_distance_px": fovea_data["disc_fovea_distance_px"]
            },
            "blood_vessels": {
                "density_percent": vessels_calculated["density_percent"],
                "total_length_px": vessels_calculated["total_length_px"],
                "estimated_mean_width_px": vessels_calculated["estimated_mean_width_px"],
                "branch_point_count": vessels_calculated["branch_point_count"]
            }
        },
        "lesions": [l for l in all_lesions_list if l["count"] > 0],
        "spatial_distribution": spatial_distribution,
        "disease_predictions": screening_result["disease_predictions"],
        "screening_status": screening_result["screening_status"],
        "clinical_disclaimer": screening_result["disclaimer"],
        "model_versions": {
            "vessel": "v1.0",
            "optic_disc_cup": "v1.0",
            "lesion": "v1.0",
            "classifier": "v1.0"
        }
    }

    # Write quality_report.json (Milestone 1 requirement) and full report.json
    with open(out_dir / "quality_report.json", "w", encoding="utf-8") as f:
        json.dump(quality_report, f, indent=2)

    with open(out_dir / "report.json", "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)

    # Save summary CSV table of measurements
    csv_rows = [
        {"metric": "Image Quality Status", "value": quality_report["status"], "unit": "category"},
        {"metric": "Blur Score", "value": quality_report["blur_score"], "unit": "variance"},
        {"metric": "FOV Area Ratio", "value": fov_meta["valid_pixel_ratio"], "unit": "ratio"},
        {"metric": "Optic Disc Area", "value": od_oc_calculated["optic_disc"]["area_px2"], "unit": "px2"},
        {"metric": "Vertical CDR", "value": od_oc_calculated["optic_cup"]["vertical_cdr"], "unit": "ratio"},
        {"metric": "Vessel Density", "value": vessels_calculated["density_percent"], "unit": "%"},
        {"metric": "Vessel Length", "value": vessels_calculated["total_length_px"], "unit": "px"},
        {"metric": "Disc-Fovea Distance", "value": fovea_data["disc_fovea_distance_px"], "unit": "px"},
        {"metric": "DR Grade", "value": dr_pred["predicted_grade"], "unit": "grade_0_to_4"},
        {"metric": "Glaucoma Risk Score", "value": glaucoma_pred["risk_score"], "unit": "score"},
        {"metric": "Overall Screening Status", "value": screening_result["screening_status"], "unit": "category"},
    ]
    pd.DataFrame(csv_rows).to_csv(out_dir / "measurements.csv", index=False)

    print(f"[OK] Analysis complete! Results successfully saved in: {out_dir.resolve()}")
    return full_report


def main():
    parser = argparse.ArgumentParser(description="FUNDUS AI Complete Retinal Analysis Pipeline")
    parser.add_argument("--image", type=str, default=None, help="Path to retinal fundus image")
    parser.add_argument("--output_dir", type=str, default="outputs/sample_001", help="Output directory")
    parser.add_argument("--target_size", type=int, default=512, help="Target processing square resolution")
    parser.add_argument("--demo", action="store_true", help="Run with synthetic fundus demo image")

    args = parser.parse_args()

    use_demo = args.demo or (args.image is None)
    run_pipeline(
        image_path=args.image,
        output_dir=args.output_dir,
        target_size=args.target_size,
        use_synthetic_demo=use_demo
    )


if __name__ == "__main__":
    main()

