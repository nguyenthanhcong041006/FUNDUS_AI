"""Multi-disease screening aggregation and decision logic with fail-safe medical gates."""

from typing import Dict, Any, Optional


class DiseaseScreeningAggregator:
    """Aggregates multi-disease predictions with rigorous clinical fail-safes."""

    def aggregate(
        self,
        dr_results: Dict[str, Any],
        glaucoma_results: Dict[str, Any],
        hypertension_results: Dict[str, Any],
        amd_results: Dict[str, Any],
        quality_status: str = "gradable"
    ) -> Dict[str, Any]:
        """Synthesizes all individual screening scores into a structured summary."""
        glauc_risk = glaucoma_results.get("risk_score")
        glauc_level = glaucoma_results.get("risk_level", "low_suspicion")
        glauc_status = glaucoma_results.get("status", "evaluable")

        disease_predictions = {
            "diabetic_retinopathy": {
                "grade": dr_results.get("predicted_grade", 0),
                "grade_label": dr_results.get("grade_label", "No Apparent DR"),
                "confidence": dr_results.get("confidence", 0.0),
            },
            "glaucoma": {
                "risk_score": glauc_risk,
                "risk_level": glauc_level,
                "status": glauc_status,
                "vertical_cdr": glaucoma_results.get("vertical_cdr")
            },
            "hypertensive_retinopathy": {
                "risk_score": hypertension_results.get("risk_score", 0.0),
                "risk_level": hypertension_results.get("risk_level", "normal_limits"),
            },
            "amd": {
                "risk_score": amd_results.get("risk_score", 0.0),
                "risk_level": amd_results.get("risk_level", "low_suspicion"),
            }
        }

        # Determine overall screening triage status with clinical quality gates
        dr_grade = dr_results.get("predicted_grade", 0)
        amd_risk = amd_results.get("risk_score", 0.0)

        # Gate 1: Image Quality Fail-Safe
        if quality_status == "ungradable":
            triage_status = "ungradable_repeat_image"
        elif quality_status == "review_required":
            triage_status = "requires_clinical_review"
        # Gate 2: Severe disease requires urgent referral
        elif dr_grade >= 3 or (glauc_risk is not None and glauc_risk >= 0.75):
            triage_status = "urgent_referral"
        # Gate 3: Structural segmentation failure or missing optic disc
        elif glauc_status in ["disc_not_evaluable", "anatomical_anomaly_detected"] or glauc_risk is None:
            triage_status = "requires_clinical_review"
        # Gate 4: Mild/Moderate abnormalities
        elif dr_grade >= 1 or (glauc_risk is not None and glauc_risk >= 0.45) or amd_risk >= 0.50:
            triage_status = "requires_clinical_review"
        # Gate 5: Everything cleanly evaluated within normal limits
        else:
            triage_status = "no_apparent_abnormality"

        return {
            "disease_predictions": disease_predictions,
            "screening_status": triage_status,
            "disclaimer": (
                "Research prototype output only. Not for standalone clinical diagnosis. "
                "Any positive, borderline, or unevaluable screening flag requires examination by a licensed ophthalmologist."
            )
        }
