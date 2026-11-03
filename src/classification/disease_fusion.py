"""Multi-disease screening aggregation and decision logic."""

from typing import Dict, Any, List


class DiseaseScreeningAggregator:
    """Aggregates multi-disease predictions, keeping tasks distinct and transparent."""

    def aggregate(
        self,
        dr_results: Dict[str, Any],
        glaucoma_results: Dict[str, Any],
        hypertension_results: Dict[str, Any],
        amd_results: Dict[str, Any],
        quality_status: str = "gradable"
    ) -> Dict[str, Any]:
        """Synthesizes all individual screening scores into a structured summary."""
        disease_predictions = {
            "diabetic_retinopathy": {
                "grade": dr_results.get("predicted_grade", 0),
                "grade_label": dr_results.get("grade_label", "No Apparent DR"),
                "confidence": dr_results.get("confidence", 0.0),
            },
            "glaucoma": {
                "risk_score": glaucoma_results.get("risk_score", 0.0),
                "risk_level": glaucoma_results.get("risk_level", "low_suspicion"),
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

        # Determine overall screening triage status
        dr_grade = dr_results.get("predicted_grade", 0)
        glauc_risk = glaucoma_results.get("risk_score", 0.0)
        amd_risk = amd_results.get("risk_score", 0.0)

        if quality_status == "ungradable":
            triage_status = "ungradable_repeat_image"
        elif dr_grade >= 3 or glauc_risk >= 0.75:
            triage_status = "urgent_referral"
        elif dr_grade >= 1 or glauc_risk >= 0.45 or amd_risk >= 0.50:
            triage_status = "requires_clinical_review"
        else:
            triage_status = "no_apparent_abnormality"

        return {
            "disease_predictions": disease_predictions,
            "screening_status": triage_status,
            "disclaimer": (
                "Research prototype output only. Not for standalone clinical diagnosis. "
                "Any positive or borderline screening flag requires examination by a licensed ophthalmologist."
            )
        }

