"""Glaucoma screening logic based on optic nerve head features."""

from typing import Dict, Any, Optional


class GlaucomaScreener:
    """Evaluates glaucoma risk indicators based on Vertical CDR and model outputs."""

    def __init__(self, model_path: Optional[str] = None, vcdr_threshold: float = 0.65):
        self.model_path = model_path
        self.vcdr_threshold = vcdr_threshold

    def evaluate(
        self,
        od_oc_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Assesses glaucoma suspicion score.

        Args:
            od_oc_metrics: Dict containing vertical_cdr and area_ratio.

        Returns:
            Dict with glaucoma risk score, risk level, and rationale.
        """
        cup_metrics = od_oc_metrics.get("optic_cup", {})
        vcdr = cup_metrics.get("vertical_cdr", 0.0)
        area_ratio = cup_metrics.get("cup_disc_area_ratio", 0.0)

        # Baseline rule-informed risk score
        if vcdr >= self.vcdr_threshold or area_ratio >= 0.50:
            risk_level = "elevated_suspicion"
            risk_score = 0.78
            rationale = (
                f"Elevated vertical cup-to-disc ratio ({vcdr:.2f} >= {self.vcdr_threshold}) "
                "or enlarged cup area. Comprehensive ophthalmological exam and OCT recommended."
            )
        elif vcdr >= 0.50:
            risk_level = "borderline"
            risk_score = 0.45
            rationale = f"Borderline cup-to-disc ratio ({vcdr:.2f}). Regular monitoring recommended."
        else:
            risk_level = "low_suspicion"
            risk_score = 0.12
            rationale = f"Cup-to-disc ratio within typical range ({vcdr:.2f})."

        return {
            "disease": "glaucoma",
            "risk_score": round(risk_score, 2),
            "risk_level": risk_level,
            "vertical_cdr": vcdr,
            "rationale": rationale,
            "clinical_limitation": "A fundus photo alone cannot diagnose glaucoma. Visual field and IOP testing are required."
        }

