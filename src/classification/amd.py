"""Age-related Macular Degeneration (AMD) signs assessment."""

from typing import Dict, Any, Optional


class AMDScreener:
    """Evaluates AMD risk indicators based on drusen presence and macular location."""

    def evaluate(
        self,
        drusen_summary: Optional[Dict[str, Any]] = None,
        spatial_summary: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Calculates AMD suspicion score."""
        drusen_count = drusen_summary.get("count", 0) if drusen_summary else 0
        macular_involvement = spatial_summary.get("macular_involvement", False) if spatial_summary else False

        if drusen_count > 10 and macular_involvement:
            risk_score = 0.75
            risk_level = "elevated_suspicion"
            note = f"Multiple drusen ({drusen_count}) with macular involvement."
        elif drusen_count > 0:
            risk_score = 0.35
            risk_level = "borderline"
            note = f"Sparse drusen detected ({drusen_count})."
        else:
            risk_score = 0.08
            risk_level = "low_suspicion"
            note = "No significant drusen observed."

        return {
            "disease": "amd",
            "risk_score": round(risk_score, 2),
            "risk_level": risk_level,
            "drusen_count": drusen_count,
            "macular_involvement": macular_involvement,
            "note": note,
            "clinical_limitation": "OCT is clinically necessary for complete retinal layer evaluation."
        }

