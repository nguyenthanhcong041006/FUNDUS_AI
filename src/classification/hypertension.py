"""Hypertensive retinopathy signs assessment."""

from typing import Dict, Any, Optional


class HypertensionScreener:
    """Evaluates hypertensive retinopathy signs (arteriolar narrowing, AV ratio)."""

    def __init__(self, av_threshold: float = 0.60):
        self.av_threshold = av_threshold

    def evaluate(
        self,
        av_metrics: Optional[Dict[str, Any]] = None,
        hemorrhage_count: int = 0
    ) -> Dict[str, Any]:
        """Calculates risk score based on vascular signs."""
        av_ratio = av_metrics.get("av_pixel_ratio", 0.67) if av_metrics else 0.67

        if av_ratio < self.av_threshold:
            risk_score = 0.65
            risk_level = "suspicious_narrowing"
            note = f"Generalized arteriolar narrowing suggested (A/V ratio: {av_ratio:.2f})."
        else:
            risk_score = 0.15
            risk_level = "normal_limits"
            note = "No obvious vascular caliber changes detected."

        return {
            "disease": "hypertensive_retinopathy",
            "risk_score": round(risk_score, 2),
            "risk_level": risk_level,
            "av_ratio": av_ratio,
            "note": note
        }

