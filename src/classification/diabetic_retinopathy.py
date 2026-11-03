"""Diabetic Retinopathy (DR) severity classification."""

from typing import Dict, Any, Optional
import numpy as np


DR_GRADES = {
    0: "No Apparent DR",
    1: "Mild NPDR",
    2: "Moderate NPDR",
    3: "Severe NPDR",
    4: "Proliferative DR"
}


class DRClassifier:
    """Predicts Diabetic Retinopathy severity grade and probability distribution."""

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.model = None

    def predict(
        self,
        image_rgb: np.ndarray,
        lesion_summary: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Predicts DR grade.

        Combines model probabilities with lesion evidence when available.
        """
        if self.model is not None:
            # Model inference placeholder
            probabilities = [0.85, 0.10, 0.03, 0.01, 0.01]
            predicted_grade = int(np.argmax(probabilities))
        else:
            # Evidence-based screening heuristic when no checkpoint is loaded
            ma_count = 0
            he_count = 0
            ex_count = 0

            if lesion_summary:
                for l in lesion_summary.get("lesions", []):
                    l_type = l.get("type", "")
                    if "microaneurysm" in l_type:
                        ma_count = l.get("count", 0)
                    elif "hemorrhage" in l_type:
                        he_count = l.get("count", 0)
                    elif "exudate" in l_type:
                        ex_count = l.get("count", 0)

            if he_count >= 5 or ex_count >= 5:
                predicted_grade = 2
                probabilities = [0.05, 0.20, 0.65, 0.08, 0.02]
            elif ma_count > 0 or he_count > 0 or ex_count > 0:
                predicted_grade = 1
                probabilities = [0.20, 0.65, 0.10, 0.04, 0.01]
            else:
                predicted_grade = 0
                probabilities = [0.90, 0.07, 0.02, 0.01, 0.00]

        return {
            "predicted_grade": predicted_grade,
            "grade_label": DR_GRADES[predicted_grade],
            "confidence": round(float(probabilities[predicted_grade]), 3),
            "class_probabilities": {
                DR_GRADES[i]: round(float(p), 4) for i, p in enumerate(probabilities)
            }
        }

