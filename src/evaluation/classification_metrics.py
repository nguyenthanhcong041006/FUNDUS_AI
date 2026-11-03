"""Classification and screening evaluation metrics."""

from typing import Dict, Any, List, Optional
import numpy as np


def calculate_classification_metrics(
    y_true: List[int] | np.ndarray,
    y_pred: List[int] | np.ndarray,
    class_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Calculates macro-averaged and per-class classification metrics."""
    yt = np.array(y_true)
    yp = np.array(y_pred)

    classes = np.unique(np.concatenate([yt, yp]))
    classes.sort()

    total = len(yt)
    correct = np.sum(yt == yp)
    accuracy = float(correct / max(total, 1))

    per_class = {}
    macro_f1 = []

    for c in classes:
        tp = np.sum((yp == c) & (yt == c))
        fp = np.sum((yp == c) & (yt != c))
        fn = np.sum((yp != c) & (yt == c))
        tn = np.sum((yp != c) & (yt != c))

        prec = float(tp / max(tp + fp, 1e-6))
        rec = float(tp / max(tp + fn, 1e-6))
        spec = float(tn / max(tn + fp, 1e-6))
        f1 = float(2 * prec * rec / max(prec + rec, 1e-6))

        c_name = class_names[c] if (class_names and c < len(class_names)) else f"class_{c}"
        per_class[c_name] = {
            "precision": round(prec, 4),
            "recall_sensitivity": round(rec, 4),
            "specificity": round(spec, 4),
            "f1_score": round(f1, 4),
            "support": int(np.sum(yt == c))
        }
        macro_f1.append(f1)

    return {
        "accuracy": round(accuracy, 4),
        "macro_f1": round(float(np.mean(macro_f1)), 4) if macro_f1 else 0.0,
        "per_class": per_class,
        "total_samples": total
    }

