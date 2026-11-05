"""Training pipelines for FUNDUS AI modules."""

from .train_vessels import train_vessels
from .train_artery_vein import train_artery_vein
from .train_lesions import train_lesions
from .train_dr_grading import train_dr_grading
from .train_multi_disease import train_multi_disease

__all__ = [
    "train_vessels",
    "train_artery_vein",
    "train_lesions",
    "train_dr_grading",
    "train_multi_disease",
]

