"""ODIR-5K Multi-Label Disease Classification Dataset Loader with letterboxing."""

from pathlib import Path
from typing import Optional, Callable
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from src.datasets.utils import resolve_path
from src.preprocessing.normalization import letterbox_rgb

ODIR_DISEASE_COLUMNS = ["N", "D", "G", "C", "A", "H", "M", "O"]


class ODIRDataset(Dataset):
    """Multi-label disease classification dataset for ODIR-5K."""

    def __init__(
        self,
        manifest_path: str = "manifests/multi_disease/odir_train.csv",
        image_size: int = 512,
        eye_mode: str = "both",  # 'left', 'right', or 'both'
        transform: Optional[Callable] = None,
        limit: Optional[int] = None
    ):
        self.image_size = image_size
        self.eye_mode = eye_mode
        self.transform = transform

        p = resolve_path(manifest_path)
        if not p.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")

        df = pd.read_csv(p)
        if limit:
            df = df.iloc[:limit]

        self.samples = []
        for _, row in df.iterrows():
            target_labels = row[ODIR_DISEASE_COLUMNS].values.astype(np.float32)

            left_path = resolve_path(row.get("left_image_path"))
            if eye_mode in ["left", "both"] and left_path.is_file():
                self.samples.append({
                    "patient_id": row["ID"],
                    "image_path": str(left_path),
                    "labels": target_labels,
                    "laterality": "left"
                })

            right_path = resolve_path(row.get("right_image_path"))
            if eye_mode in ["right", "both"] and right_path.is_file():
                self.samples.append({
                    "patient_id": row["ID"],
                    "image_path": str(right_path),
                    "labels": target_labels,
                    "laterality": "right"
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        rec = self.samples[idx]

        # 1. Load image at original resolution
        img_path = resolve_path(rec["image_path"])
        if not img_path.is_file():
            raise FileNotFoundError(f"Image not found: {img_path}")
        img = Image.open(img_path).convert("RGB")
        img_np_orig = np.array(img, dtype=np.uint8)

        # 2. Letterbox RGB image
        img_lb, _ = letterbox_rgb(img_np_orig, target_size=self.image_size)
        img_float = img_lb.astype(np.float32) / 255.0

        # 3. Apply optional data augmentation
        if self.transform is not None:
            img_float = self.transform(img_float)

        img_tensor = torch.from_numpy(img_float.transpose(2, 0, 1)).float()
        labels_tensor = torch.from_numpy(rec["labels"]).float()

        return {
            "image": img_tensor,
            "labels": labels_tensor,
            "patient_id": rec["patient_id"],
            "laterality": rec["laterality"]
        }
