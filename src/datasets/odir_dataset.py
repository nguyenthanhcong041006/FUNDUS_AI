"""ODIR-5K Multi-Label Disease Classification Dataset Loader."""

from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from src.datasets.utils import resolve_path


ODIR_DISEASE_COLUMNS = ["N", "D", "G", "C", "A", "H", "M", "O"]


class ODIRDataset(Dataset):
    """Multi-label disease classification dataset for ODIR-5K."""

    def __init__(
        self,
        manifest_path: str = "manifests/multi_disease/odir_train.csv",
        image_size: int = 512,
        eye_mode: str = "both",  # 'left', 'right', or 'both' (yielding 2 samples per patient)
        limit: Optional[int] = None
    ):
        self.image_size = image_size
        self.eye_mode = eye_mode

        p = resolve_path(manifest_path)
        if not p.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")

        df = pd.read_csv(p)
        if limit:
            df = df.iloc[:limit]

        # Expand records based on eye_mode
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

        img_path = resolve_path(rec["image_path"])
        img = Image.open(img_path).convert("RGB")
        img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
        img_np = np.array(img, dtype=np.float32) / 255.0

        img_tensor = torch.from_numpy(img_np.transpose(2, 0, 1)).float()
        labels_tensor = torch.from_numpy(rec["labels"]).float()

        return {
            "image": img_tensor,
            "labels": labels_tensor,
            "patient_id": rec["patient_id"],
            "laterality": rec["laterality"]
        }
