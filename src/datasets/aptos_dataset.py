"""APTOS 2019 Dataset Loader for 5-Class DR Severity Classification with letterboxing."""

from pathlib import Path
from typing import Optional, Callable
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from src.datasets.utils import resolve_path
from src.preprocessing.normalization import letterbox_rgb


class APTOSDataset(Dataset):
    """5-Class DR severity grading dataset for APTOS 2019."""

    def __init__(
        self,
        manifest_path: str = "manifests/dr_grading/aptos_train.csv",
        image_size: int = 512,
        transform: Optional[Callable] = None,
        limit: Optional[int] = None
    ):
        self.image_size = image_size
        self.transform = transform
        p = resolve_path(manifest_path)
        if not p.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")

        df = pd.read_csv(p)
        if limit:
            df = df.iloc[:limit]
        self.records = df.to_dict(orient="records")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int):
        rec = self.records[idx]

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

        label = int(rec.get("diagnosis", 0))

        img_tensor = torch.from_numpy(img_float.transpose(2, 0, 1)).float()
        label_tensor = torch.tensor(label, dtype=torch.long)

        return {
            "image": img_tensor,
            "label": label_tensor,
            "id_code": rec["id_code"]
        }
