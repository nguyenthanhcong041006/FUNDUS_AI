"""APTOS 2019 Dataset Loader for 5-Class DR Severity Classification."""

from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from src.datasets.utils import resolve_path


class APTOSDataset(Dataset):
    """5-Class DR severity grading dataset for APTOS 2019."""

    def __init__(
        self,
        manifest_path: str = "manifests/dr_grading/aptos_train.csv",
        image_size: int = 512,
        limit: Optional[int] = None
    ):
        self.image_size = image_size
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

        img_path = resolve_path(rec["image_path"])
        img = Image.open(img_path).convert("RGB")
        img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
        img_np = np.array(img, dtype=np.float32) / 255.0

        label = int(rec.get("diagnosis", 0))

        img_tensor = torch.from_numpy(img_np.transpose(2, 0, 1)).float()
        label_tensor = torch.tensor(label, dtype=torch.long)

        return {
            "image": img_tensor,
            "label": label_tensor,
            "id_code": rec["id_code"]
        }
