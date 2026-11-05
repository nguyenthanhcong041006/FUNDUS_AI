"""IDRiD Multi-Lesion Segmentation Dataset Loader."""

from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from src.datasets.utils import resolve_path


class IDRiDLesionDataset(Dataset):
    """4-Channel multi-lesion segmentation dataset for IDRiD (MA, HE, EX, SE)."""

    def __init__(
        self,
        manifest_path: str = "manifests/dr_lesions/idrid_lesions_train.csv",
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

    def _load_single_mask(self, mask_path) -> np.ndarray:
        p = resolve_path(mask_path)
        if p.is_file():
            m = Image.open(p).convert("L")
            m = m.resize((self.image_size, self.image_size), Image.NEAREST)
            return (np.array(m, dtype=np.float32) > 0).astype(np.float32)
        return np.zeros((self.image_size, self.image_size), dtype=np.float32)

    def __getitem__(self, idx: int):
        rec = self.records[idx]

        # Load RGB image
        img_path = resolve_path(rec["image_path"])
        img = Image.open(img_path).convert("RGB")
        img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
        img_np = np.array(img, dtype=np.float32) / 255.0

        # Load 4 lesion masks: 0: MA, 1: HE, 2: EX, 3: SE
        ma_mask = self._load_single_mask(rec.get("ma_mask_path", ""))
        he_mask = self._load_single_mask(rec.get("he_mask_path", ""))
        ex_mask = self._load_single_mask(rec.get("ex_mask_path", ""))
        se_mask = self._load_single_mask(rec.get("se_mask_path", ""))

        # 4-channel target mask
        target_masks = np.stack([ma_mask, he_mask, ex_mask, se_mask], axis=0)

        img_tensor = torch.from_numpy(img_np.transpose(2, 0, 1)).float()
        mask_tensor = torch.from_numpy(target_masks).float()

        return {
            "image": img_tensor,
            "mask": mask_tensor,
            "image_id": rec["image_id"]
        }
