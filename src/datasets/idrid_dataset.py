"""IDRiD Multi-Lesion Segmentation Dataset Loader with aspect-preserving letterboxing."""

from pathlib import Path
from typing import Optional, Callable
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from src.datasets.utils import resolve_path
from src.preprocessing.normalization import letterbox_rgb, letterbox_mask


class IDRiDLesionDataset(Dataset):
    """4-Channel multi-lesion segmentation dataset for IDRiD (MA, HE, EX, SE)."""

    def __init__(
        self,
        manifest_path: str = "manifests/dr_lesions/idrid_lesions_train.csv",
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

    def _load_raw_single_mask(self, mask_path, orig_h: int, orig_w: int) -> np.ndarray:
        if not mask_path or pd.isna(mask_path) or str(mask_path).strip() == "":
            return np.zeros((orig_h, orig_w), dtype=np.float32)
        p = resolve_path(mask_path)
        if not p.is_file():
            raise FileNotFoundError(f"Lesion mask path specified but not found on disk: {p}")
        m = Image.open(p).convert("L")
        return (np.array(m, dtype=np.float32) > 0).astype(np.float32)

    def __getitem__(self, idx: int):
        rec = self.records[idx]

        # 1. Load RGB image at native resolution
        img_path = resolve_path(rec["image_path"])
        if not img_path.is_file():
            raise FileNotFoundError(f"Image not found: {img_path}")
        img_orig = Image.open(img_path).convert("RGB")
        img_np_orig = np.array(img_orig, dtype=np.uint8)
        orig_h, orig_w = img_np_orig.shape[:2]

        # 2. Letterbox RGB image
        img_lb, pad_meta = letterbox_rgb(img_np_orig, target_size=self.image_size)
        img_float = img_lb.astype(np.float32) / 255.0

        # 3. Load 4 lesion masks at native resolution: 0: MA, 1: HE, 2: EX, 3: SE
        ma_raw = self._load_raw_single_mask(rec.get("ma_mask_path", ""), orig_h, orig_w)
        he_raw = self._load_raw_single_mask(rec.get("he_mask_path", ""), orig_h, orig_w)
        ex_raw = self._load_raw_single_mask(rec.get("ex_mask_path", ""), orig_h, orig_w)
        se_raw = self._load_raw_single_mask(rec.get("se_mask_path", ""), orig_h, orig_w)

        # 4. Letterbox each lesion mask using same pad_meta
        ma_lb = letterbox_mask(ma_raw, target_size=self.image_size, pad_meta=pad_meta)
        he_lb = letterbox_mask(he_raw, target_size=self.image_size, pad_meta=pad_meta)
        ex_lb = letterbox_mask(ex_raw, target_size=self.image_size, pad_meta=pad_meta)
        se_lb = letterbox_mask(se_raw, target_size=self.image_size, pad_meta=pad_meta)

        target_masks = np.stack([ma_lb, he_lb, ex_lb, se_lb], axis=0)  # (4, 512, 512)

        # 5. Optional transform
        if self.transform is not None:
            img_float, target_masks = self.transform(img_float, target_masks)

        img_tensor = torch.from_numpy(img_float.transpose(2, 0, 1)).float()
        mask_tensor = torch.from_numpy(target_masks).float()

        return {
            "image": img_tensor,
            "mask": mask_tensor,
            "image_id": rec["image_id"]
        }
