"""DRIVE Dataset Loader reading from standardized manifest with letterboxing."""

from pathlib import Path
from typing import Optional, Callable
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from src.datasets.utils import resolve_path
from src.preprocessing.normalization import letterbox_rgb, letterbox_mask


class DRIVEManifestDataset(Dataset):
    """Vessel segmentation dataset for DRIVE, reading paths from manifest CSV."""

    def __init__(
        self,
        manifest_path: str = "manifests/vessel/drive_train.csv",
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

        # 1. Load RGB image at native resolution
        img_path = resolve_path(rec["image_path"])
        if not img_path.is_file():
            raise FileNotFoundError(f"Image not found: {img_path}")
        img_orig = Image.open(img_path).convert("RGB")
        img_np_orig = np.array(img_orig, dtype=np.uint8)

        # 2. Letterbox RGB image
        img_lb, pad_meta = letterbox_rgb(img_np_orig, target_size=self.image_size)
        img_float = img_lb.astype(np.float32) / 255.0

        # 3. Load & Letterbox Ground Truth vessel mask
        mask_path = resolve_path(rec.get("mask_path"))
        if not mask_path.is_file():
            raise FileNotFoundError(f"Vessel mask not found: {mask_path}")
        mask = Image.open(mask_path).convert("L")
        mask_raw = (np.array(mask, dtype=np.float32) > 0).astype(np.float32)
        mask_lb = letterbox_mask(mask_raw, target_size=self.image_size, pad_meta=pad_meta)

        # 4. Optional transform (augmentation)
        if self.transform is not None:
            img_float, mask_lb = self.transform(img_float, mask_lb)

        # 5. Load FOV mask if available
        fov_path_str = rec.get("fov_path")
        if fov_path_str:
            fov_path = resolve_path(fov_path_str)
            if fov_path.is_file():
                fov = Image.open(fov_path).convert("L")
                fov_raw = (np.array(fov, dtype=np.float32) > 0).astype(np.float32)
                fov_lb = letterbox_mask(fov_raw, target_size=self.image_size, pad_meta=pad_meta)
            else:
                fov_lb = np.ones((self.image_size, self.image_size), dtype=np.float32)
        else:
            fov_lb = np.ones((self.image_size, self.image_size), dtype=np.float32)

        # Convert to PyTorch tensors (CHW)
        img_tensor = torch.from_numpy(img_float.transpose(2, 0, 1)).float()
        mask_tensor = torch.from_numpy(mask_lb).unsqueeze(0).float()
        fov_tensor = torch.from_numpy(fov_lb).unsqueeze(0).float()

        return {
            "image": img_tensor,
            "mask": mask_tensor,
            "fov": fov_tensor,
            "image_id": rec["image_id"]
        }
