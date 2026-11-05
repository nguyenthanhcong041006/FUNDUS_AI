"""DRIVE Dataset Loader reading from standardized manifest."""

from pathlib import Path
from typing import Optional, Callable
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from src.datasets.utils import resolve_path


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

        # Load RGB image
        img_path = resolve_path(rec["image_path"])
        img = Image.open(img_path).convert("RGB")
        img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
        img_np = np.array(img, dtype=np.float32) / 255.0

        # Load Ground Truth vessel mask
        mask_path = resolve_path(rec.get("mask_path"))
        if mask_path.is_file():
            mask = Image.open(mask_path).convert("L")
            mask = mask.resize((self.image_size, self.image_size), Image.NEAREST)
            mask_np = (np.array(mask, dtype=np.float32) > 0).astype(np.float32)
        else:
            mask_np = np.zeros((self.image_size, self.image_size), dtype=np.float32)

        # Load FOV mask if available
        fov_path = resolve_path(rec.get("fov_path"))
        if fov_path.is_file():
            fov = Image.open(fov_path).convert("L")
            fov = fov.resize((self.image_size, self.image_size), Image.NEAREST)
            fov_np = (np.array(fov, dtype=np.float32) > 0).astype(np.float32)
        else:
            fov_np = np.ones((self.image_size, self.image_size), dtype=np.float32)

        # Convert to PyTorch tensors (CHW)
        img_tensor = torch.from_numpy(img_np.transpose(2, 0, 1)).float()
        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).float()
        fov_tensor = torch.from_numpy(fov_np).unsqueeze(0).float()

        return {
            "image": img_tensor,
            "mask": mask_tensor,
            "fov": fov_tensor,
            "image_id": rec["image_id"]
        }
