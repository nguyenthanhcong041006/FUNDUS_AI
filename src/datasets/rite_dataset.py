"""RITE Dataset Loader for Artery and Vein 3-Class Segmentation."""

from pathlib import Path
from typing import Optional, Callable
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from src.datasets.utils import resolve_path


class RITEManifestDataset(Dataset):
    """Artery/Vein 3-class segmentation dataset for RITE."""

    def __init__(
        self,
        manifest_path: str = "manifests/artery_vein/rite_train.csv",
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

        # Load RGB image
        img_path = resolve_path(rec["image_path"])
        img = Image.open(img_path).convert("RGB")
        img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
        img_np = np.array(img, dtype=np.float32) / 255.0

        # Load AV Mask (RGB format: Red=Artery, Blue=Vein, Green=Overlap)
        av_path = resolve_path(rec["av_mask_path"])
        av_img = Image.open(av_path).convert("RGB")
        av_img = av_img.resize((self.image_size, self.image_size), Image.NEAREST)
        av_np = np.array(av_img)

        # Build 3-class mask: 0=Background, 1=Artery, 2=Vein
        # Red channel high & Blue low -> Artery
        # Blue channel high & Red low -> Vein
        mask_3class = np.zeros((self.image_size, self.image_size), dtype=np.int64)
        r, g, b = av_np[:, :, 0], av_np[:, :, 1], av_np[:, :, 2]

        is_artery = (r > 128) & (b < 128)
        is_vein = (b > 128) & (r < 128)

        mask_3class[is_artery] = 1
        mask_3class[is_vein] = 2

        img_tensor = torch.from_numpy(img_np.transpose(2, 0, 1)).float()
        mask_tensor = torch.from_numpy(mask_3class).long()

        return {
            "image": img_tensor,
            "mask": mask_tensor,
            "image_id": rec["image_id"]
        }
