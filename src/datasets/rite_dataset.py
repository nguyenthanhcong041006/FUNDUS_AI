"""RITE Dataset Loader for Artery and Vein 3-Class Segmentation with letterboxing."""

from pathlib import Path
from typing import Optional, Callable
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from src.datasets.utils import resolve_path
from src.preprocessing.normalization import letterbox_rgb, letterbox_mask


class RITEManifestDataset(Dataset):
    """Artery/Vein 3-class segmentation dataset for RITE."""

    def __init__(
        self,
        manifest_path: str = "manifests/artery_vein/rite_train.csv",
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

        # 3. Load AV Mask at native resolution (RGB format: Red=Artery, Blue=Vein, Green=Overlap)
        av_path = resolve_path(rec["av_mask_path"])
        if not av_path.is_file():
            raise FileNotFoundError(f"AV mask not found: {av_path}")
        av_img = Image.open(av_path).convert("RGB")
        av_np_orig = np.array(av_img)

        # Build native 3-class mask: 0=Background, 1=Artery, 2=Vein
        mask_3class_native = np.zeros((av_np_orig.shape[0], av_np_orig.shape[1]), dtype=np.uint8)
        r, b = av_np_orig[:, :, 0], av_np_orig[:, :, 2]
        is_artery = (r > 128) & (b < 128)
        is_vein = (b > 128) & (r < 128)
        mask_3class_native[is_artery] = 1
        mask_3class_native[is_vein] = 2

        # 4. Letterbox 3-class mask with nearest interpolation
        mask_lb = letterbox_mask(mask_3class_native, target_size=self.image_size, pad_meta=pad_meta)

        # 5. Optional transform
        if self.transform is not None:
            img_float, mask_lb = self.transform(img_float, mask_lb)

        img_tensor = torch.from_numpy(img_float.transpose(2, 0, 1)).float()
        mask_tensor = torch.from_numpy(mask_lb).long()

        return {
            "image": img_tensor,
            "mask": mask_tensor,
            "image_id": rec["image_id"]
        }
