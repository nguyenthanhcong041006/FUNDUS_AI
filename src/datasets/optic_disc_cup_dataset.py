"""Optic Disc & Cup Dataset Loaders for Drishti-GS1, PAPILA, and REFUGE2.

Features:
- Aspect-ratio preserving letterboxing (letterbox_rgb, letterbox_mask)
- Synchronized medical data augmentations (flips, rotations, scaling, CLAHE, jitter)
- Multi-channel targets: Channel 0 = Optic Disc (OD), Channel 1 = Optic Cup (OC)
- Multi-domain dataset support (REFUGE2 + Drishti-GS1)
"""

from pathlib import Path
from typing import Optional, Callable, List, Dict, Any, Union
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, ConcatDataset

from src.datasets.utils import resolve_path
from src.preprocessing.normalization import letterbox_rgb, letterbox_mask


class DrishtiDataset(Dataset):
    """Optic Disc and Cup segmentation dataset for Drishti-GS1.
    
    Provides 2-channel targets with aspect-ratio preserving letterboxing:
      Channel 0: Optic Disc (OD)
      Channel 1: Optic Cup (OC)
    """

    def __init__(
        self,
        manifest_path: str = "manifests/optic_disc_cup/drishti_train.csv",
        image_size: int = 512,
        binary_threshold: Optional[int] = 128,  # Consensus threshold (>=2 experts agree)
        transform: Optional[Callable] = None,
        limit: Optional[int] = None
    ):
        self.image_size = image_size
        self.binary_threshold = binary_threshold
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

    def _load_raw_mask(self, mask_path_str: str) -> Optional[np.ndarray]:
        if not mask_path_str:
            return None
        p = resolve_path(mask_path_str)
        if not p.is_file():
            raise FileNotFoundError(f"Drishti mask file not found: {p}")
        mask_img = Image.open(p).convert("L")
        arr = np.array(mask_img, dtype=np.float32)
        if self.binary_threshold is not None:
            return (arr >= self.binary_threshold).astype(np.float32)
        return arr / 255.0

    def __getitem__(self, idx: int):
        rec = self.records[idx]

        # 1. Load RGB Image at original resolution
        img_path = resolve_path(rec["image_path"])
        if not img_path.is_file():
            raise FileNotFoundError(f"Image not found: {img_path}")
        img_orig = Image.open(img_path).convert("RGB")
        img_np_orig = np.array(img_orig, dtype=np.uint8)

        # 2. Letterbox RGB image
        img_lb, pad_meta = letterbox_rgb(img_np_orig, target_size=self.image_size)
        img_float = img_lb.astype(np.float32) / 255.0

        # 3. Load & Letterbox OD and Cup Masks
        od_raw = self._load_raw_mask(rec.get("od_mask_path", ""))
        cup_raw = self._load_raw_mask(rec.get("cup_mask_path", ""))

        if od_raw is None or cup_raw is None:
            raise ValueError(f"Missing OD or Cup mask for record {rec.get('image_id')}")

        od_lb = letterbox_mask(od_raw, target_size=self.image_size, pad_meta=pad_meta)
        cup_lb = letterbox_mask(cup_raw, target_size=self.image_size, pad_meta=pad_meta)

        # Multi-channel target: 2 channels (Disc, Cup)
        masks_2ch = np.stack([od_lb, cup_lb], axis=0)  # (2, H, W)

        # 4. Apply optional joint data augmentations
        if self.transform is not None:
            img_float, masks_2ch = self.transform(img_float, masks_2ch)

        # 5. Reference CDR
        cdr_ref_raw = rec.get("cdr_ref")
        cdr_ref = float(cdr_ref_raw) if cdr_ref_raw != "" and not pd.isna(cdr_ref_raw) else -1.0

        # 6. Clinical Diagnosis
        diag_str = str(rec.get("diagnosis", "")).lower()
        diag_label = 1 if "glaucomatous" in diag_str else (0 if "normal" in diag_str else -1)

        img_tensor = torch.from_numpy(img_float.transpose(2, 0, 1)).float()
        mask_tensor = torch.from_numpy(masks_2ch).float()

        return {
            "image": img_tensor,
            "mask": mask_tensor,
            "cdr_ref": torch.tensor(cdr_ref, dtype=torch.float32),
            "diagnosis": torch.tensor(diag_label, dtype=torch.long),
            "image_id": rec["image_id"]
        }


class PAPILADataset(Dataset):
    """External Validation dataset for Optic Disc & Cup from PAPILA.
    
    Rasterizes expert polygon contours into 2-channel binary masks (OD, OC)
    and applies aspect-preserving letterboxing to eliminate aspect-ratio warping.
    """

    def __init__(
        self,
        manifest_path: str = "manifests/optic_disc_cup/papila_external_test.csv",
        image_size: int = 512,
        expert_mode: str = "consensus",  # 'consensus', 'exp1', 'exp2'
        transform: Optional[Callable] = None,
        limit: Optional[int] = None
    ):
        self.image_size = image_size
        self.expert_mode = expert_mode
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

    def _contour_to_mask(self, contour_path_str: str, h: int, w: int) -> np.ndarray:
        if not contour_path_str:
            return np.zeros((h, w), dtype=np.uint8)
        p = resolve_path(contour_path_str)
        mask = np.zeros((h, w), dtype=np.uint8)
        if p.is_file():
            try:
                pts = np.loadtxt(str(p))
                if len(pts) >= 3:
                    pts_int = pts.astype(np.int32)
                    cv2.fillPoly(mask, [pts_int], 1)
            except Exception:
                pass
        return mask

    def __getitem__(self, idx: int):
        rec = self.records[idx]

        # 1. Load RGB Image at original resolution
        img_path = resolve_path(rec["image_path"])
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            raise FileNotFoundError(f"Image could not be opened: {img_path}")
        h, w = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # 2. Rasterize Contours for OD and Cup at native resolution (e.g. 2576x1934)
        od_m1 = self._contour_to_mask(rec.get("od_contour_exp1", ""), h, w)
        od_m2 = self._contour_to_mask(rec.get("od_contour_exp2", ""), h, w)

        cup_m1 = self._contour_to_mask(rec.get("cup_contour_exp1", ""), h, w)
        cup_m2 = self._contour_to_mask(rec.get("cup_contour_exp2", ""), h, w)

        if self.expert_mode == "exp1":
            od_full, cup_full = od_m1, cup_m1
        elif self.expert_mode == "exp2":
            od_full, cup_full = od_m2, cup_m2
        else:  # consensus: union or agreement
            od_full = ((od_m1 + od_m2) >= 1).astype(np.float32)
            cup_full = ((cup_m1 + cup_m2) >= 1).astype(np.float32)

        # 3. Apply aspect-ratio preserving letterbox to image & masks
        img_lb, pad_meta = letterbox_rgb(img_rgb, target_size=self.image_size)
        od_lb = letterbox_mask(od_full, target_size=self.image_size, pad_meta=pad_meta)
        cup_lb = letterbox_mask(cup_full, target_size=self.image_size, pad_meta=pad_meta)

        img_float = img_lb.astype(np.float32) / 255.0
        masks_2ch = np.stack([od_lb.astype(np.float32), cup_lb.astype(np.float32)], axis=0)

        # 4. Optional augmentation
        if self.transform is not None:
            img_float, masks_2ch = self.transform(img_float, masks_2ch)

        # 5. Clinical Diagnosis (0: Normal, 1: Glaucoma, 2: Suspicious)
        diag_raw = rec.get("diagnosis")
        try:
            diag_label = int(diag_raw) if diag_raw != "" and not pd.isna(diag_raw) else -1
        except Exception:
            diag_label = -1

        img_tensor = torch.from_numpy(img_float.transpose(2, 0, 1)).float()
        mask_tensor = torch.from_numpy(masks_2ch).float()

        return {
            "image": img_tensor,
            "mask": mask_tensor,
            "diagnosis": torch.tensor(diag_label, dtype=torch.long),
            "patient_id": rec.get("patient_id", ""),
            "eye_side": rec.get("eye_side", ""),
            "image_id": rec["image_id"]
        }


class REFUGE2Dataset(Dataset):
    """Optic Disc & Cup segmentation dataset for REFUGE2 (MICCAI Challenge).

    Provides 2-channel targets from gold-standard BMP masks with letterboxing:
      Channel 0: Optic Disc (pixels <= 128: both Disc rim and Cup)
      Channel 1: Optic Cup  (pixels == 0: Cup center)
    """

    def __init__(
        self,
        manifest_path: str = "manifests/optic_disc_cup/refuge2_train.csv",
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

    def _load_raw_mask(self, mask_path_str: str) -> np.ndarray:
        """Decodes REFUGE BMP mask into 2-channel native array: [OD, Cup]."""
        p = resolve_path(mask_path_str)
        if not p.is_file():
            raise FileNotFoundError(f"REFUGE mask not found: {mask_path_str}")
        mask_img = Image.open(p)
        arr = np.array(mask_img)
        od_mask = (arr <= 128).astype(np.float32)
        cup_mask = (arr == 0).astype(np.float32)
        return np.stack([od_mask, cup_mask], axis=0)  # (2, H, W)

    def __getitem__(self, idx: int):
        rec = self.records[idx]

        # 1. Load RGB Image at native resolution
        img_path = resolve_path(rec["image_path"])
        if not img_path.is_file():
            raise FileNotFoundError(f"Image not found: {img_path}")
        img_orig = Image.open(img_path).convert("RGB")
        img_np = np.array(img_orig, dtype=np.uint8)

        # 2. Aspect-preserving Letterbox
        img_lb, pad_meta = letterbox_rgb(img_np, target_size=self.image_size)
        img_float = img_lb.astype(np.float32) / 255.0

        # 3. Load & Letterbox 2-channel OD & Cup mask
        masks_raw = self._load_raw_mask(rec.get("mask_path", ""))
        masks_2ch = letterbox_mask(masks_raw, target_size=self.image_size, pad_meta=pad_meta)

        # 4. Optional joint augmentation
        if self.transform is not None:
            img_float, masks_2ch = self.transform(img_float, masks_2ch)

        # 5. Reference CDR
        cdr_raw = rec.get("cdr_ref")
        cdr_ref = float(cdr_raw) if cdr_raw != "" and not pd.isna(cdr_raw) else -1.0

        # 6. Clinical Diagnosis (-1: Unknown, 0: Non-Glaucoma, 1: Glaucoma)
        diag_raw = rec.get("diagnosis")
        try:
            diag_label = int(diag_raw) if diag_raw != "" and not pd.isna(diag_raw) else -1
        except Exception:
            diag_label = -1

        # 7. Fovea Location (X, Y)
        fx = float(rec.get("fovea_x", 0.0)) if not pd.isna(rec.get("fovea_x")) else 0.0
        fy = float(rec.get("fovea_y", 0.0)) if not pd.isna(rec.get("fovea_y")) else 0.0

        img_tensor = torch.from_numpy(img_float.transpose(2, 0, 1)).float()
        mask_tensor = torch.from_numpy(masks_2ch).float()

        return {
            "image": img_tensor,
            "mask": mask_tensor,
            "cdr_ref": torch.tensor(cdr_ref, dtype=torch.float32),
            "diagnosis": torch.tensor(diag_label, dtype=torch.long),
            "fovea": torch.tensor([fx, fy], dtype=torch.float32),
            "image_id": rec["image_id"]
        }


class MixedOpticDiscDataset(Dataset):
    """Concatenated multi-domain dataset for robust optic disc & cup training.
    
    Allows combining REFUGE2 (Canon/Zeiss) and Drishti-GS1 (diverse optics).
    """

    def __init__(self, datasets: List[Dataset]):
        self.datasets = datasets
        self.concat_dataset = ConcatDataset(datasets)

    def __len__(self):
        return len(self.concat_dataset)

    def __getitem__(self, idx: int):
        return self.concat_dataset[idx]
