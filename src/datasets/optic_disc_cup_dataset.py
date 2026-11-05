"""Optic Disc & Cup Dataset Loaders for Drishti-GS1 and PAPILA.

Supports multi-class segmentation (Channel 0: Optic Disc, Channel 1: Optic Cup)
and CDR/Glaucoma screening evaluation per Section 2 of FUNDUS_AI_Dataset_Training_Plan_2.md.
"""

from pathlib import Path
from typing import Optional, Callable
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

from src.datasets.utils import resolve_path


class DrishtiDataset(Dataset):
    """Optic Disc and Cup segmentation dataset for Drishti-GS1.
    
    Provides 2-channel targets:
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

    def _load_mask(self, mask_path_str: str) -> np.ndarray:
        p = resolve_path(mask_path_str)
        if p.is_file():
            mask_img = Image.open(p).convert("L")
            mask_img = mask_img.resize((self.image_size, self.image_size), Image.NEAREST)
            arr = np.array(mask_img, dtype=np.float32)
            if self.binary_threshold is not None:
                return (arr >= self.binary_threshold).astype(np.float32)
            else:
                return arr / 255.0
        return np.zeros((self.image_size, self.image_size), dtype=np.float32)

    def __getitem__(self, idx: int):
        rec = self.records[idx]

        # 1. Load RGB Image
        img_path = resolve_path(rec["image_path"])
        img = Image.open(img_path).convert("RGB")
        img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
        img_np = np.array(img, dtype=np.float32) / 255.0

        # 2. Load OD and Cup Masks
        od_mask = self._load_mask(rec.get("od_mask_path", ""))
        cup_mask = self._load_mask(rec.get("cup_mask_path", ""))

        # Multi-channel target: 2 channels (Disc, Cup)
        masks_2ch = np.stack([od_mask, cup_mask], axis=0)  # (2, H, W)

        # 3. Reference CDR
        cdr_ref_raw = rec.get("cdr_ref")
        cdr_ref = float(cdr_ref_raw) if cdr_ref_raw != "" and not pd.isna(cdr_ref_raw) else -1.0

        # 4. Clinical Diagnosis
        diag_str = str(rec.get("diagnosis", "")).lower()
        diag_label = 1 if "glaucomatous" in diag_str else (0 if "normal" in diag_str else -1)

        img_tensor = torch.from_numpy(img_np.transpose(2, 0, 1)).float()
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
    
    Rasterizes expert polygon contours into 2-channel binary masks (OD, OC).
    """

    def __init__(
        self,
        manifest_path: str = "manifests/optic_disc_cup/papila_external_test.csv",
        image_size: int = 512,
        expert_mode: str = "consensus",  # 'consensus' (agreement/union), 'exp1', 'exp2'
        limit: Optional[int] = None
    ):
        self.image_size = image_size
        self.expert_mode = expert_mode

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

        # 1. Load RGB Image
        img_path = resolve_path(rec["image_path"])
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            raise FileNotFoundError(f"Image could not be opened: {img_path}")
        h, w = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # 2. Rasterize Contours for OD and Cup
        od_m1 = self._contour_to_mask(rec.get("od_contour_exp1", ""), h, w)
        od_m2 = self._contour_to_mask(rec.get("od_contour_exp2", ""), h, w)

        cup_m1 = self._contour_to_mask(rec.get("cup_contour_exp1", ""), h, w)
        cup_m2 = self._contour_to_mask(rec.get("cup_contour_exp2", ""), h, w)

        if self.expert_mode == "exp1":
            od_full, cup_full = od_m1, cup_m1
        elif self.expert_mode == "exp2":
            od_full, cup_full = od_m2, cup_m2
        else:  # consensus: union or agreement
            od_full = ((od_m1 + od_m2) >= 1).astype(np.uint8)
            cup_full = ((cup_m1 + cup_m2) >= 1).astype(np.uint8)

        # 3. Resize to target size
        img_resized = cv2.resize(img_rgb, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        od_resized = cv2.resize(od_full, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        cup_resized = cv2.resize(cup_full, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)

        img_np = img_resized.astype(np.float32) / 255.0
        masks_2ch = np.stack([od_resized.astype(np.float32), cup_resized.astype(np.float32)], axis=0)

        # 4. Clinical Diagnosis (0: Normal, 1: Glaucoma, 2: Suspicious)
        diag_raw = rec.get("diagnosis")
        try:
            diag_label = int(diag_raw) if diag_raw != "" and not pd.isna(diag_raw) else -1
        except Exception:
            diag_label = -1

        img_tensor = torch.from_numpy(img_np.transpose(2, 0, 1)).float()
        mask_tensor = torch.from_numpy(masks_2ch).float()

        return {
            "image": img_tensor,
            "mask": mask_tensor,
            "diagnosis": torch.tensor(diag_label, dtype=torch.long),
            "patient_id": rec.get("patient_id", ""),
            "eye_side": rec.get("eye_side", ""),
            "image_id": rec["image_id"]
        }

