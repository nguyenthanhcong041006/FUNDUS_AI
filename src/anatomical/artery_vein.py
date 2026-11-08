"""Artery and Vein (A/V) analysis and ratio estimation."""

from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import cv2
import numpy as np


def remove_small_objects(mask: np.ndarray, min_size: int = 30) -> np.ndarray:
    """Eliminates isolated pixel clusters smaller than min_size."""
    if np.sum(mask > 0) == 0:
        return mask
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    out = np.zeros_like(mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            out[labels == i] = 255
    return out


class ArteryVeinAnalyzer:
    """Classifies segmented vessels into arterioles and venules."""

    def __init__(self, model_path: Optional[str] = "checkpoints/artery_vein/best_av_model.pth"):
        self.model_path = model_path
        self.model = None

        if self.model_path and Path(self.model_path).exists():
            try:
                from src.models.unet import load_checkpoint_unet
                self.model = load_checkpoint_unet(self.model_path, in_channels=3, num_classes=3, map_location="cpu")
                print(f"     [U-Net Model Active] Loaded Artery/Vein checkpoint: {self.model_path}")
            except Exception:
                self.model = None

    def analyze(
        self,
        image_bgr: np.ndarray,
        vessel_mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Separates vessel pixels into artery and vein masks.

        Args:
            image_bgr: Input fundus image.
            vessel_mask: Binary blood vessel mask.

        Returns:
            artery_mask: Binary mask of arteries.
            vein_mask: Binary mask of veins.
            metrics: Arteriolar density, venular density, A/V ratio.
        """
        h, w = vessel_mask.shape[:2]

        if self.model is not None:
            import torch
            from src.preprocessing.normalization import letterbox_rgb, unpad_mask
            rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            img_lb, pad_meta = letterbox_rgb(rgb, target_size=512)
            tensor = torch.from_numpy(img_lb.astype(np.float32).transpose(2, 0, 1) / 255.0).unsqueeze(0).float()

            with torch.no_grad():
                logits = self.model(tensor)
                preds = torch.argmax(logits, dim=1).squeeze().cpu().numpy()

            art_512 = (preds == 1).astype(np.uint8) * 255
            vein_512 = (preds == 2).astype(np.uint8) * 255

            artery_mask = unpad_mask(art_512, pad_meta)
            vein_mask = unpad_mask(vein_512, pad_meta)

            # Restrict strictly to vessel tree
            if np.sum(vessel_mask > 0) > 0:
                artery_mask = cv2.bitwise_and(artery_mask, vessel_mask)
                vein_mask = cv2.bitwise_and(vein_mask, vessel_mask)
        else:
            # Color-space luminance heuristic fallback
            hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
            val = hsv[:, :, 2]
            vessel_bool = vessel_mask > 0
            if np.sum(vessel_bool) > 0:
                median_val = np.median(val[vessel_bool])
                artery_mask = np.where((val > median_val) & vessel_bool, 255, 0).astype(np.uint8)
                vein_mask = np.where((val <= median_val) & vessel_bool, 255, 0).astype(np.uint8)
            else:
                artery_mask = np.zeros((h, w), dtype=np.uint8)
                vein_mask = np.zeros((h, w), dtype=np.uint8)

        # Eliminate isolated speckles
        artery_mask = remove_small_objects(artery_mask, min_size=30)
        vein_mask = remove_small_objects(vein_mask, min_size=30)

        artery_pixels = int(np.sum(artery_mask > 0))
        vein_pixels = int(np.sum(vein_mask > 0))

        av_ratio = float(artery_pixels / max(vein_pixels, 1))

        metrics = {
            "artery_pixels": artery_pixels,
            "vein_pixels": vein_pixels,
            "av_pixel_ratio": round(av_ratio, 3),
            "status": "active" if self.model is not None else "screening_estimate"
        }

        return artery_mask, vein_mask, metrics
