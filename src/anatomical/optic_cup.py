"""Optic Cup (OC) segmentation and Cup-to-Disc Ratio (CDR) calculation."""

from typing import Dict, Any, Tuple, Optional
from pathlib import Path
import cv2
import numpy as np


class OpticCupSegmenter:
    """Segments Optic Cup within the Optic Disc region."""

    def __init__(self, model_path: Optional[str] = "checkpoints/optic_disc_cup/best_od_oc_model.pth"):
        self.model_path = model_path
        self.model = None
        if model_path:
            self._load_model(model_path)

    def _load_model(self, path: str):
        p = Path(path)
        if not p.is_file():
            cand = Path(__file__).resolve().parent.parent.parent / p
            if cand.is_file():
                p = cand
        if p.is_file():
            try:
                import torch
                from src.models.unet import UNet
                net = UNet(in_channels=3, num_classes=2)
                state = torch.load(str(p), map_location="cpu")
                net.load_state_dict(state)
                net.eval()
                self.model = net
            except Exception:
                self.model = None

    def segment(
        self,
        image_bgr: np.ndarray,
        od_mask: np.ndarray,
        od_metrics: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Segments the optic cup inside the optic disc mask.

        Args:
            image_bgr: Preprocessed fundus image.
            od_mask: Segmented binary mask of the optic disc.
            od_metrics: Geometric metrics of the optic disc.

        Returns:
            oc_mask: Binary uint8 mask (0 or 255) for Optic Cup.
            metrics: CDR ratios (VCDR, HCDR, area ratio) and cup measurements.
        """
        h, w = od_mask.shape[:2]
        oc_mask = np.zeros((h, w), dtype=np.uint8)

        if self.model is not None:
            import torch
            img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            inp = cv2.resize(img_rgb, (512, 512)).astype(np.float32) / 255.0
            inp_t = torch.from_numpy(inp.transpose(2, 0, 1)).unsqueeze(0).float()
            with torch.no_grad():
                logits = self.model(inp_t)
                prob = torch.sigmoid(logits)[0, 1].cpu().numpy()  # Channel 1: Optic Cup
            pred_512 = (prob > 0.5).astype(np.uint8) * 255
            oc_mask = cv2.resize(pred_512, (w, h), interpolation=cv2.INTER_NEAREST)
            # Anatomically constrain cup within disc
            if np.sum(od_mask > 0) > 0:
                oc_mask = cv2.bitwise_and(oc_mask, od_mask)
        else:
            # Baseline morphological cup estimation inside disc
            if np.sum(od_mask > 0) > 0:
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
                oc_mask = cv2.erode(od_mask, kernel, iterations=2)

        contours, _ = cv2.findContours(oc_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        metrics: Dict[str, Any] = {
            "area_px2": 0.0,
            "horizontal_diameter_px": 0.0,
            "vertical_diameter_px": 0.0,
            "vertical_cdr": 0.0,
            "horizontal_cdr": 0.0,
            "cup_disc_area_ratio": 0.0
        }

        if contours:
            c = max(contours, key=cv2.contourArea)
            bx, by, bw, bh = cv2.boundingRect(c)
            metrics["area_px2"] = float(cv2.contourArea(c))
            metrics["horizontal_diameter_px"] = float(bw)
            metrics["vertical_diameter_px"] = float(bh)

            if od_metrics:
                disc_vert = max(float(od_metrics.get("vertical_diameter_px", 1.0)), 1.0)
                disc_horiz = max(float(od_metrics.get("horizontal_diameter_px", 1.0)), 1.0)
                disc_area = max(float(od_metrics.get("area_px2", 1.0)), 1.0)

                metrics["vertical_cdr"] = min(round(float(bh) / disc_vert, 3), 1.0)
                metrics["horizontal_cdr"] = min(round(float(bw) / disc_horiz, 3), 1.0)
                metrics["cup_disc_area_ratio"] = min(round(metrics["area_px2"] / disc_area, 3), 1.0)

        return oc_mask, metrics
