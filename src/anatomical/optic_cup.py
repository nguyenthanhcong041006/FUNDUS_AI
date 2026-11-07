"""Optic Cup (OC) segmentation and Cup-to-Disc Ratio (CDR) calculation with letterboxing."""

from typing import Dict, Any, Tuple, Optional
from pathlib import Path
import cv2
import numpy as np

from src.preprocessing.normalization import letterbox_rgb, unpad_mask


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
            oc_mask: Binary uint8 mask (0 or 255) for Optic Cup at native resolution.
            metrics: CDR ratios (VCDR, HCDR, area ratio) and cup measurements.
        """
        h, w = od_mask.shape[:2]
        oc_mask = np.zeros((h, w), dtype=np.uint8)

        disc_is_detected = od_metrics.get("detected", False) if od_metrics else False
        if not disc_is_detected and od_metrics and od_metrics.get("area_px2", 0) > 50:
            disc_is_detected = True

        if self.model is not None:
            import torch
            img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            img_lb, pad_meta = letterbox_rgb(img_rgb, target_size=512)
            inp = img_lb.astype(np.float32) / 255.0
            inp_t = torch.from_numpy(inp.transpose(2, 0, 1)).unsqueeze(0).float()
            with torch.no_grad():
                logits = self.model(inp_t)
                prob = torch.sigmoid(logits)[0, 1].cpu().numpy()  # Channel 1: Optic Cup
            pred_512 = (prob > 0.5).astype(np.uint8) * 255
            oc_mask = unpad_mask(pred_512, pad_meta)
            # Anatomically constrain cup within disc
            if np.sum(od_mask > 0) > 0:
                oc_mask = cv2.bitwise_and(oc_mask, od_mask)
        else:
            # Baseline morphological cup estimation inside disc
            if np.sum(od_mask > 0) > 0:
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
                oc_mask = cv2.erode(od_mask, kernel, iterations=2)

        contours, _ = cv2.findContours(oc_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cup_detected = False
        cup_bh, cup_bw, cup_area = 0.0, 0.0, 0.0

        if contours:
            c = max(contours, key=cv2.contourArea)
            c_area = float(cv2.contourArea(c))
            if c_area > 10:
                bx, by, bw, bh = cv2.boundingRect(c)
                cup_bw = float(bw)
                cup_bh = float(bh)
                cup_area = c_area
                cup_detected = True

        # Calculate ratios with fail-safes
        if disc_is_detected and od_metrics:
            disc_vert = float(od_metrics.get("vertical_diameter_px", 0.0))
            disc_horiz = float(od_metrics.get("horizontal_diameter_px", 0.0))
            disc_area = float(od_metrics.get("area_px2", 0.0))

            if disc_vert > 0 and disc_horiz > 0 and disc_area > 0:
                vcdr = min(round(cup_bh / disc_vert, 3), 1.0) if cup_detected else 0.0
                hcdr = min(round(cup_bw / disc_horiz, 3), 1.0) if cup_detected else 0.0
                area_ratio = min(round(cup_area / disc_area, 3), 1.0) if cup_detected else 0.0
            else:
                vcdr = None
                hcdr = None
                area_ratio = None
        else:
            vcdr = None
            hcdr = None
            area_ratio = None

        metrics: Dict[str, Any] = {
            "detected": cup_detected,
            "area_px2": cup_area,
            "horizontal_diameter_px": cup_bw,
            "vertical_diameter_px": cup_bh,
            "vertical_cdr": vcdr,
            "horizontal_cdr": hcdr,
            "cup_disc_area_ratio": area_ratio
        }

        return oc_mask, metrics
