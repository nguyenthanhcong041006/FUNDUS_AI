"""Optic Disc (OD) segmentation and localization."""

from typing import Dict, Any, Tuple, Optional
from pathlib import Path
import cv2
import numpy as np


class OpticDiscSegmenter:
    """Segments the Optic Disc from fundus images."""

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
                print(f"     [U-Net Model Active] Loaded OD/OC checkpoint: {p}")
            except Exception as e:
                print(f"     [U-Net Load Warning] Falling back to morphology: {e}")
                self.model = None

    def segment(
        self,
        image_bgr: np.ndarray,
        fov_mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Segments the optic disc, returning a binary mask and geometry metrics.

        Args:
            image_bgr: Preprocessed fundus image.
            fov_mask: Retinal FOV binary mask.

        Returns:
            od_mask: Binary uint8 mask (0 or 255) for Optic Disc.
            metrics: Centroid, bounding box, area, vertical/horizontal diameter.
        """
        h, w = image_bgr.shape[:2]
        od_mask = np.zeros((h, w), dtype=np.uint8)

        if self.model is not None:
            import torch
            img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            inp = cv2.resize(img_rgb, (512, 512)).astype(np.float32) / 255.0
            inp_t = torch.from_numpy(inp.transpose(2, 0, 1)).unsqueeze(0).float()
            with torch.no_grad():
                logits = self.model(inp_t)
                prob = torch.sigmoid(logits)[0, 0].cpu().numpy()  # Channel 0: Optic Disc
            pred_512 = (prob > 0.5).astype(np.uint8) * 255
            od_mask = cv2.resize(pred_512, (w, h), interpolation=cv2.INTER_NEAREST)
            if fov_mask is not None:
                od_mask = cv2.bitwise_and(od_mask, fov_mask)
        else:
            # Robust candidate heuristic fallback for unweighted pipeline testing:
            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
            red = image_bgr[:, :, 2]
            bright = cv2.addWeighted(red, 0.7, gray, 0.3, 0)

            if fov_mask is not None:
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
                eroded_fov = cv2.erode(fov_mask, kernel, iterations=2)
                bright = cv2.bitwise_and(bright, eroded_fov)

            bright_blur = cv2.GaussianBlur(bright, (17, 17), 0)
            _, thresh = cv2.threshold(bright_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                c = max(contours, key=cv2.contourArea)
                (cx, cy), radius = cv2.minEnclosingCircle(c)
                cv2.circle(od_mask, (int(cx), int(cy)), int(radius), 255, -1)

        # Extract geometric landmarks
        contours, _ = cv2.findContours(od_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        metrics: Dict[str, Any] = {
            "center": (w // 2, h // 2),
            "radius": 0.0,
            "bbox": (0, 0, 0, 0),
            "area_px2": 0.0,
            "horizontal_diameter_px": 0.0,
            "vertical_diameter_px": 0.0
        }

        if contours:
            c = max(contours, key=cv2.contourArea)
            m = cv2.moments(c)
            if m["m00"] > 0:
                cx = int(m["m10"] / m["m00"])
                cy = int(m["m01"] / m["m00"])
                metrics["center"] = (cx, cy)

            (cx, cy), radius = cv2.minEnclosingCircle(c)
            metrics["radius"] = float(radius)

            bx, by, bw, bh = cv2.boundingRect(c)
            metrics["bbox"] = (int(bx), int(by), int(bw), int(bh))
            metrics["horizontal_diameter_px"] = float(bw)
            metrics["vertical_diameter_px"] = float(bh)
            metrics["area_px2"] = float(cv2.contourArea(c))

        return od_mask, metrics
