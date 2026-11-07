"""Blood vessel segmentation and vessel tree extraction."""

from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import cv2
import numpy as np


class VesselSegmenter:
    """Segments retinal blood vessels using deep learning or adaptive morphological filters."""

    def __init__(self, model_path: Optional[str] = "checkpoints/vessels/best_vessel_model.pth", threshold: float = 0.5):
        self.model_path = model_path
        self.threshold = threshold
        self.model = None

        if self.model_path and Path(self.model_path).exists():
            try:
                import torch
                from src.models.unet import UNet
                net = UNet(in_channels=3, num_classes=1)
                state = torch.load(self.model_path, map_location="cpu")
                net.load_state_dict(state)
                net.eval()
                self.model = net
                print(f"     [U-Net Model Active] Loaded vessel checkpoint: {self.model_path}")
            except Exception as e:
                print(f"     [U-Net Load Warning] Falling back to morphology: {e}")
                self.model = None

    def segment(
        self,
        image_bgr: np.ndarray,
        fov_mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Segments blood vessels and calculates basic vessel metrics.

        Args:
            image_bgr: Preprocessed fundus image.
            fov_mask: Retinal field of view mask.

        Returns:
            vessel_mask: Binary uint8 mask (0 or 255) of retinal vessels.
            metrics: Vessel density, pixel count.
        """
        h, w = image_bgr.shape[:2]

        if self.model is not None:
            # Deep Learning U-Net Inference with letterbox
            import torch
            from src.preprocessing.normalization import letterbox_rgb, unpad_mask
            rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            img_lb, pad_meta = letterbox_rgb(rgb, target_size=512)
            tensor = torch.from_numpy(img_lb.astype(np.float32).transpose(2, 0, 1) / 255.0).unsqueeze(0).float()

            with torch.no_grad():
                logits = self.model(tensor)
                probs = torch.sigmoid(logits).squeeze().cpu().numpy()

            vessel_512 = (probs > self.threshold).astype(np.uint8) * 255
            vessel_mask = unpad_mask(vessel_512, pad_meta)
        else:
            # High-performance classical vessel enhancement fallback:
            green = image_bgr[:, :, 1]
            clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
            enhanced = clahe.apply(green)
            inv = 255 - enhanced
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            tophat = cv2.morphologyEx(inv, cv2.MORPH_TOPHAT, kernel)
            thresh = cv2.adaptiveThreshold(
                tophat, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 15, -2
            )
            clean_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, clean_kernel)
            vessel_mask = cleaned

        # Mask out anything outside the FOV
        if fov_mask is not None:
            vessel_mask = cv2.bitwise_and(vessel_mask, fov_mask)

        vessel_pixels = int(np.sum(vessel_mask > 0))
        valid_fov_pixels = int(np.sum(fov_mask > 0)) if fov_mask is not None else (h * w)

        vessel_density = float((vessel_pixels / max(valid_fov_pixels, 1)) * 100.0)

        metrics = {
            "vessel_pixel_count": vessel_pixels,
            "vessel_density_percent": round(vessel_density, 2),
            "valid_fov_pixels": valid_fov_pixels
        }

        return vessel_mask, metrics

