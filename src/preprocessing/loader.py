"""Image loading and format validation module."""

from pathlib import Path
from typing import Dict, Tuple, Any, Optional
import cv2
import numpy as np
from PIL import Image


SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def load_and_validate_image(
    image_path: str | Path,
    target_color: str = "BGR"
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Loads and validates a fundus image file.

    Args:
        image_path: Path to the image file.
        target_color: Desired color format ('BGR' or 'RGB').

    Returns:
        image: Numpy array of shape (H, W, 3) in requested color space.
        metadata: Dictionary with file size, original shape, channels, etc.

    Raises:
        FileNotFoundError: If the image path does not exist.
        ValueError: If file format is unsupported or image cannot be decoded.
    """
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Image file not found: {path.resolve()}")

    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported image extension '{path.suffix}'. Supported: {SUPPORTED_EXTENSIONS}"
        )

    # Attempt cv2 read first
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)

    # Fallback to PIL if cv2 returns None (e.g., specific multi-page or uncompressed TIFF)
    if image_bgr is None:
        try:
            with Image.open(path) as pil_img:
                pil_rgb = pil_img.convert("RGB")
                image_rgb = np.array(pil_rgb)
                image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        except Exception as e:
            raise ValueError(f"Failed to decode image {path.name}: {e}") from e

    if image_bgr is None or image_bgr.size == 0:
        raise ValueError(f"Empty or corrupted image file: {path.name}")

    h, w = image_bgr.shape[:2]
    channels = 1 if len(image_bgr.shape) == 2 else image_bgr.shape[2]

    if channels != 3:
        if channels == 1:
            image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
        else:
            image_bgr = image_bgr[:, :, :3]

    metadata = {
        "file_name": path.name,
        "file_path": str(path.resolve()),
        "original_height": h,
        "original_width": w,
        "channels": 3,
        "file_size_bytes": path.stat().st_size,
    }

    if target_color.upper() == "RGB":
        output_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    else:
        output_image = image_bgr

    return output_image, metadata

