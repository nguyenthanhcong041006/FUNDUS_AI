"""Dataset path resolution utilities for cross-platform and directory-independent loading."""

from pathlib import Path
from typing import Union, Optional
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def resolve_path(p: Union[str, Path, None], base_dir: Optional[Path] = None) -> Path:
    """Resolves a file path relative to current dir, project root, or optional base_dir."""
    if p is None or (isinstance(p, float) and np.isnan(p)) or not str(p).strip():
        return Path("")
    path = Path(p)
    if path.is_file() or path.is_dir():
        return path
    if base_dir and (base_dir / path).exists():
        return base_dir / path
    cand = PROJECT_ROOT / path
    if cand.exists():
        return cand
    return path

