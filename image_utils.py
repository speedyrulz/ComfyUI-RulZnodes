"""Shared PIL -> float conversion for the RulZ image loaders."""
import numpy as np
from PIL import Image

# Greyscale modes Pillow uses for 16-bit PNG / TIFF files ("I" in older Pillow).
_HIGH_BIT_GREY_MODES = {"I", "I;16", "I;16B", "I;16L", "I;16N"}


def to_float_rgb(img: Image.Image) -> np.ndarray:
    """PIL image -> float32 [H, W, 3] array in 0..1.

    convert("RGB") clips 16-bit values at 255, so a 16-bit PNG would come out
    almost solid white. Those modes are scaled down from their full range instead.
    """
    if img.mode in _HIGH_BIT_GREY_MODES:
        grey = np.clip(np.asarray(img).astype(np.float32) / 65535.0, 0.0, 1.0)
        return np.repeat(grey[..., None], 3, axis=-1)
    return np.asarray(img.convert("RGB")).astype(np.float32) / 255.0
