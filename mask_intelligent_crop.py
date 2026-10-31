"""
Mask Intelligent Crop

Crops an image (and/or its mask) to the bounding box of the mask, then extends
every side of that box by a percentage (default 10%). The padded box is
clamped to the image bounds.

  * image and mask inputs are both optional:
      - image + mask : crop both, mask is resampled to the image size first
      - mask only    : crop the mask, image output is None
      - image only   : nothing to crop by, image is passed through unchanged
                       and a blank (all zero) mask of the image size is output
  * padding_mode      - uniform: padding_percent on every side
                        per_edge: padding_top / _bottom / _left / _right each
                        set one side
  * padding_percent   - how far each side extends beyond the masked area
  * padding_reference - what the percentage is measured against:
                          crop  - the final cropped image (default), so with 10%
                                  the subject sits inside a 10% border
                          mask  - the mask bounding box
                          image - the full input image
  * aspect_limit      - never let the crop get more elongated than this
                        (default 2:3 / 3:2). A too-tall crop is widened and a
                        too-wide crop is heightened, centred on the subject and
                        clamped to the image. The crop is never shrunk for this.
  * divisible_by      - optionally grow the crop so width and height are
                        multiples of this value (the box is only shrunk if the
                        image is too small to grow, and never into the mask)
  * mask_threshold    - mask pixels above this value count as "masked"
  * batches           - the crop box is the union over the whole mask batch so
                        every frame in the batch is cropped to the same size
"""

import math

import torch
import torch.nn.functional as F


def _normalize_mask(mask: torch.Tensor) -> torch.Tensor:
    """Return the mask as a float tensor shaped [B, H, W]."""
    if mask.dim() == 2:
        mask = mask.unsqueeze(0)
    elif mask.dim() == 4:
        # [B, 1, H, W] or [B, H, W, 1]
        if mask.shape[1] == 1:
            mask = mask[:, 0, :, :]
        elif mask.shape[-1] == 1:
            mask = mask[..., 0]
        else:
            raise ValueError(f"Unsupported mask shape {tuple(mask.shape)}")
    elif mask.dim() != 3:
        raise ValueError(f"Unsupported mask shape {tuple(mask.shape)}")
    return mask.float()


def _resize_mask(mask: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Resample a [B, H, W] mask to [B, height, width]."""
    if mask.shape[1] == height and mask.shape[2] == width:
        return mask
    resized = F.interpolate(
        mask.unsqueeze(1), size=(height, width), mode="bilinear", align_corners=False
    )
    return resized[:, 0, :, :]


def _place_span(lo: int, hi: int, limit: int, new_size: int, must_lo: int, must_hi: int):
    """
    Re-centre the half-open span [lo, hi) at a new length `new_size`, keeping it
    inside [0, limit) and making sure it still contains [must_lo, must_hi).
    """
    new_size = max(must_hi - must_lo, min(new_size, limit))
    center = (lo + hi) / 2.0
    new_lo = int(round(center - new_size / 2.0))
    new_lo = max(0, min(new_lo, limit - new_size))
    new_hi = new_lo + new_size

    # Guarantee the masked region is still inside the span (only reachable when
    # the span had to shrink to satisfy divisible_by).
    if new_lo > must_lo:
        new_lo = must_lo
        new_hi = new_lo + new_size
    if new_hi < must_hi:
        new_hi = must_hi
        new_lo = new_hi - new_size

    new_lo = max(0, new_lo)
    new_hi = min(limit, new_hi)
    return new_lo, new_hi


def _fit_span(lo: int, hi: int, limit: int, multiple: int, must_lo: int, must_hi: int):
    """
    Adjust the half-open span [lo, hi) so that its length is a multiple of
    `multiple`, stays inside [0, limit) and still fully contains [must_lo, must_hi).

    The span is grown symmetrically when possible. If growing would exceed the
    frame, the largest fitting multiple is used as long as it still covers the
    mask; otherwise the whole frame dimension is used.
    """
    size = hi - lo
    must_size = must_hi - must_lo

    if multiple > 1:
        grown = math.ceil(size / multiple) * multiple
        if grown <= limit:
            new_size = grown
        else:
            shrunk = (limit // multiple) * multiple
            new_size = shrunk if shrunk >= must_size else limit
    else:
        new_size = size

    return _place_span(lo, hi, limit, new_size, must_lo, must_hi)


ASPECT_LIMITS = {
    "off": None,
    "1:1": 1.0,
    "4:5 / 5:4": 5.0 / 4.0,
    "3:4 / 4:3": 4.0 / 3.0,
    "2:3 / 3:2": 3.0 / 2.0,
    "9:16 / 16:9": 16.0 / 9.0,
}


def _limit_aspect(x0, x1, y0, y1, width, height, max_ratio, mx0, mx1, my0, my1):
    """
    Grow the box so that long side / short side <= max_ratio. A box that is
    too tall is widened, a box that is too wide is heightened. Growth is
    centred on the existing box and clamped to the frame, so if the frame
    itself is too small the limit is met as closely as possible.
    """
    if max_ratio is None:
        return x0, x1, y0, y1
    w, h = x1 - x0, y1 - y0
    if w <= 0 or h <= 0:
        return x0, x1, y0, y1
    if h / w > max_ratio:
        target_w = int(math.ceil(h / max_ratio))
        x0, x1 = _place_span(x0, x1, width, target_w, mx0, mx1)
    elif w / h > max_ratio:
        target_h = int(math.ceil(w / max_ratio))
        y0, y1 = _place_span(y0, y1, height, target_h, my0, my1)
    return x0, x1, y0, y1


PADDING_REFERENCES = ["crop", "mask", "image"]
PADDING_MODES = ["uniform", "per_edge"]

# Largest combined share of an axis the two "crop"-referenced borders may take
# (the formula diverges at 100%).
_MAX_CROP_SHARE = 0.9


def _padding(percent_lo: float, percent_hi: float, reference: str, frame_size: int, mask_size: int):
    """
    Pixels to add before (left/top) and after (right/bottom) the mask on one
    axis, each rounded to the nearest pixel.
    """
    p_lo, p_hi = percent_lo / 100.0, percent_hi / 100.0
    if reference == "image":
        pad_lo, pad_hi = frame_size * p_lo, frame_size * p_hi
    elif reference == "mask":
        pad_lo, pad_hi = mask_size * p_lo, mask_size * p_hi
    else:
        # "crop": each side is its percentage of the final crop size.
        # crop = mask + pad_lo + pad_hi with pad = p * crop
        #   =>  crop = mask / (1 - p_lo - p_hi). The combined share is capped
        # below 100%, scaling both sides down together, where it diverges.
        total = p_lo + p_hi
        if total > _MAX_CROP_SHARE:
            p_lo, p_hi = _MAX_CROP_SHARE * (p_lo / total), _MAX_CROP_SHARE * (p_hi / total)
        denom = 1.0 - (p_lo + p_hi)
        pad_lo, pad_hi = mask_size * p_lo / denom, mask_size * p_hi / denom
    return int(math.floor(pad_lo + 0.5)), int(math.floor(pad_hi + 0.5))


def compute_crop_box(mask: torch.Tensor, padding_percent, divisible_by: int, threshold: float,
                     frame_height: int = None, frame_width: int = None,
                     padding_reference: str = "crop", max_aspect_ratio: float = None):
    """
    mask: [B, H, W] float tensor at its native resolution.
    padding_percent: one percentage for every side, or a (top, bottom, left,
    right) tuple of percentages.
    frame_height / frame_width: size of the frame being cropped (the image).
    Defaults to the mask size. The mask bounding box is scaled into frame
    coordinates before padding is applied, so a low-resolution mask never
    grows a soft halo from resampling.

    Returns (x0, y0, x1, y1) as a half-open box in frame pixel coordinates, or
    None if no pixel exceeds the threshold.
    """
    _, mask_height, mask_width = mask.shape
    height = frame_height if frame_height is not None else mask_height
    width = frame_width if frame_width is not None else mask_width

    active = mask > threshold
    rows = torch.any(active, dim=(0, 2))
    cols = torch.any(active, dim=(0, 1))
    if not bool(rows.any()):
        return None

    row_idx = torch.nonzero(rows, as_tuple=False).flatten()
    col_idx = torch.nonzero(cols, as_tuple=False).flatten()
    my0, my1 = int(row_idx.min()), int(row_idx.max()) + 1
    mx0, mx1 = int(col_idx.min()), int(col_idx.max()) + 1

    # Scale the mask bounding box into frame coordinates (outward rounding).
    if mask_height != height or mask_width != width:
        sy = height / mask_height
        sx = width / mask_width
        my0, my1 = int(math.floor(my0 * sy)), min(height, int(math.ceil(my1 * sy)))
        mx0, mx1 = int(math.floor(mx0 * sx)), min(width, int(math.ceil(mx1 * sx)))

    if isinstance(padding_percent, (int, float)):
        top = bottom = left = right = padding_percent
    else:
        top, bottom, left, right = padding_percent
    pad_left, pad_right = _padding(left, right, padding_reference, width, mx1 - mx0)
    pad_top, pad_bottom = _padding(top, bottom, padding_reference, height, my1 - my0)

    x0 = max(0, mx0 - pad_left)
    x1 = min(width, mx1 + pad_right)
    y0 = max(0, my0 - pad_top)
    y1 = min(height, my1 + pad_bottom)

    x0, x1, y0, y1 = _limit_aspect(x0, x1, y0, y1, width, height, max_aspect_ratio, mx0, mx1, my0, my1)

    x0, x1 = _fit_span(x0, x1, width, divisible_by, mx0, mx1)
    y0, y1 = _fit_span(y0, y1, height, divisible_by, my0, my1)
    return x0, y0, x1, y1


class MaskIntelligentCrop:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "padding_mode": (PADDING_MODES, {
                    "default": "uniform",
                    "tooltip": "uniform: padding_percent is used on every side. per_edge: padding_top, padding_bottom, padding_left and padding_right set each side separately.",
                }),
                "padding_percent": ("FLOAT", {
                    "default": 10.0, "min": 0.0, "max": 100.0, "step": 0.5,
                    "tooltip": "uniform mode: how far each side extends beyond the masked area, as a percentage of the padding_reference size (width for left/right, height for top/bottom).",
                }),
                **{
                    f"padding_{edge}": ("FLOAT", {
                        "default": 10.0, "min": 0.0, "max": 100.0, "step": 0.5,
                        "tooltip": f"per_edge mode: how far the {edge} side extends beyond the masked area, as a percentage of the padding_reference {axis}.",
                    })
                    for edge, axis in (("top", "height"), ("bottom", "height"), ("left", "width"), ("right", "width"))
                },
                "padding_reference": (PADDING_REFERENCES, {
                    "default": "crop",
                    "tooltip": "crop: percentage of the final cropped image, so the subject sits inside a border of that size. mask: percentage of the mask bounding box. image: percentage of the full input image.",
                }),
                "aspect_limit": (list(ASPECT_LIMITS.keys()), {
                    "default": "2:3 / 3:2",
                    "tooltip": "Never crop more elongated than this ratio. A crop that would be taller than 2:3 is widened, one wider than 3:2 is heightened. The crop is only ever grown, never shrunk.",
                }),
                "divisible_by": ("INT", {
                    "default": 1, "min": 1, "max": 1024, "step": 1,
                    "tooltip": "Grow the crop so its width and height are multiples of this value. 1 disables it.",
                }),
                "mask_threshold": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Mask pixels above this value count as masked.",
                }),
            },
            "optional": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "crop"
    CATEGORY = "RulZnodes"
    DESCRIPTION = (
        "Crop an image and/or mask to the mask's bounding box, extended on every side by "
        "a percentage of the crop, mask or image size: one value for all sides or one per "
        "edge. Both inputs are optional."
    )

    def crop(self, padding_percent, padding_reference, aspect_limit, divisible_by, mask_threshold,
             image=None, mask=None, padding_mode="uniform",
             padding_top=10.0, padding_bottom=10.0, padding_left=10.0, padding_right=10.0):
        if image is None and mask is None:
            raise ValueError("Mask Intelligent Crop needs an image, a mask, or both.")

        # No mask: nothing to crop by. Pass the image through with a blank mask.
        if mask is None:
            batch, height, width, _ = image.shape
            blank = torch.zeros((batch, height, width), dtype=torch.float32, device=image.device)
            return (image, blank)

        mask = _normalize_mask(mask)

        if image is not None:
            height, width = int(image.shape[1]), int(image.shape[2])
            mask = mask.to(image.device)
        else:
            height, width = int(mask.shape[1]), int(mask.shape[2])

        # Find the box on the native mask, then bring the mask to frame size.
        if padding_mode == "per_edge":
            padding_percent = (padding_top, padding_bottom, padding_left, padding_right)
        box = compute_crop_box(mask, padding_percent, divisible_by, mask_threshold, height, width,
                               padding_reference, ASPECT_LIMITS.get(aspect_limit))
        mask = _resize_mask(mask, height, width)
        if box is None:
            # Empty mask: return the inputs at full size.
            return (image, mask)

        x0, y0, x1, y1 = box
        cropped_mask = mask[:, y0:y1, x0:x1]
        cropped_image = image[:, y0:y1, x0:x1, :] if image is not None else None
        return (cropped_image, cropped_mask)


NODE_CLASS_MAPPINGS = {
    "MaskIntelligentCrop": MaskIntelligentCrop,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MaskIntelligentCrop": "RulZ Mask Intelligent Crop",
}
