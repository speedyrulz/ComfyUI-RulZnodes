"""
RulZ Resize Image Advanced (originally "Resize Image v2 Advanced")

A standalone clone of the KJNodes "Resize Image v2" node (ImageResizeKJv2) with
a few additions:

  * use_input_size    - "image_1" / "image_2" / "off": use that image's
                        width/height as the width/height settings, so the output
                        is the closest resolution to it that satisfies
                        divisible_by. image_2 is optional and only ever used for
                        its size; the output is always a resized image_1.
  * upscale_mode      - "upscale_by" multiplies both edges of the target size,
                        "total_megapixels" scales the target to the nearest total
                        pixel count. Both respect divisible_by.
  * output resolution - reported back to the node UI after execution.
  * defaults          - upscale_method = lanczos, keep_proportion = crop.

The pad helper and colour parser from KJNodes are vendored below so this
package has no dependency on ComfyUI-KJNodes.
"""

import logging
import math
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import ImageColor

from nodes import MAX_RESOLUTION
from comfy.utils import common_upscale
from comfy import model_management

try:
    from server import PromptServer
except ImportError:
    PromptServer = None


# ---------------------------------------------------------------------------
# Helpers (vendored from ComfyUI-KJNodes)
# ---------------------------------------------------------------------------

def string_to_color(color_string: str) -> List[int]:
    color_list = [0, 0, 0]  # Default fallback (black)

    if ',' in color_string:
        # CSV format (e.g. "255, 0, 0", "255, 0, 0, 128" or "1.0, 0.5, 0.0")
        try:
            values = [float(channel.strip()) for channel in color_string.split(',')]
            if all(0 <= v <= 1 for v in values):
                color_list = [int(v * 255) for v in values]
            else:
                color_list = [int(v) for v in values]
        except ValueError:
            logging.warning(f"Invalid color format: {color_string}. Using default black.")
    elif color_string.startswith('#') or (color_string.lstrip('#').isalnum() and not color_string.lstrip('#').replace('.', '', 1).isdigit()):
        color_string_stripped = color_string.lstrip('#')
        if len(color_string_stripped) in [6, 8] and all(c in '0123456789ABCDEFabcdef' for c in color_string_stripped):
            if len(color_string_stripped) == 6:
                color_list = [int(color_string_stripped[i:i + 2], 16) for i in (0, 2, 4)]
            else:
                color_list = [int(color_string_stripped[i:i + 2], 16) for i in (0, 2, 4, 6)]
        else:
            try:
                color_list = list(ImageColor.getrgb(color_string))
            except ValueError:
                logging.warning(f"Invalid color name or hex format: {color_string}. Using default black.")
    else:
        try:
            value = float(color_string.strip())
            value = int(value * 255) if 0 <= value <= 1 else int(value)
            color_list = [value, value, value]
        except ValueError:
            logging.warning(f"Invalid color format: {color_string}. Using default black.")

    return np.clip(color_list, 0, 255).tolist()


def pad_image(image, left, right, top, bottom, color, pad_mode, mask=None):
    """Pad an image (and optional mask). Port of ImagePadKJ.pad without target sizing."""
    B, H, W, C = image.shape
    if mask is not None:
        BM, HM, WM = mask.shape
        if HM != H or WM != W:
            mask = F.interpolate(mask.unsqueeze(1), size=(H, W), mode='nearest-exact').squeeze(1)

    color_list = string_to_color(color)
    bg_color = [x / 255.0 for x in color_list]
    if len(bg_color) == 1:
        bg_color = bg_color * 3
    bg_color = torch.tensor(bg_color, dtype=image.dtype, device=image.device)

    pad_left, pad_right, pad_top, pad_bottom = left, right, top, bottom
    padded_width = W + pad_left + pad_right
    padded_height = H + pad_top + pad_bottom

    if pad_mode == "pillarbox_blur":
        def _gaussian_blur_nchw(img_nchw, sigma_px):
            if sigma_px <= 0:
                return img_nchw
            radius = max(1, int(3.0 * float(sigma_px)))
            k = 2 * radius + 1
            x = torch.arange(-radius, radius + 1, device=img_nchw.device, dtype=img_nchw.dtype)
            k1 = torch.exp(-(x * x) / (2.0 * float(sigma_px) * float(sigma_px)))
            k1 = k1 / k1.sum()
            kx = k1.view(1, 1, 1, k)
            ky = k1.view(1, 1, k, 1)
            c = img_nchw.shape[1]
            kx = kx.repeat(c, 1, 1, 1)
            ky = ky.repeat(c, 1, 1, 1)
            img_nchw = F.conv2d(img_nchw, kx, padding=(0, radius), groups=c)
            img_nchw = F.conv2d(img_nchw, ky, padding=(radius, 0), groups=c)
            return img_nchw

        out_image = torch.zeros((B, padded_height, padded_width, C), dtype=image.dtype, device=image.device)
        for b in range(B):
            scale_fill = max(padded_width / float(W), padded_height / float(H)) if (W > 0 and H > 0) else 1.0
            bg_w = max(1, int(round(W * scale_fill)))
            bg_h = max(1, int(round(H * scale_fill)))
            src_b = image[b].movedim(-1, 0).unsqueeze(0)
            bg = common_upscale(src_b, bg_w, bg_h, "bilinear", crop="disabled")
            y0 = max(0, (bg_h - padded_height) // 2)
            x0 = max(0, (bg_w - padded_width) // 2)
            y1 = min(bg_h, y0 + padded_height)
            x1 = min(bg_w, x0 + padded_width)
            bg = bg[:, :, y0:y1, x0:x1]
            if bg.shape[2] != padded_height or bg.shape[3] != padded_width:
                pad_h = padded_height - bg.shape[2]
                pad_w = padded_width - bg.shape[3]
                pad_top_fix = max(0, pad_h // 2)
                pad_bottom_fix = max(0, pad_h - pad_top_fix)
                pad_left_fix = max(0, pad_w // 2)
                pad_right_fix = max(0, pad_w - pad_left_fix)
                bg = F.pad(bg, (pad_left_fix, pad_right_fix, pad_top_fix, pad_bottom_fix), mode="replicate")
            sigma = max(1.0, 0.006 * float(min(padded_height, padded_width)))
            bg = _gaussian_blur_nchw(bg, sigma_px=sigma)
            if C >= 3:
                r, g, bch = bg[:, 0:1], bg[:, 1:2], bg[:, 2:3]
                luma = 0.2126 * r + 0.7152 * g + 0.0722 * bch
                gray = torch.cat([luma, luma, luma], dim=1)
                desat = 0.20
                rgb = torch.cat([r, g, bch], dim=1)
                rgb = rgb * (1.0 - desat) + gray * desat
                bg[:, 0:3, :, :] = rgb
            bg = torch.clamp(bg * 0.35, 0.0, 1.0)
            out_image[b] = bg.squeeze(0).movedim(0, -1)
        out_image[:, pad_top:pad_top + H, pad_left:pad_left + W, :] = image
        out_masks = torch.ones((B, padded_height, padded_width), dtype=image.dtype, device=image.device)
        if mask is not None:
            out_masks[:, pad_top:pad_top + H, pad_left:pad_left + W] = mask
        else:
            out_masks[:, pad_top:pad_top + H, pad_left:pad_left + W] = 0.0
        return out_image, out_masks

    out_image = torch.zeros((B, padded_height, padded_width, C), dtype=image.dtype, device=image.device)
    for b in range(B):
        if pad_mode == "edge":
            out_image[b, :pad_top, :, :] = image[b, 0, :, :].mean(dim=0)
            out_image[b, pad_top + H:, :, :] = image[b, H - 1, :, :].mean(dim=0)
            out_image[b, :, :pad_left, :] = image[b, :, 0, :].mean(dim=0)
            out_image[b, :, pad_left + W:, :] = image[b, :, W - 1, :].mean(dim=0)
            out_image[b, pad_top:pad_top + H, pad_left:pad_left + W, :] = image[b]
        elif pad_mode == "edge_pixel":
            for y in range(pad_top):
                out_image[b, y, pad_left:pad_left + W, :] = image[b, 0, :, :]
            for y in range(pad_top + H, padded_height):
                out_image[b, y, pad_left:pad_left + W, :] = image[b, H - 1, :, :]
            for x in range(pad_left):
                out_image[b, pad_top:pad_top + H, x, :] = image[b, :, 0, :]
            for x in range(pad_left + W, padded_width):
                out_image[b, pad_top:pad_top + H, x, :] = image[b, :, W - 1, :]
            out_image[b, :pad_top, :pad_left, :] = image[b, 0, 0, :]
            out_image[b, :pad_top, pad_left + W:, :] = image[b, 0, W - 1, :]
            out_image[b, pad_top + H:, :pad_left, :] = image[b, H - 1, 0, :]
            out_image[b, pad_top + H:, pad_left + W:, :] = image[b, H - 1, W - 1, :]
            out_image[b, pad_top:pad_top + H, pad_left:pad_left + W, :] = image[b]
        else:
            out_image[b, :, :, :] = bg_color.unsqueeze(0).unsqueeze(0)
            out_image[b, pad_top:pad_top + H, pad_left:pad_left + W, :] = image[b]

    if mask is not None:
        out_masks = F.pad(mask, (pad_left, pad_right, pad_top, pad_bottom), mode='replicate')
    else:
        out_masks = torch.ones((B, padded_height, padded_width), dtype=image.dtype, device=image.device)
        out_masks[:, pad_top:pad_top + H, pad_left:pad_left + W] = 0.0

    return out_image, out_masks


def round_to_divisible(value, divisible_by):
    """Round to the nearest multiple of divisible_by (never below one multiple / 1 px)."""
    if divisible_by > 1:
        return max(divisible_by, int(round(value / divisible_by)) * divisible_by)
    return max(1, int(round(value)))


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class ResizeImageV2Advanced:
    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos", "nvidia_rtx_vsr"]
    keep_proportion_modes = ["stretch", "resize", "pad", "pad_edge", "pad_edge_pixel", "crop", "pillarbox_blur", "total_pixels"]
    upscale_modes = ["disabled", "upscale_by", "total_megapixels"]
    input_size_modes = ["off", "image_1", "image_2"]

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image_1": ("IMAGE", {"tooltip": "The image that gets resized."}),
                "use_input_size": (s.input_size_modes, {"default": "off", "tooltip": "Use image_1's or image_2's width and height as the width/height settings (image_2 is only used for its size). Combined with divisible_by this outputs the closest valid resolution to that image."}),
                "width": ("INT", {"default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
                "height": ("INT", {"default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
                "upscale_mode": (s.upscale_modes, {"default": "disabled", "tooltip": "disabled: use width/height as-is.\nupscale_by: multiply both edges of the target size.\ntotal_megapixels: scale the target size to the nearest total pixel count."}),
                "upscale_by": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 100.0, "step": 0.01, "tooltip": "Edge multiplier used when upscale_mode is 'upscale_by'."}),
                "total_megapixels": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 1000.0, "step": 0.01, "tooltip": "Target total pixel count in megapixels (1 MP = 1024*1024 px) used when upscale_mode is 'total_megapixels'."}),
                "upscale_method": (s.upscale_methods, {"default": "lanczos"}),
                "keep_proportion": (s.keep_proportion_modes, {"default": "crop"}),
                "pad_color": ("STRING", {"default": "0, 0, 0", "tooltip": "Color to use for padding."}),
                "crop_position": (["center", "top", "bottom", "left", "right"], {"default": "center"}),
                "divisible_by": ("INT", {"default": 32, "min": 0, "max": 512, "step": 1}),
            },
            "optional": {
                "image_2": ("IMAGE", {"tooltip": "Optional. Only its width/height are used (when use_input_size = image_2); the output is always a resized image_1."}),
                "mask": ("MASK",),
                "device": (["cpu", "gpu"],),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT", "MASK",)
    RETURN_NAMES = ("IMAGE", "width", "height", "mask",)
    FUNCTION = "resize"
    CATEGORY = "RulZnodes"
    DESCRIPTION = """
Resizes the image to the specified width and height.
Size can be taken from image_1 or image_2 (use_input_size), multiplied
(upscale_by) or fitted to a total pixel count (total_megapixels).
All target sizes are rounded to the nearest multiple of divisible_by.
The resulting output resolution is shown on the node.
"""

    # ------------------------------------------------------------------
    def compute_target_size(self, W, H, width, height, use_input_size, upscale_mode, upscale_by, total_megapixels, divisible_by, W2=None, H2=None):
        """Resolve the requested width/height after the advanced options are applied."""
        if use_input_size == "image_1":
            width, height = W, H
        elif use_input_size == "image_2":
            if W2 is None or H2 is None:
                raise ValueError("use_input_size is set to 'image_2' but no image_2 is connected.")
            width, height = W2, H2

        if upscale_mode == "disabled":
            return width, height

        # Resolve zeros (auto dimension) from the input aspect ratio so scaling has something to work on.
        if width == 0 and height == 0:
            base_w, base_h = float(W), float(H)
        elif width == 0:
            base_h = float(height)
            base_w = height * W / H
        elif height == 0:
            base_w = float(width)
            base_h = width * H / W
        else:
            base_w, base_h = float(width), float(height)

        if upscale_mode == "upscale_by":
            base_w *= upscale_by
            base_h *= upscale_by
        elif upscale_mode == "total_megapixels":
            target_pixels = total_megapixels * 1024 * 1024
            scale = math.sqrt(target_pixels / (base_w * base_h))
            base_w *= scale
            base_h *= scale

        return round_to_divisible(base_w, divisible_by), round_to_divisible(base_h, divisible_by)

    # ------------------------------------------------------------------
    def resize(self, image_1, use_input_size, width, height, upscale_mode, upscale_by, total_megapixels,
               keep_proportion, upscale_method, divisible_by, pad_color, crop_position, unique_id,
               device="cpu", image_2=None, mask=None, per_batch=64):
        image = image_1
        B, H, W, C = image.shape
        H2, W2 = (image_2.shape[1], image_2.shape[2]) if image_2 is not None else (None, None)

        width, height = self.compute_target_size(
            W, H, width, height, use_input_size, upscale_mode, upscale_by, total_megapixels, divisible_by, W2, H2
        )

        # Treat ComfyUI's 64x64 placeholder mask as no mask
        if mask is not None and mask.shape[-2:] == (64, 64) and (H != 64 or W != 64):
            mask = None

        # Scale mask to match image dimensions if they differ
        if mask is not None and mask.shape[-2:] != (H, W):
            mask = common_upscale(mask.unsqueeze(1), W, H, "bilinear", crop="disabled").squeeze(1)

        if device == "gpu":
            if upscale_method == "lanczos":
                raise ValueError("Lanczos is not supported on the GPU")
            device = model_management.get_torch_device()
        else:
            device = torch.device("cpu")

        pillarbox_blur = keep_proportion == "pillarbox_blur"
        pad_left = pad_right = pad_top = pad_bottom = 0

        if keep_proportion in ["resize", "total_pixels"] or keep_proportion.startswith("pad") or pillarbox_blur:
            if keep_proportion == "total_pixels":
                total_pixels = width * height
                aspect_ratio = W / H
                new_height = int(math.sqrt(total_pixels / aspect_ratio))
                new_width = int(math.sqrt(total_pixels * aspect_ratio))
            elif width == 0 and height == 0:
                new_width = W
                new_height = H
            elif width == 0 and height != 0:
                ratio = height / H
                new_width = round(W * ratio)
                new_height = height
            elif height == 0 and width != 0:
                ratio = width / W
                new_width = width
                new_height = round(H * ratio)
            else:
                ratio = min(width / W, height / H)
                new_width = round(W * ratio)
                new_height = round(H * ratio)

            if keep_proportion.startswith("pad") or pillarbox_blur:
                if crop_position == "center":
                    pad_left = (width - new_width) // 2
                    pad_right = width - new_width - pad_left
                    pad_top = (height - new_height) // 2
                    pad_bottom = height - new_height - pad_top
                elif crop_position == "top":
                    pad_left = (width - new_width) // 2
                    pad_right = width - new_width - pad_left
                    pad_top = 0
                    pad_bottom = height - new_height
                elif crop_position == "bottom":
                    pad_left = (width - new_width) // 2
                    pad_right = width - new_width - pad_left
                    pad_top = height - new_height
                    pad_bottom = 0
                elif crop_position == "left":
                    pad_left = 0
                    pad_right = width - new_width
                    pad_top = (height - new_height) // 2
                    pad_bottom = height - new_height - pad_top
                elif crop_position == "right":
                    pad_left = width - new_width
                    pad_right = 0
                    pad_top = (height - new_height) // 2
                    pad_bottom = height - new_height - pad_top

            width = new_width
            height = new_height
        else:
            if width == 0:
                width = W
            if height == 0:
                height = H

        # Snap to the nearest multiple of divisible_by (KJ's original floors; "advanced" picks the closest).
        if divisible_by > 1:
            width = round_to_divisible(width, divisible_by)
            height = round_to_divisible(height, divisible_by)
            # Nearest rounding can grow the content past the pad canvas; never pad negatively.
            pad_left, pad_right, pad_top, pad_bottom = (max(0, p) for p in (pad_left, pad_right, pad_top, pad_bottom))

        # NVIDIA RTX Video Super Resolution setup
        nvvfx_sr = None
        nvvfx_ctx = None
        if upscale_method == "nvidia_rtx_vsr":
            try:
                import nvvfx
            except ImportError:
                raise ImportError("NVIDIA RTX Video Super Resolution is not available. Please install the nvidia-vfx library and ensure you have a compatible NVIDIA GPU.")
            nvvfx_ctx = nvvfx.VideoSuperRes(nvvfx.effects.QualityLevel.ULTRA)
            nvvfx_sr = nvvfx_ctx.__enter__()
            nvvfx_sr.output_width = max(8, round(width / 8) * 8)
            nvvfx_sr.output_height = max(8, round(height / 8) * 8)
            nvvfx_sr.load()

        def _process_subbatch(in_image, in_mask, pad_left, pad_right, pad_top, pad_bottom):
            out_image = in_image if in_image.device == device else in_image.to(device)
            out_mask = None if in_mask is None else (in_mask if in_mask.device == device else in_mask.to(device))

            if keep_proportion == "crop":
                old_height = out_image.shape[-3]
                old_width = out_image.shape[-2]
                old_aspect = old_width / old_height
                new_aspect = width / height
                if old_aspect > new_aspect:
                    crop_w = round(old_height * new_aspect)
                    crop_h = old_height
                else:
                    crop_w = old_width
                    crop_h = round(old_width / new_aspect)
                if crop_position == "center":
                    x = (old_width - crop_w) // 2
                    y = (old_height - crop_h) // 2
                elif crop_position == "top":
                    x = (old_width - crop_w) // 2
                    y = 0
                elif crop_position == "bottom":
                    x = (old_width - crop_w) // 2
                    y = old_height - crop_h
                elif crop_position == "left":
                    x = 0
                    y = (old_height - crop_h) // 2
                elif crop_position == "right":
                    x = old_width - crop_w
                    y = (old_height - crop_h) // 2
                out_image = out_image.narrow(-2, x, crop_w).narrow(-3, y, crop_h)
                if out_mask is not None:
                    out_mask = out_mask.narrow(-1, x, crop_w).narrow(-2, y, crop_h)

            if upscale_method == "nvidia_rtx_vsr":
                frames_chw = out_image.movedim(-1, 1).cuda().contiguous()
                upscaled_frames = []
                for j in range(frames_chw.shape[0]):
                    dlpack_out = nvvfx_sr.run(frames_chw[j]).image
                    upscaled_frames.append(torch.from_dlpack(dlpack_out).clone())
                out_image = torch.stack(upscaled_frames, dim=0).movedim(1, -1).cpu()
                if out_mask is not None:
                    out_mask = common_upscale(out_mask.unsqueeze(1), width, height, "bilinear", crop="disabled").squeeze(1)
            else:
                out_image = common_upscale(out_image.movedim(-1, 1), width, height, upscale_method, crop="disabled").movedim(1, -1)
                if out_mask is not None:
                    if upscale_method == "lanczos":
                        out_mask = common_upscale(out_mask.unsqueeze(1).repeat(1, 3, 1, 1), width, height, upscale_method, crop="disabled").movedim(1, -1)[:, :, :, 0]
                    else:
                        out_mask = common_upscale(out_mask.unsqueeze(1), width, height, upscale_method, crop="disabled").squeeze(1)

            if (keep_proportion.startswith("pad") or pillarbox_blur) and (pad_left > 0 or pad_right > 0 or pad_top > 0 or pad_bottom > 0):
                padded_width = width + pad_left + pad_right
                padded_height = height + pad_top + pad_bottom
                if divisible_by > 1:
                    width_remainder = padded_width % divisible_by
                    height_remainder = padded_height % divisible_by
                    if width_remainder > 0:
                        pad_right += divisible_by - width_remainder
                    if height_remainder > 0:
                        pad_bottom += divisible_by - height_remainder

                pad_mode = (
                    "pillarbox_blur" if pillarbox_blur else
                    "edge" if keep_proportion == "pad_edge" else
                    "edge_pixel" if keep_proportion == "pad_edge_pixel" else
                    "color"
                )
                out_image, out_mask = pad_image(out_image, pad_left, pad_right, pad_top, pad_bottom, pad_color, pad_mode, mask=out_mask)

            return out_image, out_mask

        if per_batch == 0 or B <= per_batch:
            out_image, out_mask = _process_subbatch(image, mask, pad_left, pad_right, pad_top, pad_bottom)
        else:
            chunks = []
            mask_chunks = [] if mask is not None else None
            total_batches = (B + per_batch - 1) // per_batch
            current_batch = 0
            for start_idx in range(0, B, per_batch):
                current_batch += 1
                end_idx = min(start_idx + per_batch, B)
                sub_img = image[start_idx:end_idx]
                sub_mask = mask[start_idx:end_idx] if mask is not None else None
                sub_out_img, sub_out_mask = _process_subbatch(sub_img, sub_mask, pad_left, pad_right, pad_top, pad_bottom)
                chunks.append(sub_out_img.cpu())
                if mask is not None:
                    mask_chunks.append(sub_out_mask.cpu() if sub_out_mask is not None else None)
                if unique_id and PromptServer is not None:
                    try:
                        PromptServer.instance.send_progress_text(
                            f"<tr><td>Resize v2 Adv</td><td>batch {current_batch}/{total_batches} · images {end_idx}/{B}</td></tr>",
                            unique_id
                        )
                    except Exception:
                        pass
                else:
                    logging.info(f"[ResizeImageV2Advanced] batch {current_batch}/{total_batches} · images {end_idx}/{B}")
            out_image = torch.cat(chunks, dim=0)
            if mask is not None and any(m is not None for m in mask_chunks):
                out_mask = torch.cat([m for m in mask_chunks if m is not None], dim=0)
            else:
                out_mask = None

        if nvvfx_ctx is not None:
            nvvfx_ctx.__exit__(None, None, None)

        out_w, out_h = out_image.shape[2], out_image.shape[1]
        out_mp = (out_w * out_h) / (1024 * 1024)
        resolution_text = f"{out_w} x {out_h}  ({out_mp:.2f} MP)  |  image_1 {W} x {H}"
        if image_2 is not None:
            resolution_text += f"  |  image_2 {W2} x {H2}"

        if unique_id and PromptServer is not None:
            try:
                memory_size_mb = (out_image.numel() * out_image.element_size()) / (1024 * 1024)
                PromptServer.instance.send_progress_text(
                    f"<tr><td>Output: </td><td><b>{out_image.shape[0]}</b> x <b>{out_w}</b> x <b>{out_h} | {memory_size_mb:.2f}MB</b></td></tr>",
                    unique_id
                )
            except Exception:
                pass

        out_mask = out_mask.cpu() if out_mask is not None else torch.zeros(1, 64, 64, device=torch.device("cpu"), dtype=torch.float32)

        return {
            "ui": {"text": [resolution_text]},
            "result": (out_image.cpu(), out_w, out_h, out_mask),
        }


NODE_CLASS_MAPPINGS = {"ResizeImageV2Advanced": ResizeImageV2Advanced}
NODE_DISPLAY_NAME_MAPPINGS = {"ResizeImageV2Advanced": "RulZ Resize Image Advanced"}
