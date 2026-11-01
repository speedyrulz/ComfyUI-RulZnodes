import hashlib
import os

import numpy as np
import torch
from PIL import Image, ImageOps, ImageSequence

import folder_paths
import node_helpers

try:
    from pillow_heif import register_heif_opener

    register_heif_opener()
    HEIF_AVAILABLE = True
except ImportError:
    HEIF_AVAILABLE = False
    print(
        "[ImageLoadFilenameHEIC] pillow-heif is not installed; HEIC/HEIF files "
        "cannot be loaded. Run: pip install pillow-heif"
    )

IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff", ".jfif",
}
HEIF_EXTENSIONS = {".heic", ".heif"}


if HEIF_AVAILABLE:
    try:
        from . import heic_preview

        heic_preview.install()
    except Exception as e:
        print(f"[ImageLoadFilenameHEIC] HEIC preview disabled: {e}")


def _list_images():
    exts = IMAGE_EXTENSIONS | (HEIF_EXTENSIONS if HEIF_AVAILABLE else set())
    input_dir = folder_paths.get_input_directory()
    files = [
        f
        for f in os.listdir(input_dir)
        if os.path.isfile(os.path.join(input_dir, f))
        and os.path.splitext(f)[1].lower() in exts
    ]
    return sorted(files)


class LoadImageWithFilenameHEIC:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"image": (_list_images(), {"image_upload": True})},
        }

    CATEGORY = "RulZnodes"
    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("image", "mask", "filename")
    FUNCTION = "load_image"

    def load_image(self, image):
        image_path = folder_paths.get_annotated_filepath(image)
        img = node_helpers.pillow(Image.open, image_path)

        output_images = []
        output_masks = []
        w, h = None, None

        for frame in ImageSequence.Iterator(img):
            frame = node_helpers.pillow(ImageOps.exif_transpose, frame)

            if frame.mode == "I":
                frame = frame.point(lambda i: i * (1 / 255))
            rgb = frame.convert("RGB")

            if not output_images:
                w, h = rgb.size
            if rgb.size != (w, h):
                continue

            arr = np.array(rgb).astype(np.float32) / 255.0
            output_images.append(torch.from_numpy(arr)[None,])

            if "A" in frame.getbands():
                alpha = np.array(frame.getchannel("A")).astype(np.float32) / 255.0
                mask = 1.0 - torch.from_numpy(alpha)
            elif frame.mode == "P" and "transparency" in frame.info:
                alpha = np.array(frame.convert("RGBA").getchannel("A")).astype(np.float32) / 255.0
                mask = 1.0 - torch.from_numpy(alpha)
            else:
                mask = torch.zeros((h, w), dtype=torch.float32)
            output_masks.append(mask.unsqueeze(0))

        if len(output_images) > 1:
            output_image = torch.cat(output_images, dim=0)
            output_mask = torch.cat(output_masks, dim=0)
        else:
            output_image = output_images[0]
            output_mask = output_masks[0]

        # Name of the file without directory (and annotation) or extension.
        filename = os.path.splitext(os.path.basename(image_path))[0]
        return (output_image, output_mask, filename)

    @classmethod
    def IS_CHANGED(cls, image):
        image_path = folder_paths.get_annotated_filepath(image)
        m = hashlib.sha256()
        with open(image_path, "rb") as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(cls, image):
        if not folder_paths.exists_annotated_filepath(image):
            return f"Invalid image file: {image}"
        return True


class LoadImageBatchHEIC(LoadImageWithFilenameHEIC):
    """Same loader as RulZ Load Image (HEIC). The list of dropped images lives in the
    frontend (web/heic_batch.js), which moves `image` on to the next one after each
    queued run, so the backend only ever loads the current image."""


NODE_CLASS_MAPPINGS = {
    "LoadImageWithFilenameHEIC": LoadImageWithFilenameHEIC,
    "LoadImageBatchHEIC": LoadImageBatchHEIC,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadImageWithFilenameHEIC": "RulZ Load Image (HEIC)",
    "LoadImageBatchHEIC": "RulZ Load Image Batch (HEIC)",
}
