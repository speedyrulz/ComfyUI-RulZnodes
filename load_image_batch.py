import json
import os
import threading

import numpy as np
import torch
from PIL import Image, ImageOps

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tiff", ".tif")

# Progress is persisted per-directory in this file so a batch resumes from where
# it left off, even across ComfyUI restarts.
STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "batch_state.json")
_lock = threading.Lock()


def _load_state():
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_state(state):
    tmp = STATE_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, STATE_FILE)


def _list_images(directory):
    if not os.path.isdir(directory):
        return []
    files = [f for f in os.listdir(directory) if f.lower().endswith(IMAGE_EXTS)]
    files.sort(key=str.lower)
    return files


def _load_image(path):
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    img = img.convert("RGB")
    arr = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr)[None,]


class LoadImageBatchSimple:
    """
    Steps through every image in a directory, one image per queued run.
    Progress is remembered per directory, so queueing 5 then 5 again covers all 10.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "directory": ("STRING", {"default": "/home/jeff/input"}),
                "reset": ("BOOLEAN", {"default": False, "label_on": "restart from first image", "label_off": "resume"}),
                "loop": ("BOOLEAN", {"default": True, "label_on": "wrap around at end", "label_off": "stay on last image"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "INT", "INT")
    RETURN_NAMES = ("image", "filename", "total_images", "current_index")
    FUNCTION = "load"
    CATEGORY = "RulZnodes"

    @classmethod
    def IS_CHANGED(cls, directory, reset, loop):
        # Always re-run so each queued prompt advances to the next image.
        return float("nan")

    @classmethod
    def VALIDATE_INPUTS(cls, directory, reset, loop):
        if not os.path.isdir(directory):
            return f"Directory not found: {directory}"
        if not _list_images(directory):
            return f"No images found in: {directory}"
        return True

    def load(self, directory, reset, loop):
        directory = os.path.abspath(os.path.expanduser(directory))
        files = _list_images(directory)
        total = len(files)
        if total == 0:
            raise FileNotFoundError(f"No images found in {directory}")

        with _lock:
            state = _load_state()
            index = 0 if reset else int(state.get(directory, 0))

            if index >= total:
                if loop:
                    index = 0
                else:
                    index = total - 1

            filename = files[index]
            image = _load_image(os.path.join(directory, filename))

            # Store the position for the *next* run.
            next_index = index + 1
            if next_index >= total and loop:
                next_index = 0
            state[directory] = next_index
            _save_state(state)

        print(f"[LoadImageBatchSimple] {index + 1}/{total}: {filename}")

        return {
            "ui": {"text": [f"{index + 1} / {total}  —  {filename}"]},
            "result": (image, filename, total, index),
        }


NODE_CLASS_MAPPINGS = {"LoadImageBatchSimple": LoadImageBatchSimple}
NODE_DISPLAY_NAME_MAPPINGS = {"LoadImageBatchSimple": "RulZ Simple Load Image Batch"}
