import json
import os
import re
import threading

import torch
from PIL import Image, ImageOps

import folder_paths

from .image_utils import to_float_rgb

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tiff", ".tif")

# Progress is persisted per-directory so a batch resumes from where it left off,
# even across ComfyUI restarts. It lives in ComfyUI's user folder, which survives
# updates of this pack. Progress saved by older versions inside the pack folder is
# picked up until the new file exists.
STATE_FILE = os.path.join(folder_paths.get_user_directory(), "rulznodes", "batch_state.json")
_LEGACY_STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "batch_state.json")
_lock = threading.Lock()


def _load_state():
    path = STATE_FILE if os.path.exists(STATE_FILE) else _LEGACY_STATE_FILE
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_state(state):
    try:
        os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
        tmp = STATE_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
        os.replace(tmp, STATE_FILE)
    except OSError as e:
        # Losing the bookmark is better than failing the run.
        print(f"[LoadImageBatchSimple] could not save progress to {STATE_FILE}: {e}")


def _resolve_dir(directory):
    """The directory as typed -> absolute path. Accepts ~ and surrounding quotes
    (Windows "Copy as path" adds them). Empty means ComfyUI's input folder."""
    directory = directory.strip().strip("\"'").strip()
    if not directory:
        return folder_paths.get_input_directory()
    return os.path.abspath(os.path.expanduser(directory))


def _natural_key(name):
    # "img2" sorts before "img10", as in a file browser and the HEIC batch node.
    return [int(part) if part.isdecimal() else part.lower() for part in re.split(r"(\d+)", name)]


def _list_images(directory):
    if not os.path.isdir(directory):
        return []
    files = [f for f in os.listdir(directory) if f.lower().endswith(IMAGE_EXTS)]
    files.sort(key=_natural_key)
    return files


def _load_image(path):
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img)
        return torch.from_numpy(to_float_rgb(img))[None,]


class LoadImageBatchSimple:
    """
    Steps through every image in a directory, one image per queued run.
    Progress is remembered per directory, so queueing 5 then 5 again covers all 10.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "directory": ("STRING", {"default": "", "tooltip": "Folder to read. Empty = ComfyUI's input folder. ~ and quoted paths work."}),
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
        directory = _resolve_dir(directory)
        if not os.path.isdir(directory):
            return f"Directory not found: {directory}"
        if not _list_images(directory):
            return f"No images found in: {directory}"
        return True

    def load(self, directory, reset, loop):
        directory = _resolve_dir(directory)
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
