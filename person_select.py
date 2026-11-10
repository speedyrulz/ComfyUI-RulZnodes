"""RulZ Person Select (SAM3): load an image, click a person, get their mask.

SAM 3 finds every instance of the text prompt ("person") as its own mask. The
frontend (web/person_select.js) shows those masks over the image and stores the
clicked points; the mask under each point is the selection. A click that lands on
nothing SAM detected falls back to a SAM point prompt at that spot.

The preview route and the node share `_resolve`, so what is shown on the node is
what the node outputs. Detection runs through ComfyUI's own SAM3 Detect node.
"""
import asyncio
import base64
import contextlib
import io
import json
import os
import re
import threading
from collections import OrderedDict

import torch
import torch.nn.functional as F
from aiohttp import web
from PIL import Image

import comfy.sd
import comfy.utils
import folder_paths
from server import PromptServer

from .load_image_heic import LoadImageWithFilenameHEIC, _list_images

LABEL_MAX_SIDE = 1024  # longest side of the label map sent to the frontend
_DET_CACHE_MAX = 2  # each entry holds full-resolution masks for one image
_EXTRA_CACHE_MAX = 16

_lock = threading.Lock()  # one SAM inference at a time (preview route and node)
_loaded = {"name": None, "model": None, "clip": None}
_det_cache = OrderedDict()

routes = PromptServer.instance.routes


def _list_models():
    """Checkpoints, with the ones that look like SAM first so the default is right."""
    names = folder_paths.get_filename_list("checkpoints")
    return sorted(names, key=lambda n: ("sam" not in n.lower(), n.lower()))


def _detect_node():
    try:
        from comfy_extras.nodes_sam3 import SAM3_Detect
    except ImportError as e:
        raise RuntimeError(
            "RulZ Person Select needs a ComfyUI version with built-in SAM 3 support "
            "(the 'SAM3 Detect' node). Update ComfyUI."
        ) from e
    return SAM3_Detect


def _get_model(name):
    if _loaded["name"] != name:
        path = folder_paths.get_full_path_or_raise("checkpoints", name)
        out = comfy.sd.load_checkpoint_guess_config(
            path,
            output_vae=False,
            output_clip=True,
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
        )
        model, clip = out[0], out[1]
        if clip is None or not hasattr(model.model.diffusion_model, "forward_segment"):
            raise ValueError(f"'{name}' is not a SAM 3 checkpoint")
        _loaded.update(name=name, model=model, clip=clip)
    return _loaded["model"], _loaded["clip"]


def _image_path(name):
    """Annotated image name -> absolute path inside ComfyUI's input/output/temp."""
    if not isinstance(name, str) or not name.strip():
        raise FileNotFoundError("No image selected")
    path = os.path.abspath(folder_paths.get_annotated_filepath(name))
    bases = (
        folder_paths.get_input_directory(),
        folder_paths.get_output_directory(),
        folder_paths.get_temp_directory(),
    )
    if not any(path.startswith(os.path.abspath(b) + os.sep) for b in bases) or not os.path.isfile(path):
        raise FileNotFoundError(f"Image not found: {name}")
    return path


def _prompt_text(prompt, max_detections):
    """'person, dog' -> 'person:20, dog:20' (SAM 3 returns one match per prompt
    unless told a maximum). A ':N' the user typed is kept."""
    parts = [p.strip() for p in str(prompt).split(",") if p.strip()] or ["person"]
    return ", ".join(p if re.search(r":\s*[\d.]+$", p) else f"{p}:{max_detections}" for p in parts)


def _detect(image_name, model_name, prompt, max_detections, threshold):
    path = _image_path(image_name)
    text = _prompt_text(prompt, max_detections)
    key = (path, os.path.getmtime(path), model_name, text, round(float(threshold), 3))
    entry = _det_cache.get(key)
    if entry is not None:
        _det_cache.move_to_end(key)
        return entry

    image = LoadImageWithFilenameHEIC().load_image(image_name)[0][:1]
    model, clip = _get_model(model_name)
    conditioning = clip.encode_from_tokens_scheduled(clip.tokenize(text))
    masks = _detect_node().execute(
        model, image, conditioning=conditioning, threshold=threshold,
        refine_iterations=2, individual_masks=True,
    ).result[0]
    masks = masks.cpu() > 0.5  # [N, H, W]

    entry = {
        "image": image,
        "model_name": model_name,
        "masks": masks,
        "areas": masks.flatten(1).sum(1),
        "extras": OrderedDict(),  # (px, py) -> point-prompt mask [H, W], or None
        "labels": {},  # extra keys -> label map data URL
    }
    _det_cache[key] = entry
    while len(_det_cache) > _DET_CACHE_MAX:
        _det_cache.popitem(last=False)
    return entry


def _point_mask(entry, px, py):
    """SAM point prompt for a click that no detected mask covers."""
    key = (px, py)
    if key not in entry["extras"]:
        model, _ = _get_model(entry["model_name"])
        mask = _detect_node().execute(
            model, entry["image"], positive_coords=json.dumps([{"x": px, "y": py}]),
            refine_iterations=2, individual_masks=True,
        ).result[0]
        mask = mask[0].cpu() > 0.5
        entry["extras"][key] = mask if bool(mask.any()) else None
        while len(entry["extras"]) > _EXTRA_CACHE_MAX:
            entry["extras"].popitem(last=False)
            entry["labels"].clear()
    return entry["extras"][key]


def _resolve(image_name, model_name, prompt, max_detections, threshold, points):
    """Masks for the image and which of them the points select.

    Returns the image, all masks [M, H, W] (detected first, then point-prompt
    ones), the number detected, the selected indices and each point's index (-1
    for a point that selects nothing)."""
    with torch.no_grad():
        entry = _detect(image_name, model_name, prompt, max_detections, threshold)
        detected = entry["masks"]
        n, H, W = detected.shape
        extra_keys = []
        point_labels = []
        for x, y in points:
            px = min(W - 1, max(0, int(x * W)))
            py = min(H - 1, max(0, int(y * H)))
            label = -1
            hits = detected[:, py, px].nonzero().flatten()
            if len(hits):
                # Overlapping people: the smaller mask is the one drawn on top.
                label = int(hits[entry["areas"][hits].argmin()])
            else:
                for i, k in enumerate(extra_keys):
                    if entry["extras"][k][py, px]:
                        label = n + i
                        break
                if label < 0 and _point_mask(entry, px, py) is not None:
                    extra_keys.append((px, py))
                    label = n + len(extra_keys) - 1
            point_labels.append(label)

        masks = detected
        if extra_keys:
            masks = torch.cat([detected, torch.stack([entry["extras"][k] for k in extra_keys])])
    return {
        "entry": entry,
        "extra_keys": tuple(extra_keys),
        "image": entry["image"],
        "masks": masks,
        "n_detected": n,
        "selected": sorted({i for i in point_labels if i >= 0}),
        "point_labels": point_labels,
    }


def _labels_url(result):
    """Label map (pixel = mask index + 1, 0 = nothing) as a small greyscale PNG."""
    cache = result["entry"]["labels"]
    url = cache.get(result["extra_keys"])
    if url is None:
        masks = result["masks"]
        _, H, W = masks.shape
        labels = torch.zeros((H, W), dtype=torch.uint8)
        areas = masks.flatten(1).sum(1)
        for i in torch.argsort(areas, descending=True).tolist()[:255]:
            labels[masks[i]] = i + 1  # largest first, so smaller masks end up on top
        scale = min(1.0, LABEL_MAX_SIDE / max(H, W))
        if scale < 1.0:
            size = (max(1, round(H * scale)), max(1, round(W * scale)))
            labels = F.interpolate(labels[None, None].float(), size=size, mode="nearest")[0, 0].to(torch.uint8)
        buf = io.BytesIO()
        Image.fromarray(labels.numpy(), mode="L").save(buf, format="PNG")
        url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
        cache.clear()
        cache[result["extra_keys"]] = url
    return url


def _parse_points(raw):
    """Selection widget value -> [(x, y), ...] with x and y in 0..1."""
    try:
        data = json.loads(raw) if isinstance(raw, str) else raw
    except (TypeError, ValueError):
        return []
    points = []
    for p in data if isinstance(data, list) else []:
        try:
            x, y = float(p[0]), float(p[1])
        except (TypeError, ValueError, IndexError, KeyError):
            continue
        if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
            points.append((x, y))
    return points


@contextlib.contextmanager
def _no_progress():
    """ComfyUI's progress hook reports to the running workflow's current node, and
    fails before any workflow has run. The preview is not part of a workflow."""
    hook = comfy.utils.PROGRESS_BAR_HOOK
    comfy.utils.PROGRESS_BAR_HOOK = None
    try:
        yield
    finally:
        comfy.utils.PROGRESS_BAR_HOOK = hook


def _preview(body):
    with _lock, _no_progress():
        result = _resolve(
            body.get("image"),
            str(body.get("sam3_model", "")),
            body.get("prompt", "person"),
            max(1, min(64, int(body.get("max_detections", 20)))),
            float(body.get("threshold", 0.5)),
            _parse_points(body.get("points")),
        )
        _, H, W = result["masks"].shape
        return {
            "width": W,
            "height": H,
            "count": int(result["masks"].shape[0]),
            "n_detected": result["n_detected"],
            "selected": result["selected"],
            "point_labels": result["point_labels"],
            "labels": _labels_url(result),
        }


@routes.post("/rulz/person_select/detect")
async def detect(request):
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid request body"}, status=400)
    # A workflow is using the GPU; the frontend retries when the queue is empty.
    if PromptServer.instance.prompt_queue.currently_running:
        return web.json_response({"error": "busy"}, status=409)
    try:
        loop = asyncio.get_running_loop()
        return web.json_response(await loop.run_in_executor(None, _preview, body))
    except FileNotFoundError as e:
        return web.json_response({"error": str(e)}, status=404)
    except Exception as e:
        print(f"[RulZ Person Select] preview failed: {e}")
        return web.json_response({"error": str(e)}, status=500)


class RulZPersonSelect:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": (_list_images(),),
                "sam3_model": (_list_models(), {"tooltip": "SAM 3 or SAM 3.1 checkpoint (models/checkpoints)."}),
                "prompt": ("STRING", {"default": "person", "tooltip": "What SAM 3 looks for. Comma-separate several things, e.g. 'person, dog'."}),
                "max_detections": ("INT", {"default": 20, "min": 1, "max": 64, "tooltip": "Most matches to find per prompt."}),
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Detection confidence. Lower it if someone is not found."}),
                "selection": ("STRING", {"default": "[]", "tooltip": "Clicked points, set by clicking the preview."}),
            },
        }

    CATEGORY = "RulZnodes"
    RETURN_TYPES = ("IMAGE", "MASK", "MASK")
    RETURN_NAMES = ("image", "mask", "others_mask")
    OUTPUT_TOOLTIPS = (
        "The loaded image.",
        "Mask of what you clicked.",
        "Mask of everything else that was detected, minus what you clicked.",
    )
    FUNCTION = "select"
    DESCRIPTION = (
        "Load an image, click a person on the preview, and get a mask of just that "
        "person plus a mask of everyone else. Shift-click selects more than one. Uses SAM 3."
    )

    def select(self, image, sam3_model, prompt, max_detections, threshold, selection):
        points = _parse_points(selection)
        if not points:
            raise ValueError("RulZ Person Select: nothing is selected. Click a person on the node's preview first.")
        with _lock:
            result = _resolve(image, sam3_model, prompt, max_detections, threshold, points)
        masks, selected = result["masks"], result["selected"]
        if not selected:
            raise ValueError("RulZ Person Select: the clicked point did not produce a mask. Click the person again.")
        chosen = masks[selected].any(dim=0)
        rest = [i for i in range(result["n_detected"]) if i not in selected]
        others = masks[rest].any(dim=0) & ~chosen if rest else torch.zeros_like(chosen)
        return (result["image"], chosen.float().unsqueeze(0), others.float().unsqueeze(0))

    @classmethod
    def IS_CHANGED(cls, image, **kwargs):
        return LoadImageWithFilenameHEIC.IS_CHANGED(image)

    @classmethod
    def VALIDATE_INPUTS(cls, image):
        if not folder_paths.exists_annotated_filepath(image):
            return f"Invalid image file: {image}"
        return True


NODE_CLASS_MAPPINGS = {"RulZPersonSelect": RulZPersonSelect}
NODE_DISPLAY_NAME_MAPPINGS = {"RulZPersonSelect": "RulZ Person Select (SAM3)"}
