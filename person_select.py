"""RulZ Person Select (SAM3): load an image, click a person, get their mask.

SAM 3 finds every instance of the text prompt ("person") as its own mask. The
frontend (web/person_select.js) shows those masks over the image and stores the
clicked points; the mask under each point is the selection. A click that lands on
nothing SAM detected falls back to a SAM point prompt at that spot.

The preview route and the node share `_resolve`, so what is shown on the node is
what the node outputs. SAM 3 runs through ComfyUI's own SAM 3 model code:

- On the `default` device the model goes through ComfyUI's model management like
  any other model, and the preview waits while a workflow is running.
- On another GPU the node keeps its own copy of the model there, outside model
  management, so the preview can run while a workflow is using the main GPU.
"""
import asyncio
import base64
import contextlib
import gc
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

import comfy.model_management
import comfy.ops
import comfy.sd
import comfy.utils
import folder_paths
from server import PromptServer

from .load_image_heic import LoadImageWithFilenameHEIC, _list_images

DEFAULT_DEVICE = "default"
LABEL_MAX_SIDE = 1024  # longest side of the label map sent to the frontend
REFINE_ITERATIONS = 2  # SAM decoder passes per mask, as SAM3 Detect's default
_DET_CACHE_MAX = 2  # each entry holds full-resolution masks for one image
_EXTRA_CACHE_MAX = 16
_TEXT_CACHE_MAX = 32
_CPU = torch.device("cpu")

_lock = threading.Lock()  # one SAM inference at a time (preview route and node)
_models = {}  # (checkpoint, device string or None) -> loaded model handle
_text_cache = {}  # (model key, prompt text) -> conditioning
_det_cache = OrderedDict()
_tls = threading.local()  # .preview is True while a preview request runs

routes = PromptServer.instance.routes


def _install_interrupt_guard():
    """ComfyUI checks its global Cancel flag before every layer and clears the flag
    when it raises. A preview running beside a workflow would take that workflow's
    Cancel, so the check is skipped in preview threads only."""
    original = getattr(comfy.ops, "run_every_op", None)
    if original is None or getattr(original, "_rulz_preview_guard", False):
        return

    def run_every_op():
        if not getattr(_tls, "preview", False):
            original()

    run_every_op._rulz_preview_guard = True
    comfy.ops.run_every_op = run_every_op


_install_interrupt_guard()


def _sam3_helpers():
    try:
        from comfy_extras.nodes_sam3 import _extract_text_prompts, _refine_mask
    except ImportError as e:
        raise RuntimeError(
            "RulZ Person Select needs a ComfyUI version with built-in SAM 3 support "
            "(the 'SAM3 Detect' node). Update ComfyUI."
        ) from e
    return _extract_text_prompts, _refine_mask


def _list_models():
    """Checkpoints, with the ones that look like SAM first so the default is right."""
    names = folder_paths.get_filename_list("checkpoints")
    return sorted(names, key=lambda n: ("sam" not in n.lower(), n.lower()))


def _device_options():
    options = [DEFAULT_DEVICE]
    try:
        for dev in comfy.model_management.get_all_torch_devices():
            if dev.type == "cuda":
                options.append(f"cuda:{dev.index} - {torch.cuda.get_device_name(dev.index)}")
            elif dev.type != "cpu":
                options.append(str(dev))
    except Exception as e:
        print(f"[RulZ Person Select] could not list GPUs: {e}")
    # No CPU entry: SAM 3 takes minutes per image there.
    return options


def _normalize(device):
    if device.type != "cpu" and device.index is None:
        return torch.device(device.type, 0)
    return device


def _target_device(value):
    """`device` widget value -> the device for a separate copy of the model, or
    None to use ComfyUI's main device through its model management."""
    if not value or value == DEFAULT_DEVICE:
        return None
    try:
        device = _normalize(torch.device(str(value).split(" ", 1)[0]))
    except (RuntimeError, ValueError):
        raise ValueError(f"Unknown device: {value}")
    if device == _normalize(comfy.model_management.get_torch_device()):
        return None  # the main GPU picked by name: same as default
    if device.type == "cuda" and device.index >= torch.cuda.device_count():
        raise ValueError(f"{device} is not available on this machine")
    return device


def _get_model(name, device):
    key = (name, None if device is None else str(device))
    handle = _models.get(key)
    if handle is not None:
        return handle

    # One checkpoint per kind at a time. A separate copy is not known to
    # ComfyUI, so the old one is dropped here to give its memory back.
    for k in [k for k in _models if (k[1] is None) == (device is None)]:
        del _models[k]
    _text_cache.clear()
    if device is not None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    path = folder_paths.get_full_path_or_raise("checkpoints", name)
    options = {}
    if device is not None:
        options = {
            "model_options": {"load_device": device, "offload_device": device},
            # The text encoder only runs once per prompt; keep it off every GPU.
            "te_model_options": {"load_device": _CPU, "offload_device": _CPU},
        }
    out = comfy.sd.load_checkpoint_guess_config(
        path,
        output_vae=False,
        output_clip=True,
        embedding_directory=folder_paths.get_folder_paths("embeddings"),
        **options,
    )
    model, clip = out[0], out[1]
    sam = model.model.diffusion_model
    if clip is None or not hasattr(sam, "forward_segment"):
        raise ValueError(f"'{name}' is not a SAM 3 checkpoint")
    if device is not None:
        sam.to(device)
    handle = {"key": key, "model": model, "clip": clip, "sam": sam, "device": device, "dtype": model.model.get_dtype()}
    _models[key] = handle
    return handle


def _prepare(handle):
    """Device and dtype to run on, after making sure the weights are there."""
    if handle["device"] is None:
        comfy.model_management.load_model_gpu(handle["model"])
        return comfy.model_management.get_torch_device(), handle["dtype"]
    return handle["device"], handle["dtype"]


def _encode(handle, text):
    key = (handle["key"], text)
    cond = _text_cache.get(key)
    if cond is None:
        clip = handle["clip"]
        if handle["device"] is None:
            cond = clip.encode_from_tokens_scheduled(clip.tokenize(text))
        else:
            # Run the CPU text encoder directly so model management never sees it.
            encoder = clip.cond_stage_model
            encoder.reset_clip_options()
            encoder.set_clip_options({"execution_device": _CPU})
            out = encoder.encode_token_weights(clip.tokenize(text))
            extra = dict(out[2]) if len(out) > 2 else {}
            extra["pooled_output"] = out[1]
            cond = [[out[0], extra]]
        if len(_text_cache) >= _TEXT_CACHE_MAX:
            _text_cache.clear()
        _text_cache[key] = cond
    return cond


def _sam_frame(image, device, dtype):
    frame = comfy.utils.common_upscale(image[..., :3].movedim(-1, 1), 1008, 1008, "bilinear", crop="disabled")
    return frame.to(device=device, dtype=dtype)


def _detect_masks(handle, image, conditioning, threshold):
    """SAM3 Detect's text-prompt path for one image -> bool masks [N, H, W] on the CPU."""
    extract_text_prompts, refine_mask = _sam3_helpers()
    device, dtype = _prepare(handle)
    sam = handle["sam"]
    _, H, W, _ = image.shape
    frame = _sam_frame(image, device, dtype)
    masks = []
    for embeddings, text_mask, max_det in extract_text_prompts(conditioning, device, dtype):
        out = sam(frame, text_embeddings=embeddings, text_mask=text_mask, boxes=None, threshold=threshold, orig_size=(H, W))
        probs = out["scores"][0].sigmoid()
        keep = probs > threshold
        boxes, probs, coarse = out["boxes"][0][keep].cpu(), probs[keep].cpu(), out["masks"][0][keep]
        order = probs.argsort(descending=True)[:max_det]
        for i in order.tolist():
            mask = refine_mask(sam, image[0], coarse[i], boxes[i], H, W, device, dtype, REFINE_ITERATIONS)
            masks.append(mask[0].cpu() > 0.5)
    if not masks:
        return torch.zeros((0, H, W), dtype=torch.bool)
    return torch.stack(masks)


def _point_segment(handle, image, px, py):
    """SAM3 Detect's point-prompt path for one positive point -> bool mask [H, W]."""
    device, dtype = _prepare(handle)
    sam = handle["sam"]
    _, H, W, _ = image.shape
    frame = _sam_frame(image, device, dtype)
    points = {
        "point_coords": torch.tensor([[[px / W * 1008, py / H * 1008]]], dtype=dtype, device=device),
        "point_labels": torch.tensor([[1]], dtype=torch.int32, device=device),
    }
    logit = sam.forward_segment(frame, point_inputs=points)
    for _ in range(REFINE_ITERATIONS - 1):
        logit = sam.forward_segment(frame, mask_inputs=logit)
    return F.interpolate(logit, size=(H, W), mode="bilinear", align_corners=False)[0, 0].cpu() > 0


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


def _detect(image_name, get_handle, model_name, prompt, max_detections, threshold):
    path = _image_path(image_name)
    text = _prompt_text(prompt, max_detections)
    # The device is not part of the key: every device finds the same people.
    key = (path, os.path.getmtime(path), model_name, text, round(float(threshold), 3))
    entry = _det_cache.get(key)
    if entry is not None:
        _det_cache.move_to_end(key)
        return entry

    image = LoadImageWithFilenameHEIC().load_image(image_name)[0][:1]
    handle = get_handle()
    masks = _detect_masks(handle, image, _encode(handle, text), threshold)
    entry = {
        "image": image,
        "masks": masks,
        "areas": masks.flatten(1).sum(1),
        "extras": OrderedDict(),  # (px, py) -> point-prompt mask [H, W], or None
        "labels": {},  # extra keys -> label map data URL
    }
    _det_cache[key] = entry
    while len(_det_cache) > _DET_CACHE_MAX:
        _det_cache.popitem(last=False)
    return entry


def _point_mask(entry, get_handle, px, py):
    """SAM point prompt for a click that no detected mask covers."""
    key = (px, py)
    if key not in entry["extras"]:
        mask = _point_segment(get_handle(), entry["image"], px, py)
        entry["extras"][key] = mask if bool(mask.any()) else None
        while len(entry["extras"]) > _EXTRA_CACHE_MAX:
            entry["extras"].popitem(last=False)
            entry["labels"].clear()
    return entry["extras"][key]


def _resolve(image_name, model_name, device, prompt, max_detections, threshold, points):
    """Masks for the image and which of them the points select.

    Returns the image, all masks [M, H, W] (detected first, then point-prompt
    ones), the number detected, the selected indices and each point's index (-1
    for a point that selects nothing). The model is only loaded when something
    is not cached yet."""
    get_handle = lambda: _get_model(model_name, device)  # noqa: E731
    # ComfyUI's memory code works on the thread's current CUDA device, which is
    # per thread, so switching it here does not affect a running workflow.
    on_device = torch.cuda.device(device) if device is not None and device.type == "cuda" else contextlib.nullcontext()
    with torch.no_grad(), on_device:
        entry = _detect(image_name, get_handle, model_name, prompt, max_detections, threshold)
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
                if label < 0 and _point_mask(entry, get_handle, px, py) is not None:
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


def _preview(body, device):
    with _lock:
        _tls.preview = True
        try:
            result = _resolve(
                body.get("image"),
                str(body.get("sam3_model", "")),
                device,
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
        finally:
            _tls.preview = False


@routes.post("/rulz/person_select/detect")
async def detect(request):
    try:
        body = await request.json()
        device = _target_device(body.get("device"))
    except ValueError as e:
        return web.json_response({"error": str(e)}, status=400)
    except Exception:
        return web.json_response({"error": "Invalid request body"}, status=400)
    # The main GPU is in use by a workflow; the frontend retries when the queue
    # is empty. A separate device does not have to wait.
    if device is None and PromptServer.instance.prompt_queue.currently_running:
        return web.json_response({"error": "busy"}, status=409)
    try:
        loop = asyncio.get_running_loop()
        return web.json_response(await loop.run_in_executor(None, _preview, body, device))
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
                # Last, so workflows saved before it existed keep their values in place.
                "device": (_device_options(), {
                    "default": DEFAULT_DEVICE,
                    "tooltip": "Where SAM 3 runs. 'default' is ComfyUI's main GPU and the preview waits while a "
                    "workflow runs. Another GPU keeps its own copy of SAM 3 there, so you can click "
                    "and queue images while the main GPU is busy.",
                }),
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

    def select(self, image, sam3_model, prompt, max_detections, threshold, selection, device=DEFAULT_DEVICE):
        points = _parse_points(selection)
        if not points:
            raise ValueError("RulZ Person Select: nothing is selected. Click a person on the node's preview first.")
        target = _target_device(device)
        with _lock:
            result = _resolve(image, sam3_model, target, prompt, max_detections, threshold, points)
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
