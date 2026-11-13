"""RulZ Person Select (SAM3): load an image, click a person, get their mask.

SAM 3 finds every instance of the text prompt ("person") as its own mask. The
frontend (web/person_select.js) shows those masks over the image and stores the
clicked points; the mask under each point is the selection. A click that lands on
nothing SAM detected falls back to a SAM point prompt at that spot.

The preview route and the node share `_resolve`, so what is shown on the node is
what the node outputs. SAM 3 runs through ComfyUI's own SAM 3 model code, on
the node's own copy of the model, outside ComfyUI's model management:

- On the `default` device the copy waits in RAM and moves to the main GPU only
  while it runs. The preview waits while a workflow is running.
- On another GPU the copy stays there, so the preview can run while a workflow
  is using the main GPU.
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

import numpy as np
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
LABEL_MAX = 0xFFFF  # label map pixels are 16 bit (red = low byte, green = high byte)
REFINE_ITERATIONS = 2  # SAM decoder passes per mask, as SAM3 Detect's default
_DET_CACHE_MAX = 2  # each entry holds full-resolution masks for one image
_EXTRA_CACHE_MAX = 16
_TEXT_CACHE_MAX = 32
_CPU = torch.device("cpu")

_SAM3_KEY = "detector.backbone.vision_backbone.trunk.blocks.0.attn.qkv.weight"  # as ComfyUI detects SAM 3

_lock = threading.Lock()  # one SAM inference at a time (preview route and node)
_models = {}  # checkpoint name -> loaded model handle
_not_sam = {}  # checkpoint path -> mtime, for files found not to be SAM 3
_text_cache = {}  # (model key, prompt text) -> conditioning
_det_cache = OrderedDict()
_tls = threading.local()  # .preview is True while a preview request runs

routes = PromptServer.instance.routes


class _Busy(Exception):
    """A workflow started while a preview for the main GPU was waiting its turn."""


def _workflow_running():
    return bool(PromptServer.instance.prompt_queue.currently_running)


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


def _gpus():
    """{label: torch device} for the GPUs ComfyUI can see.

    Labels are GPU names, not numbers: CUDA's numbering changes with launch
    flags such as --default-device and --cuda-device, and a saved workflow must
    keep pointing at the same card. Identical cards are told apart by their PCI
    bus, which does not change."""
    found = []
    try:
        for dev in comfy.model_management.get_all_torch_devices():
            if dev.type == "cuda":
                props = torch.cuda.get_device_properties(dev.index)
                found.append((props.name, getattr(props, "pci_bus_id", dev.index), _normalize(dev)))
            elif dev.type != "cpu":
                found.append((str(dev), 0, _normalize(dev)))
    except Exception as e:
        print(f"[RulZ Person Select] could not list GPUs: {e}")
    names = [name for name, _, _ in found]
    gpus = {}
    for name, bus, dev in sorted(found, key=lambda f: (f[0], f[1])):
        gpus[f"{name} (PCI bus {bus})" if names.count(name) > 1 else name] = dev
    return gpus


def _device_options():
    # No CPU entry: SAM 3 takes minutes per image there.
    return [DEFAULT_DEVICE] + list(_gpus())


def _normalize(device):
    if device.type != "cpu" and device.index is None:
        return torch.device(device.type, 0)
    return device


def _target_device(value):
    """`device` widget value -> the GPU to keep the model on, or None for
    ComfyUI's main GPU (used only while running).

    Also takes values saved by 1.9.0/1.9.1 ("cuda:1 - NVIDIA GeForce RTX 4060
    Ti"), by the GPU's name. A GPU that is not available now (hidden by a
    launch flag, or another machine) falls back to the main GPU."""
    value = str(value or "").strip()
    if not value or value == DEFAULT_DEVICE:
        return None
    label = re.sub(r"^cuda:\d+\s*-\s*", "", value)
    device = _gpus().get(label)
    if device is None:
        print(f"[RulZ Person Select] GPU '{value}' is not available; using ComfyUI's main GPU.")
        return None
    if device == _main_device():
        return None  # the main GPU picked by name: same as default
    return device


def _main_device():
    return _normalize(comfy.model_management.get_torch_device())


def _empty_cache():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@contextlib.contextmanager
def _start_on_cpu():
    """Make load_checkpoint_guess_config create the diffusion model on the CPU.

    When VRAM looks plentiful (--highvram, or more free VRAM than RAM) it creates
    the model straight on the main GPU and hands it to ComfyUI's model management
    with load_models_gpu, which is what _get_model must never do. Loads are
    serialized by _lock. A checkpoint the workflow thread loads in the same moment
    also starts on the CPU, and ComfyUI moves it to the GPU when it is used."""
    mm = comfy.model_management
    original = getattr(mm, "unet_inital_load_device", None)
    if original is None:
        yield
        return
    mm.unet_inital_load_device = lambda *args, **kwargs: _CPU
    try:
        yield
    finally:
        mm.unet_inital_load_device = original


def _text_encoder_options():
    """The text encoder lives and runs on the CPU (see _encode). ComfyUI's CLIP()
    calls load_models_gpu when the encoder is created on its load device, so name
    the GPU as the load device and create it on the CPU instead. The dtype is the
    one a CPU load device would get (with dynamic VRAM the checkpoint's own
    weights are used as stored, fp16 in the SAM 3 releases)."""
    main = _main_device()
    if main.type == "cpu":
        return {"load_device": _CPU, "offload_device": _CPU}
    return {
        "load_device": main,
        "offload_device": _CPU,
        "initial_device": _CPU,
        "dtype": comfy.model_management.text_encoder_dtype(_CPU),
    }


def _check_sam3(name, path):
    """Refuse a checkpoint that is not SAM 3 before loading it. The list holds every
    checkpoint, and loading a 6-20 GB SDXL or Flux file on each preview refresh just
    to reject it would take a long time and push the loaded SAM 3 out."""
    mtime = os.path.getmtime(path)
    if _not_sam.get(path) == mtime:
        raise ValueError(f"'{name}' is not a SAM 3 checkpoint")
    if not path.lower().endswith((".safetensors", ".sft")):
        return  # no cheap way to look inside; the load itself checks
    try:
        header = comfy.utils.safetensors_header(path)
        keys = json.loads(header) if header else None
    except Exception:
        return
    if isinstance(keys, dict) and not any(k.endswith(_SAM3_KEY) for k in keys):
        _not_sam[path] = mtime
        raise ValueError(f"'{name}' is not a SAM 3 checkpoint")


def _get_model(name, device):
    """The node's own copy of SAM 3, set to run on `device` (None = main GPU).

    ComfyUI's model management is never used. Its model list is changed by the
    workflow thread (it unloads everything after some workflows), and loading
    into it from the preview thread at that moment breaks it for every later
    load until ComfyUI restarts."""
    handle = _models.get(name)
    if handle is None:
        path = folder_paths.get_full_path_or_raise("checkpoints", name)
        _check_sam3(name, path)
        # One checkpoint at a time; drop the old one to give its memory back.
        _models.clear()
        _text_cache.clear()
        _empty_cache()
        try:
            with _start_on_cpu():
                out = comfy.sd.load_checkpoint_guess_config(
                    path,
                    output_vae=False,
                    output_clip=True,
                    embedding_directory=folder_paths.get_folder_paths("embeddings"),
                    # load_device picks the GPU dtype; the weights stay on the CPU until used.
                    model_options={"load_device": _main_device(), "offload_device": _CPU},
                    # The text encoder only runs once per prompt; keep it off every GPU.
                    te_model_options=_text_encoder_options(),
                )
        except RuntimeError as e:
            if "Could not detect model type" not in str(e):
                raise
            # A .ckpt/.pt file the header check could not look inside.
            _not_sam[path] = os.path.getmtime(path)
            raise ValueError(f"'{name}' is not a SAM 3 checkpoint") from e
        model, clip = out[0], out[1]
        sam = model.model.diffusion_model
        if clip is None or not hasattr(sam, "forward_segment"):
            _not_sam[path] = os.path.getmtime(path)
            raise ValueError(f"'{name}' is not a SAM 3 checkpoint")
        sam.to(_CPU)
        handle = {"key": name, "clip": clip, "sam": sam, "dtype": model.model.get_dtype(), "at": _CPU}
        _models[name] = handle
    handle["target"] = device
    return handle


def _prepare(handle):
    """Device and dtype to run on, after moving the weights there."""
    device = handle["target"] or _main_device()
    if handle["at"] != device:
        if handle["target"] is None:
            if getattr(_tls, "preview", False):
                # The route checked the queue before this preview waited for the
                # lock and loaded the checkpoint; check again before using the GPU.
                if _workflow_running():
                    raise _Busy()
            else:
                # On the workflow thread it is safe to ask ComfyUI to make room.
                comfy.model_management.free_memory(3 * 1024**3, device)
        try:
            handle["sam"].to(device)
        except BaseException:
            # Out of memory or any other failure part way: the layers moved
            # before it stay on the GPU unless all of it is moved back.
            handle["at"] = None
            _park(handle)
            raise
        handle["at"] = device
    return device, handle["dtype"]


def _park(handle):
    """Move the copy off the main GPU so it never holds VRAM a workflow needs."""
    if handle["at"] != _CPU:
        handle["sam"].to(_CPU)
        handle["at"] = _CPU
        _empty_cache()


def _encode(handle, text):
    key = (handle["key"], text)
    cond = _text_cache.get(key)
    if cond is None:
        # Run the CPU text encoder directly so model management never sees it.
        clip = handle["clip"]
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
    """SAM3 Detect's text-prompt path for one image -> bool masks [N, H, W] on the CPU.

    SAM3 Detect passes orig_size, which makes the detector enlarge the masks of
    all 200 candidates to the image size before any is filtered out: several GB
    of VRAM for a phone photo. Here only the kept masks are enlarged, the same
    way (bilinear works on each mask separately), so the result is unchanged."""
    extract_text_prompts, refine_mask = _sam3_helpers()
    device, dtype = _prepare(handle)
    sam = handle["sam"]
    _, H, W, _ = image.shape
    frame = _sam_frame(image, device, dtype)
    masks = []
    for embeddings, text_mask, max_det in extract_text_prompts(conditioning, device, dtype):
        out = sam(frame, text_embeddings=embeddings, text_mask=text_mask, boxes=None, threshold=threshold, orig_size=None)
        probs = out["scores"][0].sigmoid()
        keep = probs > threshold
        boxes = out["boxes"][0][keep]  # 0..1, scaled to pixels as the detector does
        boxes = boxes * torch.tensor([W, H, W, H], device=boxes.device, dtype=boxes.dtype)
        boxes, probs, coarse = boxes.cpu(), probs[keep].cpu(), out["masks"][0][keep]
        del out
        order = probs.argsort(descending=True)[:max_det]
        for i in order.tolist():
            full = F.interpolate(coarse[i][None, None], size=(H, W), mode="bilinear", align_corners=False)[0, 0]
            mask = refine_mask(sam, image[0], full, boxes[i], H, W, device, dtype, REFINE_ITERATIONS)
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
    extras = entry["extras"]
    key = (px, py)
    if key in extras:
        extras.move_to_end(key)
        return extras[key]
    mask = _point_segment(get_handle(), entry["image"], px, py)
    mask = mask if bool(mask.any()) else None
    extras[key] = mask
    while len(extras) > _EXTRA_CACHE_MAX:
        extras.popitem(last=False)
        entry["labels"].clear()
    return mask


def _resolve(image_name, model_name, device, prompt, max_detections, threshold, points):
    """Masks for the image and which of them the points select.

    Returns the image, all masks [M, H, W] (detected first, then point-prompt
    ones), the number detected, the selected indices and each point's index (-1
    for a point that selects nothing). The model is only loaded when something
    is not cached yet."""
    run_on = device or _main_device()
    # The current CUDA device is per thread, so switching it here does not
    # affect a running workflow.
    on_device = torch.cuda.device(run_on) if run_on.type == "cuda" else contextlib.nullcontext()
    try:
        # Workflows run under torch.inference_mode, and weights or masks made
        # there cannot be moved or changed outside it later, so the preview
        # runs in it too.
        with torch.inference_mode(), on_device:
            return _resolve_on(image_name, model_name, device, prompt, max_detections, threshold, points)
    except torch.cuda.OutOfMemoryError:
        raise RuntimeError(
            f"Not enough free VRAM on {run_on} for SAM 3. Set device to another GPU, or wait until the workflow "
            "using it has finished."
        )
    finally:
        handle = _models.get(model_name)
        if device is None and handle is not None:
            with torch.inference_mode():
                _park(handle)  # the main GPU belongs to the workflows


def _resolve_on(image_name, model_name, device, prompt, max_detections, threshold, points):
    get_handle = lambda: _get_model(model_name, device)  # noqa: E731
    with torch.no_grad():
        entry = _detect(image_name, get_handle, model_name, prompt, max_detections, threshold)
        detected = entry["masks"]
        n, H, W = detected.shape
        # The point-prompt masks this selection uses are held here, not looked up
        # in entry["extras"] again: that cache keeps only the newest 16.
        extra_keys, extra_masks = [], []
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
                for i, extra in enumerate(extra_masks):
                    if extra[py, px]:
                        label = n + i
                        break
                if label < 0:
                    mask = _point_mask(entry, get_handle, px, py)
                    if mask is not None:
                        extra_keys.append((px, py))
                        extra_masks.append(mask)
                        label = n + len(extra_masks) - 1
            point_labels.append(label)

        masks = detected
        if extra_masks:
            masks = torch.cat([detected, torch.stack(extra_masks)])
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
    """Label map (pixel = mask index + 1, 0 = nothing) as a small PNG. Labels are
    16 bit, red the low byte and green the high byte, since several comma-separated
    prompts can find more than 255 things."""
    cache = result["entry"]["labels"]
    url = cache.get(result["extra_keys"])
    if url is None:
        masks = result["masks"][:LABEL_MAX]
        _, H, W = masks.shape
        labels = torch.zeros((H, W), dtype=torch.int32)
        areas = masks.flatten(1).sum(1)
        for i in torch.argsort(areas, descending=True).tolist():
            labels[masks[i]] = i + 1  # largest first, so smaller masks end up on top
        scale = min(1.0, LABEL_MAX_SIDE / max(H, W))
        if scale < 1.0:
            size = (max(1, round(H * scale)), max(1, round(W * scale)))
            labels = F.interpolate(labels[None, None].float(), size=size, mode="nearest")[0, 0].to(torch.int32)
        labels = labels.numpy()
        rgb = np.zeros((*labels.shape, 3), dtype=np.uint8)
        rgb[..., 0] = labels & 0xFF
        rgb[..., 1] = labels >> 8
        buf = io.BytesIO()
        Image.fromarray(rgb).save(buf, format="PNG")
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
    with _lock, torch.inference_mode():
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
    if device is None and _workflow_running():
        return web.json_response({"error": "busy"}, status=409)
    try:
        loop = asyncio.get_running_loop()
        return web.json_response(await loop.run_in_executor(None, _preview, body, device))
    except _Busy:
        return web.json_response({"error": "busy"}, status=409)
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
    def VALIDATE_INPUTS(cls, image, device=DEFAULT_DEVICE):
        # Taking `device` here skips ComfyUI's "Value not in list" check for it:
        # a saved GPU that is now numbered differently, or missing, is resolved
        # by name or falls back to the main GPU in _target_device.
        if not folder_paths.exists_annotated_filepath(image):
            return f"Invalid image file: {image}"
        return True


NODE_CLASS_MAPPINGS = {"RulZPersonSelect": RulZPersonSelect}
NODE_DISPLAY_NAME_MAPPINGS = {"RulZPersonSelect": "RulZ Person Select (SAM3)"}
