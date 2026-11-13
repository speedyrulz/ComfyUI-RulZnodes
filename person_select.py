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

What SAM 3 finds is kept per image (_CachedImage): every candidate of each
prompt before the threshold is applied, and each mask once it is refined. A
threshold or max_detections change only refines masks not needed before, and a
preview stops early once the same node has sent a newer one.
"""
import asyncio
import base64
import contextlib
import gc
import hashlib
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
_CACHE_MAX_BYTES = 1024**3  # all cached images together; the newest one is always kept
_DETS_PER_IMAGE = 8  # prompts whose candidates are kept per image
_EXTRA_CACHE_MAX = 16  # point-prompt masks kept per image
_TEXT_CACHE_MAX = 32
_TEXT_KEYS_MAX = 256
_CLIENTS_MAX = 256  # preview nodes whose newest request number is remembered
_CPU = torch.device("cpu")

_SAM3_KEY = "detector.backbone.vision_backbone.trunk.blocks.0.attn.qkv.weight"  # as ComfyUI detects SAM 3

_lock = threading.Lock()  # one SAM inference at a time (preview route and node)
_models = {}  # checkpoint name -> loaded model handle
_not_sam = {}  # checkpoint path -> mtime, for files found not to be SAM 3
_text_cache = {}  # (model key, prompt text) -> conditioning
_text_keys = OrderedDict()  # (model name, prompt text) -> [(prompt key, max detections)]
_images = OrderedDict()  # (image path, mtime) -> _CachedImage
_latest = OrderedDict()  # preview client id -> newest request number
_latest_lock = threading.Lock()
# .preview is True while a preview request runs; .request is its
# (client id, request number, runs on the main GPU), None for the node itself.
_tls = threading.local()

routes = PromptServer.instance.routes


class _Busy(Exception):
    """A workflow started while a preview for the main GPU was waiting its turn or running."""


class _Stale(Exception):
    """The same node has sent a newer preview request, so this one's answer would be ignored."""


def _workflow_running():
    return bool(PromptServer.instance.prompt_queue.currently_running)


def _request_id(body):
    """(client id, request number) of a preview request, or (None, 0) when the
    frontend sent none (a page loaded before 1.9.5)."""
    client = body.get("client")
    try:
        number = int(body.get("seq"))
    except (TypeError, ValueError):
        return None, 0
    if not isinstance(client, str) or not 0 < len(client) <= 64:
        return None, 0
    return client, number


def _register_request(client, number):
    """Note a preview request. False when the same node already sent a newer one."""
    if client is None:
        return True
    with _latest_lock:
        newest = _latest.get(client)
        if newest is not None and newest > number:
            return False
        _latest[client] = number
        _latest.move_to_end(client)
        while len(_latest) > _CLIENTS_MAX:
            _latest.popitem(last=False)
    return True


def _checkpoint():
    """Called before each SAM 3 pass of a preview. Stops it when the same node has
    sent a newer request (dragging the threshold, quick clicks), or when it uses
    the main GPU and a workflow has started. What it found so far stays cached for
    the next preview or the node's run. Does nothing for the node itself."""
    request = getattr(_tls, "request", None)
    if request is None:
        return
    client, number, main_gpu = request
    if client is not None:
        with _latest_lock:
            if _latest.get(client, number) > number:
                raise _Stale()
    if main_gpu and _workflow_running():
        raise _Busy()


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


def _tensor_key(*tensors):
    """A hash of the tensors' dtype, shape and contents."""
    h = hashlib.blake2b(digest_size=16)
    for t in tensors:
        t = t.detach().to(_CPU).contiguous()
        h.update(f"{t.dtype}{tuple(t.shape)}".encode())
        h.update(t.flatten().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def _prompt_keys(get_handle, model_name, text):
    """[(key, max detections)] for each comma-separated part of the prompt, in order.

    The key identifies the text embedding SAM 3 gets for that part. ":N" only sets
    the maximum, so a part's candidates are found once for every max_detections.
    Remembered per text, so a preview answered from the cache never loads the model."""
    cache_key = (model_name, text)
    keys = _text_keys.get(cache_key)
    if keys is None:
        extract_text_prompts, _ = _sam3_helpers()
        cond = _encode(get_handle(), text)
        keys = [
            (_tensor_key(embeddings, mask), max_det)
            for embeddings, mask, max_det in extract_text_prompts(cond, _CPU, torch.float32)
        ]
        _text_keys[cache_key] = keys
        while len(_text_keys) > _TEXT_KEYS_MAX:
            _text_keys.popitem(last=False)
    _text_keys.move_to_end(cache_key)
    return keys


class _Crop:
    """A bool mask [H, W] kept as its bounding box, 8 pixels to a byte.

    Masks are cached between previews, and at full size each takes a byte per
    pixel (48 MB for a 48 MP photo) though most of it is empty."""

    __slots__ = ("y0", "x0", "h", "w", "bits", "area")

    def __init__(self, mask):
        self.area = int(mask.sum())
        rows = mask.any(1).nonzero().flatten()
        cols = mask.any(0).nonzero().flatten()
        if not len(rows):
            self.y0 = self.x0 = self.h = self.w = 0
            self.bits = np.zeros((0, 0), dtype=np.uint8)
            return
        self.y0, self.x0 = int(rows[0]), int(cols[0])
        self.h, self.w = int(rows[-1]) + 1 - self.y0, int(cols[-1]) + 1 - self.x0
        box = mask[self.y0:self.y0 + self.h, self.x0:self.x0 + self.w]
        self.bits = np.packbits(box.numpy(), axis=1)

    @property
    def nbytes(self):
        return self.bits.nbytes + 64

    def at(self, y, x):
        y, x = y - self.y0, x - self.x0
        if not (0 <= y < self.h and 0 <= x < self.w):
            return False
        return bool((self.bits[y, x >> 3] >> (7 - (x & 7))) & 1)

    def _unpack(self, rows=slice(None)):
        return np.unpackbits(self.bits[rows], axis=1, count=self.w).view(np.bool_)

    def paste(self, out):
        """out |= this mask, for a bool tensor [H, W]."""
        if self.h:
            out[self.y0:self.y0 + self.h, self.x0:self.x0 + self.w] |= torch.from_numpy(self._unpack())

    def sample(self, rows, cols):
        """The mask at the given rows and columns (int arrays) -> bool array."""
        out = np.zeros((len(rows), len(cols)), dtype=np.bool_)
        if self.h:
            r = (rows >= self.y0) & (rows < self.y0 + self.h)
            c = (cols >= self.x0) & (cols < self.x0 + self.w)
            if r.any() and c.any():
                out[np.ix_(r, c)] = self._unpack(rows[r] - self.y0)[:, cols[c] - self.x0]
        return out


class _Candidates:
    """SAM 3's detector output for one prompt part: all of its candidates (200)
    with their scores, boxes (in pixels) and coarse masks, before the threshold,
    plus each candidate's refined mask once made. The detector does not use the
    threshold, so these serve every threshold and max_detections."""

    def __init__(self, out, height, width):
        self.probs = out["scores"][0].sigmoid().cpu()
        boxes = out["boxes"][0]  # 0..1, scaled to pixels as the detector does
        scale = torch.tensor([width, height, width, height], device=boxes.device, dtype=boxes.dtype)
        self.boxes = (boxes * scale).cpu()
        self.coarse = out["masks"][0].cpu()
        self.refined = {}  # candidate index -> _Crop

    @property
    def nbytes(self):
        return self.probs.nbytes + self.boxes.nbytes + self.coarse.nbytes + sum(c.nbytes for c in self.refined.values())

    def kept(self, threshold, max_det):
        """The candidates SAM3 Detect keeps: over the threshold, most confident first."""
        keep = self.probs > threshold
        order = self.probs[keep].argsort(descending=True)[:max_det]
        return keep.nonzero().flatten()[order].tolist()


_255 = np.float32(255.0)


def _from_uint8(pixels):
    # The loader's own conversion (to_float_rgb: uint8 -> float32, / 255.0 in
    # float32), in one pass, so the values match to the bit.
    return torch.from_numpy(np.divide(pixels.numpy(), _255, dtype=np.float32))


def _compact(image):
    """float32 [1, H, W, 3] -> uint8 when that converts back to exactly the same
    values (any 8-bit image), otherwise the float32 image (16-bit greyscale)."""
    pixels = image[0].numpy()
    out = np.empty(pixels.shape, dtype=np.uint8)
    for y in range(0, pixels.shape[0], 256):  # in strips, to keep the extra memory small
        strip = pixels[y:y + 256]
        work = np.multiply(strip, _255)
        np.rint(work, out=work)
        np.clip(work, 0, 255, out=work)
        quantized = out[y:y + 256]
        np.copyto(quantized, work, casting="unsafe")
        np.divide(quantized, _255, out=work, dtype=np.float32)
        if not np.array_equal(work, strip):
            return image
    return torch.from_numpy(out)[None]


class _CachedImage:
    """One image and what SAM 3 found in it, kept between previews and runs."""

    def __init__(self, image):
        self.height, self.width = int(image.shape[1]), int(image.shape[2])
        self.pixels = _compact(image)  # uint8 takes a quarter of the float32 image
        self.frame = None  # SAM 3's 1008x1008 input, float32 on the CPU
        self.trunk = {}  # model name -> (device, image encoder output for the frame, on the CPU)
        self.dets = OrderedDict()  # (model name, prompt key) -> _Candidates
        self.extras = OrderedDict()  # (model name, x, y) -> point-prompt _Crop, or None
        self.labels = {}  # the masks a label map shows -> its data URL

    def image(self):
        """The image as the loader returns it: float32 [1, H, W, 3]."""
        return _from_uint8(self.pixels) if self.pixels.dtype == torch.uint8 else self.pixels

    def sam_frame(self, get_image):
        if self.frame is None:
            image = get_image()
            self.frame = comfy.utils.common_upscale(image[..., :3].movedim(-1, 1), 1008, 1008, "bilinear", crop="disabled")
        return self.frame

    @property
    def nbytes(self):
        n = self.pixels.nbytes + (self.frame.nbytes if self.frame is not None else 0)
        n += sum(out.nbytes for _, out in self.trunk.values())
        n += sum(c.nbytes for c in self.dets.values())
        n += sum(c.nbytes for c in self.extras.values() if c is not None)
        return n + sum(len(url) for url in self.labels.values())


def _cached_image(path, image_name):
    """(_CachedImage, the float32 image when it was just loaded, else None)."""
    key = (path, os.path.getmtime(path))
    cached = _images.get(key)
    loaded = None
    if cached is None:
        for old in [k for k in _images if k[0] == path]:
            del _images[old]  # the file has changed since
        frames = LoadImageWithFilenameHEIC().load_image(image_name)[0]
        # A view of the first frame would keep every frame of a GIF alive.
        loaded = frames if len(frames) == 1 else frames[:1].clone()
        cached = _CachedImage(loaded)
        _images[key] = cached
    _images.move_to_end(key)
    return cached, loaded


def _trim_cache():
    """Drop the least recently used images while all of them together hold more
    than _CACHE_MAX_BYTES. The newest one is always kept."""
    while len(_images) > 1 and sum(c.nbytes for c in _images.values()) > _CACHE_MAX_BYTES:
        _images.popitem(last=False)


class _TrunkMemo:
    """Runs SAM 3's image encoder (the ViT trunk, most of its work) once per input.

    ComfyUI's SAM 3 runs it again for each comma-separated prompt, for both
    passes of each mask refinement (on the same crop), and for both passes of a
    point prompt. While active, a call with the input of an earlier call returns
    that call's output. The frame's output is pinned for the whole run and can be
    passed in from, or saved to, the cache."""

    def __init__(self, sam):
        try:
            self.trunk = sam.detector.backbone["vision_backbone"].trunk
        except (AttributeError, KeyError, TypeError):
            self.trunk = None  # another ComfyUI SAM 3 layout: run it as it is
        self.frame = self.frame_out = self.on_frame = None
        self.last = self.last_out = None

    def pin(self, frame, out=None, on_computed=None):
        self.frame, self.frame_out, self.on_frame = frame, out, on_computed

    def __enter__(self):
        if self.trunk is not None:
            run = self.trunk.forward

            def forward(x, *args, **kwargs):
                if args or kwargs:
                    return run(x, *args, **kwargs)
                if x is self.frame and self.frame_out is not None:
                    return self.frame_out
                if x is self.last:
                    return self.last_out
                out = run(x)
                if x is self.frame:
                    self.frame_out = out
                    if self.on_frame is not None:
                        self.on_frame(out)
                else:
                    self.last, self.last_out = x, out  # holding x keeps its id from being reused
                return out

            self.trunk.forward = forward
        return self

    def __exit__(self, *exc):
        if self.trunk is not None:
            self.trunk.__dict__.pop("forward", None)
        self.frame = self.frame_out = self.on_frame = self.last = self.last_out = None
        return False


class _Run:
    """What one preview or node run shares, each part made when first needed: a
    run answered from the cache never loads or moves the model or converts the image."""

    def __init__(self, cached, model_name, device, stack, image=None):
        self.cached = cached
        self.model_name = model_name
        self.device = device  # None = ComfyUI's main GPU
        self._stack = stack
        self._sam = None
        self._image = image  # the float32 image, when the run has it already
        self._frame = None

    def handle(self):
        return _get_model(self.model_name, self.device)

    def sam(self):
        """(handle, device, dtype, trunk memo), with the weights on the device."""
        if self._sam is None:
            handle = self.handle()
            device, dtype = _prepare(handle)
            memo = self._stack.enter_context(_TrunkMemo(handle["sam"]))
            self._sam = (handle, device, dtype, memo)
        return self._sam

    def image(self):
        if self._image is None:
            self._image = self.cached.image()
        return self._image

    def frame(self):
        """The 1008x1008 frame on the device. Its image encoder output comes from the
        cache when it was made on this device, and is saved there otherwise."""
        if self._frame is None:
            _, device, dtype, memo = self.sam()
            self._frame = self.cached.sam_frame(self.image).to(device=device, dtype=dtype)
            trunk, name = self.cached.trunk, self.model_name
            saved = trunk.get(name)
            if saved is not None and saved[0] == device:
                memo.pin(self._frame, saved[1].to(device))
            else:
                def save(out):
                    if torch.is_tensor(out):
                        trunk[name] = (device, out.cpu())

                memo.pin(self._frame, None, save)
        return self._frame


def _candidates(run, text, part, key):
    """The candidates for one prompt part, from the cache or from SAM 3's detector."""
    dets = run.cached.dets
    det_key = (run.model_name, key)
    found = dets.get(det_key)
    if found is None:
        _checkpoint()
        extract_text_prompts, _ = _sam3_helpers()
        handle, device, dtype, _ = run.sam()
        frame = run.frame()
        embeddings, text_mask, _ = extract_text_prompts(_encode(handle, text), device, dtype)[part]
        # SAM3 Detect passes orig_size, which makes the detector enlarge the masks
        # of all 200 candidates to the image size: several GB of VRAM for a phone
        # photo. Here only the kept masks are enlarged, the same way (_refined).
        # The detector ignores `threshold`; it is applied afterwards (kept).
        out = handle["sam"](frame, text_embeddings=embeddings, text_mask=text_mask, boxes=None, threshold=0.0, orig_size=None)
        found = _Candidates(out, run.cached.height, run.cached.width)
        del out
        dets[det_key] = found
        while len(dets) > _DETS_PER_IMAGE:
            dets.popitem(last=False)
    dets.move_to_end(det_key)
    return found


def _refined(run, found, index):
    """Candidate `index` refined as SAM3 Detect refines it -> _Crop. Bilinear
    enlarging works on each mask separately, so enlarging only this one gives
    the same mask as enlarging all of them."""
    crop = found.refined.get(index)
    if crop is None:
        _checkpoint()
        _, refine_mask = _sam3_helpers()
        handle, device, dtype, _ = run.sam()
        H, W = run.cached.height, run.cached.width
        coarse = found.coarse[index].to(device)
        full = F.interpolate(coarse[None, None], size=(H, W), mode="bilinear", align_corners=False)[0, 0]
        mask = refine_mask(handle["sam"], run.image()[0], full, found.boxes[index], H, W, device, dtype, REFINE_ITERATIONS)
        crop = _Crop(mask[0].cpu() > 0.5)
        found.refined[index] = crop
    return crop


def _point_mask(run, px, py):
    """SAM3 Detect's point-prompt path for a click no detected mask covers ->
    _Crop, or None when SAM finds nothing there."""
    extras = run.cached.extras
    key = (run.model_name, px, py)
    if key in extras:
        extras.move_to_end(key)
        return extras[key]
    _checkpoint()
    handle, device, dtype, _ = run.sam()
    frame = run.frame()
    sam = handle["sam"]
    H, W = run.cached.height, run.cached.width
    points = {
        "point_coords": torch.tensor([[[px / W * 1008, py / H * 1008]]], dtype=dtype, device=device),
        "point_labels": torch.tensor([[1]], dtype=torch.int32, device=device),
    }
    logit = sam.forward_segment(frame, point_inputs=points)
    for _ in range(REFINE_ITERATIONS - 1):
        logit = sam.forward_segment(frame, mask_inputs=logit)
    mask = F.interpolate(logit, size=(H, W), mode="bilinear", align_corners=False)[0, 0].cpu() > 0
    crop = _Crop(mask) if bool(mask.any()) else None
    extras[key] = crop
    while len(extras) > _EXTRA_CACHE_MAX:
        extras.popitem(last=False)
    return crop


def _resolve(image_name, model_name, device, prompt, max_detections, threshold, points):
    """Masks for the image and which of them the points select.

    Returns the masks as _Crops (detected first, then point-prompt ones), the
    number detected, the selected indices and each point's index (-1 for a point
    that selects nothing). The model is only loaded when something is not cached yet."""
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
    cached, loaded = _cached_image(_image_path(image_name), image_name)
    text = _prompt_text(prompt, max_detections)
    with contextlib.ExitStack() as stack, torch.no_grad():
        run = _Run(cached, model_name, device, stack, loaded)
        try:
            masks, shown = [], []
            for part, (key, max_det) in enumerate(_prompt_keys(run.handle, model_name, text)):
                found = _candidates(run, text, part, key)
                for index in found.kept(threshold, max_det):
                    masks.append(_refined(run, found, index))
                    shown.append((key, index))
            n = len(masks)
            H, W = cached.height, cached.width
            extra_keys = []
            point_labels = []
            for x, y in points:
                px = min(W - 1, max(0, int(x * W)))
                py = min(H - 1, max(0, int(y * H)))
                label = -1
                hits = [i for i in range(n) if masks[i].at(py, px)]
                if hits:
                    # Overlapping people: the smaller mask is the one drawn on top.
                    label = min(hits, key=lambda i: masks[i].area)
                else:
                    # Point-prompt masks this selection already made come next.
                    label = next((i for i in range(n, len(masks)) if masks[i].at(py, px)), -1)
                    if label < 0:
                        mask = _point_mask(run, px, py)
                        if mask is not None:
                            masks.append(mask)
                            extra_keys.append((px, py))
                            label = len(masks) - 1
                point_labels.append(label)
        finally:
            _trim_cache()
    return {
        "cached": cached,
        "image": run.image,
        "masks": masks,
        "shown": (model_name, tuple(shown), tuple(extra_keys)),
        "n_detected": n,
        "selected": sorted({i for i in point_labels if i >= 0}),
        "point_labels": point_labels,
    }


def _nearest_source(size, out):
    """The source indices F.interpolate(mode="nearest") reads when resizing `size`
    pixels to `out`, taken from resizing a ramp so they match it exactly."""
    ramp = torch.arange(size, dtype=torch.float32).view(1, 1, size, 1)
    return F.interpolate(ramp, size=(out, 1), mode="nearest").flatten().to(torch.int64).numpy()


def _labels_url(result):
    """Label map (pixel = mask index + 1, 0 = nothing) as a small PNG. Labels are
    16 bit, red the low byte and green the high byte, since several comma-separated
    prompts can find more than 255 things. Made from the masks at the label map's
    own size: the pixels nearest-neighbour downscaling would pick, read from each."""
    cached = result["cached"]
    url = cached.labels.get(result["shown"])
    if url is None:
        masks = result["masks"][:LABEL_MAX]
        H, W = cached.height, cached.width
        scale = min(1.0, LABEL_MAX_SIDE / max(H, W))
        if scale < 1.0:
            rows = _nearest_source(H, max(1, round(H * scale)))
            cols = _nearest_source(W, max(1, round(W * scale)))
        else:
            rows, cols = np.arange(H), np.arange(W)
        labels = np.zeros((len(rows), len(cols)), dtype=np.int32)
        areas = torch.tensor([m.area for m in masks], dtype=torch.int64)
        for i in torch.argsort(areas, descending=True).tolist():
            labels[masks[i].sample(rows, cols)] = i + 1  # largest first, so smaller masks end up on top
        rgb = np.zeros((*labels.shape, 3), dtype=np.uint8)
        rgb[..., 0] = labels & 0xFF
        rgb[..., 1] = labels >> 8
        buf = io.BytesIO()
        Image.fromarray(rgb).save(buf, format="PNG")
        url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
        cached.labels.clear()
        cached.labels[result["shown"]] = url
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


def _preview(body, device, request):
    with _lock, torch.inference_mode():
        _tls.preview = True
        _tls.request = request
        try:
            _checkpoint()  # a newer request may have come while this one waited
            result = _resolve(
                body.get("image"),
                str(body.get("sam3_model", "")),
                device,
                body.get("prompt", "person"),
                max(1, min(64, int(body.get("max_detections", 20)))),
                float(body.get("threshold", 0.5)),
                _parse_points(body.get("points")),
            )
            return {
                "width": result["cached"].width,
                "height": result["cached"].height,
                "count": len(result["masks"]),
                "n_detected": result["n_detected"],
                "selected": result["selected"],
                "point_labels": result["point_labels"],
                "labels": _labels_url(result),
            }
        finally:
            _tls.preview = False
            _tls.request = None


@routes.post("/rulz/person_select/detect")
async def detect(request):
    try:
        body = await request.json()
        device = _target_device(body.get("device"))
        client, number = _request_id(body)
    except ValueError as e:
        return web.json_response({"error": str(e)}, status=400)
    except Exception:
        return web.json_response({"error": "Invalid request body"}, status=400)
    # Only the newest request from a node is worth answering; the frontend
    # ignores "stale" replies.
    if not _register_request(client, number):
        return web.json_response({"error": "stale"}, status=409)
    # The main GPU is in use by a workflow; the frontend retries when the queue
    # is empty. A separate device does not have to wait.
    if device is None and _workflow_running():
        return web.json_response({"error": "busy"}, status=409)
    try:
        loop = asyncio.get_running_loop()
        request_info = (client, number, device is None)
        return web.json_response(await loop.run_in_executor(None, _preview, body, device, request_info))
    except _Stale:
        return web.json_response({"error": "stale"}, status=409)
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
        size = (result["cached"].height, result["cached"].width)
        chosen = torch.zeros(size, dtype=torch.bool)
        others = torch.zeros(size, dtype=torch.bool)
        for i in selected:
            masks[i].paste(chosen)
        for i in range(result["n_detected"]):
            if i not in selected:
                masks[i].paste(others)
        others &= ~chosen
        return (result["image"](), chosen.float().unsqueeze(0), others.float().unsqueeze(0))

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
