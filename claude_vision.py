"""RulZ Claude Vision: ask Claude about an image (or several) through the official `anthropic` Python SDK.

The node sends an optional image or image batch, a system prompt and a prompt to the Claude API and returns
Claude's reply and a summary of its thinking. You pick the model and how hard it thinks.

- The API key comes from the `api_key` input, else the `ANTHROPIC_API_KEY` environment variable, else the first
  line of `anthropic_api_key.txt` in this folder, else whatever the SDK finds itself (`ANTHROPIC_AUTH_TOKEN` or
  an `ant auth login` profile). The key is never logged.
- The model list is fetched from the API once per ComfyUI run (when a key is available at startup) and falls back
  to a fixed list. A saved workflow that names a model missing from the list still runs.
- Thinking is sent the way each model family accepts it. Some models cannot turn thinking off, some reject
  certain levels, and older ones take a token budget instead of an effort. `_thinking_plan` holds those rules;
  a request that broke them would come back as HTTP 400. When the node has to bend a setting, the status line says so.
- The request is streamed so that the Cancel button of ComfyUI works while Claude is still writing.
- Several prompts (a list wired into `prompt`) or `one answer per image` give one request, and one answer, each.
"""
import base64
import io
import logging
import os
import re
import threading

import numpy as np
import torch
from PIL import Image

import comfy.model_management

logger = logging.getLogger("RulZnodes.claude_vision")

MAX_SEED = 0xFFFFFFFFFFFFFFFF
THINKING_LEVELS = ["off", "low", "medium", "high", "xhigh", "max"]
IMAGE_MODES = ["all images in one request", "one answer per image"]
DEFAULT_MODEL = "claude-opus-5-5"
FALLBACK_MODELS = ["claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5", "claude-fable-5-2", "claude-opus-5",
                   "claude-sonnet-5", "claude-opus-4-8", "claude-haiku-4-5"]
KEY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "anthropic_api_key.txt")

# The API's limits for one image, and for the images of one request.
MAX_EDGE = 8000
MAX_IMAGE_B64 = 5_000_000
MAX_IMAGES = 100
# Models that can re-run a refused request on a fallback model, and the beta that switches that on.
FALLBACK_BETA = "server-side-fallback-2026-07-01"
FALLBACK_MODELS_WITH_BETA = ("claude-opus-5-5", "claude-fable-5-2", "claude-fable-5-1", "claude-opus-5", "claude-sonnet-5-5")
# Thinking token budgets for models that take a budget instead of an effort, by level.
LEGACY_BUDGETS = {"low": 2048, "medium": 8000, "high": 16000, "xhigh": 32000, "max": 32000}
HAIKU_45_MAX_OUTPUT = 64000

_AUTH_HELP = ("Claude did not accept the credentials (or none were found). Give a valid key in the api_key input, "
              "set the ANTHROPIC_API_KEY environment variable, put the key on the first line of anthropic_api_key.txt "
              "in the RulZnodes folder, or set ANTHROPIC_AUTH_TOKEN / sign in with `ant auth login`.")

_MODEL_CACHE = None
_MODEL_LOCK = threading.Lock()


def _at(values, index, default=None):
    """Item `index` of a list input, repeating the last item as ComfyUI does for shorter lists."""
    if not values:
        return default
    return values[min(index, len(values) - 1)]


def _import_anthropic():
    try:
        import anthropic
    except ImportError as e:
        raise RuntimeError("RulZ Claude Vision needs the anthropic package. Run: pip install anthropic "
                           "(with ComfyUI's own Python), then restart ComfyUI.") from e
    return anthropic


def _key_from_file():
    """First non-empty line of anthropic_api_key.txt next to this module, or ''."""
    try:
        with open(KEY_FILE, "r", encoding="utf-8-sig") as f:
            for line in f:
                if line.strip():
                    return line.strip()
    except OSError:
        pass
    return ""


def _resolve_key(widget_key=""):
    """The API key to use: the widget, then the environment, then the key file. '' means let the SDK look."""
    return ((widget_key or "").strip() or os.environ.get("ANTHROPIC_API_KEY", "").strip() or _key_from_file())


def _make_client(widget_key="", **kwargs):
    anthropic = _import_anthropic()
    key = _resolve_key(widget_key)
    if key:
        return anthropic.Anthropic(api_key=key, **kwargs)
    return anthropic.Anthropic(**kwargs)


def get_models():
    """The model ids for the dropdown: the live list when a key allows, else the fixed one. Fetched once."""
    global _MODEL_CACHE
    with _MODEL_LOCK:
        if _MODEL_CACHE is not None:
            return list(_MODEL_CACHE)
        models = []
        key = _resolve_key("")
        if key:
            try:
                client = _make_client("", timeout=5.0, max_retries=0)
                for count, item in enumerate(client.models.list(limit=100)):
                    if count >= 300:
                        break
                    model_id = getattr(item, "id", "")
                    if isinstance(model_id, str) and model_id.startswith("claude-") and model_id not in models:
                        models.append(model_id)
            except Exception as e:  # no network, a bad key, a missing package: use the fixed list
                logger.info("RulZ Claude Vision: model list not fetched (%s); using the built-in list", type(e).__name__)
                models = []
        if not models:
            models = list(FALLBACK_MODELS)
        elif DEFAULT_MODEL in models:
            models.remove(DEFAULT_MODEL)
        if models[0] != DEFAULT_MODEL:
            models.insert(0, DEFAULT_MODEL)
        _MODEL_CACHE = models
        return list(models)


def _is(model, base):
    """True for `base` itself or a dated snapshot of it (`base-20261101`)."""
    return model == base or re.fullmatch(re.escape(base) + r"-\d{8}", model) is not None


def _is_legacy(model):
    """True for models older than 4.6, which take a thinking budget and no effort."""
    m = re.match(r"claude-(?:opus|sonnet|haiku)-(\d+)(?:-(\d{1,2}))?(?:-|$)", model)
    if m:
        return (int(m.group(1)), int(m.group(2) or 0)) < (4, 6)
    return re.match(r"claude-\d+(?:-\d+)?-(?:opus|sonnet|haiku)", model) is not None or model.startswith("claude-instant")


def _thinking_plan(model, level, max_tokens):
    """The thinking, effort and max_tokens to send for `model` at `level`, plus notes for the status line.

    Returns (thinking, effort, max_tokens, notes); `thinking` or `effort` is None when it is not sent.
    """
    notes = []
    adaptive = {"type": "adaptive", "display": "summarized"}

    if _is_legacy(model):
        # No effort and no adaptive thinking on these: a token budget, which must stay below max_tokens.
        if _is(model, "claude-haiku-4-5") and max_tokens > HAIKU_45_MAX_OUTPUT:
            max_tokens = HAIKU_45_MAX_OUTPUT
            notes.append("Haiku 4.5 writes at most 64000 tokens; max_tokens lowered to 64000")
        if level == "off":
            return None, None, max_tokens, notes
        budget = LEGACY_BUDGETS[level]
        if budget >= max_tokens:
            max_tokens = budget + 4096
            notes.append(f"max_tokens raised to {max_tokens} to fit the {budget}-token thinking budget")
            if _is(model, "claude-haiku-4-5") and max_tokens > HAIKU_45_MAX_OUTPUT:
                budget = HAIKU_45_MAX_OUTPUT - 4096
                max_tokens = HAIKU_45_MAX_OUTPUT
                notes[-1] = f"thinking budget lowered to {budget} to fit the 64000-token output limit"
        return {"type": "enabled", "budget_tokens": budget}, None, max_tokens, notes

    if _is(model, "claude-sonnet-5-5"):
        if level == "off":
            # between_tools is rejected at xhigh and max; it carries no other field.
            return {"type": "between_tools"}, "low", max_tokens, notes
        return adaptive, level, max_tokens, notes

    if _is(model, "claude-haiku-5-5") or _is(model, "claude-opus-5"):
        if level == "off":
            # disabled is accepted only at effort high or below.
            return {"type": "disabled"}, "low", max_tokens, notes
        return adaptive, level, max_tokens, notes

    if (_is(model, "claude-opus-4-8") or _is(model, "claude-opus-4-7") or _is(model, "claude-sonnet-5")):
        if level == "off":
            return {"type": "disabled"}, "low", max_tokens, notes
        return adaptive, level, max_tokens, notes

    if _is(model, "claude-opus-4-6") or _is(model, "claude-sonnet-4-6"):
        if level == "off":
            return {"type": "disabled"}, None, max_tokens, notes
        if level == "xhigh":
            level = "high"
            notes.append("xhigh is not available on this model; used high")
        return adaptive, level, max_tokens, notes

    # claude-opus-5-5, claude-fable-5*, claude-mythos-* and any newer model: thinking cannot be switched off.
    if level == "off":
        notes.append(f"off is not available on {_pretty(model)}; used low")
        level = "low"
    return adaptive, level, max_tokens, notes


def _pretty(model):
    """`claude-opus-5-5` -> `Opus 5.5`, for the status line; unknown shapes come back unchanged."""
    m = re.fullmatch(r"claude-(opus|sonnet|haiku|fable|mythos)-(\d+)(?:-(\d{1,2}))?(?:-\d{8})?", model)
    if not m:
        return model
    version = m.group(2) + ("." + m.group(3) if m.group(3) else "")
    return f"{m.group(1).capitalize()} {version}"


def _to_pil(frame):
    """One ComfyUI image (H, W, C floats 0..1) as an 8-bit RGB PIL image. Alpha is dropped."""
    array = (frame.detach().float().clamp(0, 1) * 255.0).round().to(torch.uint8).cpu().numpy()
    if array.ndim == 2:
        array = array[:, :, None]
    if array.shape[2] == 1:
        array = np.repeat(array, 3, axis=2)
    return Image.fromarray(array[:, :, :3], "RGB")


def _png(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _jpeg(image):
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=92)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def encode_image(frame):
    """An image block for the API from one ComfyUI image: PNG, or JPEG and smaller when the limits need it.

    The API takes at most 8000 pixels per edge and 5 MB of base64 per image.
    """
    image = _to_pil(frame)
    width, height = image.size
    if max(width, height) > MAX_EDGE:
        scale = MAX_EDGE / max(width, height)
        image = image.resize((max(1, int(width * scale)), max(1, int(height * scale))), Image.Resampling.LANCZOS)
    data, media = _png(image), "image/png"
    if len(data) > MAX_IMAGE_B64:
        data, media = _jpeg(image), "image/jpeg"
        while len(data) > MAX_IMAGE_B64 and min(image.size) > 1:
            image = image.resize((max(1, int(image.width * 0.8)), max(1, int(image.height * 0.8))), Image.Resampling.LANCZOS)
            data = _jpeg(image)
    return {"type": "image", "source": {"type": "base64", "media_type": media, "data": data}}


def _status_line(requested, final, level, notes):
    serving = getattr(final, "model", None)
    parts = [str(serving or requested)]
    if serving and serving != requested:
        parts[0] = f"{serving} (asked for {requested})"
    parts.append(f"effort {level}")
    usage = getattr(final, "usage", None)
    if usage is not None:
        parts.append(f"{getattr(usage, 'input_tokens', 0)} tokens in, {getattr(usage, 'output_tokens', 0)} out")
    return "  |  ".join(parts + list(notes))


def _blocks(final):
    texts, thoughts = [], []
    for block in getattr(final, "content", None) or []:
        kind = getattr(block, "type", None)
        if kind == "text":
            texts.append(getattr(block, "text", "") or "")
        elif kind == "thinking":
            thoughts.append(getattr(block, "thinking", "") or "")
    return "".join(texts), "\n\n".join(t for t in thoughts if t)


def ask_claude(client, model, level, system_prompt, prompt, images, max_tokens):
    """One request. Returns (text, thinking, status)."""
    anthropic = _import_anthropic()
    thinking, effort, max_tokens, notes = _thinking_plan(model, level, int(max_tokens))

    content = list(images)
    if prompt:
        content.append({"type": "text", "text": prompt})
    request = dict(model=model, max_tokens=max_tokens, messages=[{"role": "user", "content": content}])
    if system_prompt:
        request["system"] = system_prompt
    if thinking is not None:
        request["thinking"] = thinking
    if effort is not None:
        request["output_config"] = {"effort": effort}

    if any(_is(model, base) for base in FALLBACK_MODELS_WITH_BETA):
        request.update(betas=[FALLBACK_BETA], fallbacks="default")
        open_stream = client.beta.messages.stream
    else:
        open_stream = client.messages.stream

    try:
        comfy.model_management.throw_exception_if_processing_interrupted()
        with open_stream(**request) as stream:
            for _event in stream:
                comfy.model_management.throw_exception_if_processing_interrupted()
            final = stream.get_final_message()
    except anthropic.AuthenticationError as e:
        raise RuntimeError(_AUTH_HELP) from e
    except TypeError as e:  # the SDK raises this when it finds no credentials at all
        if "authentication" in str(e).lower():
            raise RuntimeError(_AUTH_HELP) from e
        raise
    except anthropic.PermissionDeniedError as e:
        raise RuntimeError(f"This API key may not use {model} (permission denied): {getattr(e, 'message', e)}") from e
    except anthropic.NotFoundError as e:
        raise RuntimeError(f"Claude has no model called {model} for this account (not found): {getattr(e, 'message', e)}") from e
    except anthropic.RateLimitError as e:
        raise RuntimeError(f"Claude is rate limiting requests for {model}. Wait a little and run again.") from e
    except anthropic.BadRequestError as e:
        raise RuntimeError(f"Claude refused the request for {model} as invalid: {getattr(e, 'message', e)}") from e
    except anthropic.APIStatusError as e:
        raise RuntimeError(f"Claude returned an error for {model} (HTTP {getattr(e, 'status_code', '?')}): "
                           f"{getattr(e, 'message', e)}") from e
    except anthropic.APIConnectionError as e:
        raise RuntimeError(f"Could not reach the Claude API for {model}. Check the network connection.") from e

    stop = getattr(final, "stop_reason", None)
    if stop == "refusal":
        details = getattr(final, "stop_details", None)
        category = getattr(details, "category", None) or "unspecified"
        explanation = getattr(details, "explanation", None) or "no explanation given"
        raise RuntimeError(f"Claude declined this request ({model}): category {category}: {explanation}")
    if stop == "max_tokens":
        notes.append("cut off at max_tokens")
    text, thoughts = _blocks(final)
    return text, thoughts, _status_line(model, final, effort or level, notes)


class RulZClaudeVision:
    """Ask Claude about an image through the Anthropic API. See the module docstring."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (get_models(), {"default": DEFAULT_MODEL,
                                         "tooltip": "Which Claude model answers. The list comes from your account when a key was set "
                                                    "at startup, else a built-in one."}),
                "thinking": (THINKING_LEVELS, {"default": "medium",
                                               "tooltip": "How hard Claude thinks before it answers. Higher is slower and costs more. "
                                                          "Some models cannot switch thinking off; the status line says when a level was changed."}),
                "system_prompt": ("STRING", {"multiline": True, "default": "",
                                             "tooltip": "Standing instructions for Claude, such as a role or an output format. Can stay empty."}),
                "prompt": ("STRING", {"multiline": True, "default": "",
                                      "tooltip": "What to ask. Wire a list of prompts to get an answer for each of them."}),
                "max_tokens": ("INT", {"default": 16000, "min": 1, "max": 128000,
                                       "tooltip": "The most tokens Claude may write, thinking included. A token is about three quarters of a word."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": MAX_SEED, "control_after_generate": True,
                                 "tooltip": "Not sent to Claude, which has no seed. It only changes so you can force the node to run again."}),
            },
            "optional": {
                "image": ("IMAGE", {"tooltip": "A picture, or a batch of pictures, to ask about. Can stay empty for a text-only question."}),
                "image_mode": (IMAGE_MODES, {"default": IMAGE_MODES[0],
                                             "tooltip": "What an image batch means: every image in one request, or the prompt asked once per image."}),
                "api_key": ("STRING", {"default": "",
                                       "tooltip": "Your Anthropic API key. WARNING: a key typed here is saved in the workflow file and in the "
                                                  "metadata of every PNG ComfyUI saves, so anyone you share them with gets the key. Safer: set "
                                                  "the ANTHROPIC_API_KEY environment variable, or put the key on the first line of "
                                                  "anthropic_api_key.txt in the RulZnodes folder, and leave this empty."}),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, model=None):
        # Naming `model` here makes ComfyUI skip its dropdown check for that input only, so a workflow saved
        # with a model that is not in today's list (or when the list could not be fetched) still runs. No
        # **kwargs: with them ComfyUI would skip the checks of every input. A linked model arrives as None.
        value = model[0] if isinstance(model, list) and model else model
        if value is None or (isinstance(value, str) and value.strip().startswith("claude-")):
            return True
        return "Pick a Claude model: its id must start with claude-."

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("text", "thinking")
    OUTPUT_IS_LIST = (True, True)
    INPUT_IS_LIST = True
    OUTPUT_NODE = True
    FUNCTION = "ask"
    CATEGORY = "RulZnodes"
    DESCRIPTION = ("Send an image (or several), a system prompt and a prompt to Claude through the Anthropic API and get "
                   "its answer and a summary of its thinking. Needs an Anthropic API key and the anthropic Python package.")

    def ask(self, model, thinking, system_prompt, prompt, max_tokens, seed, image=None, image_mode=None, api_key=None):
        model_id = str(_at(model, 0, DEFAULT_MODEL) or "").strip()
        if not model_id.startswith("claude-"):
            raise ValueError("RulZ Claude Vision needs a model id that starts with claude-.")
        level = _at(thinking, 0, "medium")
        per_image = _at(image_mode, 0, IMAGE_MODES[0]) == IMAGE_MODES[1]
        tokens = int(_at(max_tokens, 0, 16000))
        widget_key = str(_at(api_key, 0, "") or "")

        # One job per item of the list inputs, paired by position (a shorter list repeats its last item),
        # and per image when asked.
        jobs = []
        paired = (prompt, image, system_prompt, seed)
        for i in range(max(1, *(len(values or []) for values in paired))):
            text = str(_at(prompt, i, "") or "").strip()
            system = str(_at(system_prompt, i, "") or "").strip()
            picture = _at(image, i)
            frames = [] if picture is None else [picture[j] for j in range(picture.shape[0])]
            if frames and per_image:
                jobs += [(text, system, [frame]) for frame in frames]
            else:
                if len(frames) > MAX_IMAGES:
                    raise ValueError(f"Claude takes at most {MAX_IMAGES} images in one request and {len(frames)} were given. "
                                     f"Use image_mode 'one answer per image' or send fewer.")
                jobs.append((text, system, frames))
        for text, _system, frames in jobs:
            if not text and not frames:
                raise ValueError("RulZ Claude Vision needs a prompt or an image: both are empty.")

        client = _make_client(widget_key)
        answers, thoughts, statuses = [], [], []
        for text, system, frames in jobs:
            blocks = [encode_image(frame) for frame in frames]
            answer, thought, status = ask_claude(client, model_id, level, system, text, blocks, tokens)
            answers.append(answer)
            thoughts.append(thought)
            statuses.append(status)
            logger.info("RulZ Claude Vision: %s", status)
        return {"ui": {"text": ["\n".join(statuses)], "generated": answers}, "result": (answers, thoughts)}


NODE_CLASS_MAPPINGS = {"RulZClaudeVision": RulZClaudeVision}
NODE_DISPLAY_NAME_MAPPINGS = {"RulZClaudeVision": "RulZ Claude Vision"}
