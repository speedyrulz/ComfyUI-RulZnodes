"""RulZ OpenRouter Vision: ask any OpenRouter model about an image (or several) through OpenRouter's HTTP API.

The node sends an optional image or image batch, a system prompt and a prompt to a model on openrouter.ai and returns
the model's reply and, when the model shows it, a summary of its thinking. It is the sibling of RulZ Claude Vision for
the hundreds of models OpenRouter offers. It needs no package beyond `requests`, which ComfyUI itself requires.

- The API key comes from the `api_key` input, else the `OPENROUTER_API_KEY` environment variable, else the first line
  of `openrouter_api_key.txt` in this folder. The key is never logged and never appears in an error message.
- The model list is fetched from `GET /models`, which needs no key and sends no information about you, when the node
  list is first loaded (never waiting more than a few seconds) and falls back to a fixed list. The metadata of every
  model is kept: it says which parameters a model accepts, so the request only carries what the model takes. A saved
  workflow that names a model missing from the list still runs, with the request built without that knowledge.
- Thinking is sent the way each model accepts it (`_reasoning_plan`). Some models cannot turn thinking off and some
  take only certain levels; when the node has to bend a setting, the status line says so.
- The request is streamed and run in a worker thread, so the Cancel button of ComfyUI works at once, also while
  the node waits for the first byte. Whether OpenRouter stops billing then depends on the provider.
- Several prompts (a list wired into `prompt`) or `one answer per image` give one request, and one answer, each.
"""
import json
import logging
import math
import os
import queue
import re
import threading
import time

import requests

import comfy.model_management

from .claude_vision import IMAGE_MODES, MAX_SEED, _at, encode_image

logger = logging.getLogger("RulZnodes.openrouter_vision")

BASE_URL = "https://openrouter.ai/api/v1"
KEY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "openrouter_api_key.txt")
THINKING_LEVELS = ["off", "low", "medium", "high", "xhigh", "max"]
DATA_POLICIES = ["any provider", "no training on data", "zero data retention"]
DEFAULT_MODEL = "~openai/gpt-mini-latest"
FALLBACK_MODELS = ["~openai/gpt-mini-latest", "openrouter/auto", "~anthropic/claude-opus-latest",
                   "~anthropic/claude-sonnet-latest", "~anthropic/claude-haiku-latest", "~google/gemini-pro-latest",
                   "~google/gemini-flash-latest", "~openai/gpt-terra-latest", "~x-ai/grok-latest",
                   "qwen/qwen3-vl-235b-a22b-instruct", "qwen/qwen3-vl-30b-a3b-instruct", "google/gemma-4-31b-it"]
MAX_IMAGES = 100

# How long INPUT_TYPES or a run may wait for the model list, and how soon a failed fetch is tried again.
LIST_WAIT = 6.0
LIST_RETRY = 300.0
LIST_TIMEOUT = (5, 15)
# requests' timeout is per read, not in total: connecting may take 15 s, then a pause of 300 s between two bytes.
TIMEOUT = (15, 300)
# Only before any answer arrived, only on these statuses, at most twice: the wait is Retry-After (up to 30 s) or this.
RETRY_STATUSES = (408, 429, 502, 503)
RETRY_WAITS = (2.0, 5.0)
RETRY_AFTER_CAP = 30.0
POLL = 0.1

# `reasoning.effort` values OpenRouter knows, from the lowest to the highest.
EFFORT_RANK = {"minimal": 0, "low": 1, "medium": 2, "high": 3, "xhigh": 4, "max": 5}
PROVIDER_FOR_POLICY = {"no training on data": {"data_collection": "deny"},
                       "zero data retention": {"data_collection": "deny", "zdr": True}}
GENERIC_ERRORS = ("provider returned error", "internal server error", "error", "unknown error", "bad request")

_NO_KEY_HELP = ("No OpenRouter API key found. Give one in the api_key input, set the OPENROUTER_API_KEY environment "
                "variable, or put the key on the first line of openrouter_api_key.txt in the RulZnodes folder.")
_AUTH_HELP = ("OpenRouter did not accept the API key. Give a valid key in the api_key input, set the OPENROUTER_API_KEY "
              "environment variable, or put the key on the first line of openrouter_api_key.txt in the RulZnodes folder.")

_LOCK = threading.Lock()
_META = {}            # every model OpenRouter listed, by id
_DROPDOWN = None      # the ids for the dropdown once a list was fetched; None until then
_INFLIGHT = None      # an Event while a fetch runs
_NEXT_TRY = 0.0       # time.monotonic() before which a failed fetch is not repeated


# ---- the model list ----

def _usable(entry):
    """True when `entry` belongs in the dropdown: it takes images, answers in text and is not batch-priced."""
    model_id = entry["id"]
    arch = entry.get("architecture")
    if not isinstance(arch, dict):
        return False
    inputs = arch.get("input_modalities")
    if not isinstance(inputs, list) or "image" not in inputs or model_id.endswith(":batch"):
        return False
    return arch.get("output_modalities") == ["text"] or model_id.startswith("openrouter/")


def _parse_models(payload):
    """(metadata by id, dropdown ids) from the body of GET /models. Entries that look wrong are skipped."""
    entries = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(entries, list):
        raise ValueError("no model list in the reply")
    meta, ids = {}, []
    for entry in entries:
        try:
            if not isinstance(entry, dict) or not isinstance(entry.get("id"), str) or not entry["id"].strip():
                continue
            model_id = entry["id"].strip()
            meta[model_id] = entry
            if _usable(entry):
                ids.append(model_id)
        except Exception:  # an odd entry is skipped, never fatal
            continue
    if not meta:
        raise ValueError("the model list is empty")
    ids = sorted(set(ids) - {DEFAULT_MODEL}, key=lambda m: (m.lower(), m))
    return meta, [DEFAULT_MODEL] + ids


def _fetch_models(done):
    """Worker: GET /models and keep the result. Runs in a daemon thread; its result is kept even when it comes late."""
    global _META, _DROPDOWN, _INFLIGHT, _NEXT_TRY
    ok = False
    try:
        response = requests.get(f"{BASE_URL}/models", timeout=LIST_TIMEOUT, allow_redirects=False)
        if response.status_code != 200:
            raise ValueError(f"HTTP {response.status_code}")
        meta, ids = _parse_models(response.json())
        with _LOCK:
            _META, _DROPDOWN = meta, ids
        ok = True
    except Exception as e:  # no network, an odd reply: the fixed list is used
        logger.info("RulZ OpenRouter Vision: model list not fetched (%s); using the built-in list", type(e).__name__)
    finally:
        with _LOCK:
            _INFLIGHT = None
            if not ok:
                _NEXT_TRY = time.monotonic() + LIST_RETRY
        done.set()


def get_models(wait=LIST_WAIT, poll=None):
    """The model ids for the dropdown: the live list when it can be had within `wait` seconds, else the fixed one.

    One fetch runs at a time, however many callers ask. A list that arrives after the wait is kept for the next
    caller. A failed fetch is tried again at most every five minutes; a good list is kept for the whole run.
    `poll`, when given, is called every tenth of a second while waiting (to check for Cancel).
    """
    global _INFLIGHT
    with _LOCK:
        if _DROPDOWN is None and _INFLIGHT is None and time.monotonic() >= _NEXT_TRY:
            _INFLIGHT = threading.Event()
            threading.Thread(target=_fetch_models, args=(_INFLIGHT,), daemon=True).start()
        event = _INFLIGHT
    if event is not None:
        deadline = time.monotonic() + wait
        while not event.is_set():
            left = deadline - time.monotonic()
            if left <= 0:
                break
            event.wait(min(POLL, left))
            if poll is not None:
                poll()
    with _LOCK:
        return list(_DROPDOWN) if _DROPDOWN is not None else list(FALLBACK_MODELS)


def _meta_for(model):
    """The listed metadata of `model`: its exact id, else the id without a `:suffix` (`:free`, `:nitro`), else None."""
    with _LOCK:
        found = _META.get(model)
        if found is None and ":" in model:
            found = _META.get(model.rsplit(":", 1)[0])
        return found


def _params(meta):
    """The `supported_parameters` of a model as a set, or None when the metadata says nothing usable."""
    values = meta.get("supported_parameters") if isinstance(meta, dict) else None
    return set(v for v in values if isinstance(v, str)) if isinstance(values, list) else None


def _has_image_input(meta):
    """False only when the metadata clearly lists the model's input kinds and `image` is not one of them."""
    arch = meta.get("architecture") if isinstance(meta, dict) else None
    inputs = arch.get("input_modalities") if isinstance(arch, dict) else None
    return not isinstance(inputs, list) or "image" in inputs


# ---- the key ----

def _key_from_file():
    """First non-empty line of openrouter_api_key.txt next to this module, or ''."""
    try:
        with open(KEY_FILE, "r", encoding="utf-8-sig") as f:
            for line in f:
                if line.strip():
                    return line.strip()
    except OSError:
        pass
    return ""


def _resolve_key(widget_key=""):
    """The API key to use: the widget, then the environment, then the key file. '' means none was found."""
    return ((widget_key or "").strip() or os.environ.get("OPENROUTER_API_KEY", "").strip() or _key_from_file())


def _checked_key(widget_key=""):
    """The key to send, or an error that says what is wrong without containing the key."""
    key = _resolve_key(widget_key)
    if not key:
        raise RuntimeError(_NO_KEY_HELP)
    if re.fullmatch(r"[\x21-\x7e]+", key) is None:
        # requests would refuse such a header with a message that quotes the key.
        raise RuntimeError("The OpenRouter API key has characters that cannot be sent: spaces, line breaks or "
                           "characters that are not plain ASCII. Check it was copied whole, with nothing around it.")
    return key


# ---- what to send ----

def _reasoning_plan(meta, level):
    """The `reasoning` object to send for a model with metadata `meta` at `level`, and notes for the status line.

    Returns (reasoning, label, notes); `reasoning` is None when nothing is sent and `label` is the thinking level
    that is in effect, for the status line. `meta` None means the model is not in the list.
    """
    notes = []
    params = _params(meta)
    if meta is None or params is None:
        # Nothing is known, so send the plain request; a model that has no thinking ignores it.
        return ({"effort": "none"}, "off", notes) if level == "off" else ({"effort": level}, level, notes)

    name = str(meta.get("id") or "this model")
    if "reasoning" not in params:
        if level != "off":
            notes.append(f"{name} has no thinking; thinking not sent")
        return None, "off", notes

    info = meta.get("reasoning")
    if not isinstance(info, dict):
        # A router such as openrouter/auto: the model it picks decides.
        if level == "off":
            notes.append(f"{name} picks the model that answers, which decides whether it thinks")
            return None, "default", notes
        return {"effort": level}, level, notes

    mandatory = info.get("mandatory") is True
    efforts = info.get("supported_efforts")
    if not isinstance(efforts, list):
        if level == "off":
            if mandatory:
                notes.append(f"thinking cannot be turned off on {name}")
                return None, "default", notes
            return {"enabled": False}, "off", notes
        return {"effort": level}, level, notes

    # The model lists the efforts it accepts. "none" means off; the rest are ranked.
    efforts = [e for e in efforts if isinstance(e, str)]
    ranked = sorted((e for e in set(efforts) if e in EFFORT_RANK), key=lambda e: EFFORT_RANK[e])
    if level == "off":
        if mandatory:
            if not ranked:
                notes.append(f"thinking cannot be turned off on {name} and it lists no effort; nothing sent")
                return None, "default", notes
            notes.append(f"off is not available on {name}; used {ranked[0]}")
            return {"effort": ranked[0]}, ranked[0], notes
        if "none" in efforts:
            return {"effort": "none"}, "off", notes
        return {"enabled": False}, "off", notes
    if level in ranked:
        return {"effort": level}, level, notes
    if not ranked:
        notes.append(f"{name} accepts no thinking level but off; thinking not sent")
        return None, "default", notes
    # Nearest level it accepts; a tie goes to the lower one.
    nearest = min(ranked, key=lambda e: (abs(EFFORT_RANK[e] - EFFORT_RANK[level]), EFFORT_RANK[e]))
    notes.append(f"{level} is not available on {name}; used {nearest}")
    return {"effort": nearest}, nearest, notes


def _positive_int(value):
    """`value` as an int above 0, else None. Booleans and odd shapes are not numbers here."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 1:
        return None
    return int(value)


def _build_request(model, meta, level, system_prompt, prompt, images, max_tokens, temperature, seed, data_policy):
    """The JSON body for POST /chat/completions, the thinking level in effect and notes for the status line.

    `images` are the blocks `encode_image` makes. Parameters the model's metadata says it does not take are left
    out; with no metadata (`meta` None) everything asked for is sent.
    """
    notes = []
    params = _params(meta)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    content = []
    if prompt:
        content.append({"type": "text", "text": prompt})
    for block in images:
        source = block["source"]
        content.append({"type": "image_url",
                        "image_url": {"url": f"data:{source['media_type']};base64,{source['data']}"}})
    messages.append({"role": "user", "content": content})
    body = {"model": model, "messages": messages, "stream": True}

    field = "max_tokens"
    if params is not None and "max_tokens" not in params and "max_completion_tokens" in params:
        field = "max_completion_tokens"
    top = meta.get("top_provider") if isinstance(meta, dict) else None
    limit = _positive_int(top.get("max_completion_tokens")) if isinstance(top, dict) else None
    if limit is not None and limit < max_tokens:
        notes.append(f"max_tokens lowered to {limit}, the most this model writes")
        max_tokens = limit
    body[field] = max_tokens

    if temperature >= 0:
        if params is None or "temperature" in params:
            body["temperature"] = temperature
        else:
            notes.append("temperature not sent: the model does not take it")
    if params is None or "seed" in params:
        body["seed"] = int(seed) & 0x7FFFFFFF
    else:
        notes.append("seed not sent: the model does not take it")

    reasoning, label, reasoning_notes = _reasoning_plan(meta, level)
    if reasoning is not None:
        body["reasoning"] = reasoning
    notes += reasoning_notes
    if data_policy in PROVIDER_FOR_POLICY:
        body["provider"] = dict(PROVIDER_FOR_POLICY[data_policy])
    return body, label, notes


# ---- reading the answer ----

def _short(text, limit=300):
    """`text` on one line, tags removed (an HTML error page), and cut to about `limit` characters."""
    text = re.sub(r"<(script|style)\b.*?</\1>", " ", str(text), flags=re.S | re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= limit else text[:limit].rstrip() + "..."


def _error_parts(err):
    """(message, provider, raw, reasons, code) from the `error` of a reply, whatever its shape."""
    message = provider = raw = code = None
    reasons = []
    if isinstance(err, dict):
        message = err.get("message")
        code = err.get("code")
        extra = err.get("metadata")
        if isinstance(extra, dict):
            provider = extra.get("provider_name")
            raw = extra.get("raw")
            if isinstance(extra.get("reasons"), list):
                reasons = [str(r) for r in extra["reasons"]]
    elif err is not None:
        message = err
    if raw is not None and not isinstance(raw, str):
        try:
            raw = json.dumps(raw)
        except (TypeError, ValueError):
            raw = str(raw)
    message = _short(message) if isinstance(message, (str, int, float)) and str(message).strip() else ""
    return (message, _short(provider, 80) if isinstance(provider, str) and provider else "",
            raw or "", reasons, code)


def _error_text(err):
    """The API's message for an `error` plus the provider's details when they help."""
    message, provider, raw, reasons, _code = _error_parts(err)
    parts = [message or "no message given"]
    if provider:
        parts.append(f"provider {provider}")
    if reasons:
        parts.append("flagged for " + ", ".join(_short(r, 60) for r in reasons[:8]))
    if raw and (not message or message.lower().rstrip(".") in GENERIC_ERRORS):
        parts.append(f"provider said: {_short(raw)}")
    return "; ".join(parts)


def _http_error(status, text, model, data_policy):
    """The RuntimeError for an HTTP error reply `text` (JSON or not) with status `status`."""
    try:
        err = json.loads(text)
        err = err.get("error", err) if isinstance(err, dict) else err
        detail = _error_text(err) if isinstance(err, (dict, str)) else _short(text, 200)
    except ValueError:
        detail = _short(text, 200) or "no message given"
    if status == 401:
        return RuntimeError(f"{_AUTH_HELP} ({detail})")
    if status == 402:
        return RuntimeError(f"OpenRouter says the account has too little credit for {model} (HTTP 402): {detail}. "
                            f"Add credits at https://openrouter.ai/credits, or lower max_tokens: OpenRouter "
                            f"holds back credit for the full max_tokens of a request.")
    if status == 403:
        return RuntimeError(f"OpenRouter refused the request for {model} (HTTP 403): {detail}")
    if status == 404:
        hint = ""
        if data_policy != DATA_POLICIES[0]:
            hint = (f" The data_policy setting '{data_policy}' may leave this model without a provider: "
                    f"try 'any provider'.")
        return RuntimeError(f"OpenRouter has no model or no provider for {model} (not found): {detail}.{hint}")
    if status == 429:
        return RuntimeError(f"OpenRouter is rate limiting requests for {model} (HTTP 429): {detail}. "
                            f"Wait a little and run again.")
    if status == 400:
        return RuntimeError(f"OpenRouter refused the request for {model} as invalid (HTTP 400): {detail}")
    return RuntimeError(f"OpenRouter returned an error for {model} (HTTP {status}): {detail}")


def _transport_error(exc, model):
    """The RuntimeError for an exception the worker met. Its text is not used: it may name the request."""
    if isinstance(exc, (requests.ConnectionError, requests.Timeout)):
        return RuntimeError(f"Could not reach OpenRouter for {model}. Check the network connection.")
    if isinstance(exc, requests.RequestException):
        return RuntimeError(f"The connection to OpenRouter broke while {model} was answering ({type(exc).__name__}).")
    return RuntimeError(f"RulZ OpenRouter Vision failed while talking to OpenRouter ({type(exc).__name__}).")


def _retry_wait(header, default):
    """Seconds to wait before a retry: the Retry-After header when it is a number of seconds, else `default`."""
    try:
        value = float(header)
        if math.isfinite(value) and value >= 0:
            return min(value, RETRY_AFTER_CAP)
    except (TypeError, ValueError):
        pass
    return default


def _read_body(response, cap=8_000_000):
    """The body of `response` as text, at most `cap` bytes."""
    data = b""
    for chunk in response.iter_content(65536):
        data += chunk
        if len(data) >= cap:
            break
    return data.decode("utf-8", errors="replace")


def _stream_worker(q, stop, url, headers, data):
    """Worker thread: post the request, retry where allowed, and put events on `q`.

    Events: ("http", status, text), ("body", text) for a reply that is not a stream, ("chunk", object),
    ("done",), ("end",) when the stream ended without [DONE], and ("exc", exception). It closes the response
    itself and stops reading as soon as `stop` is set.
    """
    response = None
    try:
        attempt = 0
        while True:
            response = requests.post(url, data=data, headers=headers, stream=True, timeout=TIMEOUT,
                                     allow_redirects=False)
            if stop.is_set():
                return
            status = response.status_code
            if status in RETRY_STATUSES and attempt < len(RETRY_WAITS):
                wait = _retry_wait(response.headers.get("Retry-After"), RETRY_WAITS[attempt])
                attempt += 1
                response.close()
                response = None
                if stop.wait(wait):
                    return
                continue
            break
        if status != 200:
            q.put(("http", status, _read_body(response, 100_000)))
            return
        if "text/event-stream" not in (response.headers.get("Content-Type") or "").lower():
            q.put(("body", _read_body(response)))
            return
        # Lines are read as bytes: str.splitlines would also split at U+2028 inside the text.
        for raw in response.iter_lines(chunk_size=256):
            if stop.is_set():
                return
            line = raw.decode("utf-8", errors="replace")
            if not line.startswith("data:"):
                continue  # blank lines, ": OPENROUTER PROCESSING" comments, other SSE fields
            payload = line[5:].strip()
            if payload == "[DONE]":
                q.put(("done",))
                return
            try:
                q.put(("chunk", json.loads(payload)))
            except ValueError:
                continue
        q.put(("end",))
    except Exception as e:
        q.put(("exc", e))
    finally:
        if response is not None:
            try:
                response.close()
            except Exception:
                pass


class _Answer:
    """What the chunks of one reply add up to."""

    def __init__(self, model):
        self.asked = model
        self.text, self.thinking, self.refusal = [], [], []
        self.model = self.provider = self.finish = self.usage = None

    def feed(self, obj, field="delta"):
        """Take one parsed chunk (or, with field='message', one whole reply). Raises on an error in it."""
        if not isinstance(obj, dict):
            return
        if obj.get("error") is not None:
            raise RuntimeError(f"OpenRouter reported an error while {self.asked} was answering: "
                               f"{_error_text(obj['error'])}")
        if isinstance(obj.get("model"), str) and obj["model"]:
            self.model = obj["model"]
        if isinstance(obj.get("provider"), str) and obj["provider"]:
            self.provider = obj["provider"]
        if isinstance(obj.get("usage"), dict):
            self.usage = obj["usage"]
        choices = obj.get("choices")
        choice = next((c for c in choices if isinstance(c, dict)), None) if isinstance(choices, list) else None
        if choice is None:
            return
        part = choice.get(field)
        if isinstance(part, dict):
            if isinstance(part.get("content"), str):
                self.text.append(part["content"])
            thought = part.get("reasoning")
            if not (isinstance(thought, str) and thought):
                thought = part.get("reasoning_content")
            if isinstance(thought, str) and thought:
                self.thinking.append(thought)
            else:
                details = part.get("reasoning_details")
                for item in details if isinstance(details, list) else []:
                    if isinstance(item, dict):
                        piece = item.get("text") if item.get("type") == "reasoning.text" else (
                            item.get("summary") if item.get("type") == "reasoning.summary" else None)
                        if isinstance(piece, str) and piece:
                            self.thinking.append(piece)
            if isinstance(part.get("refusal"), str):
                self.refusal.append(part["refusal"])
        reason = choice.get("finish_reason")
        if isinstance(reason, str) and reason:
            self.finish = reason
            if reason == "error":
                raise RuntimeError(f"OpenRouter reported an error while {self.asked} was answering: "
                                   f"{_error_text(choice.get('error')) if choice.get('error') else 'the model stopped with an error'}")


def _cost(value):
    """A cost in dollars as text that never shows a tiny cost as zero, or None."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        return None
    if value == 0:
        return "free"
    if value >= 1:
        return f"${value:.2f}"
    if value >= 0.01:
        return f"${value:.4f}"
    return f"${value:.{min(12, 2 - math.floor(math.log10(value)))}f}"


def _status_line(requested, answer, label, notes):
    serving = answer.model
    parts = [f"{serving} (asked for {requested})" if serving and serving != requested else str(serving or requested)]
    if answer.provider:
        parts.append(f"via {answer.provider}")
    parts.append(f"thinking {label}")
    usage = answer.usage
    if isinstance(usage, dict):
        tokens_in, tokens_out = usage.get("prompt_tokens"), usage.get("completion_tokens")
        if isinstance(tokens_in, (int, float)) and isinstance(tokens_out, (int, float)):
            parts.append(f"{int(tokens_in)} tokens in, {int(tokens_out)} out")
        cost = _cost(usage.get("cost"))
        if cost:
            parts.append(cost)
    return "  |  ".join(parts + list(notes))


def _request(model, key, body, data_policy):
    """Send `body` and wait for the whole answer, checking for Cancel every tenth of a second. Returns an _Answer."""
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json",
               "Accept": "text/event-stream", "Accept-Encoding": "identity"}
    data = json.dumps(body).encode("utf-8")
    q, stop = queue.Queue(), threading.Event()
    answer, complete = _Answer(model), False
    try:
        comfy.model_management.throw_exception_if_processing_interrupted()
        threading.Thread(target=_stream_worker, args=(q, stop, f"{BASE_URL}/chat/completions", headers, data),
                         daemon=True).start()
        while True:
            comfy.model_management.throw_exception_if_processing_interrupted()
            try:
                event = q.get(timeout=POLL)
            except queue.Empty:
                continue
            kind = event[0]
            if kind == "chunk":
                answer.feed(event[1])
            elif kind == "done":
                complete = True
                break
            elif kind == "end":
                break
            elif kind == "body":
                try:
                    reply = json.loads(event[1])
                except ValueError:
                    raise RuntimeError(f"OpenRouter sent a reply for {model} that is neither a stream nor JSON: "
                                       f"{_short(event[1], 200)}") from None
                answer.feed(reply, "message")
                complete = True
                break
            elif kind == "http":
                raise _http_error(event[1], event[2], model, data_policy)
            else:
                raise _transport_error(event[1], model)
    except RuntimeError as e:
        text = str(e)
        if key in text:
            raise RuntimeError(text.replace(key, "***")) from None
        raise
    finally:
        stop.set()
    if not complete and answer.finish is None:
        raise RuntimeError(f"The connection to OpenRouter ended before {model} finished its answer.")
    return answer


def ask_openrouter(model, meta, key, level, system_prompt, prompt, images, max_tokens, temperature, seed, data_policy):
    """One request. Returns (text, thinking, status)."""
    if images and not _has_image_input(meta):
        raise ValueError(f"{model} does not take images, according to OpenRouter's model list. Pick a model "
                         f"that does, or remove the image.")
    body, label, notes = _build_request(model, meta, level, system_prompt, prompt, images, max_tokens,
                                        temperature, seed, data_policy)
    answer = _request(model, key, body, data_policy)

    text, thoughts, refusal = "".join(answer.text), "".join(answer.thinking), "".join(answer.refusal).strip()
    if answer.finish == "content_filter" or (refusal and not text.strip()):
        raise RuntimeError(f"{model} declined this request: {refusal or 'content filter, no reason given'}")
    if answer.finish == "length":
        notes.append("cut off at max_tokens")
        if not text.strip():
            notes.append("no answer text: thinking used up all the tokens, raise max_tokens or lower thinking")
    elif not text.strip():
        notes.append("the model returned no text")
    return text, thoughts, _status_line(model, answer, label, notes)


_MODEL_ID = re.compile(r"[^\s/]+/\S+")


class RulZOpenRouterVision:
    """Ask any OpenRouter model about an image. See the module docstring."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (get_models(), {"default": DEFAULT_MODEL,
                                         "tooltip": "Which model answers. The list of models that take images comes from "
                                                    "openrouter.ai when the node list loads, else a built-in one. Any id "
                                                    "from openrouter.ai/models also works in a saved workflow."}),
                "thinking": (THINKING_LEVELS, {"default": "medium",
                                               "tooltip": "How hard the model thinks before it answers. Higher is slower and costs "
                                                          "more. Not every model can switch thinking off or takes every level; "
                                                          "the status line says when a level was changed."}),
                "system_prompt": ("STRING", {"multiline": True, "default": "",
                                             "tooltip": "Standing instructions for the model, such as a role or an output format. Can stay empty."}),
                "prompt": ("STRING", {"multiline": True, "default": "",
                                      "tooltip": "What to ask. Wire a list of prompts to get an answer for each of them."}),
                "max_tokens": ("INT", {"default": 16000, "min": 1, "max": 128000,
                                       "tooltip": "The most tokens the model may write, thinking included. OpenRouter holds back "
                                                  "credit for all of it, so a lower value suits a small balance."}),
                "temperature": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 2.0, "step": 0.05,
                                          "tooltip": "How varied the answers are. -1 = the model's own default, nothing is sent."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": MAX_SEED, "control_after_generate": True,
                                 "tooltip": "Sent to models that take a seed (lowered to 31 bits), which makes their answers "
                                            "repeatable but not guaranteed identical. It also lets you force the node to run again."}),
            },
            "optional": {
                "image": ("IMAGE", {"tooltip": "A picture, or a batch of pictures, to ask about. Can stay empty for a text-only question."}),
                "image_mode": (IMAGE_MODES, {"default": IMAGE_MODES[0],
                                             "tooltip": "What an image batch means: every image in one request, or the prompt asked once per image."}),
                "data_policy": (DATA_POLICIES, {"default": DATA_POLICIES[1],
                                                "tooltip": "Which providers OpenRouter may use for your prompt and images. "
                                                           "'no training on data' skips providers that train on them, "
                                                           "'zero data retention' only uses providers that keep nothing. "
                                                           "Stricter settings leave some models without a provider."}),
                "api_key": ("STRING", {"default": "",
                                       "tooltip": "Your OpenRouter API key. WARNING: a key typed here is saved in the workflow file and in the "
                                                  "metadata of every PNG ComfyUI saves, so anyone you share them with gets the key. Safer: set "
                                                  "the OPENROUTER_API_KEY environment variable, or put the key on the first line of "
                                                  "openrouter_api_key.txt in the RulZnodes folder, and leave this empty."}),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, model=None):
        # Naming `model` here makes ComfyUI skip its dropdown check for that input only, so a workflow saved with a
        # model that is not in today's list (text-only, :free, :nitro, a new id) still runs. No **kwargs: with them
        # ComfyUI would skip the checks of every input. A linked model arrives as None.
        value = model[0] if isinstance(model, list) and model else model
        if value is None or (isinstance(value, str) and _MODEL_ID.fullmatch(value.strip())):
            return True
        return "Pick a model: its id looks like provider/model-name."

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("text", "thinking")
    OUTPUT_IS_LIST = (True, True)
    INPUT_IS_LIST = True
    OUTPUT_NODE = True
    FUNCTION = "ask"
    CATEGORY = "RulZnodes"
    DESCRIPTION = ("Send an image (or several), a system prompt and a prompt to any model on OpenRouter and get its "
                   "answer and, when the model shows it, a summary of its thinking. Needs an OpenRouter API key. "
                   "Every run is billed to your OpenRouter account, except free models.")

    def ask(self, model, thinking, system_prompt, prompt, max_tokens, temperature, seed, image=None,
            image_mode=None, data_policy=None, api_key=None):
        model_id = str(_at(model, 0, DEFAULT_MODEL) or "").strip()
        if not _MODEL_ID.fullmatch(model_id):
            raise ValueError("RulZ OpenRouter Vision needs a model id like provider/model-name.")
        level = _at(thinking, 0, "medium")
        per_image = _at(image_mode, 0, IMAGE_MODES[0]) == IMAGE_MODES[1]
        tokens = int(_at(max_tokens, 0, 16000))
        degrees = float(_at(temperature, 0, -1.0))
        policy = _at(data_policy, 0, DATA_POLICIES[1])
        if policy not in DATA_POLICIES:
            policy = DATA_POLICIES[1]
        widget_key = str(_at(api_key, 0, "") or "")

        # One job per item of the list inputs, paired by position (a shorter list repeats its last item),
        # and per image when asked.
        jobs = []
        paired = (prompt, image, system_prompt, seed)
        for i in range(max(1, *(len(values or []) for values in paired))):
            text = str(_at(prompt, i, "") or "").strip()
            system = str(_at(system_prompt, i, "") or "").strip()
            picture = _at(image, i)
            number = int(_at(seed, i, 0) or 0)
            frames = [] if picture is None else [picture[j] for j in range(picture.shape[0])]
            if frames and per_image:
                jobs += [(text, system, [frame], number) for frame in frames]
            else:
                if len(frames) > MAX_IMAGES:
                    raise ValueError(f"OpenRouter takes at most {MAX_IMAGES} images in one request here and {len(frames)} were given. "
                                     f"Use image_mode 'one answer per image' or send fewer.")
                jobs.append((text, system, frames, number))
        for text, _system, frames, _number in jobs:
            if not text and not frames:
                raise ValueError("RulZ OpenRouter Vision needs a prompt or an image: both are empty.")

        key = _checked_key(widget_key)
        # The metadata says what the model takes. A run goes on without it when the list cannot be had.
        get_models(poll=comfy.model_management.throw_exception_if_processing_interrupted)
        meta = _meta_for(model_id)
        answers, thoughts, statuses = [], [], []
        for text, system, frames, number in jobs:
            blocks = [encode_image(frame) for frame in frames]
            answer, thought, status = ask_openrouter(model_id, meta, key, level, system, text, blocks, tokens,
                                                     degrees, number, policy)
            answers.append(answer)
            thoughts.append(thought)
            statuses.append(status)
            logger.info("RulZ OpenRouter Vision: %s", status)
        return {"ui": {"text": ["\n".join(statuses)], "generated": answers}, "result": (answers, thoughts)}


NODE_CLASS_MAPPINGS = {"RulZOpenRouterVision": RulZOpenRouterVision}
NODE_DISPLAY_NAME_MAPPINGS = {"RulZOpenRouterVision": "RulZ OpenRouter Vision"}
