"""RulZ Generate Text: write text with the language model inside a loaded CLIP, several answers at once.

Started from the Fast Generate Text node of WAS Node Suite (MIT, WASasquatch), which switches a model
onto ComfyUI's fixed-cache, graph-captured decode for one run. This node makes the same switch, with the
same guards, and then decodes rows together. Measured on ComfyUI 0.45 with Qwen3-VL 4B int8 on an
RTX 4060 Ti: the plain decode writes about 3 tokens/s and the graph decode about 45, and one graph
decode step takes 22 ms for one answer, 24 ms for eight and 30 ms for sixty-four, because the step is
spent reading the model's weights from VRAM whatever the number of rows. So:

- several prompts (a list wired into `prompt`), several images (one answer per image), or several
  `variations` of one prompt are written in one pass through ComfyUI's own decode loop;
- prompts of different lengths share a batch: shorter rows are padded on the left and masked;
- every row draws its random numbers from its own seed (`seed + variation`), the same numbers in
  whichever batch it lands. A batch still changes the rounding of the model's arithmetic a little, so a
  row can drift from the same row in another batch; the same rows in the same batches write the same
  text, and a run that writes a single answer draws it exactly as core Generate Text does;
- the batch is cut to what the key/value cache can take in free VRAM, and a batch that still runs
  out of VRAM is halved and retried.

The decode loop itself stays ComfyUI's (`BaseGenerate.generate`). For the length of one run this module
replaces two methods on the model instance: `generate` (to merge the rows) and `sample_token` (to sample
and track each row), plus the model's `forward` for the prompt pass of a padded batch. They are put back
afterwards. A model that decodes some other way runs one row at a time, exactly as ComfyUI would: Qwen
3.5, which has its own decoder, and Gemma 4, whose decoder takes inputs that cannot be stacked.
"""
import contextlib
import logging
import time
import weakref

import torch

import comfy.model_management
import comfy.utils

try:
    import comfy.text_encoders.llama as _llama
except ImportError:  # a ComfyUI without language models
    _llama = None

logger = logging.getLogger("RulZnodes.generate_text")

MAX_SEED = 0xFFFFFFFFFFFFFFFF
MTP_CHOICES = ["auto", "off", "2", "3", "4", "5", "tree2s1", "tree2s2", "tree3s1", "tree4s1m8"]
IMAGE_MODES = ["all images in one prompt", "one answer per image"]
THINK_OPEN, THINK_CLOSE = "<think>", "</think>"

# The keyword arguments a model's wrapper may hand to the decoder that rows can be merged over.
# Anything else (Gemma 4's initial_input_ids, say) sends the rows through one at a time.
MERGEABLE = {"stop_tokens", "initial_tokens", "presence_penalty", "position_ids", "visual_pos_masks",
             "deepstack_embeds", "mtp", "min_tokens", "execution_dtype"}
# The switches on a layer stack that select ComfyUI's fixed-cache, graph-captured decode.
FLAGS = ("fixed_kv", "graph_dynamic_vbar_blocks", "prefetch_dynamic_vbars")
# Share of the free VRAM a batch's key/value cache may take, and the least it is ever allowed.
KV_VRAM_SHARE = 0.6
KV_VRAM_FLOOR = 1 << 30
# Embedding positions assumed for an image or other media item when sizing a batch before embedding it.
MEDIA_TOKENS_GUESS = 1024
_MISSING = object()
# The wired CLIP's model -> the patcher of a second copy of it that was laid out for the graph decode.
_FAST_COPIES = weakref.WeakKeyDictionary()


class _Unbatchable(Exception):
    """These rows cannot share a batch; the message says why."""


class _Ragged(_Unbatchable):
    """These rows cannot share a batch because their prompts differ in length; rows of one length still can."""


def _fresh(tokens):
    """A copy of a tokenize result whose media items a model may consume (Gemma 4 pops their data).

    Only the item dicts are copied: the image tensors inside stay the same objects, so a vision
    encoder cache keyed on them still finds a variation's image.
    """
    def item(value):
        return dict(value) if isinstance(value, dict) else value

    def row(r):
        return [tuple(item(v) for v in t) if isinstance(t, tuple) else item(t) for t in r]

    if isinstance(tokens, dict):
        return {key: [row(r) for r in rows] for key, rows in tokens.items()}
    return [row(r) for r in tokens]


@contextlib.contextmanager
def _shadow(obj, name, value):
    """Put an attribute on one instance for the length of a block, then restore what was there."""
    saved = obj.__dict__.get(name, _MISSING)
    setattr(obj, name, value)
    try:
        yield
    finally:
        if saved is _MISSING:
            obj.__dict__.pop(name, None)
        else:
            setattr(obj, name, saved)


def _decoder_in(cond):
    """(transformer, layer stack) of the language model in a text encoder model, or (None, None)."""
    name = getattr(cond, "clip", None)
    inner = getattr(cond, name, None) if isinstance(name, str) else None
    transformer = getattr(inner, "transformer", None)
    model = getattr(transformer, "model", None)
    if transformer is None or model is None:
        return None, None
    return transformer, model


def _decoder_of(clip):
    """(transformer, layer stack) of the language model in a CLIP, or (None, None)."""
    return _decoder_in(getattr(clip, "cond_stage_model", None))


def _batch_refusal(transformer, model):
    """Why this model's rows cannot be decoded together, or "" when they can."""
    if _llama is None or transformer is None:
        return "this CLIP holds no language model this node knows how to batch"
    if getattr(type(transformer), "generate", None) is not _llama.BaseGenerate.generate:
        owner = next((c for c in type(transformer).__mro__ if "generate" in c.__dict__), type(transformer))
        return f"{owner.__name__} writes through its own decoder"
    for name in ("sample_token", "processed_probs"):
        if not callable(getattr(transformer, name, None)):
            return f"this ComfyUI's decoder has no {name}"
    if not all(hasattr(model, flag) for flag in FLAGS) or not hasattr(model, "layers"):
        return f"{type(model).__name__} has no fixed key/value cache"
    return ""


def _flash_decode(model, device):
    """Whether ComfyUI's fixed-cache decode kernel runs this model's attention shape on this device."""
    try:
        import comfy_kitchen

        probe = getattr(comfy_kitchen, "flash_attention_decode_is_available", None)
        if probe is None or not probe(device):
            return False
        cfg = model.config
        query = torch.zeros((1, 1, cfg.num_attention_heads, cfg.head_dim), device=device, dtype=torch.bfloat16)
        key = torch.zeros((1, 2, cfg.num_key_value_heads, cfg.head_dim), device=device, dtype=torch.bfloat16)
        comfy_kitchen.flash_attention_decode(query, key, torch.zeros_like(key), torch.full((1,), 2, device=device, dtype=torch.int32))
        return True
    except Exception as e:
        logger.debug("fixed-cache decode is not available: %s", e)
        return False


def _graph_refusal(clip, model, device):
    """Why the fixed-cache decode would run without captured graphs, or "" when it has them."""
    import comfy.model_prefetch

    theta = getattr(getattr(model, "config", None), "rope_theta", None)
    if isinstance(theta, (list, tuple)) and len(theta) > 1:
        return "the model keeps two rotary tables, local and global"
    if getattr(comfy.model_management.args, "disable_cuda_graphs", False):
        return "ComfyUI was started with --disable-cuda-graphs"
    if not comfy.model_prefetch.malloc_graph_enabled(device):
        return "graph allocation is not available on this device"
    if not clip.patcher.is_dynamic():
        return "the model is not loaded with dynamic VRAM"
    return ""


def _smallest_window(model):
    """The narrowest sliding attention window among the layers, or None when every layer sees everything."""
    windows = [layer.sliding_attention for layer in getattr(model, "layers", [])
               if type(getattr(layer, "sliding_attention", None)) is int and layer.sliding_attention > 0]
    return min(windows) if windows else None


def _layout(model):
    """How dynamic VRAM laid this layer stack's weights out: "none", "units" or "scattered".

    The graph decode joins each layer's q/k/v and gate/up weights in place, which is only right when
    ComfyUI gave them back-to-back VRAM addresses. It does that when the graph switch is on at the
    model's first load. Loaded first with the switch off (a plain text encode, say), the weights sit
    scattered, and turning the switch on then overwrites neighbouring weights: the model writes
    garbage from then on, on every decode path.
    """
    seen = False
    scattered = False
    for layer in getattr(model, "layers", []):
        order = []
        for module in layer.modules():
            if getattr(module, "_v", None) is None or module in order:
                continue
            group = getattr(module, "_combined_linear", None)
            order.extend(group.linears if group is not None else (module,))
        slots = [getattr(m, "_v", None) for m in order]
        if any(v is None for v in slots):
            scattered = True
            continue
        seen = seen or bool(slots)
        # rising order, not only adjacent group members: ModelPatcherDynamic.load spans a graph unit from
        # its first module's slot, so a module placed below that would fall outside the unit
        if any(b[1] <= a[1] for a, b in zip(slots, slots[1:])):
            scattered = True
        for module in order:
            group = getattr(module, "_combined_linear", None)
            if group is not None and module is group.linears[0]:
                members = [m._v for m in group.linears]
                if any(a[1] + a[2] != b[1] for a, b in zip(members, members[1:])):
                    scattered = True
    if not seen and not scattered:
        return "none"
    return "scattered" if scattered else "units"


def _fast_copy(clip):
    """A CLIP over a second copy of the model, built with the graph decode on. None when the loader cannot rebuild it."""
    cond = clip.cond_stage_model
    base = _FAST_COPIES.get(cond)
    if base is False:
        return None
    if base is None:
        init = getattr(clip.patcher, "cached_patcher_init", None)
        if init is None:
            return None
        try:
            # with dynamic VRAM the loader leaves the new copy in RAM, so nothing loads it with the switches off
            base = init[0](*init[1])
            if len(init) > 2:
                base = base[init[2]]
        except Exception as e:
            logger.warning("RulZ Generate Text could not build a second copy of the model: %s", e)
            return None
        _, model = _decoder_in(base.model)
        if model is None or not all(hasattr(model, flag) for flag in FLAGS):
            _FAST_COPIES[cond] = False
            return None
        # Nothing but this node loads the copy, and always with the switches on, so its weights are
        # laid out block by block from its first load.
        _FAST_COPIES[cond] = base
        # the wired copy is not needed on the GPU while this one writes
        comfy.model_management.unload_model_and_clones(clip.patcher)
    copy = clip.clone()
    # this run's patches (a LoRA on the text encoder, say) over the copy's model
    patcher = clip.patcher.clone(model_override=base.get_clone_model_override())
    # ComfyUI's model management moves its bookkeeping to a dropped clone's parent, which must own this model
    patcher.parent = base
    patcher.hook_backup = {}
    copy.patcher = patcher
    copy.cond_stage_model = base.model
    return copy


def _prepare_fast(clip):
    """Pick the CLIP and the switches for ComfyUI's fixed-cache, graph-captured decode.

    Returns:
        (clip, flags, note): the CLIP to generate with (the wired one, or a copy laid out for the graph
        decode), the switches to set on its layer stack for the run (empty when it is left as it is),
        and "" on the graph decode or else which slower path runs and why.
    """
    transformer, model = _decoder_of(clip)
    device = clip.patcher.load_device
    if getattr(model, "fixed_kv", False):
        # the model already decodes on its fixed cache (Qwen3-VL does, graphs included): leave it be
        return clip, {}, ""
    if not _flash_decode(model, device):
        return clip, {}, "plain decode: the fixed-cache kernel does not run this model on this device"
    reason = _graph_refusal(clip, model, device)
    if reason:
        return clip, {"fixed_kv": True, "graph_dynamic_vbar_blocks": False, "prefetch_dynamic_vbars": False}, f"fixed-cache decode without graphs: {reason}"
    flags = {flag: True for flag in FLAGS}
    if _layout(model) != "scattered":
        return clip, flags, ""
    copy = _fast_copy(clip)
    if copy is None:
        return clip, {}, ("plain decode: this CLIP was first loaded without the graph decode, and its loader cannot build "
                          "a second copy. Reload it and run RulZ Generate Text before anything else uses it")
    return copy, flags, "a second copy of the model, laid out for the graph decode, writes the text (this CLIP was first loaded for encoding)"


def _is_plain_positions(position_ids, length):
    """True when explicit position ids are just 0..length-1 on every axis (a prompt without images)."""
    if position_ids is None:
        return True
    if not torch.is_tensor(position_ids) or position_ids.shape[-1] != length:
        return False
    return bool((position_ids == torch.arange(length, device=position_ids.device)).all())


def _same(a, b):
    """Equality that also covers tensors and lists of tensors."""
    if torch.is_tensor(a) or torch.is_tensor(b):
        return torch.is_tensor(a) and torch.is_tensor(b) and a.shape == b.shape and bool((a == b).all())
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return a == b


class _Rows:
    """Samples every row of a batch and keeps each row's tokens. Stands in for `sample_token`."""

    def __init__(self, transformer, seeds, stop_tokens, max_length, device, initial_tokens, exact=False):
        self.processed_probs = transformer.processed_probs
        self.count = len(seeds)
        # a run of one answer draws as ComfyUI does; every other run draws each row from its seed's own numbers
        self.exact = exact and self.count == 1
        self.device = device
        self.seeds = [int(s) & MAX_SEED for s in seeds]
        self.max_length = max_length
        self.generator = None
        self.uniform = None
        stops = sorted(int(t) for t in stop_tokens)
        self.stop = torch.tensor(stops, device=device, dtype=torch.long)
        filler = next(i for i in range(len(stops) + 1) if i not in stops)
        self.filler = torch.tensor(filler, device=device, dtype=torch.long)
        self.history = torch.zeros((self.count, max_length), device=device, dtype=torch.long)
        self.ends = torch.full((self.count,), max_length, device=device, dtype=torch.long)
        self.finished = torch.zeros((self.count,), device=device, dtype=torch.bool)
        self.initial_tokens = initial_tokens
        self.mask = None
        self.step = 0
        self.first_token_time = None

    def __call__(self, logits, temperature, top_k, top_p, min_p, repetition_penalty, token_history, generator,
                 do_sample=True, presence_penalty=0.0, penalty_mask=None):
        if logits.shape[0] != self.count:
            raise RuntimeError(f"the decoder handed over {logits.shape[0]} rows for a batch of {self.count}")
        if _llama.penalty_active(repetition_penalty, presence_penalty):
            if self.mask is None:
                # one row of "already used" flags per answer; ComfyUI's own mask covers a single row
                self.mask = torch.zeros(logits.shape, device=logits.device, dtype=torch.bool)
                for row, used in enumerate(self.initial_tokens):
                    if len(used) > 0:
                        self.mask[row].index_fill_(0, torch.tensor(list(used), device=logits.device), True)
            logits = torch.where(self.mask, _llama.apply_penalty(logits, repetition_penalty, presence_penalty), logits)

        if not do_sample or temperature == 0.0:
            token = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            probs, top = self.processed_probs(logits, temperature, top_k, top_p, min_p)
            if self.exact:
                # exactly ComfyUI's draw, so one answer with a seed writes what core Generate Text writes
                if self.generator is None:
                    self.generator = torch.Generator(device=logits.device).manual_seed(self.seeds[0])
                token = torch.multinomial(probs, num_samples=1, generator=self.generator)
            else:
                if self.uniform is None:
                    # every row's random numbers for the whole run, drawn up front from that row's own seed
                    draws = [torch.rand(self.max_length, generator=torch.Generator().manual_seed(s)) for s in self.seeds]
                    self.uniform = torch.stack(draws).to(logits.device)
                # one draw per row by inverse CDF: the first candidate whose running total passes the row's number
                cdf = probs.float().cumsum(dim=-1)
                point = self.uniform[:, self.step:self.step + 1] * cdf[:, -1:]
                token = torch.searchsorted(cdf, point, right=True).clamp_(max=probs.shape[-1] - 1)
            if top is not None:
                token = top.gather(1, token)

        if self.first_token_time is None:
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            self.first_token_time = time.perf_counter()

        # ComfyUI replays each decode step's GPU allocations at the same addresses, so a tensor made in
        # one step is overwritten in the next. Everything kept across steps (history, ends, finished,
        # mask) was allocated up front or on the first step and is only ever written in place here.
        # A finished row keeps running until the whole batch is done; it is fed a filler token.
        token = torch.where(self.finished.unsqueeze(1), self.filler, token)
        flat = token[:, 0]
        self.history[:, self.step] = flat
        hit = torch.isin(flat, self.stop).logical_and_(self.finished.logical_not())
        self.ends.masked_fill_(hit, self.step + 1)
        self.finished.logical_or_(hit)
        if self.mask is not None:
            self.mask.scatter_(1, token, True)
        # ComfyUI's loop stops when row 0 writes a stop token, so row 0 says when every row has.
        feed = token.clone()
        feed[0, 0] = torch.where(self.finished.all(), self.stop[0], torch.where(self.finished[0], self.filler, flat[0]))
        self.step += 1
        return feed

    def results(self):
        """Each row's token ids, its stop token included, as ComfyUI's generate returns them."""
        history = self.history[:, :self.step].cpu()
        ends = self.ends.cpu().tolist()
        return [history[row, :min(ends[row], self.step)].tolist() for row in range(self.count)]


def _merge(rows):
    """Merge the decoder calls captured for each row into one call.

    Args:
        rows: [(embeds, args, kwargs)] as each row's wrapper handed them to `transformer.generate`.

    Returns:
        (embeds [B, L, H], args, kwargs, attention_mask or None, initial_tokens per row).

    Raises:
        _Unbatchable: the rows differ in something that cannot be stacked.
    """
    _, first_args, first_kwargs = rows[0]
    unknown = set().union(*(set(k) for _, _, k in rows)) - MERGEABLE
    if unknown:
        raise _Unbatchable(f"the model passes {', '.join(sorted(unknown))} to its decoder")

    embeds = [e.unsqueeze(0) if e.ndim == 2 else e for e, _, _ in rows]
    if any(e.shape[0] != 1 for e in embeds):
        raise _Unbatchable("a prompt already holds more than one row")
    lengths = [e.shape[1] for e in embeds]
    longest = max(lengths)
    ragged = any(n != longest for n in lengths)

    for _, args, kwargs in rows[1:]:
        # the seed is the last positional argument and differs per row by design; each row has its own generator
        if len(args) != len(first_args) or not _same(args[:7], first_args[:7]):
            raise _Unbatchable("the rows were given different sampling settings")
        for key in ("stop_tokens", "presence_penalty", "mtp", "min_tokens", "execution_dtype"):
            if not _same(kwargs.get(key), first_kwargs.get(key)):
                raise _Unbatchable(f"the rows differ in {key}")

    kwargs = dict(first_kwargs)
    positions = [k.get("position_ids") for _, _, k in rows]
    visual = [k.get("visual_pos_masks") for _, _, k in rows]
    deepstack = [k.get("deepstack_embeds") for _, _, k in rows]
    has_media = any(d is not None for d in deepstack) or any(v is not None and bool(v.any()) for v in visual)

    attention_mask = None
    if ragged:
        if has_media or not all(_is_plain_positions(p, n) for p, n in zip(positions, lengths)):
            raise _Ragged("prompts with images must have the same length to share a batch")
        device, dtype = embeds[0].device, embeds[0].dtype
        attention_mask = torch.zeros((len(rows), longest), device=device, dtype=torch.long)
        padded = []
        for row, e in enumerate(embeds):
            pad = longest - e.shape[1]
            attention_mask[row, pad:] = 1
            padded.append(torch.cat([torch.zeros((1, pad, e.shape[2]), device=device, dtype=dtype), e], dim=1) if pad else e)
        embeds = padded
        if any(p is not None for p in positions):
            # plain 0..n-1 on every row: the longest row's ids serve all of them (rotary positions are relative)
            kwargs["position_ids"] = positions[lengths.index(longest)]
        if any(v is not None for v in visual):
            kwargs["visual_pos_masks"] = torch.zeros((len(rows), longest), device=device, dtype=torch.bool)
    else:
        if any(not _same(p, positions[0]) for p in positions[1:]):
            raise _Unbatchable("the prompts lay their images out differently")
        if any((v is None) != (visual[0] is None) for v in visual) or any((d is None) != (deepstack[0] is None) for d in deepstack):
            raise _Unbatchable("only some of the prompts hold an image")
        if visual[0] is not None:
            kwargs["visual_pos_masks"] = torch.cat(list(visual), dim=0)
        if deepstack[0] is not None:
            if any(len(d) != len(deepstack[0]) for d in deepstack):
                raise _Unbatchable("the prompts carry different image features")
            # image positions are read row by row, so each level's features are stacked in row order
            kwargs["deepstack_embeds"] = [torch.cat([d[i] for d in deepstack], dim=0) for i in range(len(deepstack[0]))]

    initial = [list(k.get("initial_tokens") or []) for _, _, k in rows]
    return torch.cat(embeds, dim=0), first_args, kwargs, attention_mask, initial


class _Group:
    """One batch of rows going through `clip.generate` together."""

    def __init__(self, clip, tokens, seeds, options, flags=None, exact=False):
        self.clip = clip
        # every attempt gets its own copy: a model may consume the media items, and a failed attempt is retried
        self.tokens = [_fresh(t) for t in tokens]
        self.exact = exact
        self.seeds = seeds
        self.options = options
        self.flags = dict(flags or {})
        self.rows = None
        self.captured = None
        self.note = ""
        self.times = {}

    def run(self):
        """Decode the group. Returns each row's token ids."""
        clip = self.clip
        transformer, model = _decoder_of(clip)
        cond = clip.cond_stage_model
        device = clip.patcher.load_device
        original = transformer.generate
        original_forward = model.forward
        max_length = self.options["max_length"]
        call = dict(self.options)
        group = self
        flags = self.flags
        window = _smallest_window(model)

        def generate(embeds=None, *args, **kwargs):
            if group.captured is not None:
                group.captured.append((embeds, args, kwargs))
                return []
            rows = [(embeds, args, kwargs)]
            group.captured = []
            try:
                for tokens in group.tokens[1:]:
                    cond.generate(tokens, **call)
                rows += group.captured
            finally:
                group.captured = None
            if len(rows) != len(group.tokens):
                raise _Unbatchable("the model did not reach its decoder for every prompt")
            if len(rows) == 1 and set(kwargs) - MERGEABLE:
                # one row, and its decoder takes inputs this node does not know how to stack (Gemma 4's
                # initial_input_ids): run it exactly as ComfyUI would, sampler included
                return original(embeds, *args, **kwargs)
            embeds, args, kwargs, attention_mask, initial = _merge(rows)
            stop_tokens = kwargs.get("stop_tokens")
            if stop_tokens is None:
                stop_tokens = model.config.stop_tokens
            if stop_tokens is None:
                raise ValueError(f"{type(model.config).__name__} names no token that ends an answer, so ComfyUI cannot "
                                 "write text with it. Load a text encoder that writes, such as qwen3vl_4b with type krea2.")
            limit = int(kwargs.get("max_length", args[1] if len(args) > 1 else max_length))
            with contextlib.ExitStack() as stack:
                if flags and window is not None and embeds.shape[1] + limit > window:
                    # past a sliding attention window the fixed cache would see too much: decode ComfyUI's plain way
                    group.note = f"plain decode: prompt plus max_length passes the model's {window} token sliding window"
                    for name in flags:
                        stack.enter_context(_shadow(model, name, False))
                if attention_mask is not None:
                    if not getattr(model, "fixed_kv", False) or not _flash_decode(model, embeds.device):
                        raise _Ragged("prompts of different lengths need the fixed-cache decode")
                    pending = [attention_mask]

                    def forward(*a, **k):
                        # the prompt pass only: later steps run on the cache's own per-row lengths
                        if pending:
                            k["attention_mask"] = pending.pop()
                        return original_forward(*a, **k)
                    stack.enter_context(_shadow(model, "forward", forward))
                group.rows = _Rows(transformer, group.seeds, stop_tokens, limit, embeds.device, initial, exact=group.exact)
                stack.enter_context(_shadow(transformer, "sample_token", group.rows))
                group.times["decoder"] = time.perf_counter()
                return original(embeds, *args, **kwargs)

        self.times["start"] = time.perf_counter()
        try:
            with contextlib.ExitStack() as stack:
                for name, value in flags.items():
                    # set before the model loads, so each block is registered as a graph unit
                    stack.enter_context(_shadow(model, name, value))
                stack.enter_context(_shadow(transformer, "generate", generate))
                if hasattr(type(transformer), "vision_cache"):
                    # Qwen3-VL encodes an image once for all the variations of it in this batch
                    stack.enter_context(_shadow(transformer, "vision_cache", {}))
                ids = clip.generate(self.tokens[0], **call)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
        finally:
            self.times["end"] = time.perf_counter()
            if flags.get("graph_dynamic_vbar_blocks"):
                _release_graphs()
        if self.rows is None:
            # the wrapper never called the decoder this module replaced: these are row 0's ids, its own way
            if len(self.tokens) > 1:
                raise _Unbatchable("the model did not call the decoder this node batches")
            return [ids]
        return self.rows.results()


def _release_graphs():
    """Drop the decode graphs, prefetch queues and cast buffers a run that switched graphs on left behind."""
    import comfy.memory_management

    if not getattr(comfy.memory_management, "aimdo_enabled", False):
        return
    import comfy.model_prefetch
    import comfy_aimdo.model_vbar

    # captured graphs hold the finished generation's cache addresses; the next one allocates anew. Called
    # from `finally` blocks, so a failure here is logged rather than raised over the error leaving them.
    try:
        comfy.model_prefetch.cleanup_prefetch_queues()
        comfy.model_management.reset_cast_buffers()
        comfy_aimdo.model_vbar.vbars_reset_watermark_limits()
    except Exception as e:
        logger.warning("RulZ Generate Text could not release the decode graphs: %s", e)


def _generate_one(clip, tokens, seed, options, flags):
    """One row ComfyUI's own way, sampler included, under the same switches as the batches."""
    _, model = _decoder_of(clip)
    with contextlib.ExitStack() as stack:
        for name, value in flags.items():
            stack.enter_context(_shadow(model, name, value))
        try:
            return clip.generate(_fresh(tokens), seed=seed, **options)
        finally:
            if flags.get("graph_dynamic_vbar_blocks"):
                _release_graphs()


def _row_key(tokens):
    """(media signature, token count) of a tokenized prompt, to plan batches before embedding anything."""
    batch = next(iter(tokens.values())) if isinstance(tokens, dict) else tokens
    row = batch[-1] if len(batch) > 0 else []
    media = []
    for item in row:
        value = item[0] if isinstance(item, tuple) else item
        if not isinstance(value, (int, float)):
            data = value.get("data") if isinstance(value, dict) else value
            media.append(tuple(data.shape) if torch.is_tensor(data) else type(data).__name__)
    return tuple(media), len(row)


def _rows_that_fit(clip, model, keys, max_length, batch_size):
    """How many rows the key/value cache leaves room for, between 1 and `batch_size`."""
    device = clip.patcher.load_device
    cfg = getattr(model, "config", None)
    if device.type != "cuda" or cfg is None or not keys:
        return batch_size
    try:
        longest = max(count + MEDIA_TOKENS_GUESS * len(media) for media, count in keys)
        # keys and values, two bytes each, for every layer and every position a row may reach
        per_row = cfg.num_hidden_layers * 2 * cfg.num_key_value_heads * cfg.head_dim * 2 * (longest + max_length)
        unloaded = max(0, clip.patcher.model_size() - clip.patcher.loaded_size())
        free = comfy.model_management.get_free_memory(device) - unloaded
        budget = max(KV_VRAM_FLOOR, KV_VRAM_SHARE * free)
        return int(max(1, min(batch_size, budget // per_row)))
    except Exception as e:
        logger.debug("could not size the batch from free VRAM: %s", e)
        return batch_size


def _plan(keys, batch_size, ragged=True):
    """Split row indices into batches: rows with the same images together, text rows by length.

    With `ragged` off (no fixed-cache kernel to mask padded rows), text rows batch only with rows of
    exactly their length.
    """
    text = sorted((i for i, k in enumerate(keys) if not k[0]), key=lambda i: keys[i][1])
    groups = [text[i:i + batch_size] for i in range(0, len(text), batch_size)] if ragged else []
    media = {}
    for i, k in enumerate(keys):
        if k[0] or not ragged:
            media.setdefault(k, []).append(i)
    for rows in media.values():
        groups += [rows[i:i + batch_size] for i in range(0, len(rows), batch_size)]
    return groups


def generate_rows(clip, tokens, seeds, options, batch_size=8):
    """Write one answer per row.

    Args:
        clip: A ComfyUI CLIP holding a language model.
        tokens: One `clip.tokenize` result per row.
        seeds: One seed per row.
        options: Keyword arguments of `clip.generate` other than the tokens and the seed.
        batch_size: The most rows decoded at once.

    Returns:
        (ids per row, report dict).
    """
    transformer, model = _decoder_of(clip)
    refusal = _batch_refusal(transformer, model)
    flags, fast_note = {}, ""
    if not refusal:
        clip, flags, fast_note = _prepare_fast(clip)
        transformer, model = _decoder_of(clip)
    results = [None] * len(tokens)
    report = {"rows": len(tokens), "batches": 0, "largest": 0, "tokens": 0, "prompt_seconds": 0.0, "decode_seconds": 0.0,
              "steps": 0, "seconds": 0.0, "notes": []}

    def note(text):
        if text and text not in report["notes"]:
            report["notes"].append(text)

    started = time.perf_counter()
    if refusal:
        if len(tokens) > 1:
            note(f"one row at a time: {refusal}")
        for i, (t, seed) in enumerate(zip(tokens, seeds)):
            results[i] = clip.generate(_fresh(t), seed=seed, **options)
            report["tokens"] += len(results[i])
            comfy.model_management.throw_exception_if_processing_interrupted()
        report["batches"] = len(tokens)
        report["largest"] = 1 if tokens else 0
        report["seconds"] = time.perf_counter() - started
        return results, report

    note(fast_note)
    keys = [_row_key(t) for t in tokens]
    asked = max(1, int(batch_size))
    fit = _rows_that_fit(clip, model, keys, int(options["max_length"]), asked)
    if fit < min(asked, len(tokens)):
        note(f"batches of {fit}: the key/value cache for {min(asked, len(tokens))} rows of max_length would not fit in free VRAM")
    device = clip.patcher.load_device
    ragged = bool(flags.get("fixed_kv", getattr(model, "fixed_kv", False))) and _flash_decode(model, device)
    exact = len(tokens) == 1
    pending = _plan(keys, fit, ragged)
    while pending:
        rows = pending.pop(0)
        group = _Group(clip, [tokens[i] for i in rows], [seeds[i] for i in rows], dict(options, seed=seeds[rows[0]]), flags, exact)
        try:
            ids = group.run()
        except _Unbatchable as e:
            if len(rows) == 1:
                # one row that this node cannot batch even with itself: let ComfyUI write it its own way
                note(f"ComfyUI's own decode for a row: {e}")
                ids = [_generate_one(clip, tokens[rows[0]], seeds[rows[0]], options, flags)]
                group.rows = None
            else:
                lengths = {}
                for i in rows:
                    lengths.setdefault(keys[i], []).append(i)
                if isinstance(e, _Ragged) and len(lengths) > 1:
                    note(f"rows of one length per batch: {e}")
                    pending = list(lengths.values()) + pending
                else:
                    note(f"one row at a time: {e}")
                    pending = [[i] for i in rows] + pending
                continue
        except comfy.model_management.OOM_EXCEPTION:
            if len(rows) == 1:
                raise
            # not enough VRAM for this many rows at once: free what the attempt held and halve every batch left
            del group
            comfy.model_management.soft_empty_cache()
            half = (len(rows) + 1) // 2
            note(f"out of VRAM at {len(rows)} rows, batches of {half} from there")
            pending = [g[j:j + half] for g in [rows] + pending for j in range(0, len(g), half)]
            continue
        for i, row_ids in zip(rows, ids):
            results[i] = row_ids
            report["tokens"] += len(row_ids)
        report["batches"] += 1
        report["largest"] = max(report["largest"], len(rows))
        note(group.note)
        if group.rows is not None and group.rows.first_token_time is not None:
            report["prompt_seconds"] += group.rows.first_token_time - group.times.get("decoder", group.times["start"])
            report["decode_seconds"] += group.times["end"] - group.rows.first_token_time
            report["steps"] += max(0, group.rows.step - 1)
        comfy.model_management.throw_exception_if_processing_interrupted()
    report["seconds"] = time.perf_counter() - started
    return results, report


def split_reasoning(clip, tokens, prompt, text):
    """(answer, thinking) of a decoded generation, as core Generate Text separates them."""
    head, mark, tail = text.partition(THINK_CLOSE)
    if mark:
        return tail.strip(), head.replace(THINK_OPEN, "", 1).strip()
    opened = head.lstrip().startswith(THINK_OPEN) or str(prompt or "").rstrip().endswith(THINK_OPEN)
    if not opened:
        try:
            batch = next(iter(tokens.values())) if isinstance(tokens, dict) else tokens
            ids = [t[0] for t in batch[-1] if isinstance(t, tuple) and isinstance(t[0], int)]
            opened = clip.decode(ids[-4:]).rstrip().endswith(THINK_OPEN)
        except Exception as e:
            logger.debug("could not decode the end of the prompt: %s", e)
    if opened:
        # thinking the length limit cut off before it closed is all thinking and no answer
        return "", head.replace(THINK_OPEN, "", 1).strip()
    return text.strip(), ""


def _status(report, variations_note=""):
    """One line for the node: what was written, how fast, and anything worth knowing."""
    tokens, rows = report["tokens"], report["rows"]
    parts = [f"{rows} answer{'s' if rows != 1 else ''}, {tokens} tokens in {report['seconds']:.1f}s"]
    if report["decode_seconds"] > 0 and report["steps"] > 0:
        steps = report["steps"] / report["decode_seconds"]
        total = max(0, tokens - rows) / report["decode_seconds"]
        if report["largest"] > 1:
            parts.append(f"{total:.0f} tokens/s over batches of up to {report['largest']} ({steps:.0f} steps/s)")
        else:
            parts.append(f"{steps:.0f} tokens/s")
    elif report["seconds"] > 0 and tokens:
        parts.append(f"{tokens / report['seconds']:.0f} tokens/s overall")
    notes = list(report["notes"])
    if variations_note:
        notes.append(variations_note)
    return "  |  ".join(parts + notes)


def _at(values, index, default=None):
    """Item `index` of a list input, repeating the last item as ComfyUI does for shorter lists."""
    if not values:
        return default
    return values[min(index, len(values) - 1)]


class RulZGenerateText:
    """Generate Text that writes several answers in one pass. See the module docstring."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP", {"tooltip": "A CLIP holding a language model, such as qwen3vl_4b from Load CLIP with type krea2."}),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True, "default": "",
                                      "tooltip": "What to ask the model. Wire a list of prompts to write an answer for each of them in one pass."}),
                "max_length": ("INT", {"default": 512, "min": 1, "max": 32768,
                                       "tooltip": "The most tokens to write per answer. A token is about three quarters of a word."}),
                "sampling": (["on", "off"], {"default": "on",
                                             "tooltip": "on draws each token at random within the limits below. off always takes the likeliest token."}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.01, "max": 2.0, "step": 0.01,
                                          "tooltip": "How adventurous each pick is. 0.7 is balanced, 0.2 stays close to the likeliest words."}),
                "top_k": ("INT", {"default": 64, "min": 0, "max": 1000, "tooltip": "Pick only among this many likeliest tokens. 0 turns the limit off."}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01,
                                    "tooltip": "Pick only among the likeliest tokens whose chances add up to this. 1.0 turns it off."}),
                "min_p": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01,
                                    "tooltip": "Drop any token less likely than this fraction of the likeliest one. 0 turns it off."}),
                "repetition_penalty": ("FLOAT", {"default": 1.05, "min": 0.0, "max": 5.0, "step": 0.01,
                                                 "tooltip": "Above 1.0 makes a token already used less likely again. 1.0 turns it off."}),
                "presence_penalty": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 5.0, "step": 0.01,
                                               "tooltip": "Subtracted from every token already used. 0 turns it off."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": MAX_SEED, "control_after_generate": True,
                                 "tooltip": "The same seed, settings and batch write the same text. Variation n of a prompt uses seed + n."}),
                "variations": ("INT", {"default": 1, "min": 1, "max": 64,
                                       "tooltip": "How many answers to write for each prompt, all at once. Needs sampling on."}),
                "batch_size": ("INT", {"default": 8, "min": 1, "max": 64,
                                       "tooltip": "The most answers written at once. More is faster in total until VRAM runs out; a batch that does not fit is halved and retried."}),
            },
            "optional": {
                "image": ("IMAGE", {"tooltip": "A picture to ask about, for a model that reads images."}),
                "image_mode": (IMAGE_MODES, {"default": IMAGE_MODES[0],
                                             "tooltip": "What an image batch means: every image in one prompt, or the prompt asked once per image."}),
                "video": ("IMAGE", {"tooltip": "Video frames as an image batch, read as 24 fps."}),
                "audio": ("AUDIO", {"tooltip": "Sound to ask about, for a model that hears it."}),
                "thinking": ("BOOLEAN", {"default": False, "tooltip": "Let a model that reasons think before it answers."}),
                "use_default_template": ("BOOLEAN", {"default": True, "advanced": True,
                                                     "tooltip": "Wrap the prompt in the model's own chat template and system prompt."}),
                "mtp": (MTP_CHOICES, {"default": "auto", "advanced": True,
                                      "tooltip": "Multi-token prediction for a checkpoint that carries those heads. Such a model writes one row at a time."}),
                "system_prompt": ("STRING", {"forceInput": True,
                                             "tooltip": "Replaces the system prompt in the model's chat template. Read only while use_default_template is on."}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("generated_text", "thinking")
    OUTPUT_IS_LIST = (True, True)
    INPUT_IS_LIST = True
    OUTPUT_NODE = True
    FUNCTION = "generate"
    CATEGORY = "RulZnodes"
    DESCRIPTION = ("Write text with the language model inside a loaded CLIP, as core Generate Text does, but several answers "
                   "in one pass: a list of prompts, one answer per image, or several variations of one prompt. Decoding "
                   "eight answers together takes about as long as one.")

    def generate(self, clip, prompt, max_length, sampling, temperature, top_k, top_p, min_p, repetition_penalty,
                 presence_penalty, seed, variations, batch_size, image=None, image_mode=None, video=None, audio=None,
                 thinking=None, use_default_template=None, mtp=None, system_prompt=None):
        clip = _at(clip, 0)
        if clip is None:
            raise ValueError("RulZ Generate Text needs a CLIP. Connect Load CLIP with a language model such as qwen3vl_4b.")
        do_sample = _at(sampling, 0, "on") == "on"
        template = bool(_at(use_default_template, 0, True))
        per_image = _at(image_mode, 0, IMAGE_MODES[0]) == IMAGE_MODES[1]
        mtp_choice = _at(mtp, 0, "auto")
        count = max(1, int(_at(variations, 0, 1)))
        variations_note = ""
        if count > 1 and not do_sample:
            count = 1
            variations_note = "variations need sampling on"

        # One job per item of the list inputs, paired up by position as ComfyUI pairs lists (a shorter list
        # repeats its last item), and per image when asked. The settings above use their first value: rows
        # with different sampling settings could not share a batch.
        think = bool(_at(thinking, 0, False))
        jobs = []
        paired = (prompt, image, video, audio, system_prompt, seed)
        for i in range(max(1, *(len(values or []) for values in paired))):
            text = _at(prompt, i, "") or ""
            picture = _at(image, i)
            first_seed = int(_at(seed, i, 0))
            extra = dict(skip_template=not template, min_length=1, thinking=think, video=_at(video, i), audio=_at(audio, i),
                         system_prompt=str(_at(system_prompt, i, "") or "") if template else "")
            if picture is not None and per_image and picture.shape[0] > 1:
                jobs += [(text, picture[j:j + 1], extra, first_seed) for j in range(picture.shape[0])]
            else:
                jobs.append((text, picture, extra, first_seed))

        tokens, seeds, owners = [], [], []
        for index, (text, picture, extra, first_seed) in enumerate(jobs):
            row = clip.tokenize(text, image=picture, **extra)
            for n in range(count):
                tokens.append(row)
                seeds.append((first_seed + n) & MAX_SEED)
                owners.append(index)

        options = dict(do_sample=do_sample, max_length=int(_at(max_length, 0, 512)),
                       temperature=float(_at(temperature, 0, 1.0)) if do_sample else 1.0,
                       top_k=int(_at(top_k, 0, 50)) if do_sample else 50,
                       top_p=float(_at(top_p, 0, 1.0)) if do_sample else 1.0,
                       min_p=float(_at(min_p, 0, 0.0)) if do_sample else 0.0,
                       repetition_penalty=float(_at(repetition_penalty, 0, 1.0)),
                       presence_penalty=float(_at(presence_penalty, 0, 0.0)),
                       mtp={"off": False, "auto": True}.get(mtp_choice, mtp_choice))

        ids, report = generate_rows(clip, tokens, seeds, options, batch_size=int(_at(batch_size, 0, 8)))

        answers, thoughts = [], []
        for row_ids, row_tokens, owner in zip(ids, tokens, owners):
            answer, thought = split_reasoning(clip, row_tokens, jobs[owner][0], clip.decode(row_ids))
            answers.append(answer)
            thoughts.append(thought)

        status = _status(report, variations_note)
        logger.info("RulZ Generate Text: %s", status)
        return {"ui": {"text": [status], "generated": answers}, "result": (answers, thoughts)}


NODE_CLASS_MAPPINGS = {"RulZGenerateText": RulZGenerateText}
NODE_DISPLAY_NAME_MAPPINGS = {"RulZGenerateText": "RulZ Generate Text"}
