"""Block-sparse SageAttention for Qwen Image 2.1.

While sampling, Qwen Image 2.1 caches the text and reference-image keys, so from the second step on the
target image's queries attend [text, references, target]. Most of that attention lands on a minority of
64-token key blocks. On the second step this patch estimates, per layer, head and 128-query block, the
fewest key blocks that hold `tau` of the attention mass, always keeps the text blocks, and reuses that
block list for the rest of the sampling run. SageAttention's own kernel, changed to walk the list
(sparse_sage/), computes only those blocks, so a full list gives sageattn's exact output.

Everything else runs dense through the attention override already on the model (Patch Sage Attention KJ,
say) or sageattn: the first step, masked prefix attention, the dense_layers, layers whose estimate keeps
most blocks anyway, and any model that is not Qwen Image 2.1.
"""
import logging
import re

import torch

import comfy.model_management
import comfy.patcher_extension

from .sparse_sage import BK, mask_to_lut, predict, sage_nhd, sageattn, sparse_sageattn, unsupported_reason

try:
    from comfy.ldm.qwen_image21.model import QwenImage21Transformer2DModel
except ImportError:  # ComfyUI without Qwen Image 2.1
    QwenImage21Transformer2DModel = None

KEY = "rulz_qwen_sparse_sage"
MAX_DENSITY = 0.85  # above this the block list saves too little over dense Sage


def parse_layers(text):
    """'0, 30-31' -> {0, 30, 31}."""
    layers = set()
    for part in re.findall(r"\d+\s*-\s*\d+|\d+", text or ""):
        if "-" in part:
            a, b = (int(x) for x in part.split("-"))
            layers.update(range(min(a, b), max(a, b) + 1))
        else:
            layers.add(int(part))
    return layers


def sequence_layout(x, context, ref_latents, image_slots):
    """(target tokens, total tokens, text key blocks) of the joint sequence Qwen Image 2.1 builds:
    text with each reference spliced in at its slot, then the target (QwenImage21Transformer2DModel.build_sequence)."""
    refs = [r.shape[-2] * r.shape[-1] for r in (ref_latents or [])]
    ctx = context.shape[1]
    slots = (list(image_slots or []) + [ctx] * len(refs))[:len(refs)]
    bounds = [0] + slots + [ctx]
    blocks, pos = set(), 0
    for i, (start, end) in enumerate(zip(bounds[:-1], bounds[1:])):
        if end > start:
            blocks.update(range(pos // BK, (pos + end - start + BK - 1) // BK))
            pos += end - start
        if i < len(refs):
            pos += refs[i]
    target = x.shape[-2] * x.shape[-1]
    return target, pos + target, sorted(blocks)


class SparsePatch:
    """Options and per-sampling-run state; the ON_CLEANUP callback drops the block lists after each run."""

    def __init__(self, tau, dense_layers, previous):
        self.tau = tau
        self.dense_layers = dense_layers
        self.previous = previous
        self.layout = None
        self.masks = {}
        self.logged = False
        self.warned = False

    def reset(self):
        self.masks = {}
        self.layout = None
        self.logged = False

    def forward(self, executor, args, kwargs):
        x, context, ref_latents, image_slots, transformer_options = args[0], args[2], args[3], args[4], args[5]
        sigmas, schedule = transformer_options.get("sigmas"), transformer_options.get("sample_sigmas")
        # the first step fills the text/reference K/V cache and its target attention stays dense
        first = sigmas is None or schedule is None or float(sigmas[0]) >= float(schedule[0])
        if first:
            self.reset()
        else:
            self.layout = sequence_layout(x, context, ref_latents, image_slots)
        try:
            return executor(*args, **kwargs)
        finally:
            self.layout = None
            if self.masks and not self.logged:
                self.logged = True
                sparse = [e[2] for e in self.masks.values() if e]
                if sparse:
                    logging.info(f"RulZ Qwen Sparse Sage: {len(sparse)} attention layers sparse at {sum(sparse) / len(sparse):.0%} "
                                 f"of key blocks on average, {len(self.masks) - len(sparse)} dense")

    def attention(self, func, q, k, v, heads, mask, kwargs):
        if mask is None and self.layout is not None and q.dim() == 3 and not kwargs.get("skip_reshape", False):
            out = self.sparse(q, k, v, heads, kwargs.get("transformer_options") or {})
            if out is not None:
                return out
        if self.previous is not None:
            return self.previous(func, q, k, v, heads, mask=mask, **kwargs)
        if mask is None and sageattn is not None and q.dim() == 3 and not kwargs.get("skip_reshape", False) and q.dtype in (torch.float16, torch.bfloat16):
            return sage_nhd(q, k, v, heads)
        return func(q, k, v, heads, mask=mask, **kwargs)

    def sparse(self, q, k, v, heads, transformer_options):
        target, total, text_blocks = self.layout
        layer = transformer_options.get("block_index")
        b, nq, c = q.shape
        nk = k.shape[1]
        d = c // heads
        if nq != target or nk != total or layer in self.dense_layers or q.dtype not in (torch.float16, torch.bfloat16) or d not in (64, 128):
            return None
        key = (layer, b, nq, nk)
        entry = self.masks.get(key)
        qn, kn, vn = q.view(b, nq, heads, d), k.view(b, nk, heads, d), v.view(b, nk, heads, d)
        if entry is None:
            entry = False
            reason = unsupported_reason(q.device)
            if reason is not None:
                if not self.warned:
                    self.warned = True
                    logging.warning(f"RulZ Qwen Sparse Sage: running dense, {reason}")
            else:
                m = predict(qn, kn, self.tau, text_blocks)
                density = m.float().mean().item()
                if density <= MAX_DENSITY:
                    entry = mask_to_lut(m) + (density,)
            self.masks[key] = entry
        if not entry:
            return None
        return sparse_sageattn(qn, kn, vn, entry[0], entry[1]).reshape(b, nq, c)


class QwenSparseSageAttention:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "tau": ("FLOAT", {"default": 0.99, "min": 0.9, "max": 1.0, "step": 0.001,
                                  "tooltip": "Share of the estimated attention each query block keeps. Lower skips more blocks: faster, less exact."}),
                "dense_layers": ("STRING", {"default": "0",
                                            "tooltip": "Transformer blocks that always run dense, e.g. '0, 30-31'. Block 0 attends almost everywhere."}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "RulZnodes"

    def patch(self, model, tau=0.99, dense_layers="0"):
        m = model.clone()
        transformer_options = m.model_options.setdefault("transformer_options", {})
        patch = SparsePatch(tau, parse_layers(dense_layers), transformer_options.get("optimized_attention_override"))

        # closures, not bound methods: a deep copy of the model options must keep sharing this patch's state
        def override(func, q, k, v, heads, mask=None, **kwargs):
            return patch.attention(func, q, k, v, heads, mask, kwargs)

        def wrapper(executor, *args, **kwargs):
            return patch.forward(executor, args, kwargs)

        transformer_options["optimized_attention_override"] = override
        if QwenImage21Transformer2DModel is None or not isinstance(m.get_model_object("diffusion_model"), QwenImage21Transformer2DModel):
            logging.info("RulZ Qwen Sparse Sage: not a Qwen Image 2.1 model, attention runs dense")
            return (m,)
        # load (or on first use build) the kernel now rather than in the middle of the second sampling step
        reason = unsupported_reason(comfy.model_management.get_torch_device())
        if reason is not None:
            logging.warning(f"RulZ Qwen Sparse Sage: running dense, {reason}")
            return (m,)
        m.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, KEY, wrapper)
        m.add_callback_with_key(comfy.patcher_extension.CallbacksMP.ON_CLEANUP, KEY, lambda model_patcher: patch.reset())
        return (m,)


NODE_CLASS_MAPPINGS = {"QwenSparseSageAttention": QwenSparseSageAttention}
NODE_DISPLAY_NAME_MAPPINGS = {"QwenSparseSageAttention": "RulZ Qwen Sparse Sage Attention"}
