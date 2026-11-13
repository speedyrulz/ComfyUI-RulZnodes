"""Block-sparse SageAttention 2 (int8 QK, fp8 PV) for NVIDIA sm89 / sm120 GPUs.

csrc/ is SageAttention's sm89 kernel (Apache-2.0, see LICENSE) changed to walk a per-query-block list of
64-key blocks. It is built on first use with the CUDA toolkit; quantization reuses the installed
sageattention 2 package, so a full block list is as accurate as sageattn: the same error against exact
attention, though the outputs are not bit-identical.
"""
import logging
import os

import torch
import torch.nn.functional as F

BQ, BK = 128, 64
# sm89 (Ada) and sm120/121 (consumer Blackwell) run SageAttention's sm89 kernels
ARCHS = {(8, 9): "compute_89,code=sm_89", (12, 0): "compute_120a,code=sm_120a", (12, 1): "compute_121a,code=sm_121a"}
CSRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csrc")

try:
    from sageattention import sageattn
    from sageattention.quant import per_channel_fp8, per_warp_int8
    SAGE_ERROR = None
except ImportError as e:
    sageattn = None
    SAGE_ERROR = f"SageAttention 2 is not installed ({e})"

_ext = None
_ext_error = None


def unsupported_reason(device):
    """Why the sparse kernel can't run on device, or None."""
    if SAGE_ERROR:
        return SAGE_ERROR
    if device.type != "cuda":
        return f"device {device} is not CUDA"
    cap = torch.cuda.get_device_capability(device)
    if cap not in ARCHS:
        return f"{torch.cuda.get_device_name(device)} is sm{cap[0]}{cap[1]}, not sm89 or sm120"
    load()
    return _ext_error


def load():
    """Builds (first use only, a few minutes) and loads the extension for every supported GPU present."""
    global _ext, _ext_error
    if _ext is not None or _ext_error is not None:
        return _ext
    from torch.utils.cpp_extension import load as load_extension
    caps = sorted({torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count())} & ARCHS.keys())
    name = "rulz_sparse_sage_" + "_".join(f"sm{a}{b}" for a, b in caps)
    cflags = ["/O2", "/std:c++20"] if os.name == "nt" else ["-O3", "-std=c++20"]
    logging.info(f"RulZ Qwen Sparse Sage: loading kernel {name}; the first use builds it, which takes a few minutes")
    try:
        _ext = load_extension(
            name=name,
            sources=[os.path.join(CSRC, "qattn", "sparse_attn.cu")],
            extra_cflags=cflags,
            extra_cuda_cflags=["-O3", "-std=c++20", "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__",
                               "--use_fast_math", "-diag-suppress=174"] + [f"-gencode=arch={ARCHS[c]}" for c in caps],
        )
    except Exception as e:  # no nvcc / host compiler, or the build failed: attention stays dense
        _ext_error = f"the sparse kernel could not be built ({e})"
        logging.warning(f"RulZ Qwen Sparse Sage: {_ext_error}; attention runs dense")
    return _ext


def sage_dense(q, k, v, heads, skip_output_reshape=False):
    """sageattn as Patch Sage Attention KJ's 'auto' mode calls it, on (B, N, heads * D) tensors or skip_reshape
    (B, H, N, D) ones (Flux). Returns (B, N, heads * D), or (B, H, N, D) with skip_output_reshape."""
    if q.dim() == 4:
        out = sageattn(q, k, v, tensor_layout="HND", is_causal=False)
        return out if skip_output_reshape else out.transpose(1, 2).flatten(2)
    b, nq, c = q.shape
    d = c // heads
    out = sageattn(q.view(b, nq, heads, d), k.view(b, -1, heads, d), v.view(b, -1, heads, d), tensor_layout="NHD", is_causal=False)
    return out.transpose(1, 2) if skip_output_reshape else out.reshape(b, nq, c)


def sparse_sageattn(q, k, v, lut, count):
    """q, k, v: (B, N, H, D) fp16/bf16. sageattn's sm89/sm120 quantization: per-warp int8 QK, fp8 PV."""
    km = k.mean(dim=1, keepdim=True)
    q_int8, q_scale, k_int8, k_scale = per_warp_int8(q, k, km, BLKQ=BQ, WARPQ=32, BLKK=BK, tensor_layout="NHD")
    v_fp8, v_scale, _ = per_channel_fp8(v, tensor_layout="NHD", scale_max=2.25, smooth_v=False)
    o = torch.empty_like(q)
    _ext.qk_int8_sv_f8_sparse_attn(q_int8, k_int8, v_fp8, o, q_scale, k_scale, v_scale, lut, count, q.shape[-1] ** -0.5)
    return o


def block_mean(x, bs):
    # (B, N, H, D) -> (B, H, ceil(N / bs), D) fp32 block means, plus the token count of each block
    B, N, H, D = x.shape
    nb = (N + bs - 1) // bs
    pad = nb * bs - N
    s = F.pad(x, (0, 0, 0, 0, 0, pad)).view(B, nb, bs, H, D).sum(2, dtype=torch.float32)
    cnt = torch.full((nb,), float(bs), device=x.device)
    cnt[-1] = bs - pad
    return (s / cnt[None, :, None, None]).transpose(1, 2), cnt


def predict(q, k, tau, sink_blocks, group=8, head_chunk=8):
    """(B, H, nqb, nkb) bool: per 128-query block, the fewest 64-key blocks whose estimated softmax mass
    reaches tau. The mass comes from per-group query means against per-block key means, weighted by block
    size; sink_blocks are always kept."""
    B, nq, H, D = q.shape
    nqb = (nq + BQ - 1) // BQ
    kb, kcnt = block_mean(k, BK)
    qg, _ = block_mean(q, group)
    qg = F.pad(qg, (0, 0, 0, nqb * (BQ // group) - qg.shape[2]))
    bias = kcnt.log()
    est = torch.empty(B, H, nqb, kb.shape[2], device=q.device)
    for h0 in range(0, H, head_chunk):
        logits = torch.matmul(qg[:, h0:h0 + head_chunk].to(q.dtype), (kb[:, h0:h0 + head_chunk] * D ** -0.5).to(q.dtype).transpose(-1, -2))
        p = torch.softmax(logits.float() + bias, dim=-1)
        est[:, h0:h0 + head_chunk] = p.view(B, p.shape[1], nqb, BQ // group, -1).sum(3)
    srt, idx = est.sort(-1, descending=True)
    keep = (srt.cumsum(-1) - srt) < tau * est.sum(-1, keepdim=True)
    mask = torch.zeros_like(keep).scatter_(-1, idx, keep)
    mask[..., sink_blocks] = True
    return mask


def mask_to_lut(mask):
    """(B, H, nqb, nkb) bool -> ascending int32 block ids (B, H, nqb, max count) and per-row counts."""
    nkb = mask.shape[-1]
    idx = torch.arange(nkb, device=mask.device, dtype=torch.int32)
    order = torch.where(mask, idx, idx + nkb).sort(-1).values
    count = mask.sum(-1, dtype=torch.int32)
    return order[..., :int(count.max())].contiguous(), count.contiguous()
