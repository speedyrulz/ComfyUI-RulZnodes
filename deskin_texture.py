"""
RulZ DeskinTexture: removes the honeycomb / "orange-peel" skin texture that
image-restoration models (Qwen-Image etc.) sometimes paint onto skin. Runs on
torch, on the GPU when available, with a CPU fallback on out-of-memory.

  1. detect_band()      finds the frequency band the fake texture lives in
  2. band_reject()      cuts only that band (soft annular notch, Gaussian in
                        octaves, so DC / overall tone is never touched)
  3. auto_skin_mask()   rough skin-colour mask so eyes/hair/background survive
  4. regrain()          refills the band with random (non-lattice) texture +
                        fine grain, luminance only

All image tensors here are [B, C, H, W] float32 in 0..1 unless stated.
Based on ComfyUI-DeSkin 1.4.0 (MIT).
"""
import math

import numpy as np
import torch
import torch.nn.functional as F

try:
    import comfy.model_management as mm
    _DEVICE = mm.get_torch_device()
except Exception:                       # running outside ComfyUI (tests)
    _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# torch.OutOfMemoryError (torch 2.5+) also covers non-CUDA backends.
_OOM_ERROR = getattr(torch, "OutOfMemoryError", torch.cuda.OutOfMemoryError)


# ----------------------------------------------------------------------------
# frequency helpers (rfft2 layout: last dim is W//2+1)
# ----------------------------------------------------------------------------
def freq_radius(h, w, device):
    fy = torch.fft.fftfreq(h, device=device)[:, None]
    fx = torch.fft.rfftfreq(w, device=device)[None, :]
    return torch.sqrt(fy * fy + fx * fx)


def _reflect_pad(x, pad):
    """Reflect-pad [B,C,H,W] by `pad` on all sides (clamped to what reflect allows)."""
    h, w = x.shape[-2:]
    pad = int(min(pad, h - 1, w - 1))
    if pad <= 0:
        return x, 0
    return F.pad(x, (pad, pad, pad, pad), mode="reflect"), pad


def gaussian_blur(x, sigma):
    """Gaussian blur [B,C,H,W] via FFT with reflect padding."""
    if sigma <= 0:
        return x.clone()
    h, w = x.shape[-2:]
    p, pad = _reflect_pad(x, int(3 * sigma) + 1)
    G = torch.exp(-2 * (math.pi * sigma) ** 2 * freq_radius(p.shape[-2], p.shape[-1], x.device) ** 2)
    out = torch.fft.irfft2(torch.fft.rfft2(p) * G, s=p.shape[-2:])
    return out[..., pad:pad + h, pad:pad + w]


def band_reject_filter(h, w, f0, sigma_oct, strength, device):
    fr = freq_radius(h, w, device)
    d = torch.log2(torch.clamp(fr, min=1e-9) / f0)
    notch = torch.exp(-0.5 * (d / sigma_oct) ** 2)
    notch[fr < 1e-9] = 0.0
    return 1.0 - strength * notch


GRID_FREQS = (0.5, 0.25, 0.125)      # 2 px, 4 px, 8 px pixel-aligned lines (VAE / 8x upsampler checkerboard)


def grid_notch_filter(h, w, strength, device, sigma_f=None):
    """Narrow notches at the pixel-grid frequencies on the rfft2 grid: exact 2/4/8 px
    vertical, horizontal and diagonal lines.  Real photos have almost no energy at
    these exact points, so this kills the 'screen door' without blurring anything."""
    fy = torch.fft.fftfreq(h, device=device)[:, None]
    fx = torch.fft.rfftfreq(w, device=device)[None, :]
    keep = torch.ones(h, w // 2 + 1, device=device)
    pts = [(py, px) for py in (0.0,) + GRID_FREQS for px in (0.0,) + GRID_FREQS if (py, px) != (0.0, 0.0)]
    for py, px in pts:
        # the 2 px and 4 px lines are razor-sharp spectral spikes; the 8 px one is
        # broader (it is modulated by the image), so it gets a wider notch
        sf = sigma_f if sigma_f else (0.016 if max(py, px) <= 0.125 else 0.007)
        for sy in ((py, -py) if py else (0.0,)):
            d2 = (fy - sy) ** 2 + (fx - px) ** 2
            keep = keep * (1.0 - strength * torch.exp(-0.5 * d2 / sf ** 2))
    return keep


def apply_grid_notch(x, strength):
    """Remove the 2/4/8 px pixel grid from [B,C,H,W]."""
    h, w = x.shape[-2:]
    p, pad = _reflect_pad(x, 32)
    G = grid_notch_filter(p.shape[-2], p.shape[-1], strength, x.device)
    out = torch.fft.irfft2(torch.fft.rfft2(p) * G, s=p.shape[-2:])
    return out[..., pad:pad + h, pad:pad + w]


def band_reject(x, f0, sigma_oct, strength):
    """Band-reject every channel of [B,C,H,W]."""
    h, w = x.shape[-2:]
    p, pad = _reflect_pad(x, max(32, int(8 / f0)))
    H = band_reject_filter(p.shape[-2], p.shape[-1], f0, sigma_oct, strength, x.device)
    out = torch.fft.irfft2(torch.fft.rfft2(p) * H, s=p.shape[-2:])
    return out[..., pad:pad + h, pad:pad + w]


# ----------------------------------------------------------------------------
# detection
# ----------------------------------------------------------------------------
def radial_profile(power, h, w, nbins=100):
    """power: [H, W//2+1] real.  Returns (bin centres in cycles/px, mean power)."""
    fr = freq_radius(h, w, power.device)
    idx = torch.clamp((fr * (2 * nbins)).long(), 0, nbins - 1)      # bins of 0.5/nbins
    sums = torch.bincount(idx.flatten(), weights=power.flatten(), minlength=nbins)
    cnts = torch.bincount(idx.flatten(), minlength=nbins).clamp(min=1)
    prof = (sums / cnts).cpu().numpy()
    centers = (np.arange(nbins) + 0.5) * (0.5 / nbins)
    return centers, prof


def detect_band(lum, region=None):
    """lum: [H, W] float tensor (any scale).  region: [H, W] 0..1 weight (e.g. the skin
    mask) so hair / fabric / background can't swamp the skin's spectrum.
    Returns (f0, sigma_oct, excess) for the strongest bump; excess is ln(bump / natural)."""
    h, w = lum.shape
    x = lum[None, None]
    hp = (x - gaussian_blur(x, 10.0))[0, 0]                           # drop lighting gradients
    win = torch.outer(torch.hann_window(h, periodic=False, device=lum.device),
                      torch.hann_window(w, periodic=False, device=lum.device))
    if region is not None:
        win = win * region.to(lum.device)
    power = torch.fft.rfft2(hp * win).abs() ** 2
    f, p = radial_profile(power, h, w)
    logp = np.log(p + 1e-9)
    valid = (f > 0.03) & (f < 0.48)
    # Fit the natural 1/f^k baseline (straight line in log-log) to the LOWER
    # envelope of the profile; the fake texture is whatever sticks up above it.
    xs, ys = np.log(f[valid]), logp[valid]
    keep = np.ones_like(xs, bool)
    m = c = 0.0
    for _ in range(6):
        A = np.vstack([xs[keep], np.ones(keep.sum())]).T
        m, c = np.linalg.lstsq(A, ys[keep], rcond=None)[0]
        resid = ys - (m * xs + c)
        next_keep = resid < np.percentile(resid[keep], 60)
        if next_keep.sum() < 2:
            break
        keep = next_keep
    excess = np.full_like(logp, -np.inf)
    excess[valid] = ys - (m * xs + c)
    i0 = int(np.argmax(excess))
    peak = float(excess[i0])
    half = peak / 2
    lo = i0
    while lo > 0 and excess[lo] > half:
        lo -= 1
    hi = i0
    while hi < len(f) - 1 and excess[hi] > half:
        hi += 1
    fwhm_oct = max(math.log2(f[hi]) - math.log2(f[lo]), 0.3)
    return float(f[i0]), float(fwhm_oct / 2.355), peak


# ----------------------------------------------------------------------------
# mask + grain
# ----------------------------------------------------------------------------
def auto_skin_mask(rgb):
    """rgb: [B,3,H,W] in 0..1 -> [B,1,H,W] feathered skin mask.
    Colour-only, so it is only a fallback: YCrCb skin box AND a hue/saturation
    test that rejects the near-neutral stuff (cream walls, beige knitwear, grey
    floors) the plain YCrCb box lets through."""
    r, g, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cr = (r - y) * 0.713 * 255 + 128
    cb = (b - y) * 0.564 * 255 + 128
    mx = torch.maximum(torch.maximum(r, g), b)
    mn = torch.minimum(torch.minimum(r, g), b)
    sat = (mx - mn) / mx.clamp(min=1e-6)
    # hue in degrees, only valid where r is the max (skin is always r-dominant)
    hue = 60.0 * (g - b) / (mx - mn).clamp(min=1e-6)
    m = ((cr > 135) & (cr < 175) & (cb > 80) & (cb < 125)
         & (sat > 0.22) & (sat < 0.85) & (mx > 0.2)
         & (r >= g) & (g >= b) & (hue > 2) & (hue < 30)).float()
    m = (gaussian_blur(m, 3.0) > 0.5).float()                         # despeckle
    return gaussian_blur(m, 8.0).clamp(0, 1)                          # feather


_MEDIAN_SAMPLES = 200000


def _masked_median(values, mask):
    """Median of `values` where mask > 0.5, from an evenly strided subsample of at most
    _MEDIAN_SAMPLES pixels: deterministic, and no full-size random permutation."""
    sel = values[mask > 0.5]
    if sel.numel() == 0:
        return 0.0
    if sel.numel() > _MEDIAN_SAMPLES:
        sel = sel[::math.ceil(sel.numel() / _MEDIAN_SAMPLES)]
    return sel.median().item()


def strong_edges(lum, mask, k=8.0):
    """[B,1,H,W] 0/1 map of strong edges, or None if the mask has no gradient to measure.
    'Strong' = gradient > k x the median gradient inside the mask (the honeycomb's own
    gradient is ~1/255 per px; silhouettes, hems and lips are 10-40x that)."""
    g = gaussian_blur(lum, 1.0)
    gx = torch.zeros_like(g); gy = torch.zeros_like(g)
    gx[..., :, 1:-1] = (g[..., :, 2:] - g[..., :, :-2]) * 0.5
    gy[..., 1:-1, :] = (g[..., 2:, :] - g[..., :-2, :]) * 0.5
    grad = torch.sqrt(gx * gx + gy * gy)
    med = _masked_median(grad, mask)
    if med <= 0:
        return None
    return (grad > k * med).float()


def edge_zone(strong, radius):
    """Weight that is 0 within `radius` px of a strong edge (from strong_edges) and 1
    elsewhere.  Any linear filter rings for about its own spatial extent on either side
    of such an edge, so the fix is simply not to apply it there."""
    if strong is None or radius <= 0:
        return 1.0
    z = gaussian_blur(strong, radius / 2.0)                      # soft dilation
    return (1.0 - (z / 0.02).clamp(0, 1)) ** 2                    # 0 inside the zone, 1 outside


def amplitude_core(component, mask, factor):
    """Soft-threshold a removed component: keep it where it is pattern-sized, drop it
    where it is much stronger than typical (that is an edge, not pattern)."""
    if factor <= 0:
        return component, 0.0
    amp = gaussian_blur(component.abs().mean(dim=1, keepdim=True), 1.0)
    med = _masked_median(amp, mask)
    if med <= 0:
        return component, 0.0
    keep = torch.exp(-(amp / (factor * med)) ** 2)
    held = (1 - keep[mask > 0.5]).mean().item()
    return component * keep, held


def regrain(filtered, removed, f0, sigma_oct, amount, seed):
    """Add band-limited random texture + fine grain to [B,C,H,W], luminance only.
    Amplitude is LOCAL: proportional to how much texture was removed right there,
    so grain lands where the honeycomb was and nowhere else."""
    if amount <= 0:
        return filtered
    B, C, h, w = filtered.shape
    local_rms = torch.sqrt(gaussian_blur(removed.mean(dim=1, keepdim=True) ** 2, 2.0 / f0).clamp(min=0))
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    white = torch.randn(B, 1, h, w, generator=gen).to(filtered.device)
    band = 1.0 - band_reject_filter(h, w, f0, sigma_oct, 1.0, filtered.device)
    pore = torch.fft.irfft2(torch.fft.rfft2(white) * band, s=(h, w))
    pore = pore / pore.std().clamp(min=1e-6)
    fine = gaussian_blur(torch.randn(B, 1, h, w, generator=gen).to(filtered.device), 0.5)
    fine = fine / fine.std().clamp(min=1e-6)
    noise = (0.8 * pore + 0.6 * fine) * local_rms * amount
    tone = filtered.mean(dim=1, keepdim=True).clamp(0, 1)             # more grain in mid-tones
    noise = noise * (0.6 + 0.8 * (1 - (tone - 0.5).abs() * 2))
    return filtered + noise


# ----------------------------------------------------------------------------
# one-call pipeline
# ----------------------------------------------------------------------------
def deskin(rgb, mask=None, period_px=None, band_width=1.0, strength=1.0,
           grain=0.5, seed=0, mask_mode="auto_skin", auto_threshold=1.3, edge_protect=4.0,
           screen_door=1.0):
    """
    rgb:  [B,3,H,W] float 0..1 (any device)
    mask: [B,1,H,W] float 0..1 or None (overrides mask_mode when given)
    auto_threshold: auto mode applies only if the bump is >= this many times the
                    natural level inside the mask; otherwise passes through.
    Returns (out [B,3,H,W], removed [B,3,H,W], mask_used [B,1,H,W], info str)
    """
    B, C, h, w = rgb.shape
    lum = rgb.mean(dim=1)                                             # [B,H,W]

    # 1. mask (first, because detection is measured inside it) ---------------
    if mask is not None:
        m = mask
        info = "input mask"
    elif mask_mode == "whole_image":
        m = torch.ones(B, 1, h, w, device=rgb.device)
        info = "whole image"
    else:
        m = auto_skin_mask(rgb)
        info = f"auto skin mask {m.mean().item()*100:.0f}% of frame"
    if m.mean().item() < 0.005:
        return rgb.clone(), torch.zeros_like(rgb), m, info + " - mask is empty, nothing done"

    # 2. band -----------------------------------------------------------------
    if period_px:
        f0, sigma = 1.0 / float(period_px), 0.5
        info += f" | manual cell size {period_px:.1f}px"
    else:
        dets = [detect_band(lum[i] * 255, m[i, 0]) for i in range(B)]
        f0 = float(np.median([d[0] for d in dets]))
        sigma = float(np.median([d[1] for d in dets]))
        ratio = math.exp(float(np.median([d[2] for d in dets])))
        if ratio < auto_threshold:
            return (rgb.clone(), torch.zeros_like(rgb), m,
                    info + f" | auto: strongest bump is {1/f0:.1f}px at only {ratio:.2f}x natural "
                           f"(threshold {auto_threshold:.2f}) - passed through. Lower auto_threshold, "
                           f"or use manual with period_px={1/f0:.1f}")
        sigma = max(sigma, 0.5)       # model textures are ~±1 octave wide; never go narrower than manual
        info += f" | auto: cell size ~ {1/f0:.1f}px, {ratio:.1f}x natural level"
    sigma = max(sigma * band_width, 0.1)

    # 3. filter: each artefact is isolated as its own "removed" component -----
    lum4 = lum[:, None]
    honeycomb = rgb - band_reject(rgb, f0, sigma, strength)
    grid = rgb - apply_grid_notch(rgb, screen_door) if screen_door > 0 else None

    # 3b. edge protection, two layers, applied to each component separately:
    #   (a) amplitude coring - the pattern is a few /255, real edges are tens of
    #       /255, so anything much stronger than typical is an edge: put it back;
    #   (b) an edge ZONE from the original image's gradient - a filter rings for
    #       about its own spatial extent on both sides of a strong edge, and the
    #       outer ripples are small enough to slip past (a), so within that
    #       distance of a silhouette / hem / lip the filter is simply not applied.
    #       The band-reject's extent is ~1.5 cells; the razor-thin grid notches
    #       ring for ~20 px.
    if edge_protect > 0:
        strong = strong_edges(lum4, m, k=2 * edge_protect)       # shared by both zones
        honeycomb, held = amplitude_core(honeycomb, m, edge_protect)
        honeycomb = honeycomb * edge_zone(strong, 1.5 / f0)
        info += f" | edge-protect {edge_protect:.1f}x ({held*100:.0f}% of mask held back)"
        if grid is not None:
            grid, _ = amplitude_core(grid, m, edge_protect)
            grid = grid * edge_zone(strong, 20.0)
    removed = honeycomb if grid is None else honeycomb + grid
    if grid is not None:
        info += f" | screen-door {screen_door:.2f}"
    filtered = rgb - removed

    # 4. grain ----------------------------------------------------------------
    removed_rms = math.sqrt(((removed.mean(dim=1, keepdim=True) ** 2) * m).sum().item() / max(m.sum().item(), 1))
    filtered = regrain(filtered, removed, f0, sigma, grain, seed)
    info += f" | removed RMS {removed_rms*255:.2f}/255, grain {grain:.2f}"

    out = (rgb * (1 - m) + filtered * m).clamp(0, 1)
    return out, removed * m, m, info


# ----------------------------------------------------------------------------
# node
# ----------------------------------------------------------------------------
def _prep_mask(mask, B, h, w, device):
    """ComfyUI MASK ([H,W], [B,H,W] or [B,1,H,W]) -> [B,1,H,W] float matching the image."""
    if mask.dim() == 2:
        mask = mask[None]
    if mask.dim() == 3:
        mask = mask[:, None]
    mask = mask.float().to(device)
    if mask.shape[-2:] != (h, w):
        mask = F.interpolate(mask, size=(h, w), mode="bilinear", align_corners=False)
    if mask.shape[0] != B:
        mask = mask[torch.arange(B, device=device) % mask.shape[0]]
    return mask.clamp(0, 1)


class DeSkinTexture:
    """Remove honeycomb / orange-peel skin texture left by restoration models."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "detect": (["auto", "manual"], {"default": "auto",
                           "tooltip": "auto: find the pattern's size from the image's spectrum. manual: use period_px."}),
                "period_px": ("FLOAT", {"default": 8.0, "min": 2.0, "max": 256.0, "step": 0.5,
                              "tooltip": "Pattern cell size in pixels (manual mode only)."}),
                "band_width": ("FLOAT", {"default": 1.0, "min": 0.2, "max": 3.0, "step": 0.05,
                               "tooltip": "How wide a band of detail sizes to remove, relative to detected. Lower keeps more real pores."}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                             "tooltip": "How completely the band is removed."}),
                "grain": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 2.0, "step": 0.05,
                          "tooltip": "Random texture put back, as a fraction of what was removed. 0 = none (waxy), 0.7-0.9 = more skin-like."}),
                "mask_mode": (["auto_skin", "whole_image"], {"default": "auto_skin",
                              "tooltip": "Used only when no mask is connected. auto_skin = rough skin-colour mask."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff,
                         "tooltip": "Grain pattern seed."}),
                "auto_threshold": ("FLOAT", {"default": 1.3, "min": 1.0, "max": 10.0, "step": 0.05,
                                   "tooltip": "auto mode only: apply if the detected bump is at least this many times the natural level (measured inside the mask). Lower it if auto keeps passing the image through."}),
                "screen_door": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                                "tooltip": "Removes the fine pixel-aligned 2/4/8 px grid (VAE 'screen door') inside the mask. Surgical - leaves other detail alone. 0 = off."}),
                "edge_protect": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 20.0, "step": 0.5,
                                 "tooltip": "Leave alone anything stronger than N x the typical pattern amplitude (edges, lips, eyes, hems). 0 = off. Lower protects more: lower it if real edges still soften or get a halo; raise it if strong pattern survives next to edges."}),
            },
            "optional": {
                "mask": ("MASK", {"tooltip": "Optional. White = apply fix. Overrides mask_mode. Use a face-parsing / SAM skin mask for best results."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("image", "texture_removed", "mask_used", "info")
    OUTPUT_TOOLTIPS = (
        "The fixed image.",
        "What was taken out, amplified (grey = nothing). Useful for tuning.",
        "The mask that was applied.",
        "Detected cell size and settings.",
    )
    FUNCTION = "run"
    CATEGORY = "RulZnodes"
    DESCRIPTION = ("Removes the honeycomb / orange-peel texture that restoration models "
                   "(Qwen-Image etc.) paint onto skin, by cutting only that band of detail "
                   "sizes and refilling it with natural-looking grain.")

    def run(self, image, detect, period_px, band_width, strength, grain, mask_mode, seed,
            auto_threshold=1.3, screen_door=1.0, edge_protect=4.0, mask=None):
        B, h, w, C = image.shape
        rgb = image[..., :3].permute(0, 3, 1, 2).float()           # [B,3,H,W]
        m = _prep_mask(mask, B, h, w, _DEVICE) if mask is not None else None

        def _go(device):
            out, removed, m_used, info = deskin(
                rgb.to(device), None if m is None else m.to(device),
                period_px=period_px if detect == "manual" else None,
                band_width=band_width, strength=strength, grain=grain,
                seed=seed, mask_mode=mask_mode, auto_threshold=auto_threshold,
                edge_protect=edge_protect, screen_door=screen_door)
            return out, removed, m_used, info

        result = None
        try:
            result = _go(_DEVICE)
        except _OOM_ERROR:
            pass
        if result is None:
            # Retry outside the except block: inside it, the traceback still holds the
            # failed attempt's GPU tensors and empty_cache() could not free them.
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("[DeskinTexture] out of GPU memory, running on the CPU instead")
            result = _go(torch.device("cpu"))
        out, removed, m_used, info = result

        out = out.permute(0, 2, 3, 1).cpu()
        if C == 4:                                                  # keep alpha
            out = torch.cat([out, image[..., 3:4]], dim=-1)
        removed_vis = (removed * 8 + 0.5).clamp(0, 1).permute(0, 2, 3, 1).cpu()
        mask_out = m_used[:, 0].cpu()
        print(f"[DeskinTexture] {info}")
        return (out, removed_vis, mask_out, info)


NODE_CLASS_MAPPINGS = {"DeSkinTexture": DeSkinTexture}
NODE_DISPLAY_NAME_MAPPINGS = {"DeSkinTexture": "RulZ DeskinTexture"}
