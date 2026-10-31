<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/speedyrulz/ComfyUI-RulZnodes/main/assets/logo-dark.png">
    <img alt="RulZ" src="https://raw.githubusercontent.com/speedyrulz/ComfyUI-RulZnodes/main/assets/logo.png" width="520">
  </picture>
</p>

<h1 align="center">ComfyUI-RulZnodes</h1>

<p align="center">
  Nodes for ComfyUI: load HEIC photos with their filename, step through a folder one image per run,
  crop to a mask with sensible padding, resize with megapixel and input-size targets, and turn off the
  Comfy compiler per workflow.
</p>

## Nodes

| Node | What it does |
|---|---|
| [**RulZ Load Image (HEIC)**](#rulz-load-image-heic) | `LoadImage` replacement that also loads `.heic` / `.heif` and outputs the filename. |
| [**RulZ Simple Load Image Batch**](#rulz-simple-load-image-batch) | Walks through every image in a folder, one image per queued run, and resumes where it stopped. |
| [**RulZ Mask Intelligent Crop**](#rulz-mask-intelligent-crop) | Crops an image and/or mask to the mask's bounding box plus padding, with an aspect limit and divisible-by snapping. |
| [**RulZ Resize Image Advanced**](#rulz-resize-image-advanced) | KJNodes-style Resize Image v2 with input-size passthrough, upscale-by and total-megapixel targets, and the output resolution shown on the node. |
| [**RulZ Disable Comfy Compiler**](#rulz-disable-comfy-compiler) | Per-workflow replacement for the `--disable-comfy-compiler` launch flag. |

All nodes are in the **RulZnodes** category of the node menu.

## Install

```
cd ComfyUI/custom_nodes
git clone https://github.com/speedyrulz/ComfyUI-RulZnodes
pip install -r ComfyUI-RulZnodes/requirements.txt
```

Use ComfyUI's own Python for the `pip install`. Restart ComfyUI and hard-refresh the browser.

`pillow-heif` is only needed for HEIC/HEIF. Without it the other nodes work as normal and the HEIC loader
handles the usual formats only.

> **Coming from the standalone packs?** Remove `ComfyUI-image-load-filename-heic`, `load-image-batch-simple`,
> `mask-intelligent-crop`, `resize-image-v2-advanced` and `ComfyUI-DisableComfyCompiler` from `custom_nodes`.
> The node IDs are unchanged (only the display names gained a "RulZ" prefix), so existing workflows keep
> working, but with both installed the same nodes are registered twice.

---

## RulZ Load Image (HEIC)

`LoadImageWithFilenameHEIC`: a `LoadImage` replacement that also loads `.heic` / `.heif` files and outputs the filename.

**Outputs:** `image`, `mask`, `filename` (name without directory or extension).

- The upload button's file picker is widened to allow HEIC/HEIF, and HEIC files are accepted even when
  Windows reports no MIME type for them (`web/heic_upload.js`).
- Previews: browsers can't render HEIC, so `heic_preview.py` adds a middleware that serves `.heic` / `.heif`
  files as JPEG from ComfyUI's `/view` endpoint. The conversion happens in memory; no extra files are created.
- EXIF rotation is applied, so phone photos come out upright.

## RulZ Simple Load Image Batch

`LoadImageBatchSimple`: walks through every image in a directory, one image per queued run.

| Input | Description |
|---|---|
| `directory` | Folder to read (default `/home/jeff/input`). The image count is detected automatically. |
| `reset` | On: start again from the first image. Off: resume from where the last run stopped. |
| `loop` | On: wrap around to the first image after the last one. Off: keep returning the last image. |

**Outputs:** `image`, `filename`, `total_images`, `current_index` (0-based). The node also shows
`current / total — filename` after each run.

Progress is saved per directory in `batch_state.json` in the pack folder, so queueing 5 runs and then 5 more
covers all 10 images, even across a ComfyUI restart. Files added to or removed from the folder are picked up
on the next run.

## RulZ Mask Intelligent Crop

`MaskIntelligentCrop`: crops an image and/or mask to the bounding box of the mask, then extends **every side**
of that box by a percentage (default 10%), either the same on all sides or set per edge. By default the percentage is measured against the **final crop**,
so the subject ends up inside a 10% border. The padded box is clamped to the image bounds.

| Input | Description |
|---|---|
| `image` (optional) | Image batch to crop. |
| `mask` (optional) | Mask that defines the crop. If its resolution differs from the image it is resampled to the image size first. |
| `padding_mode` | `uniform` (default): `padding_percent` is used on every side. `per_edge`: `padding_top`, `padding_bottom`, `padding_left` and `padding_right` set each side separately. The node only shows the fields for the selected mode. |
| `padding_percent` | `uniform` mode. How far each side extends beyond the masked area. Left/right use this percentage of the reference **width**, top/bottom of the reference **height**. Default `10`. |
| `padding_top` / `padding_bottom` / `padding_left` / `padding_right` | `per_edge` mode. How far that one side extends beyond the masked area, as a percentage of the reference **height** (top/bottom) or **width** (left/right). Default `10` each. |
| `padding_reference` | What the percentages are measured against. `crop` (default): the final cropped image, so each border is that share of the output (with `crop`, two opposite borders together are capped at 90% and scaled down evenly beyond that). `mask`: the mask bounding box. `image`: the full input image. |
| `aspect_limit` | Never crop more elongated than this ratio. Default `2:3 / 3:2`: a crop that would be taller than 2:3 is widened, one wider than 3:2 is heightened, centred on the subject and clamped to the image. The crop is only ever grown, never shrunk. `off` disables it. |
| `divisible_by` | Grow the crop so width and height are multiples of this value. `1` disables it. The crop is only ever shrunk when the image is too small to grow, and never into the masked area. |
| `mask_threshold` | Mask pixels above this value count as masked. Default `0` (any non-zero pixel). |

| Output | Description |
|---|---|
| `image` | Cropped image. `None` if no image was connected. |
| `mask` | Cropped mask, in image coordinates. |

- **image + mask**: both are cropped to the same box.
- **mask only**: the mask is cropped, the image output is `None`.
- **image only**: nothing to crop by, so the image passes through unchanged with a blank (all zero) mask of the same size.
- **empty mask**: inputs are returned at full size.
- **batches**: the crop box is the union over the whole mask batch, so every frame is cropped to the same size.

## RulZ Resize Image Advanced

`ResizeImageV2Advanced`: a standalone clone of the KJNodes **Resize Image v2** node with extra sizing controls.
No dependency on ComfyUI-KJNodes.

| Setting | Description |
|---|---|
| `image_2` (optional) | A second image used **only for its width/height**. The output is always a resized `image_1`. |
| `use_input_size` | `off` / `image_1` / `image_2`. Uses that image's width/height as the `width`/`height` settings. With `divisible_by` set, the output is the closest resolution to that image that is divisible. |
| `upscale_mode` | `disabled` / `upscale_by` / `total_megapixels`. Applied to the target size (after `use_input_size`), before the `keep_proportion` logic runs. |
| `upscale_by` | Float ≥ 0.01. Multiplies both edges of the target size. |
| `total_megapixels` | Float ≥ 0.01. Rescales the target size (keeping its aspect ratio) to the nearest total pixel count. 1 MP = 1024×1024 px, same as ComfyUI's *ImageScaleToTotalPixels*. |
| Output resolution | Shown on the node after execution (`W x H (MP) \| image_1 W x H \| image_2 W x H`). |
| Defaults | `upscale_method = lanczos`, `keep_proportion = crop`, `divisible_by = 32`. |
| Rounding | All computed sizes are rounded to the **nearest** multiple of `divisible_by` (KJ's original rounds down). |

**Outputs:** `IMAGE`, `width`, `height`, `mask`.

Everything else (stretch / resize / pad / pad_edge / pad_edge_pixel / crop / pillarbox_blur / total_pixels,
crop position, pad colour, mask handling, GPU device option, NVIDIA RTX VSR method) matches the original.
The `nvidia_rtx_vsr` method needs the `nvidia-vfx` library and an NVIDIA RTX GPU.

## RulZ Disable Comfy Compiler

`DisableComfyCompiler`: a per-workflow replacement for the `--disable-comfy-compiler` launch flag.

The node takes a `MODEL` and returns a clone that turns the Comfy model compiler (and its CUDA graph
subfeature) off only while that model's diffusion forward pass runs. The flags are restored afterwards, so
other workflows are unaffected. Set `disable` to off to pass the model through untouched.

Start ComfyUI without `--disable-comfy-compiler` and place the node between your model loader and sampler.

- Tested on ComfyUI 0.37.0 with Qwen Image 2.1.
- Only the diffusion model is covered; text encoders and latent previews are not affected.

## Credits

RulZ Resize Image Advanced is based on the Resize Image v2 node and pad helpers from
[ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes) by kijai.

## License

[GPL-3.0](LICENSE). RulZ Resize Image Advanced is derived from GPL-3.0 code in ComfyUI-KJNodes, so the
pack as a whole is released under the same license.
