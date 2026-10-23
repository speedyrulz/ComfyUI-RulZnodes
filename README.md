<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/speedyrulz/ComfyUI-RulZnodes/main/assets/logo-dark.png">
    <img alt="RulZ" src="https://raw.githubusercontent.com/speedyrulz/ComfyUI-RulZnodes/main/assets/logo.png" width="520">
  </picture>
</p>

<h1 align="center">ComfyUI-RulZnodes</h1>

<p align="center">
  Image nodes for ComfyUI: load HEIC photos with their filename, step through a folder one image per run,
  and crop to a mask with sensible padding.
</p>

## Nodes

| Node | What it does |
|---|---|
| [**Load Image (with filename, HEIC)**](#load-image-with-filename-heic) | `LoadImage` replacement that also loads `.heic` / `.heif` and outputs the filename. |
| [**Load Image Batch (Simple)**](#load-image-batch-simple) | Walks through every image in a folder, one image per queued run, and resumes where it stopped. |
| [**Mask Intelligent Crop**](#mask-intelligent-crop) | Crops an image and/or mask to the mask's bounding box plus padding, with an aspect limit and divisible-by snapping. |

All three are in the **RulZnodes** category of the node menu.

## Install

```
cd ComfyUI/custom_nodes
git clone https://github.com/speedyrulz/ComfyUI-RulZnodes
pip install -r ComfyUI-RulZnodes/requirements.txt
```

Use ComfyUI's own Python for the `pip install`. Restart ComfyUI and hard-refresh the browser.

`pillow-heif` is only needed for HEIC/HEIF. Without it the other nodes work as normal and the HEIC loader
handles the usual formats only.

> **Coming from the standalone packs?** Remove `ComfyUI-image-load-filename-heic`, `load-image-batch-simple`
> and `mask-intelligent-crop` from `custom_nodes`. The node IDs are unchanged, so existing workflows keep
> working, but with both installed the same nodes are registered twice.

---

## Load Image (with filename, HEIC)

`LoadImageWithFilenameHEIC`: a `LoadImage` replacement that also loads `.heic` / `.heif` files and outputs the filename.

**Outputs:** `image`, `mask`, `filename` (name without directory or extension).

- The upload button's file picker is widened to allow HEIC/HEIF, and HEIC files are accepted even when
  Windows reports no MIME type for them (`web/heic_upload.js`).
- Previews: browsers can't render HEIC, so `heic_preview.py` adds a middleware that serves `.heic` / `.heif`
  files as JPEG from ComfyUI's `/view` endpoint. The conversion happens in memory; no extra files are created.
- EXIF rotation is applied, so phone photos come out upright.

## Load Image Batch (Simple)

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

## Mask Intelligent Crop

`MaskIntelligentCrop`: crops an image and/or mask to the bounding box of the mask, then extends **every side**
of that box by a percentage (default 10%). By default the percentage is measured against the **final crop**,
so the subject ends up inside a 10% border. The padded box is clamped to the image bounds.

| Input | Description |
|---|---|
| `image` (optional) | Image batch to crop. |
| `mask` (optional) | Mask that defines the crop. If its resolution differs from the image it is resampled to the image size first. |
| `padding_percent` | How far each side extends beyond the masked area. Left/right use this percentage of the reference **width**, top/bottom of the reference **height**. Default `10`. |
| `padding_reference` | What the percentage is measured against. `crop` (default): the final cropped image, so each border is that share of the output. `mask`: the mask bounding box. `image`: the full input image. |
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

## License

[MIT](LICENSE)
