<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/speedyrulz/ComfyUI-RulZnodes/main/assets/logo-dark.png">
    <img alt="RulZ" src="https://raw.githubusercontent.com/speedyrulz/ComfyUI-RulZnodes/main/assets/logo.png" width="520">
  </picture>
</p>

<h1 align="center">ComfyUI-RulZnodes</h1>

<p align="center">
  Nodes for ComfyUI: load HEIC photos with their filename, step through a folder one image per run,
  crop to a mask with sensible padding, resize with megapixel and input-size targets, remove honeycomb skin
  texture, turn off the Comfy compiler per workflow, pick a random prompt from a text file, mask a person by
  clicking them, and run Qwen Image 2.1 attention block-sparse.
</p>

## Nodes

| Node | What it does |
|---|---|
| [**RulZ Load Image (HEIC)**](#rulz-load-image-heic) | `LoadImage` replacement that also loads `.heic` / `.heif` and outputs the filename. |
| [**RulZ Load Image Batch (HEIC)**](#rulz-load-image-batch-heic) | The HEIC loader with a list: drop several photos on it and each queued run uses the next one. |
| [**RulZ Image Browser**](#rulz-image-browser) | Sidebar tab, not a graph node: browse a folder as thumbnails, select images and send them to RulZ Load Image Batch (HEIC). |
| [**RulZ Simple Load Image Batch**](#rulz-simple-load-image-batch) | Walks through every image in a folder, one image per queued run, and resumes where it stopped. |
| [**RulZ Mask Intelligent Crop**](#rulz-mask-intelligent-crop) | Crops an image and/or mask to the mask's bounding box plus padding, with an aspect limit and divisible-by snapping. |
| [**RulZ Resize Image Advanced**](#rulz-resize-image-advanced) | KJNodes-style Resize Image v2 with input-size passthrough, upscale-by and total-megapixel targets, and the output resolution shown on the node. |
| [**RulZ DeskinTexture**](#rulz-deskintexture) | Removes the honeycomb / orange-peel skin texture restoration models leave behind and puts natural grain back. |
| [**RulZ Disable Comfy Compiler**](#rulz-disable-comfy-compiler) | Per-workflow replacement for the `--disable-comfy-compiler` launch flag. |
| [**RulZ Random Prompt Selector**](#rulz-random-prompt-selector) | Outputs a random line from a text file of prompts, chosen by the seed. |
| [**RulZ Person Select (SAM3)**](#rulz-person-select-sam3) | Load an image, click a person on the node, and get a mask of that person and a mask of everyone else. |
| [**RulZ Qwen Sparse Sage Attention**](#rulz-qwen-sparse-sage-attention) | Block-sparse SageAttention for Qwen Image 2.1: about 20% faster sampling steps at 4 MP with SageAttention's quality. |

All nodes are in the **RulZnodes** category of the node menu.

## Install

**ComfyUI-Manager:** open **Custom Nodes Manager**, search for **RulZnodes** and install it. Or, with
[comfy-cli](https://github.com/Comfy-Org/comfy-cli): `comfy node install comfyui-rulznodes`.

**Manually:**

```
cd ComfyUI/custom_nodes
git clone https://github.com/speedyrulz/ComfyUI-RulZnodes
pip install -r ComfyUI-RulZnodes/requirements.txt
```

Use ComfyUI's own Python for the `pip install`. Restart ComfyUI and hard-refresh the browser.

`pillow-heif` is only needed for HEIC/HEIF. Without it the other nodes work as normal and the HEIC loader
handles the usual formats only.

> **Coming from the standalone packs?** Remove `ComfyUI-image-load-filename-heic`, `load-image-batch-simple`,
> `mask-intelligent-crop`, `resize-image-v2-advanced`, `ComfyUI-DeSkin` and `ComfyUI-DisableComfyCompiler` from
> `custom_nodes`.
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
- 16-bit greyscale PNG / TIFF files load at their full range (the stock conversion turns them almost white).
- Multi-picture JPEGs (MPO, from many phones and cameras) load as the main photo only, not as a batch with
  the hidden second view or depth map.

## RulZ Load Image Batch (HEIC)

`LoadImageBatchHEIC`: the same loader and outputs as **RulZ Load Image (HEIC)**, but it holds a list of images.

- Drag several images onto the node (or pick several with the upload button) to add them to the list. They
  are uploaded to ComfyUI's input folder and added in filename order. Dropping more adds them to the end.
- The node shows how many images are available, and the preview shows the first one.
- Each queued run uses the first image and then moves on: the count drops by 1 and the preview shows the next
  image. On the last image it stays put and the count stays at 1. Images added after that replace it.
- Picking a file from the `image` dropdown also adds it to the list. **clear list** empties the list.
- The list is saved with the workflow (in the node's properties), so it survives a reload.

## RulZ Image Browser

A tab in ComfyUI's left sidebar (the **RulZ Images** button) rather than a node in the node menu. It fills
**RulZ Load Image Batch (HEIC)** from a folder without dragging files in from a file manager.

- Type or paste a folder path and press Enter, or move around with the folder list and **Up**. Above a drive
  root, **Up** lists the drives. The last folder is remembered.
- The folder's images are shown as one column of thumbnails, by name or newest first. HEIC / HEIF files are
  included.
- Click a thumbnail to select or deselect it; Shift-click selects the range from the last clicked one.
  **All** and **None** select or clear the whole folder. The counter to the right of the thumbnails shows how
  many are selected; it and the **Load** button stay in view while you scroll.
- **Load** copies the selected images into ComfyUI's input folder and adds them to the
  batch node's list in the order shown, the same as dropping them on the node. A file that is already in the
  input folder with the same content is reused instead of copied again.
- The images go to the selected RulZ Load Image Batch (HEIC) node, or the first one in the workflow if none
  is selected. If the workflow has none, one is added.
- The folder is read on the machine that runs ComfyUI, so with a remote ComfyUI it browses that machine. Any
  browser that can reach your ComfyUI can list folders and view image thumbnails through this tab's routes;
  keep that in mind if you start ComfyUI with `--listen`.

## RulZ Simple Load Image Batch

`LoadImageBatchSimple`: walks through every image in a directory, one image per queued run.

| Input | Description |
|---|---|
| `directory` | Folder to read. Empty (the default) reads ComfyUI's `input` folder. `~` and paths in quotes (Windows **Copy as path**) work. The image count is detected automatically. |
| `reset` | On: start again from the first image. Off: resume from where the last run stopped. |
| `loop` | On: wrap around to the first image after the last one. Off: keep returning the last image. |

**Outputs:** `image`, `filename`, `total_images`, `current_index` (0-based). The node also shows
`current / total — filename` after each run.

Images are taken in natural filename order (`img2` before `img10`). Progress is saved per directory in
`ComfyUI/user/rulznodes/batch_state.json`, so queueing 5 runs and then 5 more covers all 10 images, even across
a ComfyUI restart or an update of this pack. Files added to or removed from the folder are picked up on the
next run.

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
| Rounding | All computed sizes are rounded to the **nearest** multiple of `divisible_by` (KJ's original rounds down). In the pad modes (`pad`, `pad_edge`, `pad_edge_pixel`, `pillarbox_blur`) the rounded size is the canvas: the image is fitted inside it and the rest is padding, so the output is exactly that size. |
| Pad colour | `R, G, B`, `R, G, B, A`, `#RRGGBB`, `#RRGGBBAA`, a colour name or one grey value. An alpha value is dropped for RGB images. |

**Outputs:** `IMAGE`, `width`, `height`, `mask`.

Everything else (stretch / resize / pad / pad_edge / pad_edge_pixel / crop / pillarbox_blur / total_pixels,
crop position, pad colour, mask handling, GPU device option, NVIDIA RTX VSR method) matches the original.
The `nvidia_rtx_vsr` method needs the `nvidia-vfx` library and an NVIDIA RTX GPU.

## RulZ DeskinTexture

`DeSkinTexture`: removes the honeycomb / "orange-peel" texture that image restoration models (Qwen-Image and
others) sometimes paint onto skin, and puts believable grain back so the result doesn't look like wax.

It measures the image's frequency spectrum, finds the band of detail sizes where the fake pores stick up above
the natural curve of a real photo, cuts only that band, and refills it with random, non-lattice texture. Larger
shapes, shading and finer edges are untouched. A separate step (`screen_door`) notches out the exact 2 / 4 / 8 px
grid that a VAE leaves behind. It needs no extra packages and falls back to the CPU if the GPU runs out of memory.

| Input | Description |
|---|---|
| `image` | Image batch. RGBA is fine; the alpha channel is kept. |
| `mask` (optional) | White = apply the fix. Overrides `mask_mode`. A face-parsing or SAM skin mask gives the cleanest result. |
| `detect` | `auto` (default) finds the pattern size from the spectrum. `manual` uses `period_px`. |
| `period_px` | Pattern cell size in pixels, `manual` mode only. Default `8`. |
| `band_width` | How wide a band of detail sizes to remove. Lower keeps more real pores but may leave some pattern. Default `1`. |
| `strength` | How completely the band is removed. Default `1`. |
| `grain` | Random texture put back, as a fraction of what was removed at each spot. `0` = none (waxy), `0.7`-`0.9` = more skin-like. Default `0.5`. |
| `mask_mode` | Used only with no mask connected. `auto_skin` (default): a rough skin-colour mask. `whole_image`: everything. |
| `seed` | Grain pattern seed. |
| `auto_threshold` | `auto` only applies the fix if the pattern is at least this many times the natural level inside the mask. Otherwise the image passes through and `info` names the size it found. Default `1.3`. |
| `screen_door` | Removes the fine pixel-aligned 2 / 4 / 8 px grid, inside the mask only. `0` = off. Default `1`. |
| `edge_protect` | Treats anything stronger than this many times the typical pattern as a real edge (lips, eyes, hems) and leaves it, plus a no-filter zone next to strong edges so silhouettes don't pick up a halo. `0` = off. Default `4`. |

| Output | Description |
|---|---|
| `image` | The fixed image. |
| `texture_removed` | What was taken out, amplified ×8 around mid-grey. A clean lattice on grey means the settings are right. |
| `mask_used` | The mask that was applied. Use it to check the auto skin mask. |
| `info` | Detected cell size and strength, mask coverage and amount removed. Connect it to a **Preview Any** node to read it; it is also printed in the console. |

**Tuning:**

- **Use a real skin mask when you can.** `auto_skin` is only a colour test: it can take wooden doors and warm
  fabrics and miss very pale or warm-lit skin. Detection is measured inside the mask, so a good mask also makes
  `auto` far more reliable.
- **Pattern still visible:** raise `band_width` to `1.3`-`1.5`, or switch to `manual` with the cell size measured
  at 100% zoom. A zoomed screenshot will fool you.
- **Pattern survives next to an edge:** lower `edge_protect` to `2`-`3`. A band or halo along a silhouette means the
  opposite: raise it.
- **Too smooth or too noisy:** raise or lower `grain`.
- **Fine striping at 100%:** that's the screen door, which is only fixed inside the mask. For a background that has
  it too, run a second node with `mask_mode` = `whole_image`, `strength` `0` and `screen_door` `1`.
- **Wrong size detected:** `auto` picks the strongest pattern, which can be fabric or hair if the honeycomb is
  faint. `info` shows what it chose; use `manual` in that case.

## RulZ Disable Comfy Compiler

`DisableComfyCompiler`: a per-workflow replacement for the `--disable-comfy-compiler` launch flag.

The node takes a `MODEL` and returns a clone that turns the Comfy model compiler (and its CUDA graph
subfeature) off only while that model's diffusion forward pass runs. The flags are restored afterwards, so
other workflows are unaffected. Set `disable` to off to pass the model through untouched.

Start ComfyUI without `--disable-comfy-compiler` and place the node between your model loader and sampler.

- Tested on ComfyUI 0.37.0 with Qwen Image 2.1.
- Only the diffusion model is covered; text encoders and latent previews are not affected.

## RulZ Random Prompt Selector

`RandomPromptSelector`: outputs one line from a text file of prompts, chosen by the seed.

| Input | Description |
|---|---|
| `prompt_file` | The `.txt` file to read, one prompt per line. Lists the `.txt` files in ComfyUI's `input` folder. |
| `seed` | Picks the line. The same seed and file always give the same prompt. |

**Output:** `prompt`.

- **browse for text file** uploads a file from your computer into ComfyUI's `input` folder and selects it.
  Browsers can't give a node the path of a local file, so the node reads that copy: after editing the original,
  browse for it again to replace the copy. A file with the same name in `input` is overwritten.
- Blank lines and spaces around a line are ignored.
- Set `control after generate` to `randomize` for a new prompt each run, or `fixed` to keep the same one. Editing
  the file in `input` is picked up on the next run.

## RulZ Person Select (SAM3)

`RulZPersonSelect`: an image loader with a clickable preview. SAM 3 finds every person in the image, the node
tints each one, and the one you click becomes the mask.

Needs a ComfyUI version with built-in SAM 3 support (the **SAM3 Detect** node) and a SAM 3 or SAM 3.1
checkpoint in `models/checkpoints`. Nothing else is installed or downloaded.

| Input | Description |
|---|---|
| `image` | The image to load, from ComfyUI's `input` folder. **Upload image** or dropping a file on the node adds one. HEIC/HEIF works. |
| `sam3_model` | The SAM 3 checkpoint. The node loads it itself; no loader node is needed. |
| `prompt` | What SAM 3 looks for. Default `person`. Separate several things with commas, e.g. `person, dog`. |
| `max_detections` | Most matches to find per prompt. |
| `threshold` | Detection confidence. Lower it if someone is not found. |
| `device` | Where SAM 3 runs. `default` is ComfyUI's main GPU. Pick another GPU (listed by name) to click and queue images while a workflow is using the main GPU. |

**Outputs:** `image`, `mask` (what you clicked), `others_mask` (everything else that was detected, minus what
you clicked).

- **Click** a tinted person to select them; they turn green. Click them again to deselect.
- **Shift-click** (or Ctrl-click) adds or removes a person, so several can be selected.
- Clicking something that was not detected asks SAM 3 for whatever is at that point and selects that.
- The selection is saved with the workflow as click positions, so it is cleared when you change the image.
- The preview runs SAM 3 when the image or a setting changes: a few seconds the first time while the model
  loads, then about 3 seconds per new image on `default` and about a second on a second GPU. On `default` it
  waits while a workflow is running.
- SAM 3 runs on the node's own copy of the model, outside ComfyUI's model management. On `default` that copy
  waits in RAM (about 1.7 GB) and is moved to the main GPU only while it runs, so it never holds VRAM your
  workflows need.
- Queuing with nothing selected stops the workflow with an error, so an empty mask is never passed on.

**Using a second GPU (`device`):**

- The preview runs right away even while a workflow is running, so you can pick the person in the next image
  and queue it without waiting. The queued run itself still waits its turn, because ComfyUI runs one workflow
  at a time.
- The node's copy of SAM 3 stays on that GPU (about 1.3 GB of VRAM, plus about 1 GB while it runs).
  ComfyUI does not count that memory, so if your workflows also use that GPU (multi-GPU nodes, for example)
  they share its VRAM. Switching `device` back to `default` frees it.
- The SAM 3 text encoder runs on the CPU, once per prompt.
- GPUs are listed with their names because CUDA's numbering can differ from `nvidia-smi`'s.
- Picking the main GPU by name is the same as `default`.
- Cancelling the running workflow still works while a preview runs on the other GPU.

## RulZ Qwen Sparse Sage Attention

`QwenSparseSageAttention`: block-sparse SageAttention 2 for Qwen Image 2.1.

From the second sampling step on, Qwen Image 2.1 reuses cached text and reference-image keys, and the image being
generated puts most of its attention on a minority of them. On that step the node estimates, for every layer, which
64-token key blocks hold `tau` of the attention, then computes only those for the rest of the run with a copy of
SageAttention's kernel that skips the other blocks. With every block kept the kernel gives SageAttention's exact
output.

| Input | Description |
|---|---|
| `model` | A Qwen Image 2.1 model. Other models pass through with plain SageAttention. |
| `tau` | Share of the estimated attention kept. 0.99 matched plain SageAttention's quality in testing; lower is faster and less exact. |
| `dense_layers` | Transformer blocks that always run dense, e.g. `0, 30-31`. Block 0 attends almost everywhere, so it is dense by default. |

**Output:** `model`.

Use it instead of **Patch Sage Attention KJ**, or after it: anything the node doesn't make sparse (the first step,
masked prefix attention, the dense layers) runs through the attention patch before it, or SageAttention if there is
none. A Sage patch placed after this node replaces it.

On an RTX 5060 Ti with an int8 Qwen Image 2.1 Edit model, 8 steps at 1728x2304 (4 MP) with one 4 MP reference,
sampling steps took 3.45 s instead of 4.5 s, and the output matched plain SageAttention as closely as SageAttention
matches full-precision attention. At 1 MP the steps are about 18% faster, but the one-time block estimate on the
second step takes most of that back.

Flux 2 (Klein) is not made sparse by this node: every token, reference included, attends every other token, and
ComfyUI's built-in **Model Sparse Attention** node (method `sol-attn`) handles that layout faster. With Klein 9B
int8 at 1792x2336 and a 4 MP reference, 4 steps, it took steps from 10.8 s to 8.4 s on an RTX 5060 Ti and from 14.4 s
to 10.8 s on an RTX 4060 Ti at `tau` 0.8, with no visible change.

- Needs SageAttention 2.2 (`sageattention`) and an RTX 40 (sm89) or RTX 50 (sm120) GPU.
- Needs the CUDA toolkit (`nvcc`) and a C++ compiler: the first run builds the kernel, which takes 2 to 3 minutes,
  and later runs load it. Without them, or on other GPUs, attention stays dense and the console says why.
- Tested on Linux (WSL2) with ComfyUI 0.44.0, PyTorch 2.15 and CUDA 13.4.

## Credits

RulZ Resize Image Advanced is based on the Resize Image v2 node and pad helpers from
[ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes) by kijai.

RulZ DeskinTexture is based on ComfyUI-DeSkin 1.4.0 (MIT).

RulZ Qwen Sparse Sage Attention's kernel in `sparse_sage/csrc` is SageAttention 2
([thu-ml/SageAttention](https://github.com/thu-ml/SageAttention)), changed to skip key blocks.

## License

[GPL-3.0](LICENSE). RulZ Resize Image Advanced is derived from GPL-3.0 code in ComfyUI-KJNodes, so the
pack as a whole is released under the same license. `sparse_sage/csrc` keeps SageAttention's
[Apache-2.0](sparse_sage/LICENSE) license.
