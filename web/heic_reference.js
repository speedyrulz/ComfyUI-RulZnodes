import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// "RulZ Load Image (HEIC) - Reference": up to 16 image slots on one node. `count`
// decides how many are shown: slots above it have their widgets hidden and their
// outputs removed (always from the end, so the remaining outputs keep their
// indices). Every slot has a file dropdown and an upload button; a grid of
// numbered thumbnails below shows the pictures and takes dropped files.
const NODE_NAME = "LoadImageReferenceHEIC";
const TITLE = "RulZ Load Image (HEIC) - Reference";
const MAX_SLOTS = 16;
const IMAGE_RE = /\.(png|jpe?g|webp|bmp|gif|tiff?|jfif|heic|heif)$/i;
const ACCEPT = "image/*,.heic,.heif,image/heic,image/heif";
const MIN_NODE_WIDTH = 340;
const THUMB_MIN = 76; // px, smallest thumbnail edge
const GAP = 6; // px
const GRID_PAD = 4; // px, above and below the grid

const slotName = (i) => `image_${i}`;
const buttonName = (i) => `upload_${i}`;

function isImageFile(file) {
  return file.type.startsWith("image/") || IMAGE_RE.test(file.name);
}

// Newer frontends give widgets a `hidden` accessor. Mixing it with the legacy
// type swap leaves the widget stuck hidden, so use one or the other.
function hasNativeHidden(widget) {
  for (let o = widget; o; o = Object.getPrototypeOf(o)) {
    if (Object.getOwnPropertyDescriptor(o, "hidden")?.set) return true;
  }
  return false;
}

function setWidgetVisible(widget, visible) {
  if (!widget) return;
  if (hasNativeHidden(widget)) {
    widget.hidden = !visible;
    return;
  }
  if (widget._rulzOrigType === undefined) {
    widget._rulzOrigType = widget.type;
    widget._rulzOrigComputeSize = widget.computeSize;
  }
  widget.type = visible ? widget._rulzOrigType : "hidden";
  widget.computeSize = visible ? widget._rulzOrigComputeSize : () => [0, -4];
}

function viewUrl(name) {
  const slash = name.lastIndexOf("/");
  const params = new URLSearchParams({
    filename: slash < 0 ? name : name.slice(slash + 1),
    subfolder: slash < 0 ? "" : name.slice(0, slash),
    type: "input",
  });
  return api.apiURL(`/view?${params}`);
}

function toast(severity, detail) {
  const t = app.extensionManager?.toast;
  if (t?.add) t.add({ severity, summary: TITLE, detail, life: 8000 });
  else alert(detail);
}

async function uploadFile(file) {
  const body = new FormData();
  body.append("image", file);
  const resp = await api.fetchApi("/upload/image", { method: "POST", body });
  if (resp.status !== 200) throw new Error(`${resp.status} ${resp.statusText}`);
  const data = await resp.json();
  return data.subfolder ? `${data.subfolder}/${data.name}` : data.name;
}

// --- node state ----------------------------------------------------------------

function countWidget(node) {
  return node.widgets?.find((w) => w.name === "count");
}

function comboOf(node, i) {
  return node.widgets?.find((w) => w.name === slotName(i));
}

function visibleCount(node) {
  const n = Math.round(Number(countWidget(node)?.value));
  return Number.isFinite(n) ? Math.max(1, Math.min(MAX_SLOTS, n)) : 1;
}

// --- thumbnail grid ------------------------------------------------------------

function gridMinHeight(node) {
  const n = visibleCount(node);
  const avail = Math.max(THUMB_MIN, (node.size?.[0] ?? MIN_NODE_WIDTH) - 24);
  const cols = Math.max(1, Math.floor((avail + GAP) / (THUMB_MIN + GAP)));
  const cell = (avail - (cols - 1) * GAP) / cols;
  const rows = Math.ceil(n / cols);
  return Math.ceil(rows * cell + (rows - 1) * GAP + GRID_PAD * 2);
}

function buildGrid(node) {
  const root = document.createElement("div");
  root.style.cssText = `box-sizing:border-box;width:100%;height:100%;overflow:hidden;padding:${GRID_PAD}px 0;`;
  const grid = document.createElement("div");
  grid.style.cssText = `display:grid;grid-template-columns:repeat(auto-fill,minmax(${THUMB_MIN}px,1fr));gap:${GAP}px;`;
  root.append(grid);

  const cells = [];
  for (let i = 1; i <= MAX_SLOTS; i++) {
    const cell = document.createElement("div");
    cell.style.cssText =
      "position:relative;aspect-ratio:1;overflow:hidden;box-sizing:border-box;border-radius:4px;" +
      "background:var(--comfy-input-bg,#222);border:1px solid var(--border-color,#4e4e4e);" +
      "color:var(--descrip-text,#999);font:11px sans-serif;display:flex;align-items:center;justify-content:center;";
    const img = document.createElement("img");
    img.draggable = false;
    img.style.cssText = "position:absolute;inset:0;width:100%;height:100%;object-fit:contain;user-select:none;display:none;";
    img.addEventListener("error", () => {
      img.style.display = "none";
      empty.textContent = "Cannot show";
      empty.style.display = "";
    });
    img.addEventListener("load", () => {
      img.style.display = "";
      empty.style.display = "none";
    });
    const empty = document.createElement("span");
    empty.textContent = "No image";
    const badge = document.createElement("span");
    badge.textContent = String(i);
    badge.style.cssText =
      "position:absolute;top:2px;left:2px;min-width:16px;padding:0 4px;box-sizing:border-box;border-radius:3px;" +
      "background:var(--comfy-menu-bg,#353535);color:var(--fg-color,#fff);border:1px solid var(--border-color,#4e4e4e);" +
      "font:bold 11px/15px sans-serif;text-align:center;pointer-events:none;";
    const caption = document.createElement("span");
    caption.style.cssText =
      "position:absolute;left:0;right:0;bottom:0;padding:1px 4px;box-sizing:border-box;overflow:hidden;" +
      "text-overflow:ellipsis;white-space:nowrap;background:var(--comfy-menu-bg,#353535);" +
      "color:var(--input-text,#ddd);font:10px/14px sans-serif;pointer-events:none;display:none;";
    cell.append(img, empty, badge, caption);

    cell.addEventListener("dragover", (e) => {
      if (!e.dataTransfer?.types?.includes?.("Files")) return;
      e.preventDefault();
      e.stopPropagation();
      cell.style.outline = "2px dashed var(--fg-color,#fff)";
    });
    cell.addEventListener("dragleave", () => {
      cell.style.outline = "";
    });
    cell.addEventListener("drop", (e) => {
      cell.style.outline = "";
      const files = e.dataTransfer?.files;
      if (!files?.length) return;
      e.preventDefault();
      e.stopPropagation();
      fillSlots(node, i, files);
    });
    grid.append(cell);
    cells.push({ cell, img, empty, caption, name: null });
  }

  // Dropping on the grid but outside a thumbnail works like dropping on the node.
  root.addEventListener("dragover", (e) => {
    if (e.dataTransfer?.types?.includes?.("Files")) e.preventDefault();
  });
  root.addEventListener("drop", (e) => {
    const files = e.dataTransfer?.files;
    if (!files?.length) return;
    e.preventDefault();
    e.stopPropagation();
    fillSlots(node, 1, files);
  });

  node._rulzRef.cells = cells;
  return root;
}

function refreshGrid(node) {
  const cells = node._rulzRef?.cells;
  if (!cells) return;
  const n = visibleCount(node);
  cells.forEach((c, idx) => {
    const i = idx + 1;
    c.cell.style.display = i <= n ? "" : "none";
    if (i > n) return;
    const name = comboOf(node, i)?.value;
    const value = typeof name === "string" && name ? name : null;
    if (value === c.name) return;
    c.name = value;
    c.cell.title = value ?? "";
    c.caption.textContent = value ? value.split("/").pop() : "";
    c.caption.style.display = value ? "" : "none";
    if (value) {
      c.empty.textContent = "Loading";
      c.empty.style.display = "";
      c.img.src = viewUrl(value);
    } else {
      c.img.removeAttribute("src");
      c.img.style.display = "none";
      c.empty.textContent = "No image";
      c.empty.style.display = "";
    }
  });
}

// --- slot count: widgets, outputs, size ----------------------------------------

function syncOutputs(node, n) {
  const outs = node.outputs ?? [];
  // Always from the end, so the outputs that stay keep their indices.
  while (outs.length > n) node.removeOutput(outs.length - 1);
  while ((node.outputs ?? []).length < n) {
    node.addOutput(slotName(node.outputs.length + 1), "IMAGE");
  }
}

function applyCount(node, { resize = true } = {}) {
  const countW = countWidget(node);
  if (!countW) return;
  const n = visibleCount(node);
  if (countW.value !== n) countW.value = n;
  for (let i = 1; i <= MAX_SLOTS; i++) {
    const visible = i <= n;
    setWidgetVisible(comboOf(node, i), visible);
    setWidgetVisible(node.widgets?.find((w) => w.name === buttonName(i)), visible);
  }
  syncOutputs(node, n);
  refreshGrid(node);
  if (resize) resizeNode(node);
  app.graph?.setDirtyCanvas(true, true);
}

function resizeNode(node) {
  requestAnimationFrame(() => {
    const [w, h] = node.computeSize();
    node.setSize([Math.max(node.size[0], w), h]);
    app.graph?.setDirtyCanvas(true, true);
  });
}

// --- filling slots -------------------------------------------------------------

function addToCombos(node, name) {
  for (let i = 1; i <= MAX_SLOTS; i++) {
    const values = comboOf(node, i)?.options?.values;
    if (Array.isArray(values) && !values.includes(name)) values.push(name);
  }
}

// Upload the images in `fileList` (natural filename order) into slots start,
// start + 1, ... and grow `count` to fit, up to the 16th slot.
async function fillSlots(node, start, fileList) {
  const files = [...(fileList ?? [])].filter(isImageFile);
  if (!files.length) return false;
  files.sort((a, b) => a.name.localeCompare(b.name, undefined, { numeric: true }));
  const room = MAX_SLOTS - start + 1;
  const ignored = Math.max(0, files.length - room);
  const take = files.slice(0, room);

  // One at a time so the slots keep the sorted order.
  const paths = [];
  const failed = [];
  for (const file of take) {
    try {
      paths.push(await uploadFile(file));
    } catch (e) {
      console.error(`[LoadImageReferenceHEIC] upload of ${file.name} failed`, e);
      failed.push(`${file.name}: ${e?.message ?? e}`);
    }
  }

  paths.forEach((path, k) => {
    addToCombos(node, path);
    const combo = comboOf(node, start + k);
    if (!combo) return;
    combo.value = path;
    combo.callback?.(path);
  });
  if (paths.length) {
    const countW = countWidget(node);
    const needed = start - 1 + paths.length;
    if (countW && needed > visibleCount(node)) countW.value = needed;
    applyCount(node);
  }
  if (failed.length) {
    toast(
      "error",
      `Could not upload ${failed.length} image${failed.length === 1 ? "" : "s"}: ${failed.slice(0, 3).join("; ")}${
        failed.length > 3 ? "; ..." : ""
      }`
    );
  }
  if (ignored) {
    toast(
      "warn",
      `Only ${MAX_SLOTS} images fit: ${ignored} image${ignored === 1 ? " was" : "s were"} ignored (the last ${
        ignored === 1 ? "one" : ignored
      } in name order).`
    );
  }
  return true;
}

function openPicker(node, start) {
  const input = document.createElement("input");
  input.type = "file";
  input.multiple = true;
  input.accept = ACCEPT;
  input.onchange = () => fillSlots(node, start, input.files);
  input.click();
}

// --- extension -----------------------------------------------------------------

function setup(node) {
  node._rulzRef = { cells: null };

  // One upload button per slot, placed right after that slot's dropdown.
  const buttons = [];
  for (let i = 1; i <= MAX_SLOTS; i++) {
    const btn = node.addWidget("button", buttonName(i), null, () => openPicker(node, i), { serialize: false });
    btn.serialize = false;
    btn.label = `Upload image ${i}`;
    buttons.push(btn);
  }
  const ordered = [];
  const used = new Set(buttons);
  for (const w of node.widgets) {
    if (used.has(w)) continue;
    ordered.push(w);
    const m = /^image_(\d+)$/.exec(w.name);
    if (m) ordered.push(buttons[Number(m[1]) - 1]);
  }
  node.widgets.splice(0, node.widgets.length, ...ordered);

  // Dropdown picks refresh the thumbnail; a changed count refreshes everything.
  for (let i = 1; i <= MAX_SLOTS; i++) {
    const combo = comboOf(node, i);
    if (!combo) continue;
    const callback = combo.callback;
    combo.callback = function (...args) {
      const r = callback?.apply(this, args);
      refreshGrid(node);
      return r;
    };
  }
  const countW = countWidget(node);
  if (countW) {
    const callback = countW.callback;
    countW.callback = function (...args) {
      const r = callback?.apply(this, args);
      applyCount(node);
      return r;
    };
  }

  const root = buildGrid(node);
  const dom = node.addDOMWidget("preview", "rulz_reference_grid", root, {
    serialize: false,
    hideOnZoom: false,
    getMinHeight: () => gridMinHeight(node),
  });
  dom.serialize = false;

  // Dropping files anywhere else on the node fills from slot 1.
  node.onDragOver = (e) => !!e?.dataTransfer?.types?.includes?.("Files");
  node.onDragDrop = (e) => {
    const files = e?.dataTransfer?.files;
    if (!files?.length) return false;
    fillSlots(node, 1, files);
    return true;
  };

  applyCount(node, { resize: false });
  // A new node gets a size that suits one slot; a node being loaded from a
  // workflow (or pasted) keeps the size it was saved with.
  requestAnimationFrame(() => {
    if (node._rulzConfigured) return;
    const [w, h] = node.computeSize();
    node.setSize([Math.max(MIN_NODE_WIDTH, w), h]);
    app.graph?.setDirtyCanvas(true, true);
  });
}

app.registerExtension({
  name: "RulZnodes.heicReference",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE_NAME) return;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onNodeCreated?.apply(this, arguments);
      setup(this);
      return r;
    };

    // Runs after a saved workflow's values and outputs are restored (also for
    // copy and paste). The frontend gives the node all 16 outputs again here, so
    // trim them to `count`.
    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      const r = onConfigure?.apply(this, arguments);
      this._rulzConfigured = true;
      if (this._rulzRef) applyCount(this, { resize: false });
      return r;
    };
  },
});
