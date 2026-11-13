import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// "RulZ Person Select (SAM3)": shows the image on the node with every mask SAM 3
// found drawn over it. Click a mask to select it (Shift-click for several). The
// clicked points are kept in the hidden `selection` widget; the backend
// (person_select.py) turns them into masks the same way for this preview and for
// the node's outputs.

const NODE_NAME = "RulZPersonSelect";
const ROUTE = "/rulz/person_select/detect";
const DETECT_WIDGETS = ["sam3_model", "prompt", "max_detections", "threshold", "device"];
const MIN_HEIGHT = 220;
const MAX_AUTO_HEIGHT = 560;
const TOOLBAR_HEIGHT = 30;
// While a workflow keeps the main GPU busy, ask again this often in case the
// "queue empty" status event arrives before the busy reply does.
const BUSY_RETRY_MS = 4000;

const COLOR_SELECTED = [40, 230, 120, 120];
const COLOR_HOVER = [255, 215, 60, 110];
const COLOR_OTHER = [70, 140, 255, 60];

function hasNativeHidden(widget) {
  for (let o = widget; o; o = Object.getPrototypeOf(o)) {
    if (Object.getOwnPropertyDescriptor(o, "hidden")?.set) return true;
  }
  return false;
}

function hideWidget(widget) {
  if (!widget) return;
  if (hasNativeHidden(widget)) {
    widget.hidden = true;
    return;
  }
  widget.type = "hidden";
  widget.computeSize = () => [0, -4];
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

async function uploadFile(file) {
  const body = new FormData();
  body.append("image", file);
  const resp = await api.fetchApi("/upload/image", { method: "POST", body });
  if (resp.status !== 200) throw new Error(`${resp.status} ${resp.statusText}`);
  const data = await resp.json();
  return data.subfolder ? `${data.subfolder}/${data.name}` : data.name;
}

function decodeLabels(url) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      const canvas = document.createElement("canvas");
      canvas.width = img.naturalWidth;
      canvas.height = img.naturalHeight;
      const ctx = canvas.getContext("2d", { willReadFrequently: true });
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(img, 0, 0);
      const rgba = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
      // 16-bit labels: red is the low byte, green the high byte.
      const labels = new Uint16Array(canvas.width * canvas.height);
      for (let i = 0; i < labels.length; i++) labels[i] = rgba[i * 4] | (rgba[i * 4 + 1] << 8);
      resolve({ labels, width: canvas.width, height: canvas.height });
    };
    img.onerror = () => reject(new Error("could not decode the mask preview"));
    img.src = url;
  });
}

// Identifies this node's preview requests to the server, which skips the ones a
// newer request from the same node has replaced. Random rather than the node id:
// two tabs with the same workflow must not replace each other's requests.
function clientId() {
  return globalThis.crypto?.randomUUID?.() ?? `${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`;
}

function setup(node) {
  const widget = (name) => node.widgets?.find((w) => w.name === name);
  const imageWidget = widget("image");
  const selectionWidget = widget("selection");
  hideWidget(selectionWidget);

  // state.map: label map from the server; state.selected / pointLabels: its answer
  // for the current points.
  const state = {
    map: null,
    selected: new Set(),
    pointLabels: [],
    hover: 0,
    seq: 0,
    client: clientId(),
    waiting: false,
    autoHeight: MIN_HEIGHT,
  };

  const root = document.createElement("div");
  root.style.cssText = "display:flex;flex-direction:column;width:100%;height:100%;min-height:0;gap:4px;";
  const bar = document.createElement("div");
  bar.style.cssText = `display:flex;align-items:center;gap:6px;height:${TOOLBAR_HEIGHT - 4}px;flex:none;font:12px sans-serif;`;
  const uploadBtn = document.createElement("button");
  uploadBtn.textContent = "Upload image";
  const clearBtn = document.createElement("button");
  clearBtn.textContent = "Clear";
  for (const b of [uploadBtn, clearBtn]) b.style.cssText = "cursor:pointer;padding:2px 8px;flex:none;";
  const status = document.createElement("span");
  status.style.cssText = "flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;opacity:0.85;";
  bar.append(uploadBtn, clearBtn, status);

  const stage = document.createElement("div");
  stage.style.cssText = "position:relative;flex:1;min-height:0;display:flex;align-items:center;justify-content:center;overflow:hidden;";
  const frame = document.createElement("div");
  frame.style.cssText = "position:relative;flex:none;";
  const img = document.createElement("img");
  img.draggable = false;
  img.style.cssText = "display:block;width:100%;height:100%;user-select:none;";
  const overlay = document.createElement("canvas");
  overlay.style.cssText = "position:absolute;inset:0;width:100%;height:100%;cursor:crosshair;";
  frame.append(img, overlay);
  stage.append(frame);
  root.append(bar, stage);

  const fileInput = document.createElement("input");
  fileInput.type = "file";
  fileInput.accept = "image/*,.heic,.heif";
  fileInput.style.display = "none";
  root.append(fileInput);

  const setStatus = (text, isError = false) => {
    status.textContent = text;
    status.title = text;
    status.style.color = isError ? "#ff6b6b" : "";
  };

  const getPoints = () => {
    try {
      const v = JSON.parse(selectionWidget.value || "[]");
      return Array.isArray(v) ? v.filter((p) => Array.isArray(p) && p.length === 2) : [];
    } catch {
      return [];
    }
  };
  const setPoints = (points) => {
    selectionWidget.value = JSON.stringify(points.map(([x, y]) => [+x.toFixed(4), +y.toFixed(4)]));
  };

  // Fit the frame (image + overlay) inside the stage, keeping the image's shape.
  function layout() {
    const w = img.naturalWidth, h = img.naturalHeight;
    const sw = stage.clientWidth, sh = stage.clientHeight;
    if (!w || !h || !sw || !sh) return;
    const scale = Math.min(sw / w, sh / h);
    frame.style.width = `${Math.floor(w * scale)}px`;
    frame.style.height = `${Math.floor(h * scale)}px`;
  }
  const resizeObserver = new ResizeObserver(layout);
  resizeObserver.observe(stage);

  function draw() {
    const map = state.map;
    if (!map) {
      overlay.width = overlay.height = 1;
      return;
    }
    if (overlay.width !== map.width || overlay.height !== map.height) {
      overlay.width = map.width;
      overlay.height = map.height;
    }
    const ctx = overlay.getContext("2d");
    const out = ctx.createImageData(map.width, map.height);
    const data = out.data;
    for (let i = 0; i < map.labels.length; i++) {
      const label = map.labels[i];
      if (!label) continue;
      const c = state.selected.has(label - 1) ? COLOR_SELECTED : label === state.hover ? COLOR_HOVER : COLOR_OTHER;
      const o = i * 4;
      data[o] = c[0];
      data[o + 1] = c[1];
      data[o + 2] = c[2];
      data[o + 3] = c[3];
    }
    ctx.putImageData(out, 0, 0);
    const r = Math.max(3, map.width / 140);
    getPoints().forEach(([x, y], i) => {
      if (state.pointLabels[i] === undefined) return;
      ctx.beginPath();
      ctx.arc(x * map.width, y * map.height, r, 0, Math.PI * 2);
      ctx.fillStyle = state.pointLabels[i] >= 0 ? "#fff" : "#ff6b6b";
      ctx.fill();
      ctx.lineWidth = r / 2;
      ctx.strokeStyle = "#000";
      ctx.stroke();
    });
  }

  function describe(count) {
    const n = state.selected.size;
    if (!count) return "Nothing found. Try a lower threshold, or click the image.";
    return n ? `${n} selected of ${count}. Shift-click for more.` : `${count} found. Click one to select it.`;
  }

  async function refresh() {
    const image = imageWidget?.value;
    const model = widget("sam3_model")?.value;
    if (!image || !model) {
      state.map = null;
      draw();
      setStatus(image ? "No SAM 3 checkpoint in models/checkpoints." : "Upload an image to begin.", !!image);
      return;
    }
    const seq = ++state.seq;
    state.waiting = false;
    setStatus(state.map ? "Updating..." : "Finding people (first run loads SAM 3)...");
    const body = { image, sam3_model: model, points: getPoints(), client: state.client, seq };
    for (const name of ["prompt", "max_detections", "threshold", "device"]) body[name] = widget(name)?.value;
    try {
      const resp = await api.fetchApi(ROUTE, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const data = await resp.json();
      if (seq !== state.seq) return;
      // Replaced by a newer request from this node; its reply is the one to show.
      if (resp.status === 409 && data?.error === "stale") return;
      if (resp.status === 409) {
        state.waiting = true;
        setStatus("Waiting for the running workflow. Set device to another GPU to preview now.");
        scheduleRefresh(BUSY_RETRY_MS);
        return;
      }
      if (!resp.ok) throw new Error(data?.error ?? `${resp.status} ${resp.statusText}`);
      const map = await decodeLabels(data.labels);
      if (seq !== state.seq) return;
      state.map = map;
      state.selected = new Set(data.selected);
      state.pointLabels = data.point_labels;
      draw();
      setStatus(describe(data.count));
    } catch (e) {
      if (seq !== state.seq) return;
      setStatus(String(e?.message ?? e), true);
    }
  }

  let timer = null;
  const scheduleRefresh = (delay = 150) => {
    clearTimeout(timer);
    timer = setTimeout(refresh, delay);
  };

  function showImage() {
    const name = imageWidget?.value;
    state.map = null;
    state.selected = new Set();
    state.pointLabels = [];
    draw();
    if (!name) {
      img.removeAttribute("src");
      return;
    }
    img.src = viewUrl(name);
  }

  img.addEventListener("load", () => {
    // Give the node a height that suits the image, once per image.
    const width = Math.max(node.size[0] - 20, 100);
    state.autoHeight = Math.min(MAX_AUTO_HEIGHT, Math.max(MIN_HEIGHT, (width * img.naturalHeight) / img.naturalWidth + TOOLBAR_HEIGHT));
    node.setSize([node.size[0], node.computeSize()[1]]);
    state.autoHeight = MIN_HEIGHT;
    app.graph.setDirtyCanvas(true, true);
    requestAnimationFrame(layout);
  });
  img.addEventListener("error", () => setStatus("Could not display this image.", true));

  // The user picked or uploaded a different image: its old selection is meaningless.
  function imageChanged() {
    setPoints([]);
    showImage();
    scheduleRefresh(0);
  }

  async function upload(file) {
    if (!file) return;
    try {
      setStatus(`Uploading ${file.name}...`);
      const name = await uploadFile(file);
      const values = imageWidget.options?.values;
      if (Array.isArray(values) && !values.includes(name)) values.push(name);
      imageWidget.value = name;
      imageChanged();
    } catch (e) {
      setStatus(`Upload failed: ${e?.message ?? e}`, true);
    }
  }

  const toNorm = (e) => {
    const rect = overlay.getBoundingClientRect();
    return [
      Math.min(1, Math.max(0, (e.clientX - rect.left) / rect.width)),
      Math.min(1, Math.max(0, (e.clientY - rect.top) / rect.height)),
    ];
  };
  const labelAt = ([x, y]) => {
    const map = state.map;
    if (!map) return 0;
    const px = Math.min(map.width - 1, Math.floor(x * map.width));
    const py = Math.min(map.height - 1, Math.floor(y * map.height));
    return map.labels[py * map.width + px];
  };

  overlay.addEventListener("pointermove", (e) => {
    const label = labelAt(toNorm(e));
    if (label !== state.hover) {
      state.hover = label;
      draw();
    }
  });
  overlay.addEventListener("pointerleave", () => {
    if (state.hover) {
      state.hover = 0;
      draw();
    }
  });
  overlay.addEventListener("click", (e) => {
    if (!state.map) return;
    e.preventDefault();
    e.stopPropagation();
    const point = toNorm(e);
    const index = labelAt(point) - 1;
    const points = getPoints();
    const additive = e.shiftKey || e.ctrlKey || e.metaKey;
    const isSelected = index >= 0 && state.selected.has(index);
    let next;
    if (additive) {
      // Toggle: clicking a selected mask drops the points that selected it.
      next = isSelected ? points.filter((_, i) => state.pointLabels[i] !== index) : [...points, point];
    } else {
      next = isSelected && state.selected.size === 1 ? [] : [point];
    }
    setPoints(next);
    // Show the result at once when the click is on a mask we already have.
    if (index >= 0 || !next.length) {
      state.pointLabels = next.map((p) => labelAt(p) - 1);
      state.selected = new Set(state.pointLabels.filter((i) => i >= 0));
      draw();
    }
    scheduleRefresh(0);
  });

  uploadBtn.addEventListener("click", () => fileInput.click());
  fileInput.addEventListener("change", () => {
    upload(fileInput.files?.[0]);
    fileInput.value = "";
  });
  clearBtn.addEventListener("click", () => {
    setPoints([]);
    scheduleRefresh(0);
  });
  root.addEventListener("dragover", (e) => {
    if (e.dataTransfer?.types?.includes?.("Files")) e.preventDefault();
  });
  root.addEventListener("drop", (e) => {
    const file = e.dataTransfer?.files?.[0];
    if (!file) return;
    e.preventDefault();
    e.stopPropagation();
    upload(file);
  });
  node.onDragOver = (e) => !!e?.dataTransfer?.types?.includes?.("Files");
  node.onDragDrop = (e) => {
    const file = e?.dataTransfer?.files?.[0];
    if (!file) return false;
    upload(file);
    return true;
  };

  node.addDOMWidget("preview", "rulz_person_select", root, {
    serialize: false,
    hideOnZoom: false,
    getMinHeight: () => state.autoHeight,
  });

  const chain = (w, fn) => {
    if (!w) return;
    const callback = w.callback;
    w.callback = function (...args) {
      const r = callback?.apply(this, args);
      fn();
      return r;
    };
  };
  chain(imageWidget, imageChanged);
  for (const name of DETECT_WIDGETS) chain(widget(name), () => scheduleRefresh(500));

  // The preview is skipped while a workflow runs; pick it up when the queue empties.
  const onStatus = ({ detail }) => {
    if (state.waiting && detail?.exec_info?.queue_remaining === 0) scheduleRefresh(300);
  };
  api.addEventListener("status", onStatus);
  const onRemoved = node.onRemoved;
  node.onRemoved = function () {
    api.removeEventListener("status", onStatus);
    resizeObserver.disconnect();
    clearTimeout(timer);
    state.seq++;
    return onRemoved?.apply(this, arguments);
  };

  // Called on creation and again after a saved workflow's values are restored.
  node._rulzPersonSelectReload = () => {
    showImage();
    scheduleRefresh();
  };
  node._rulzPersonSelectReload();
}

// GPU numbers change with ComfyUI's launch flags, so the list names GPUs only.
// Map a value saved as "cuda:1 - NVIDIA GeForce RTX 4060 Ti" (1.9.x) to the
// entry with that name, or to "default" when that GPU is not available now.
function fixDeviceValue(node) {
  const w = node.widgets?.find((x) => x.name === "device");
  const values = w?.options?.values;
  if (!w || !Array.isArray(values) || values.includes(w.value)) return;
  const name = String(w.value ?? "").replace(/^cuda:\d+\s*-\s*/, "");
  w.value = values.includes(name) ? name : "default";
}

app.registerExtension({
  name: "RulZnodes.personSelect",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE_NAME) return;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onNodeCreated?.apply(this, arguments);
      setup(this);
      return r;
    };

    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      const r = onConfigure?.apply(this, arguments);
      fixDeviceValue(this);
      this._rulzPersonSelectReload?.();
      return r;
    };
  },
});
