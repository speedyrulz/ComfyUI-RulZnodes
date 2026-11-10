import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { addImages, NODE_NAME } from "./heic_batch.js";

// "RulZ Image Browser": a sidebar tab that browses a folder on the machine running
// ComfyUI, shows its images as one column of thumbnails, and sends the selected ones
// to a RulZ Load Image Batch (HEIC) node, the same as dropping the files on it.
// The backend is image_browser.py.
const TAB_ID = "rulzImageBrowser";
const STORE_DIR = "RulZnodes.imageBrowser.dir";
const STORE_SORT = "RulZnodes.imageBrowser.sort";
const ROUTE = "/rulz/image_browser";

const CSS = `
.rulz-ib { display: flex; flex-direction: column; height: 100%; min-height: 0; box-sizing: border-box;
  color: var(--fg-color, #ddd); font-size: 13px; }
.rulz-ib * { box-sizing: border-box; }
.rulz-ib-head { padding: 8px; display: flex; flex-direction: column; gap: 6px;
  border-bottom: 1px solid var(--border-color, #444); }
.rulz-ib-title { font-weight: 600; }
.rulz-ib-row { display: flex; gap: 4px; align-items: center; }
.rulz-ib input[type=text], .rulz-ib select { flex: 1; min-width: 0; padding: 4px 6px; border-radius: 4px;
  border: 1px solid var(--border-color, #444); background: var(--comfy-input-bg, #222);
  color: var(--input-text, #ddd); font-size: 12px; }
.rulz-ib select { flex: 0 1 auto; }
.rulz-ib button { padding: 4px 8px; border-radius: 4px; cursor: pointer; font-size: 12px; white-space: nowrap;
  border: 1px solid var(--border-color, #444); background: var(--comfy-input-bg, #222);
  color: var(--input-text, #ddd); }
.rulz-ib button:hover:not(:disabled) { filter: brightness(1.25); }
.rulz-ib button:disabled { opacity: 0.45; cursor: default; }
.rulz-ib-dirs { max-height: 132px; overflow-y: auto; border: 1px solid var(--border-color, #444);
  border-radius: 4px; }
.rulz-ib-dirs:empty { display: none; }
.rulz-ib-dir { padding: 3px 8px; cursor: pointer; white-space: nowrap; overflow: hidden;
  text-overflow: ellipsis; }
.rulz-ib-dir:hover { background: var(--comfy-input-bg, #333); }
.rulz-ib-info { color: var(--descrip-text, #999); font-size: 12px; flex: 1; min-width: 0;
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.rulz-ib-error { color: #ff6b6b; white-space: normal; }
.rulz-ib-body { flex: 1; min-height: 0; display: flex; align-items: stretch; }
.rulz-ib-list { flex: 1; min-width: 0; min-height: 0; overflow-y: auto; padding: 8px; display: flex;
  flex-direction: column; gap: 8px; }
/* Beside the thumbnails and sticky, so it is in reach wherever the column is
   scrolled to, whether the list or the whole sidebar panel is what scrolls. */
.rulz-ib-rail { flex: none; width: 72px; align-self: flex-start; position: sticky; top: 0; z-index: 1;
  padding: 8px 6px 8px 0; display: flex; flex-direction: column; gap: 6px; text-align: center; }
.rulz-ib-item { position: relative; flex: none; border: 2px solid transparent; border-radius: 6px;
  overflow: hidden; cursor: pointer; background: var(--comfy-input-bg, #222); user-select: none; }
.rulz-ib-item:hover { border-color: var(--border-color, #666); }
.rulz-ib-item.selected { border-color: var(--p-primary-color, #4a90e2); }
.rulz-ib-item img { display: block; width: 100%; height: auto; min-height: 120px; max-height: 340px;
  object-fit: contain; pointer-events: none; }
.rulz-ib-item img.loaded { min-height: 0; }
.rulz-ib-item img.failed { visibility: hidden; height: 40px; min-height: 0; }
.rulz-ib-name { padding: 3px 6px; font-size: 11px; white-space: nowrap; overflow: hidden;
  text-overflow: ellipsis; }
.rulz-ib-check { position: absolute; top: 6px; right: 6px; width: 22px; height: 22px; border-radius: 50%;
  display: none; align-items: center; justify-content: center; font-size: 13px; font-weight: 700;
  background: var(--p-primary-color, #4a90e2); color: var(--p-primary-contrast-color, #fff); }
.rulz-ib-item.selected .rulz-ib-check { display: flex; }
.rulz-ib-count { padding: 6px 2px; border-radius: 6px; background: var(--comfy-input-bg, #222);
  font-size: 11px; line-height: 1.3; color: var(--descrip-text, #999); }
.rulz-ib-count b { display: block; font-size: 22px; line-height: 1.1; color: var(--fg-color, #ddd); }
.rulz-ib button.rulz-ib-load { padding: 12px 2px; font-size: 13px; font-weight: 600; white-space: normal;
  background: var(--p-primary-color, #4a90e2); color: var(--p-primary-contrast-color, #fff);
  border-color: transparent; }
`;

function h(tag, props = {}, ...children) {
  const el = Object.assign(document.createElement(tag), props);
  el.append(...children);
  return el;
}

function stored(key) {
  try {
    return localStorage.getItem(key);
  } catch {
    return null;
  }
}

function store(key, value) {
  try {
    localStorage.setItem(key, value);
  } catch {
    // Private window or blocked storage: the pane just won't remember.
  }
}

function toast(severity, detail) {
  const add = app.extensionManager?.toast?.add;
  if (add) {
    add.call(app.extensionManager.toast, { severity, summary: "RulZ Image Browser", detail, life: 5000 });
  } else {
    console.log(`[RulZ Image Browser] ${detail}`);
  }
}

// ---- state -----------------------------------------------------------------
let dir = null; // folder being shown ("" = list of drives)
let parent = null; // its parent, null when there is none
let images = []; // [{name, path, mtime, size}] in display order
const selected = new Set(); // paths
let anchor = -1; // last clicked index, for shift-click ranges
let sort = stored(STORE_SORT) === "newest" ? "newest" : "name";
let requestId = 0;
let busy = false;

// ---- DOM -------------------------------------------------------------------
const pathInput = h("input", { type: "text", placeholder: "Folder path, e.g. D:\\Photos", spellcheck: false });
const upButton = h("button", { title: "Parent folder" }, "Up");
const refreshButton = h("button", { title: "Reload this folder" }, "Refresh");
const dirList = h("div", { className: "rulz-ib-dirs" });
const info = h("div", { className: "rulz-ib-info" });
const sortSelect = h(
  "select",
  { title: "Thumbnail order. Selected images are loaded in this order." },
  h("option", { value: "name" }, "By name"),
  h("option", { value: "newest" }, "Newest first")
);
const allButton = h("button", {}, "All");
const noneButton = h("button", {}, "None");
const list = h("div", { className: "rulz-ib-list" });
const count = h("div", { className: "rulz-ib-count" });
const loadButton = h("button", { className: "rulz-ib-load" });
const root = h(
  "div",
  { className: "rulz-ib" },
  h(
    "div",
    { className: "rulz-ib-head" },
    h("div", { className: "rulz-ib-title" }, "RulZ Image Browser"),
    h("div", { className: "rulz-ib-row" }, pathInput, upButton, refreshButton),
    dirList,
    h("div", { className: "rulz-ib-row" }, info, sortSelect, allButton, noneButton)
  ),
  h("div", { className: "rulz-ib-body" }, list, h("div", { className: "rulz-ib-rail" }, count, loadButton))
);
sortSelect.value = sort;

function updateCount() {
  const n = selected.size;
  const label = `${n} image${n === 1 ? "" : "s"} selected`;
  count.replaceChildren(h("b", {}, String(n)), "selected");
  count.title = label;
  loadButton.textContent = busy ? "Loading..." : "Load";
  loadButton.title = n ? `Load the ${label} into RulZ Load Image Batch (HEIC)` : "Select images first";
  loadButton.disabled = busy || n === 0;
  noneButton.disabled = n === 0;
  allButton.disabled = images.length === 0 || n === images.length;
}

function setInfo(text, isError = false) {
  info.textContent = text;
  info.title = text;
  info.classList.toggle("rulz-ib-error", isError);
}

function sortImages() {
  if (sort === "newest") {
    images.sort((a, b) => b.mtime - a.mtime || a.name.localeCompare(b.name, undefined, { numeric: true }));
  } else {
    images.sort((a, b) => a.name.localeCompare(b.name, undefined, { numeric: true }));
  }
}

function renderImages() {
  anchor = -1;
  list.replaceChildren(
    ...images.map((image, index) => {
      const img = h("img", { loading: "lazy", decoding: "async", alt: "" });
      img.onload = () => img.classList.add("loaded");
      img.onerror = () => img.classList.add("failed");
      img.src = api.apiURL(`${ROUTE}/thumb?path=${encodeURIComponent(image.path)}&t=${image.mtime}`);
      const item = h(
        "div",
        { className: "rulz-ib-item", title: image.name },
        img,
        h("div", { className: "rulz-ib-check" }, "\u2713"),
        h("div", { className: "rulz-ib-name" }, image.name)
      );
      item.dataset.index = index;
      item.classList.toggle("selected", selected.has(image.path));
      return item;
    })
  );
  list.scrollTop = 0;
  updateCount();
}

function renderDirs(dirs) {
  dirList.replaceChildren(
    ...dirs.map((d) => {
      const el = h("div", { className: "rulz-ib-dir", title: d.path }, `\u{1F4C1} ${d.name}`);
      el.onclick = () => open(d.path);
      return el;
    })
  );
  dirList.scrollTop = 0;
}

function applySelection() {
  for (const item of list.children) {
    item.classList.toggle("selected", selected.has(images[item.dataset.index].path));
  }
  updateCount();
}

// Show a folder. The selection belongs to one folder, so it is cleared on a change
// of folder and kept (for the files that still exist) on a refresh.
async function open(path) {
  const id = ++requestId;
  setInfo("Loading...");
  let data;
  try {
    const resp = await api.fetchApi(`${ROUTE}/list?path=${encodeURIComponent(path)}`);
    data = await resp.json();
    if (!resp.ok) throw new Error(data?.error ?? `${resp.status} ${resp.statusText}`);
  } catch (e) {
    if (id !== requestId) return false;
    setInfo(e.message || String(e), true);
    return false;
  }
  if (id !== requestId) return false; // a newer request has taken over

  if (data.path !== dir) selected.clear();
  dir = data.path;
  parent = data.parent;
  images = data.images;
  const present = new Set(images.map((i) => i.path));
  for (const p of [...selected]) if (!present.has(p)) selected.delete(p);

  pathInput.value = dir;
  upButton.disabled = parent === null || parent === undefined;
  store(STORE_DIR, dir);
  sortImages();
  renderDirs(data.dirs);
  renderImages();
  if (dir === "") setInfo("Pick a drive");
  else setInfo(images.length ? `${images.length} image${images.length === 1 ? "" : "s"}` : "No images in this folder");
  return true;
}

// ---- sending to the node ---------------------------------------------------
function batchNodes(graph, out = []) {
  for (const node of graph?.nodes ?? graph?._nodes ?? []) {
    if (node.comfyClass === NODE_NAME || node.type === NODE_NAME) out.push(node);
    if (node.subgraph) batchNodes(node.subgraph, out);
  }
  return out;
}

function createBatchNode() {
  const node = globalThis.LiteGraph?.createNode(NODE_NAME);
  if (!node) throw new Error("RulZ Load Image Batch (HEIC) is not available");
  const canvas = app.canvas;
  const rect = canvas?.canvas?.getBoundingClientRect?.();
  const ds = canvas?.ds;
  if (rect && ds) {
    // Centre of what is currently visible, in graph coordinates.
    node.pos = [
      rect.width / 2 / ds.scale - ds.offset[0] - node.size[0] / 2,
      rect.height / 2 / ds.scale - ds.offset[1] - node.size[1] / 2,
    ];
  }
  (canvas?.graph ?? app.graph).add(node);
  return node;
}

// A selected batch node wins; otherwise the first one in the workflow. With none
// in the workflow, one is added.
function targetNode() {
  const nodes = batchNodes(app.graph);
  if (!nodes.length) return { node: createBatchNode(), note: " (added a new RulZ Load Image Batch (HEIC) node)" };
  const picked = Object.values(app.canvas?.selected_nodes ?? {}).find((n) => nodes.includes(n));
  if (picked) return { node: picked, note: "" };
  const note = nodes.length > 1 ? ` (node #${nodes[0].id}; select a batch node first to send to another one)` : "";
  return { node: nodes[0], note };
}

async function loadSelected() {
  const paths = images.filter((i) => selected.has(i.path)).map((i) => i.path);
  if (!paths.length || busy) return;
  busy = true;
  updateCount();
  try {
    const resp = await api.fetchApi(`${ROUTE}/import`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ paths }),
    });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data?.error ?? `${resp.status} ${resp.statusText}`);

    if (data.files.length) {
      const { node, note } = targetNode();
      addImages(node, data.files);
      const n = data.files.length;
      toast("success", `Added ${n} image${n === 1 ? "" : "s"} to the batch node${note}.`);
      selected.clear();
    }
    if (data.errors.length) {
      console.error("[RulZ Image Browser] could not load:", data.errors);
      toast("error", `Could not load ${data.errors.length} file(s): ${data.errors.slice(0, 3).join("; ")}`);
    }
  } catch (e) {
    console.error("[RulZ Image Browser] load failed", e);
    toast("error", `Loading failed: ${e.message || e}`);
  } finally {
    busy = false;
    applySelection();
  }
}

// ---- events ----------------------------------------------------------------
list.addEventListener("click", (e) => {
  const item = e.target.closest?.(".rulz-ib-item");
  if (!item) return;
  const index = Number(item.dataset.index);
  if (e.shiftKey && anchor >= 0 && anchor < images.length) {
    // Shift-click: select everything between the last clicked image and this one.
    const [from, to] = anchor < index ? [anchor, index] : [index, anchor];
    for (let i = from; i <= to; i++) selected.add(images[i].path);
  } else {
    const path = images[index].path;
    if (!selected.delete(path)) selected.add(path);
  }
  anchor = index;
  applySelection();
});
// Shift-click would otherwise select the text in between.
list.addEventListener("mousedown", (e) => e.shiftKey && e.preventDefault());

pathInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") open(pathInput.value);
  e.stopPropagation(); // keep ComfyUI's canvas shortcuts out of the text box
});
upButton.onclick = () => parent !== null && parent !== undefined && open(parent);
refreshButton.onclick = () => open(pathInput.value.trim() || dir || "");
sortSelect.onchange = () => {
  sort = sortSelect.value;
  store(STORE_SORT, sort);
  sortImages();
  renderImages();
};
allButton.onclick = () => {
  for (const image of images) selected.add(image.path);
  applySelection();
};
noneButton.onclick = () => {
  selected.clear();
  applySelection();
};
loadButton.onclick = loadSelected;

async function firstOpen() {
  // Last folder used, else the home folder, else the list of drives.
  const last = stored(STORE_DIR);
  if (last && (await open(last))) return;
  if (await open("~")) return;
  await open("");
}

app.registerExtension({
  name: "RulZnodes.imageBrowser",
  async setup() {
    if (!app.extensionManager?.registerSidebarTab) {
      console.warn("[RulZ Image Browser] this ComfyUI frontend has no sidebar tab API; the browser is unavailable.");
      return;
    }
    document.head.append(h("style", { textContent: CSS }));
    updateCount();
    app.extensionManager.registerSidebarTab({
      id: TAB_ID,
      icon: "pi pi-images",
      title: "RulZ Images",
      tooltip: "RulZ Image Browser",
      type: "custom",
      // Called every time the tab is opened; the pane itself is built once and kept.
      render: (el) => {
        el.replaceChildren(root);
        if (dir === null) firstOpen();
      },
    });
  },
});
