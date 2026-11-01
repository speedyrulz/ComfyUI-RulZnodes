import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// "RulZ Load Image Batch (HEIC)": keeps a list of images on the node. Dropping (or
// picking) images adds them to the list, the `image` combo always shows the first
// one, and every queued run moves on to the next image. The last image stays
// selected (count 1) until more are added, which then replace it.
const NODE_NAME = "LoadImageBatchHEIC";
const IMAGE_RE = /\.(png|jpe?g|webp|bmp|gif|tiff?|jfif|heic|heif)$/i;
const ACCEPT = "image/*,.heic,.heif,image/heic,image/heif";

function isImageFile(file) {
  return file.type.startsWith("image/") || IMAGE_RE.test(file.name);
}

function state(node) {
  node.properties ??= {};
  if (!Array.isArray(node.properties.rulz_queue)) node.properties.rulz_queue = [];
  return node.properties;
}

function comboWidget(node) {
  return node.widgets?.find((w) => w.name === "image");
}

function updateStatus(node) {
  const { rulz_queue: queue } = state(node);
  const status = node.widgets?.find((w) => w.name === "queue");
  if (status) {
    const n = queue.length;
    status.value = n ? `${n} image${n === 1 ? "" : "s"} available` : "drop images here";
  }
  app.graph.setDirtyCanvas(true, true);
}

// Point the combo at the head of the list. Calling its callback makes the
// frontend refresh the preview, as when the user picks a file.
function showHead(node) {
  const combo = comboWidget(node);
  const head = state(node).rulz_queue[0];
  if (combo && head && combo.value !== head) {
    const values = combo.options?.values;
    if (Array.isArray(values) && !values.includes(head)) values.push(head);
    node._rulzSyncing = true;
    try {
      combo.value = head;
      combo.callback?.(head);
    } finally {
      node._rulzSyncing = false;
    }
  }
  updateStatus(node);
}

function addImages(node, paths) {
  if (!paths.length) return;
  const s = state(node);
  // Once the last image has been queued it is spent: new images replace it.
  if (s.rulz_spent) s.rulz_queue = [];
  s.rulz_queue.push(...paths);
  s.rulz_spent = false;
  showHead(node);
}

async function uploadFile(file) {
  const body = new FormData();
  body.append("image", file);
  const resp = await api.fetchApi("/upload/image", { method: "POST", body });
  if (resp.status !== 200) throw new Error(`${resp.status} ${resp.statusText}`);
  const data = await resp.json();
  return data.subfolder ? `${data.subfolder}/${data.name}` : data.name;
}

async function uploadFiles(node, fileList) {
  const files = [...(fileList ?? [])].filter(isImageFile);
  files.sort((a, b) => a.name.localeCompare(b.name, undefined, { numeric: true }));
  const paths = [];
  // One at a time so the list keeps the sorted order.
  for (const file of files) {
    try {
      paths.push(await uploadFile(file));
    } catch (e) {
      console.error(`[LoadImageBatchHEIC] upload of ${file.name} failed`, e);
    }
  }
  addImages(node, paths);
  return files.length > 0;
}

function openPicker(node) {
  const input = document.createElement("input");
  input.type = "file";
  input.multiple = true;
  input.accept = ACCEPT;
  input.onchange = () => uploadFiles(node, input.files);
  input.click();
}

app.registerExtension({
  name: "RulZnodes.heicBatch",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE_NAME) return;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onNodeCreated?.apply(this, arguments);
      state(this);

      const status = this.addWidget("text", "queue", "", () => {}, { serialize: false });
      status.disabled = true;

      const combo = comboWidget(this);
      if (combo) {
        // Uploads via the stock button, pastes and dropdown picks all land here.
        const callback = combo.callback;
        combo.callback = (...args) => {
          const cr = callback?.apply(combo, args);
          if (!this._rulzSyncing && combo.value) addImages(this, [combo.value]);
          return cr;
        };

        const afterQueued = combo.afterQueued;
        combo.afterQueued = (...args) => {
          afterQueued?.apply(combo, args);
          const s = state(this);
          if (s.rulz_queue.length > 1) {
            s.rulz_queue.shift();
            showHead(this);
          } else if (s.rulz_queue.length === 1) {
            s.rulz_spent = true;
          }
        };
      }

      // The stock upload button only takes one file; let it take several.
      const upload = this.widgets?.find((w) => w.type === "button" && w !== status);
      if (upload) {
        upload.callback = () => openPicker(this);
        upload.label = "choose images to upload";
      }

      this.addWidget("button", "clear list", null, () => {
        Object.assign(state(this), { rulz_queue: [], rulz_spent: false });
        updateStatus(this);
      }, { serialize: false });

      // Replace the stock single-file drop handler.
      this.onDragOver = (e) => !!e?.dataTransfer?.types?.includes?.("Files");
      this.onDragDrop = (e) => {
        const files = e?.dataTransfer?.files;
        if (!files?.length) return false;
        uploadFiles(this, files);
        return true;
      };

      updateStatus(this);
      return r;
    };

    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      const r = onConfigure?.apply(this, arguments);
      showHead(this);
      return r;
    };
  },
});
