import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// "RulZ Random Prompt Selector": a browser can't hand the server a local file path,
// so the button uploads the chosen file to ComfyUI's input folder and selects it in
// the `prompt_file` combo. Uploading overwrites, so picking an edited copy of a file
// replaces the old one rather than creating "name (1).txt".
const NODE_NAME = "RandomPromptSelector";

async function uploadFile(file) {
  const body = new FormData();
  body.append("image", file);
  body.append("overwrite", "true");
  const resp = await api.fetchApi("/upload/image", { method: "POST", body });
  if (resp.status !== 200) throw new Error(`${resp.status} ${resp.statusText}`);
  return (await resp.json()).name;
}

function selectFile(node, name) {
  const combo = node.widgets?.find((w) => w.name === "prompt_file");
  if (!combo) return;
  const values = combo.options?.values;
  if (Array.isArray(values) && !values.includes(name)) {
    values.push(name);
    values.sort((a, b) => a.localeCompare(b, undefined, { sensitivity: "base" }));
  }
  combo.value = name;
  combo.callback?.(name);
  app.graph.setDirtyCanvas(true, true);
}

function openPicker(node) {
  const input = document.createElement("input");
  input.type = "file";
  input.accept = ".txt,text/plain";
  input.onchange = async () => {
    const file = input.files?.[0];
    if (!file) return;
    try {
      selectFile(node, await uploadFile(file));
    } catch (e) {
      console.error(`[RandomPromptSelector] upload of ${file.name} failed`, e);
      alert(`Could not upload ${file.name}: ${e.message}`);
    }
  };
  input.click();
}

app.registerExtension({
  name: "RulZnodes.randomPromptSelector",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE_NAME) return;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onNodeCreated?.apply(this, arguments);
      const browse = this.addWidget("button", "browse for text file", null, () => openPicker(this));
      browse.serialize = false; // keep it out of the workflow's saved widget values
      return r;
    };
  },
});
