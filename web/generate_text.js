import { app } from "../../scripts/app.js";
import { ComfyWidgets } from "../../scripts/widgets.js";

// "RulZ Generate Text": after a run, shows what was written on the node itself. The backend
// returns a status line (answers, tokens, tokens per second, notes) in `ui.text` and the
// answers in `ui.generated`.
const NODE_NAME = "RulZGenerateText";
const WIDGET_NAME = "written";

function getOrCreateWidget(node) {
  let widget = node.widgets?.find((w) => w.name === WIDGET_NAME);
  if (widget) return widget;

  widget = ComfyWidgets["STRING"](node, WIDGET_NAME, ["STRING", { multiline: true }], app).widget;
  // Frontends that draw text widgets without a DOM text box (Vue nodes mode) have no inputEl.
  if (widget.inputEl) {
    widget.inputEl.readOnly = true;
    widget.inputEl.style.opacity = 0.75;
    widget.inputEl.placeholder = "the answers appear here after a run";
  }
  widget.options = widget.options || {};
  widget.options.read_only = true;
  // Display only: never saved with the workflow and never sent to the backend as an input.
  widget.options = widget.options || {};
  widget.options.serialize = false;
  widget.serialize = false;
  widget.serializeValue = async () => undefined;
  return widget;
}

function show(node, status, answers) {
  const widget = getOrCreateWidget(node);
  const parts = [];
  if (status) parts.push(status);
  if (answers.length === 1) {
    parts.push(answers[0]);
  } else {
    answers.forEach((answer, i) => parts.push(`[${i + 1}] ${answer}`));
  }
  widget.value = parts.join("\n\n");
  requestAnimationFrame(() => {
    // Grow to fit, but keep any size the user dragged the node to.
    const size = node.computeSize();
    node.setSize([Math.max(node.size[0], size[0]), Math.max(node.size[1], size[1])]);
    app.graph.setDirtyCanvas(true, true);
  });
}

app.registerExtension({
  name: "RulZnodes.generateText",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE_NAME) return;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onNodeCreated?.apply(this, arguments);
      getOrCreateWidget(this);
      return r;
    };

    const onExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (message) {
      onExecuted?.apply(this, arguments);
      const status = Array.isArray(message?.text) ? message.text.join("\n") : message?.text ?? "";
      const answers = Array.isArray(message?.generated) ? message.generated : [];
      show(this, status, answers);
    };
  },
});
