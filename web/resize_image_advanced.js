import { app } from "../../scripts/app.js";
import { ComfyWidgets } from "../../scripts/widgets.js";

// Shows the computed output resolution on the "RulZ Resize Image Advanced" node
// after it executes (the backend returns it via the node's `ui.text` output).
const NODE_NAME = "ResizeImageV2Advanced";
const WIDGET_NAME = "output_resolution";

function getOrCreateWidget(node) {
    let widget = node.widgets?.find((w) => w.name === WIDGET_NAME);
    if (widget) return widget;

    widget = ComfyWidgets["STRING"](node, WIDGET_NAME, ["STRING", { multiline: true }], app).widget;
    widget.inputEl.readOnly = true;
    widget.inputEl.style.opacity = 0.7;
    widget.inputEl.style.textAlign = "center";
    widget.inputEl.style.fontWeight = "bold";
    widget.inputEl.placeholder = "output resolution (run to compute)";
    // Never send this widget's value to the backend as an input.
    widget.options = widget.options || {};
    widget.options.serialize = false;
    widget.serializeValue = async () => undefined;
    return widget;
}

function setResolution(node, text) {
    const widget = getOrCreateWidget(node);
    widget.value = text ?? "";
    requestAnimationFrame(() => {
        const size = node.computeSize();
        node.setSize([Math.max(node.size[0], size[0]), Math.max(node.size[1], size[1])]);
        app.graph.setDirtyCanvas(true, true);
    });
}

app.registerExtension({
    name: "RulZnodes.resizeOutputResolution",
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
            const text = Array.isArray(message?.text) ? message.text.join("\n") : message?.text;
            setResolution(this, text);
        };

        // Restore the last displayed value when a saved workflow is loaded.
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (info) {
            onConfigure?.apply(this, arguments);
            const widget = getOrCreateWidget(this);
            const values = info?.widgets_values;
            if (Array.isArray(values) && values.length) {
                const last = values[values.length - 1];
                if (typeof last === "string" && /\d+\s*x\s*\d+/.test(last)) {
                    widget.value = last;
                }
            }
        };
    },
});
