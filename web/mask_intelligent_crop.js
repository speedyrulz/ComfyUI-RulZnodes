import { app } from "../../scripts/app.js";

// "RulZ Mask Intelligent Crop": shows padding_percent in uniform mode and the
// four per-edge paddings in per_edge mode. Hidden widgets keep their values.
const NODE_NAME = "MaskIntelligentCrop";
const UNIFORM_WIDGETS = ["padding_percent"];
const EDGE_WIDGETS = ["padding_top", "padding_bottom", "padding_left", "padding_right"];
// Widget order before padding_mode and the per-edge inputs existed.
const LEGACY_WIDGETS = ["padding_percent", "padding_reference", "aspect_limit", "divisible_by", "mask_threshold"];

function setWidgetVisible(widget, visible) {
    if (!widget) return;
    if (widget._rulzOrigType === undefined) {
        widget._rulzOrigType = widget.type;
        widget._rulzOrigComputeSize = widget.computeSize;
    }
    widget.hidden = !visible;
    widget.type = visible ? widget._rulzOrigType : "hidden";
    widget.computeSize = visible ? widget._rulzOrigComputeSize : () => [0, -4];
}

function updateVisibility(node) {
    const mode = node.widgets?.find((w) => w.name === "padding_mode")?.value;
    const perEdge = mode === "per_edge";
    for (const name of UNIFORM_WIDGETS) setWidgetVisible(node.widgets.find((w) => w.name === name), !perEdge);
    for (const name of EDGE_WIDGETS) setWidgetVisible(node.widgets.find((w) => w.name === name), perEdge);
    requestAnimationFrame(() => {
        node.setSize([node.size[0], node.computeSize()[1]]);
        app.graph.setDirtyCanvas(true, true);
    });
}

app.registerExtension({
    name: "RulZnodes.maskCropPadding",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_NAME) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated?.apply(this, arguments);
            const modeWidget = this.widgets?.find((w) => w.name === "padding_mode");
            if (modeWidget) {
                const callback = modeWidget.callback;
                modeWidget.callback = (...args) => {
                    const cr = callback?.apply(modeWidget, args);
                    updateVisibility(this);
                    return cr;
                };
            }
            updateVisibility(this);
            return r;
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (info) {
            const r = onConfigure?.apply(this, arguments);
            // Workflows saved before padding_mode existed store 5 values in the
            // old order; put each back on its widget by name and start the
            // per-edge paddings from the old uniform value.
            const values = info?.widgets_values;
            if (Array.isArray(values) && values.length === LEGACY_WIDGETS.length && typeof values[0] === "number") {
                LEGACY_WIDGETS.forEach((name, i) => {
                    const widget = this.widgets?.find((w) => w.name === name);
                    if (widget) widget.value = values[i];
                });
                for (const name of EDGE_WIDGETS) {
                    const widget = this.widgets?.find((w) => w.name === name);
                    if (widget) widget.value = values[0];
                }
                const modeWidget = this.widgets?.find((w) => w.name === "padding_mode");
                if (modeWidget) modeWidget.value = "uniform";
            }
            updateVisibility(this);
            return r;
        };
    },
});
