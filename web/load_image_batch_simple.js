import { app } from "../../scripts/app.js";

// Shows "current / total — filename" on the node after each run.
app.registerExtension({
  name: "RulZnodes.batchStatus",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "LoadImageBatchSimple") return;
    const onExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (message) {
      onExecuted?.apply(this, arguments);
      const text = message?.text?.[0];
      if (!text) return;
      let w = this.widgets?.find((w) => w.name === "status");
      if (!w) {
        // Display only. Not `disabled`: current frontends draw a disabled widget
        // without its value. read_only makes it read-only in Vue nodes mode, and the
        // no-op click keeps the canvas from opening an edit prompt.
        w = this.addWidget("text", "status", "", () => {}, { serialize: false, read_only: true });
        w.onClick = () => {};
      }
      w.value = text;
      // Grow to fit the status line, but keep any size the user dragged it to.
      const size = this.computeSize();
      this.setSize([Math.max(this.size[0], size[0]), Math.max(this.size[1], size[1])]);
      app.graph.setDirtyCanvas(true, true);
    };
  },
});
