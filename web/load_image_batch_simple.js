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
        w = this.addWidget("text", "status", "", () => {}, { serialize: false });
        w.disabled = true;
      }
      w.value = text;
      // Grow to fit the status line, but keep any size the user dragged it to.
      const size = this.computeSize();
      this.setSize([Math.max(this.size[0], size[0]), Math.max(this.size[1], size[1])]);
      app.graph.setDirtyCanvas(true, true);
    };
  },
});
