import { app } from "../../scripts/app.js";

// The stock image upload widget has two problems with HEIC/HEIF:
//   1. Its file picker only offers png/jpeg/webp (or image/*, which on Windows
//      usually excludes .heic).
//   2. It drops any selected file whose MIME type doesn't start with "image/".
//      Windows reports an empty type for .heic unless the HEIF codec is installed,
//      so the file is silently ignored.
// The picker <input> is created detached from the document inside a closure, so it
// can't be reached through the DOM. Instead we hook input.click() (and real clicks
// for the Vue node renderer) to widen `accept`, and tag HEIC files with an image
// MIME type before the frontend's own change/drop handlers look at them.

const HEIF_RE = /\.(heic|heif)$/i;
const EXTRA_ACCEPT = ".heic,.heif,image/heic,image/heif";
const PREPARED = Symbol("heicPrepared");

function fixFileTypes(files) {
  for (const file of files ?? []) {
    if (HEIF_RE.test(file.name) && !file.type.startsWith("image/")) {
      const type = /\.heif$/i.test(file.name) ? "image/heif" : "image/heic";
      try {
        Object.defineProperty(file, "type", { value: type, configurable: true });
      } catch (e) {
        console.warn("[ImageLoadFilenameHEIC] could not tag file type", e);
      }
    }
  }
}

function prepareInput(input) {
  if (!(input instanceof HTMLInputElement) || input.type !== "file") return;
  const accept = input.accept ?? "";
  if (accept.includes("image/") && !accept.includes(".heic")) {
    input.accept = accept + "," + EXTRA_ACCEPT;
  }
  if (!input[PREPARED]) {
    input[PREPARED] = true;
    // Capture listener on the input itself runs before the frontend's onchange.
    input.addEventListener("change", () => fixFileTypes(input.files), true);
  }
}

app.registerExtension({
  name: "RulZnodes.heicUpload",
  init() {
    // Classic (canvas) nodes: the picker is opened with input.click() on a detached input.
    const origClick = HTMLInputElement.prototype.click;
    HTMLInputElement.prototype.click = function (...args) {
      prepareInput(this);
      return origClick.apply(this, args);
    };

    // Vue nodes: the user clicks a real <input type="file"> in the document.
    document.addEventListener("click", (e) => prepareInput(e.target), true);
    document.addEventListener(
      "change",
      (e) => {
        if (e.target instanceof HTMLInputElement && e.target.type === "file") {
          fixFileTypes(e.target.files);
        }
      },
      true
    );

    // Drag-and-drop onto a node goes through the same MIME type filter.
    document.addEventListener("drop", (e) => fixFileTypes(e.dataTransfer?.files), true);
  },
});
