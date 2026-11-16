import hashlib

import folder_paths

from .load_image_heic import LoadImageWithFilenameHEIC, _list_images

MAX_SLOTS = 16
SLOT_NAMES = tuple(f"image_{i}" for i in range(1, MAX_SLOTS + 1))


def _count_of(value):
    """The slot count as an int clamped to 1..MAX_SLOTS (for use where the value is
    not validated yet)."""
    try:
        return max(1, min(MAX_SLOTS, int(value)))
    except (TypeError, ValueError):
        return 1


class LoadImageReferenceHEIC:
    """Several RulZ Load Image (HEIC) loaders in one node, for reference images.
    `count` slots are shown, validated, hashed and loaded; the frontend
    (web/heic_reference.js) hides the rest and removes their outputs."""

    @classmethod
    def INPUT_TYPES(cls):
        files = _list_images()
        # Only image_1 is required, so a workflow with fewer slots in use carries no
        # stale file names for the hidden ones.
        optional = {name: (files,) for name in SLOT_NAMES[1:]}
        return {
            "required": {
                "count": ("INT", {"default": 1, "min": 1, "max": MAX_SLOTS, "step": 1}),
                SLOT_NAMES[0]: (files,),
            },
            "optional": optional,
        }

    CATEGORY = "RulZnodes"
    RETURN_TYPES = ("IMAGE",) * MAX_SLOTS
    RETURN_NAMES = SLOT_NAMES
    FUNCTION = "load_images"

    def load_images(self, count=1, **slots):
        count = _count_of(count)
        loader = LoadImageWithFilenameHEIC()
        out = []
        for i, name in enumerate(SLOT_NAMES, start=1):
            if i > count:
                out.append(None)
                continue
            # Same loading as the single node (frames, EXIF, MPO, 16-bit); its mask
            # and filename are not used here.
            out.append(loader.load_image(slots[name])[0])
        return tuple(out)

    @classmethod
    def IS_CHANGED(cls, count=1, **slots):
        count = _count_of(count)
        m = hashlib.sha256()
        m.update(str(count).encode())
        for name in SLOT_NAMES[:count]:
            value = slots.get(name)
            m.update(f"\0{name}\0{value}\0".encode("utf-8", "replace"))
            try:
                with open(folder_paths.get_annotated_filepath(value), "rb") as f:
                    for chunk in iter(lambda: f.read(1 << 20), b""):
                        m.update(chunk)
            except (OSError, TypeError, ValueError, KeyError):
                # Validation reports a missing file; nothing to hash here.
                m.update(b"\0missing")
        return m.hexdigest()

    # `**slots` makes ComfyUI skip its own combo check (an image that is not in the
    # list yet, or a stale file in a hidden slot, must not fail the run), so the
    # count and every visible slot are checked here instead.
    @classmethod
    def VALIDATE_INPUTS(cls, count=1, **slots):
        try:
            n = int(count)
        except (TypeError, ValueError):
            return f"Invalid count: {count}"
        if not 1 <= n <= MAX_SLOTS:
            return f"Count must be from 1 to {MAX_SLOTS}, not {n}"
        for i, name in enumerate(SLOT_NAMES[:n], start=1):
            value = slots.get(name)
            if not isinstance(value, str) or not value:
                return f"Image {i}: no image file selected"
            if not folder_paths.exists_annotated_filepath(value):
                return f"Image {i}: invalid image file: {value}"
        return True


NODE_CLASS_MAPPINGS = {"LoadImageReferenceHEIC": LoadImageReferenceHEIC}
NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadImageReferenceHEIC": "RulZ Load Image (HEIC) - Reference"
}
