import importlib
import traceback

# Each module is loaded on its own, so one that fails to import (a ComfyUI version
# without an API it needs, a missing optional package) disables only its own nodes.
_MODULES = (
    "load_image_heic",
    "load_image_batch",
    "mask_intelligent_crop",
    "resize_image_advanced",
    "deskin_texture",
    "disable_comfy_compiler",
    "random_prompt_selector",
    "qwen_sparse_attention",
    "person_select",
)

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
for _name in _MODULES:
    try:
        _module = importlib.import_module(f".{_name}", __name__)
    except Exception as e:
        print(f"[RulZnodes] {_name} disabled: {e}")
        traceback.print_exc()
        continue
    NODE_CLASS_MAPPINGS.update(_module.NODE_CLASS_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(_module.NODE_DISPLAY_NAME_MAPPINGS)

# Sidebar tab backend (no node class): directory listing, thumbnails and import routes.
try:
    from . import image_browser  # noqa: F401
except Exception as e:
    print(f"[RulZnodes] Image Browser sidebar disabled: {e}")

WEB_DIRECTORY = "./web"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
