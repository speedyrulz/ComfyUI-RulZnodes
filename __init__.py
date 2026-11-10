from .deskin_texture import NODE_CLASS_MAPPINGS as _DESKIN_NODES
from .deskin_texture import NODE_DISPLAY_NAME_MAPPINGS as _DESKIN_NAMES
from .disable_comfy_compiler import NODE_CLASS_MAPPINGS as _COMPILER_NODES
from .disable_comfy_compiler import NODE_DISPLAY_NAME_MAPPINGS as _COMPILER_NAMES
from .load_image_batch import NODE_CLASS_MAPPINGS as _BATCH_NODES
from .load_image_batch import NODE_DISPLAY_NAME_MAPPINGS as _BATCH_NAMES
from .load_image_heic import NODE_CLASS_MAPPINGS as _HEIC_NODES
from .load_image_heic import NODE_DISPLAY_NAME_MAPPINGS as _HEIC_NAMES
from .mask_intelligent_crop import NODE_CLASS_MAPPINGS as _CROP_NODES
from .mask_intelligent_crop import NODE_DISPLAY_NAME_MAPPINGS as _CROP_NAMES
from .qwen_sparse_attention import NODE_CLASS_MAPPINGS as _SPARSE_NODES
from .qwen_sparse_attention import NODE_DISPLAY_NAME_MAPPINGS as _SPARSE_NAMES
from .random_prompt_selector import NODE_CLASS_MAPPINGS as _PROMPT_NODES
from .random_prompt_selector import NODE_DISPLAY_NAME_MAPPINGS as _PROMPT_NAMES
from .resize_image_advanced import NODE_CLASS_MAPPINGS as _RESIZE_NODES
from .resize_image_advanced import NODE_DISPLAY_NAME_MAPPINGS as _RESIZE_NAMES

NODE_CLASS_MAPPINGS = {**_HEIC_NODES, **_BATCH_NODES, **_CROP_NODES, **_RESIZE_NODES, **_DESKIN_NODES, **_COMPILER_NODES, **_PROMPT_NODES, **_SPARSE_NODES}
NODE_DISPLAY_NAME_MAPPINGS = {**_HEIC_NAMES, **_BATCH_NAMES, **_CROP_NAMES, **_RESIZE_NAMES, **_DESKIN_NAMES, **_COMPILER_NAMES, **_PROMPT_NAMES, **_SPARSE_NAMES}

# Sidebar tab backend (no node class): directory listing, thumbnails and import routes.
try:
    from . import image_browser  # noqa: F401
except Exception as e:
    print(f"[RulZnodes] Image Browser sidebar disabled: {e}")

WEB_DIRECTORY = "./web"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
