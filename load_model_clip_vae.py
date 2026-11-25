"""Load Diffusion Model, Load CLIP and Load VAE in one node.

The node has the settings of the three ComfyUI nodes and hands them on: it expands into the three nodes, so
ComfyUI loads and caches each part on its own (another VAE does not load the model again), and everything they
learn in a later ComfyUI version (a new CLIP type, for example) is here as well.
"""
from comfy_execution.graph_utils import GraphBuilder
import nodes

# The three nodes, in the order of their outputs here. Their settings have different names.
_LOADERS = ("UNETLoader", "CLIPLoader", "VAELoader")


def _settings(name):
    """The settings of one of the loaders, required and optional, as it has them now."""
    types = nodes.NODE_CLASS_MAPPINGS[name].INPUT_TYPES()
    return {**types.get("required", {}), **types.get("optional", {})}


class RulZLoadModelClipVae:
    @classmethod
    def INPUT_TYPES(cls):
        required = {}
        for name in _LOADERS:
            required.update(_settings(name))
        return {"required": required}

    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    FUNCTION = "load"
    CATEGORY = "RulZnodes"
    DESCRIPTION = "Load Diffusion Model, Load CLIP and Load VAE in one node, with all of their settings."
    SEARCH_ALIASES = ["load diffusion model", "load clip", "load vae", "unet loader", "combined loader"]

    def load(self, **kwargs):
        graph = GraphBuilder()
        outputs = []
        for name in _LOADERS:
            values = {key: kwargs[key] for key in _settings(name) if key in kwargs}
            outputs.append(graph.node(name, **values).out(0))
        return {"result": tuple(outputs), "expand": graph.finalize()}


NODE_CLASS_MAPPINGS = {"RulZLoadModelClipVae": RulZLoadModelClipVae}
NODE_DISPLAY_NAME_MAPPINGS = {"RulZLoadModelClipVae": "RulZ Load Model, CLIP and VAE"}
