"""Per-workflow replacement for the --disable-comfy-compiler launch flag.

ComfyUI reads `args.disable_comfy_compiler` / `args.disable_cuda_graphs` at call time
(comfy/model_prefetch.py: malloc_graph_enabled / prefetch_queue_pop), so we flip them only
while the patched model's diffusion forward runs, then restore them.

With CFG split across GPUs, ComfyUI runs the forward passes for cond and uncond at the
same time in pool threads. The flags are global, so they are set when the first pass
starts and restored when the last one ends; restoring per pass would let one thread
switch the compiler back on under another, or leave the flags set for good.
"""
import threading

from comfy.cli_args import args

_FLAGS = ("disable_comfy_compiler", "disable_cuda_graphs")
_state_lock = threading.Lock()
_active = 0  # forward passes currently inside the wrapper
_saved = None  # flag values from before the first of them


def _enter():
    global _active, _saved
    with _state_lock:
        if _active == 0:
            # getattr: ComfyUI builds from before the compiler don't have these flags.
            _saved = tuple(getattr(args, name, False) for name in _FLAGS)
            for name in _FLAGS:
                setattr(args, name, True)
        _active += 1


def _exit():
    global _active
    with _state_lock:
        _active -= 1
        if _active == 0:
            for name, value in zip(_FLAGS, _saved):
                setattr(args, name, value)


class DisableComfyCompiler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "disable": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "RulZnodes"

    def patch(self, model, disable=True):
        if not disable:
            return (model,)

        m = model.clone()
        previous = m.model_options.get("model_function_wrapper")

        def wrapper(apply_model, wrapper_args):
            _enter()
            try:
                if previous is not None:
                    return previous(apply_model, wrapper_args)
                return apply_model(wrapper_args["input"], wrapper_args["timestep"], **wrapper_args["c"])
            finally:
                _exit()

        m.set_model_unet_function_wrapper(wrapper)
        return (m,)


NODE_CLASS_MAPPINGS = {"DisableComfyCompiler": DisableComfyCompiler}
NODE_DISPLAY_NAME_MAPPINGS = {"DisableComfyCompiler": "RulZ Disable Comfy Compiler"}
