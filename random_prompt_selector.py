import hashlib
import os
import random

import folder_paths


def _list_text_files():
    input_dir = folder_paths.get_input_directory()
    files = [
        f
        for f in os.listdir(input_dir)
        if f.lower().endswith(".txt") and os.path.isfile(os.path.join(input_dir, f))
    ]
    return sorted(files, key=str.lower)


def _file_path(prompt_file):
    return os.path.join(folder_paths.get_input_directory(), prompt_file)


def _read_prompts(path):
    # utf-8-sig drops the BOM Windows Notepad can add to the start of a file.
    with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
        lines = (line.strip() for line in f)
        return [line for line in lines if line]


class RandomPromptSelector:
    """
    Outputs one line of a text file, chosen by the seed. The file is uploaded into
    ComfyUI's input folder by the button on the node (web/random_prompt_selector.js).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt_file": (_list_text_files(),),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "select"
    CATEGORY = "RulZnodes"

    @classmethod
    def IS_CHANGED(cls, prompt_file, seed):
        # The seed is already part of the cache key; this catches edits to the file.
        with open(_file_path(prompt_file), "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()

    def select(self, prompt_file, seed):
        prompts = _read_prompts(_file_path(prompt_file))
        if not prompts:
            raise ValueError(f"No prompts found in {prompt_file}")
        # A private generator, so the same seed and file always give the same line
        # and the global random state other nodes use is left alone.
        return (random.Random(seed).choice(prompts),)


NODE_CLASS_MAPPINGS = {"RandomPromptSelector": RandomPromptSelector}
NODE_DISPLAY_NAME_MAPPINGS = {"RandomPromptSelector": "RulZ Random Prompt Selector"}
