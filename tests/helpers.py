"""Tiny IntroSVG-shaped checkpoint for the tests."""
import json
import os
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration

JSON_FILES = ("tokenizer.json", "tokenizer_config.json", "generation_config.json", "preprocessor_config.json")


def checkpoint() -> Path:
    """The IntroSVG Hugging Face checkpoint directory named by INTROSVG_CHECKPOINT; the test is skipped without it."""
    path = os.environ.get("INTROSVG_CHECKPOINT")
    if not path or not Path(path).is_dir():
        raise unittest.SkipTest("set INTROSVG_CHECKPOINT to the IntroSVG checkpoint directory")
    return Path(path)


def make_tiny_checkpoint(folder: Path) -> dict:
    """Write a random small model in the real checkpoint's layout to `folder`; return its config dict."""
    source = checkpoint()
    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    config.update(hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
                  num_key_value_heads=2, max_window_layers=2)
    config["rope_scaling"]["mrope_section"] = [2, 3, 3]
    config["vision_config"].update(hidden_size=32, num_heads=2, intermediate_size=64, out_hidden_size=64)

    torch.manual_seed(0)
    model = Qwen2_5_VLForConditionalGeneration(Qwen2_5_VLConfig(**config)).to(torch.bfloat16)
    # Transformers names the submodules model.language_model.* and model.visual.*; the checkpoint uses model.* and visual.*.
    tensors = {}
    for name, value in model.state_dict().items():
        name = name.replace("model.language_model.", "model.").replace("model.visual.", "visual.")
        tensors[name] = value.contiguous()
    save_file(tensors, folder / "model.safetensors")

    (folder / "config.json").write_text(json.dumps(config), encoding="utf-8")
    for name in JSON_FILES:
        (folder / name).write_bytes((source / name).read_bytes())
    return config
