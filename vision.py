"""IntroSVG vision path: Qwen2-VL image preprocessing and the vision tower that yields image embeddings."""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from torch import nn

from comfy.ops import manual_cast
from comfy.text_encoders.qwen_vl import Qwen2VLVisionTransformer, qwen2vl_patchify, smart_resize

from .qwen2_decoder import TensorStore

PATCH_SIZE, TEMPORAL_PATCH_SIZE, MERGE_SIZE = 14, 2, 2
MIN_PIXELS, MAX_PIXELS = 3136, 12845056
IMAGE_MEAN = (0.48145466, 0.4578275, 0.40821073)
IMAGE_STD = (0.26862954, 0.26130258, 0.27577711)


def preprocess(image: Image.Image):
    """Return the flattened patches [P, 1176] and `image_grid_thw` [1, 3] of an image."""
    image = image.convert("RGB")
    height, width = smart_resize(image.height, image.width, PATCH_SIZE * MERGE_SIZE, MIN_PIXELS, MAX_PIXELS)
    image = image.resize((width, height), Image.BICUBIC)
    pixels = torch.from_numpy(np.asarray(image, dtype=np.float32) / 255).permute(2, 0, 1)
    mean = torch.tensor(IMAGE_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGE_STD).view(3, 1, 1)
    pixels = ((pixels - mean) / std).unsqueeze(0).repeat(TEMPORAL_PATCH_SIZE, 1, 1, 1)
    return qwen2vl_patchify(pixels, PATCH_SIZE, TEMPORAL_PATCH_SIZE, MERGE_SIZE)


class IntroSVGVision(nn.Module):
    def __init__(self, store: TensorStore, config: dict, dtype: torch.dtype):
        super().__init__()
        vision = config["vision_config"]
        self.dtype = dtype
        self.visual = Qwen2VLVisionTransformer(
            hidden_size=vision["hidden_size"], output_hidden_size=vision["out_hidden_size"],
            intermediate_size=vision["intermediate_size"], num_heads=vision["num_heads"], num_layers=vision["depth"],
            patch_size=vision["patch_size"], temporal_patch_size=vision["temporal_patch_size"],
            spatial_merge_size=vision["spatial_merge_size"], window_size=vision["window_size"],
            device="meta", dtype=dtype, ops=manual_cast)
        state = {name: store.dense("visual." + name, dtype) for name in self.visual.state_dict()}
        self.visual.load_state_dict(state, assign=True)

    def encode(self, image: Image.Image):
        """Return the image embeddings [T, out_hidden_size] and `image_grid_thw` [1, 3]."""
        patches, grid = preprocess(image)
        device = next(self.visual.parameters()).device
        # ComfyUI's vision attention rotates queries and keys in FP32, so the tower runs in FP32 over `dtype` weights (cast per operation).
        return self.visual(patches.to(device, torch.float32), grid).to(self.dtype), grid
