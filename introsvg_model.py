"""IntroSVG model: GGUF decoder and vision tower as ComfyUI model patchers, and the generation step of the loop."""
import logging
from contextlib import contextmanager

import torch
from tokenizers import Tokenizer

import comfy.model_management
import comfy.model_patcher
from comfy.utils import ProgressBar
from comfy.text_encoders.qwen_vl import qwen2vl_mrope_position_ids

from .decoding import generate
from .introsvg import IM_END, IMAGE_PAD, PromptTooLong
from .qwen2_decoder import DecodeSession, Qwen2Decoder, TensorStore
from .vision import IntroSVGVision

log = logging.getLogger(__name__)

END_OF_TEXT = 151643
MAX_CAPACITY = 16384


def session_capacity(max_new_tokens):
    return min(MAX_CAPACITY, 2 * max_new_tokens + 4096)


class IntroSVGModel:
    """The tokenizer and two patchers over one IntroSVG GGUF; the device and dtype follow ComfyUI (FP16 on GPU, FP32 on CPU)."""

    def __init__(self, reference, device=None):
        self.device = comfy.model_management.get_torch_device() if device is None else torch.device(device)
        dtype = torch.float32 if self.device.type == "cpu" else torch.float16
        store = TensorStore(reference.tensor_index)
        self.tokenizer = Tokenizer.from_str(reference.assets["tokenizer.json"])
        self.decoder = Qwen2Decoder(store, reference.config).to(dtype=dtype)
        self.vision = IntroSVGVision(store, reference.config, dtype)
        offload_device = torch.device("cpu")
        self.decoder_patcher = comfy.model_patcher.ModelPatcher(self.decoder, load_device=self.device, offload_device=offload_device)
        self.vision_patcher = comfy.model_patcher.ModelPatcher(self.vision, load_device=self.device, offload_device=offload_device)
        self.decode_session = None
        self.capture_seconds = 0.0

    @contextmanager
    def session(self, capacity):
        """Load both patchers and hold one KV cache and decode session, with their CUDA graphs, for all phases of one call."""
        comfy.model_management.load_models_gpu([self.decoder_patcher, self.vision_patcher], memory_required=self.decoder.session_bytes(capacity), force_full_load=True)
        self.decode_session = DecodeSession(self.decoder, self.decoder.new_cache(capacity))
        try:
            yield self.decode_session
        finally:
            self.capture_seconds = self.decode_session.capture_seconds
            self.decode_session = None
        comfy.model_management.soft_empty_cache()

    def generate(self, prompt_ids, image, params, max_new_tokens, generator, speculative, loop_guard=True):
        """Prefill the prompt (its `<|image_pad|>` tokens take the embeddings of `image`) and generate; returns the text and a GenerationResult.

        Every generated token advances a progress bar of `max_new_tokens` and checks for an interrupt of the workflow."""
        session = self.decode_session
        capacity = session.cache.keys.shape[3]
        length = len(prompt_ids)
        if length >= capacity:
            raise PromptTooLong(f"The prompt has {length} tokens, the KV cache {capacity} slots")
        if length + max_new_tokens > capacity:
            log.warning("IntroSVG: max_new_tokens %d does not fit the %d-slot KV cache after %d prompt tokens; generating %d", max_new_tokens, capacity, length, capacity - length)
            max_new_tokens = capacity - length
        ids = torch.tensor([prompt_ids], device=self.device)
        embeddings = self.decoder.embed(ids)
        if image is None:
            positions = torch.arange(length, device=self.device).expand(3, -1)
        else:
            image_embeddings, grid = self.vision.encode(image)
            start = prompt_ids.index(IMAGE_PAD)
            embeddings[0, start:start + image_embeddings.shape[0]] = image_embeddings
            positions = qwen2vl_mrope_position_ids([{"type": "image", "index": start, "size": image_embeddings.shape[0], "extra": grid}], length, self.device).long()
        logits = self.decoder.prefill(embeddings, positions, session.cache)
        progress = ProgressBar(max_new_tokens)

        def on_tokens(token):
            comfy.model_management.throw_exception_if_processing_interrupted()
            progress.update(1)

        result = generate(session, logits, prompt_ids, params, max_new_tokens, {IM_END, END_OF_TEXT}, generator, speculative, loop_guard=loop_guard, on_tokens=on_tokens)
        return self.tokenizer.decode(result.tokens), result
