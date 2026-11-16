#!/usr/bin/env python3
"""Convert the IntroSVG Safetensors checkpoint into a self-contained quantized GGUF.

Tensors keep the checkpoint's names. Decoder matrices take the requested
quantization, the token embedding takes Q8_0 in the Q5_0 and Q4_0 files, the
output head takes Q8_0 in the Q5_0 file and Q5_0 in the Q4_0 file, the vision tower is F16, and biases and norms are F32. The GGUF
embeds the model config, tokenizer, generation and image preprocessing files
that ComfyUI-IntroSVG reads.

Runs on the CPU with gguf, safetensors, numpy and torch, one tensor at a time.
"""
import argparse
import json
from math import prod
from pathlib import Path

import gguf
import numpy as np
import torch
from safetensors import safe_open

ARCHITECTURE_KEY = "introsvg.architecture"
CONFIG_KEY = "introsvg.config_json"
ASSET_PREFIX = "introsvg.embedded_file."
ASSET_FILES = ("tokenizer.json", "tokenizer_config.json", "generation_config.json", "preprocessor_config.json")
FILE_TYPES = {"f16": "F16", "q8_0": "Q8_0", "q5_0": "Q5_0", "q4_0": "Q4_0"}
HEAD_TENSORS = ("model.embed_tokens.weight", "lm_head.weight")


def target_type(name: str, shape: tuple[int, ...], quantization: str) -> str:
    """GGML type name for a tensor of the checkpoint."""
    if len(shape) == 1:
        return "F32"
    if name.startswith("visual."):
        return "F16"
    file_type = FILE_TYPES[quantization]
    if name == "lm_head.weight" and quantization == "q4_0":
        return "Q5_0"
    if name in HEAD_TENSORS:
        return "Q8_0" if quantization in ("q4_0", "q5_0") else file_type
    if len(shape) == 2 and shape[-1] % 32 == 0:
        return file_type
    return "F16"


def packed_shape(shape: tuple[int, ...], ggml_type: gguf.GGMLQuantizationType) -> tuple[int, ...]:
    """The GGUF writer takes the packed byte shape of quantized tensors."""
    block_size, block_bytes = gguf.GGML_QUANT_SIZES[ggml_type]
    return (*shape[:-1], shape[-1] // block_size * block_bytes)


def convert_tensor(sf, name: str, shape: tuple[int, ...], ggml_type: gguf.GGMLQuantizationType, chunk_mib: int) -> np.ndarray:
    if ggml_type == gguf.GGMLQuantizationType.F32:
        return sf.get_tensor(name).to(torch.float32).numpy()
    if ggml_type == gguf.GGMLQuantizationType.F16:
        return sf.get_tensor(name).to(torch.float16).numpy()
    result = np.empty(packed_shape(shape, ggml_type), dtype=np.uint8)
    tensor_slice = sf.get_slice(name)
    step = max(1, chunk_mib * 1024 * 1024 // (prod(shape[1:]) * 4))
    for start in range(0, shape[0], step):
        block = tensor_slice[start:start + step].to(torch.float32).numpy()
        # Blocks whose largest magnitude underflows the FP16 scale are stored with scale 0 and dequantize to 0.
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            result[start:start + step] = gguf.quants.quantize(block, ggml_type)
    return result


def convert(model_dir: Path, output: Path, quantization: str, chunk_mib: int = 32) -> None:
    model_dir = Path(model_dir)
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Output exists; refusing to overwrite: {output}")
    shards = sorted(model_dir.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"No *.safetensors files in {model_dir}")

    plans = []
    for shard in shards:
        with safe_open(shard, framework="pt") as sf:
            for name in sf.keys():
                shape = tuple(sf.get_slice(name).get_shape())
                plans.append((shard, name, shape, gguf.GGMLQuantizationType[target_type(name, shape, quantization)]))
    names = [name for _, name, _, _ in plans]
    if len(set(names)) != len(names):
        raise ValueError("Duplicate tensor names across the Safetensors files")

    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    assets = {name: (model_dir / name).read_text(encoding="utf-8") for name in ASSET_FILES}

    output.parent.mkdir(parents=True, exist_ok=True)
    writer = gguf.GGUFWriter(str(output), arch="introsvg")
    try:
        writer.add_string(ARCHITECTURE_KEY, "introsvg")
        writer.add_string(CONFIG_KEY, json.dumps(config, ensure_ascii=False, separators=(",", ":")))
        for name, text in assets.items():
            writer.add_string(ASSET_PREFIX + name, text)
        for _, name, shape, ggml_type in plans:
            if ggml_type in (gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16):
                dtype = np.float32 if ggml_type == gguf.GGMLQuantizationType.F32 else np.float16
                writer.add_tensor_info(name, shape, np.dtype(dtype), prod(shape) * np.dtype(dtype).itemsize, raw_dtype=ggml_type)
            else:
                packed = packed_shape(shape, ggml_type)
                writer.add_tensor_info(name, packed, np.dtype(np.uint8), prod(packed), raw_dtype=ggml_type)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_ti_data_to_file()
        position = 0
        for shard in shards:
            with safe_open(shard, framework="pt") as sf:
                for plan_shard, name, shape, ggml_type in plans:
                    if plan_shard != shard:
                        continue
                    position += 1
                    print(f"[{position}/{len(plans)}] {name}: {ggml_type.name}", flush=True)
                    writer.write_tensor_data(convert_tensor(sf, name, shape, ggml_type, chunk_mib))
        writer.close()
    except BaseException:
        writer.close()
        output.unlink(missing_ok=True)
        raise
    print(f"Wrote {len(plans)} tensors to {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path, help="IntroSVG Safetensors directory")
    parser.add_argument("--output", required=True, type=Path, help="Path of the new GGUF file")
    parser.add_argument("--quantization", required=True, choices=tuple(FILE_TYPES), help="Type of the decoder matrices")
    parser.add_argument("--chunk-mib", type=int, default=32, help="Largest float32 chunk, in MiB, held during quantization")
    args = parser.parse_args()
    convert(args.model_dir, args.output, args.quantization, args.chunk_mib)


if __name__ == "__main__":
    main()
