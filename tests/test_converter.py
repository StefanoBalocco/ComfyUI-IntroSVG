"""Converter tests: per-tensor types and a round trip through a tiny checkpoint."""
import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
import warnings
from pathlib import Path

import gguf
import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

import package_alias  # noqa: F401
from helpers import make_tiny_checkpoint
from introsvg_node.gguf_support import inspect_introsvg_gguf

SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "introsvg_safetensors_to_gguf.py"
spec = importlib.util.spec_from_file_location("_introsvg_converter_test", SCRIPT)
converter = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = converter
spec.loader.exec_module(converter)


class ConverterTests(unittest.TestCase):
    def test_target_types(self):
        t = converter.target_type
        self.assertEqual(t("model.layers.0.mlp.up_proj.weight", (18944, 3584), "q4_0"), "Q4_0")
        self.assertEqual(t("model.layers.0.self_attn.q_proj.bias", (3584,), "q4_0"), "F32")
        self.assertEqual(t("model.layers.0.input_layernorm.weight", (3584,), "q8_0"), "F32")
        self.assertEqual(t("model.embed_tokens.weight", (152064, 3584), "q4_0"), "Q8_0")
        self.assertEqual(t("lm_head.weight", (152064, 3584), "q4_0"), "Q5_0")
        self.assertEqual(t("lm_head.weight", (152064, 3584), "q5_0"), "Q8_0")
        self.assertEqual(t("lm_head.weight", (152064, 3584), "q8_0"), "Q8_0")
        self.assertEqual(t("visual.blocks.0.mlp.down_proj.weight", (1280, 3420), "q4_0"), "F16")
        self.assertEqual(t("visual.patch_embed.proj.weight", (1280, 3, 2, 14, 14), "q8_0"), "F16")
        self.assertEqual(t("visual.blocks.0.norm1.weight", (1280,), "q8_0"), "F32")
        self.assertEqual(t("model.layers.0.mlp.down_proj.weight", (3584, 18944), "f16"), "F16")

    def test_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            source = tmp / "source"
            source.mkdir()
            config = make_tiny_checkpoint(source)
            with safe_open(source / "model.safetensors", framework="pt") as sf:
                expected = {name: sf.get_tensor(name).float() for name in sf.keys()}
            for quantization in ("f16", "q8_0", "q5_0", "q4_0"):
                with self.subTest(quantization=quantization):
                    output = tmp / f"{quantization}.gguf"
                    with contextlib.redirect_stdout(io.StringIO()):
                        converter.convert(source, output, quantization)
                    reference = inspect_introsvg_gguf(output)
                    self.assertEqual(reference.config["vision_config"]["depth"], 32)
                    self.assertEqual(reference.config["num_hidden_layers"], config["num_hidden_layers"])
                    json.loads(reference.assets["tokenizer.json"])
                    self.assertEqual(set(reference.tensor_index), set(expected))
                    for name, tensor in reference.tensor_index.items():
                        source_tensor = expected[name]
                        self.assertEqual(tuple(tensor.shape), tuple(reversed(source_tensor.shape)), name)
                        target = converter.target_type(name, tuple(source_tensor.shape), quantization)
                        self.assertEqual(tensor.tensor_type.name, target, name)
                        if target in ("F16", "F32"):
                            self.assertTrue(torch.allclose(torch.from_numpy(tensor.data.astype("float32")).reshape(source_tensor.shape),
                                                           source_tensor, atol=1e-3), name)
                        else:
                            restored = gguf.quants.dequantize(tensor.data, tensor.tensor_type)
                            error = (torch.from_numpy(restored).reshape(source_tensor.shape) - source_tensor).abs().max()
                            self.assertLess(error, 0.1 * source_tensor.abs().max(), name)

    def test_underflowing_blocks_quantize_to_zero(self):
        data = torch.zeros(2, 64)
        data[1] = 1e-40
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "t.safetensors"
            save_file({"w": data}, path)
            for quantization in ("q4_0", "q5_0", "q8_0"):
                with self.subTest(quantization=quantization):
                    ggml_type = gguf.GGMLQuantizationType[converter.FILE_TYPES[quantization]]
                    with safe_open(path, framework="pt") as sf, warnings.catch_warnings():
                        warnings.simplefilter("error")
                        packed = converter.convert_tensor(sf, "w", (2, 64), ggml_type, 32)
                        restored = gguf.quants.dequantize(packed, ggml_type)
                    self.assertTrue(np.isfinite(restored).all())
                    self.assertFalse(restored.any())

    def test_rejects_missing_architecture(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "other.gguf"
            writer = gguf.GGUFWriter(str(path), arch="introsvg")
            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_ti_data_to_file()
            writer.close()
            with self.assertRaisesRegex(ValueError, "introsvg.architecture"):
                inspect_introsvg_gguf(path)


if __name__ == "__main__":
    unittest.main()
