"""Vision path tests against Transformers: image preprocessing and the vision tower on a tiny checkpoint."""
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

import package_alias  # noqa: F401
import cairosvg
import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration
from transformers.models.qwen2_vl.image_processing_pil_qwen2_vl import Qwen2VLImageProcessorPil

from helpers import make_tiny_checkpoint
from introsvg_node.gguf_support import inspect_introsvg_gguf
from introsvg_node.qwen2_decoder import TensorStore
from introsvg_node.vision import IntroSVGVision, preprocess

SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "introsvg_safetensors_to_gguf.py"
spec = importlib.util.spec_from_file_location("_introsvg_converter_vision_test", SCRIPT)
converter = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = converter
spec.loader.exec_module(converter)

SVG = b'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200"><circle cx="100" cy="100" r="60" fill="#d33"/></svg>'


def render():
    png = cairosvg.svg2png(bytestring=SVG, output_width=200, output_height=200, background_color="white")
    return Image.open(io.BytesIO(png)).convert("RGB")


class VisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        cls.source = root / "source"
        cls.source.mkdir()
        make_tiny_checkpoint(cls.source)
        converter.convert(cls.source, root / "f16.gguf", "f16")
        cls.reference = inspect_introsvg_gguf(root / "f16.gguf")
        cls.processor = Qwen2VLImageProcessorPil(**{k: v for k, v in json.loads((cls.source / "preprocessor_config.json").read_text()).items()
                                                  if k not in ("image_processor_type", "processor_class")})
        cls.image = render()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_preprocess_matches_processor(self):
        expected = self.processor(images=self.image, return_tensors="pt")
        patches, grid = preprocess(self.image)
        self.assertEqual(tuple(patches.shape), (196, 1176))
        self.assertEqual(grid.tolist(), [[1, 14, 14]])
        self.assertTrue(torch.equal(grid, expected["image_grid_thw"]))
        self.assertLess((patches - expected["pixel_values"]).abs().max().item(), 1e-5)

    def test_encode_matches_transformers(self):
        vision = IntroSVGVision(TensorStore(self.reference.tensor_index), self.reference.config, torch.float32)
        embeds, grid = vision.encode(self.image)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(self.source, torch_dtype=torch.float32).eval()
        pixels = self.processor(images=self.image, return_tensors="pt")
        with torch.no_grad():
            visual = model.model.visual
            expected = visual(pixels["pixel_values"], grid_thw=pixels["image_grid_thw"])
        expected = getattr(expected, "pooler_output", expected)
        self.assertEqual(tuple(embeds.shape), (49, 64))
        self.assertEqual(grid.tolist(), [[1, 14, 14]])
        self.assertLess((embeds - expected).abs().max().item(), 1e-4)


if __name__ == "__main__":
    unittest.main()
