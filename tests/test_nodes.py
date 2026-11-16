"""Node schemas, loader validation, KV capacity clamp and an end-to-end run of the tiny checkpoint on the CPU."""
import importlib.util
import logging
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import package_alias  # noqa: F401
import torch
from comfy_api.latest import io

import comfy.model_management

from helpers import make_tiny_checkpoint
from introsvg_node import introsvg_model, nodes
from introsvg_node.decoding import GenerationResult, SamplingParams
from introsvg_node.gguf_support import inspect_introsvg_gguf
from introsvg_node import introsvg
from introsvg_node.introsvg import CRITIC_TEMPLATE, chat_ids, render_png

SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "introsvg_safetensors_to_gguf.py"
spec = importlib.util.spec_from_file_location("_introsvg_converter_nodes_test", SCRIPT)
converter = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = converter
spec.loader.exec_module(converter)


class SchemaTests(unittest.TestCase):
    def test_loader_schema(self):
        schema = nodes.LoadIntroSVGGGUF.define_schema()
        self.assertEqual((schema.node_id, schema.display_name, schema.category), ("LoadIntroSVGGGUF", "Load IntroSVG GGUF", "IntroSVG"))
        self.assertEqual([i.id for i in schema.inputs], ["gguf_name"])
        self.assertEqual([o.io_type for o in schema.outputs], ["INTROSVG"])

    def test_text_to_svg_schema(self):
        schema = nodes.IntroSVGTextToSVG.define_schema()
        self.assertEqual((schema.node_id, schema.display_name, schema.category), ("IntroSVGTextToSVG", "IntroSVG Text to SVG", "IntroSVG"))
        defaults = {i.id: getattr(i, "default", None) for i in schema.inputs}
        self.assertEqual(defaults, {"model": None, "prompt": "a red fox sitting, flat icon", "seed": 0, "critic_rounds": 3, "score_threshold": 9.5,
                                    "temperature": 0.5, "max_new_tokens": 6000, "speculative": True, "refine_temperature": 0.0, "repetition_penalty": 1.05,
                                    "critique_temperature": 0.0, "loop_guard": True})
        self.assertEqual([o.io_type for o in schema.outputs], ["SVG"])
        critic = next(i for i in schema.inputs if i.id == "critic_rounds")
        self.assertEqual((critic.min, critic.max), (0, 10))


class TinyModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        source = root / "source"
        source.mkdir()
        make_tiny_checkpoint(source)
        converter.convert(source, root / "q8_0.gguf", "q8_0")
        cls.path = root / "q8_0.gguf"
        cls.model = introsvg_model.IntroSVGModel(inspect_introsvg_gguf(cls.path), device="cpu")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_loader_rejects_other_architecture(self):
        other = Path(self.tmp.name) / "other.gguf"
        import gguf
        writer = gguf.GGUFWriter(str(other), "llama")
        writer.add_string("introsvg.architecture", "starvector")
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
        with mock.patch.object(nodes.folder_paths, "get_full_path", return_value=str(other)), \
                mock.patch.object(nodes.folder_paths, "get_folder_paths", return_value=[self.tmp.name]):
            with self.assertRaisesRegex(ValueError, "introsvg"):
                nodes.LoadIntroSVGGGUF.execute("other.gguf")

    def test_symlink_to_extensionless_file_loads(self):
        blob = Path(self.tmp.name) / "blob"
        shutil.copy(self.path, blob)
        link = Path(self.tmp.name) / "linked.gguf"
        link.symlink_to(blob)
        self.assertEqual(inspect_introsvg_gguf(link).config, inspect_introsvg_gguf(self.path).config)

    def test_capacity(self):
        self.assertEqual(introsvg_model.session_capacity(6000), 16096)
        self.assertEqual(introsvg_model.session_capacity(16000), 16384)
        self.assertEqual(introsvg_model.session_capacity(64), 4224)

    def test_capacity_clamp(self):
        calls = []

        def fake_generate(session, logits, prompt_ids, params, max_new_tokens, stop_ids, generator, speculative, loop_guard, on_tokens):
            calls.append(max_new_tokens)
            return GenerationResult([1000, 1001], 1, 0, 0, 0.1, "max_new_tokens")

        prompt = [1, 2, 3, 4, 5]
        sampling = SamplingParams(0.0)
        with mock.patch.object(introsvg_model, "generate", fake_generate), self.model.session(32):
            with self.assertLogs(introsvg_model.log, logging.WARNING):
                self.model.generate(prompt, None, sampling, 100, None, False)
            self.assertEqual(calls, [27])
            with self.assertNoLogs(introsvg_model.log, logging.WARNING):
                self.model.generate(prompt, None, sampling, 20, None, False)
            self.assertEqual(calls, [27, 20])
            with self.assertRaisesRegex(ValueError, "prompt has 40 tokens"):
                self.model.generate(list(range(40)), None, sampling, 1, None, False)
        self.assertIsNone(self.model.decode_session)

    def test_generate_with_image(self):
        image = render_png('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200"><circle cx="100" cy="100" r="60" fill="#d33"/></svg>')
        ids = chat_ids(self.model.tokenizer, CRITIC_TEMPLATE.format(origin_prompt="a ball"), 49)
        generator = torch.Generator().manual_seed(0)
        with torch.inference_mode(), self.model.session(len(ids) + 16):
            text, result = self.model.generate(ids, image, SamplingParams(0.0), 8, generator, True)
        self.assertEqual(len(result.tokens), 8)
        self.assertIsInstance(text, str)

    def test_loader_accepts_a_symlinked_model(self):
        with tempfile.TemporaryDirectory() as folder:
            os.symlink(self.path, Path(folder) / "linked.gguf")
            with mock.patch.dict(nodes.folder_paths.folder_names_and_paths, {"introsvg": ([folder], {".gguf"})}):
                loaded = nodes.LoadIntroSVGGGUF.execute("linked.gguf").result[0]
        self.assertIsInstance(loaded, introsvg_model.IntroSVGModel)

    def scripted(self, replies):
        """A stand-in for decoding.generate that answers each phase with the next reply, tokenized."""
        ids = [self.model.tokenizer.encode(reply, add_special_tokens=False).ids for reply in replies]

        def fake_generate(*args, **kwargs):
            return GenerationResult(ids.pop(0), 1, 0, 0, 0.1, "stop_token")

        return mock.patch.object(introsvg_model, "generate", fake_generate)

    def test_text_to_svg_end_to_end(self):
        svg = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200"><circle cx="100" cy="100" r="60" fill="#d33"/></svg>'
        critique = '{"score": 9.8, "critique": "good", "suggestions": "none"}'
        with self.scripted([svg, critique]), torch.inference_mode():
            output = nodes.IntroSVGTextToSVG.execute(self.model, "a red ball", 0, 1, 9.5, 0.5, 64, True)
        self.assertIsInstance(output.result[0], io.SVG.Type)
        self.assertEqual(output.result[0].data[0].getvalue(), svg.encode("utf-8"))
        self.assertIsNone(self.model.decode_session)

    def test_text_to_svg_draft_failure_names_the_draft(self):
        with self.scripted(["no svg here"] * 3), self.assertLogs(introsvg.log, logging.WARNING), self.assertRaisesRegex(ValueError, "draft failed in 3 attempts"):
            nodes.IntroSVGTextToSVG.execute(self.model, "a red ball", 0, 0, 9.5, 0.5, 64, True)
        self.assertIsNone(self.model.decode_session)

    def test_tokens_advance_the_progress_bar(self):
        ids = [1000, 1001, 1002]
        bar = mock.Mock()

        def fake_generate(*args, on_tokens, **kwargs):
            for token in ids:
                on_tokens(token)
            return GenerationResult(ids, 1, 0, 0, 0.1, "max_new_tokens")

        with mock.patch.object(introsvg_model, "generate", fake_generate), mock.patch.object(introsvg_model, "ProgressBar", return_value=bar) as bar_class, self.model.session(64):
            self.model.generate([1, 2, 3], None, SamplingParams(0.0), 10, None, False)
        bar_class.assert_called_once_with(10)
        self.assertEqual(bar.update.call_count, 3)

    def test_interrupt_stops_a_running_phase_and_releases_the_session(self):
        ids = [1, 2, 3, 4, 5]
        generator = torch.Generator().manual_seed(0)
        comfy.model_management.interrupt_current_processing(True)
        self.addCleanup(comfy.model_management.interrupt_current_processing, False)
        with self.assertRaises(comfy.model_management.InterruptProcessingException), self.model.session(64):
            self.model.generate(ids, None, SamplingParams(0.0), 8, generator, False)
        self.assertIsNone(self.model.decode_session)

    def test_interrupt_between_tokens_ends_the_phase(self):
        bar = mock.Mock()
        updates = []

        def fake_generate(*args, on_tokens, **kwargs):
            for token in range(1000, 1010):
                on_tokens(token)
            return GenerationResult([], 1, 0, 0, 0.1, "max_new_tokens")

        def update(count):
            updates.append(count)
            if len(updates) == 3:
                comfy.model_management.interrupt_current_processing(True)

        bar.update.side_effect = update
        self.addCleanup(comfy.model_management.interrupt_current_processing, False)
        with mock.patch.object(introsvg_model, "generate", fake_generate), mock.patch.object(introsvg_model, "ProgressBar", return_value=bar):
            with self.assertRaises(comfy.model_management.InterruptProcessingException), self.model.session(64):
                self.model.generate([1, 2, 3], None, SamplingParams(0.0), 10, None, False)
        self.assertEqual(len(updates), 3)
        self.assertIsNone(self.model.decode_session)



@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class CudaGenerateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        source = root / "source"
        source.mkdir()
        make_tiny_checkpoint(source)
        converter.convert(source, root / "q8_0.gguf", "q8_0")
        cls.model = introsvg_model.IntroSVGModel(inspect_introsvg_gguf(root / "q8_0.gguf"), device="cuda")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_phases_share_one_session_and_speculation_matches_plain_greedy(self):
        image = render_png('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200"><circle cx="100" cy="100" r="60" fill="#d33"/></svg>')
        repeated = chat_ids(self.model.tokenizer, "ab cd ab cd ab cd ab cd ab cd ab cd")
        critic = chat_ids(self.model.tokenizer, CRITIC_TEMPLATE.format(origin_prompt="a ball"), 49)
        phases = [(repeated, None), (critic, image), (repeated, None)]
        sampling = SamplingParams(0.0, repetition_penalty=1.0)  # greedy decoding of a random model cycles, which gives the drafts
        outputs, drafted = {}, 0
        for speculative in (False, True):
            outputs[speculative] = []
            with torch.inference_mode(), self.model.session(1024):
                for ids, phase_image in phases:
                    text, result = self.model.generate(ids, phase_image, sampling, 64, torch.Generator(device="cuda").manual_seed(0), speculative)
                    outputs[speculative].append(result.tokens)
                    drafted += result.drafted
            self.assertIsNone(self.model.decode_session)
        self.assertGreater(drafted, 0)
        for plain, fast in zip(outputs[False], outputs[True]):
            self.assertEqual(len(plain), 64)
            self.assertEqual(plain, fast)


if __name__ == "__main__":
    unittest.main()
