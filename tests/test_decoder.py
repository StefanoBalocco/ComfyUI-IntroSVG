"""Qwen2 decoder tests against Transformers on a tiny checkpoint: prefill, M-RoPE, decode and verify steps."""
import importlib.util
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import package_alias  # noqa: F401
import torch
from transformers import Qwen2_5_VLForConditionalGeneration

from helpers import make_tiny_checkpoint
from introsvg_node.gguf_support import inspect_introsvg_gguf
from introsvg_node import quantization as quant
from introsvg_node.qwen2_decoder import DecodeSession, Qwen2Decoder, TensorStore

SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "introsvg_safetensors_to_gguf.py"
spec = importlib.util.spec_from_file_location("_introsvg_converter_decoder_test", SCRIPT)
converter = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = converter
spec.loader.exec_module(converter)

IMAGE_PAD, VISION_START, VISION_END = 151655, 151652, 151653
CUDA = torch.cuda.is_available()


def last_logits(model, **inputs):
    with torch.no_grad():
        return model(**inputs).logits[0].float()


class DecoderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        cls.source = root / "source"
        cls.source.mkdir()
        cls.config = make_tiny_checkpoint(cls.source)
        for quantization in ("f16", "q8_0", "q4_0"):
            converter.convert(cls.source, root / f"{quantization}.gguf", quantization)
        cls.root = root
        cls.reference = Qwen2_5_VLForConditionalGeneration.from_pretrained(cls.source, torch_dtype=torch.float32).eval()
        generator = torch.Generator().manual_seed(1)
        cls.ids = torch.randint(0, 1000, (1, 12), generator=generator)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def decoder(self, quantization="f16", dtype=torch.float32, device="cpu"):
        reference = inspect_introsvg_gguf(self.root / f"{quantization}.gguf")
        decoder = Qwen2Decoder(TensorStore(reference.tensor_index), reference.config)
        return decoder.to(device=device, dtype=dtype).eval()

    @staticmethod
    def text_positions(length, device="cpu"):
        return torch.arange(length, device=device)[None].expand(3, -1)

    def prefill(self, decoder, ids, device="cpu", capacity=1024):
        cache = decoder.new_cache(capacity)
        logits = decoder.prefill(decoder.embed(ids.to(device)), self.text_positions(ids.shape[1], device), cache)
        return cache, logits

    def test_prefill_matches_transformers(self):
        ids = self.ids[:, :7]
        decoder = self.decoder()
        cache, logits = self.prefill(decoder, ids)
        expected = last_logits(self.reference, input_ids=ids)[-1]
        torch.testing.assert_close(logits, expected, atol=1e-4, rtol=0)
        self.assertEqual((cache.length, cache.offset), (7, 0))

    def test_mrope_with_image_positions(self):
        ids = torch.tensor([[1, 2, VISION_START] + [IMAGE_PAD] * 4 + [VISION_END, 5, 6, 7]])
        grid = torch.tensor([[1, 4, 4]])
        positions, _ = self.reference.model.get_rope_index(ids, image_grid_thw=grid, mm_token_type_ids=(ids == IMAGE_PAD).long(), attention_mask=torch.ones_like(ids))
        decoder = self.decoder()
        embeddings = decoder.embed(ids)
        cache = decoder.new_cache(64)
        logits = decoder.prefill(embeddings, positions[:, 0], cache)
        expected = last_logits(self.reference, inputs_embeds=embeddings, position_ids=positions)[-1]
        torch.testing.assert_close(logits, expected, atol=1e-4, rtol=0)
        self.assertEqual(cache.offset, int(positions.max()) + 1 - ids.shape[1])
        self.assertLess(cache.offset, 0)
        # Generated tokens continue at slot + offset.
        session = DecodeSession(decoder, cache)
        step = session.step([9])[0]
        full = torch.cat([ids, torch.tensor([[9]])], 1)
        full_positions = torch.cat([positions, (torch.arange(1) + positions.max() + 1).expand(3, 1, 1)], 2)
        expected = last_logits(self.reference, inputs_embeds=decoder.embed(full), position_ids=full_positions)[-1]
        torch.testing.assert_close(step, expected, atol=1e-4, rtol=0)

    def run_decode_equals_prefill(self, quantization, dtype, device, atol):
        decoder = self.decoder(quantization, dtype, device)
        ids = self.ids[:, :9]
        full = torch.stack([self.prefill(decoder, ids[:, :n], device)[1] for n in range(6, 10)])
        cache, _ = self.prefill(decoder, ids[:, :5], device)
        session = DecodeSession(decoder, cache)
        rows = torch.stack([session.step([int(t)])[0] for t in ids[0, 5:9]])
        torch.testing.assert_close(rows, full, atol=atol, rtol=0)
        self.assertEqual(cache.length, 9)

    def test_decode_equals_prefill(self):
        self.run_decode_equals_prefill("f16", torch.float32, "cpu", 1e-4)

    def test_span_growth(self):
        decoder = self.decoder()
        ids = torch.randint(0, 1000, (1, 514), generator=torch.Generator().manual_seed(2))
        cache, _ = self.prefill(decoder, ids[:, :510])
        session = DecodeSession(decoder, cache)
        rows = session.step(ids[0, 510:].tolist())
        torch.testing.assert_close(rows[-1], self.prefill(decoder, ids)[1], atol=1e-4, rtol=0)
        torch.testing.assert_close(session.step([5])[0], self.prefill(decoder, torch.cat([ids, torch.tensor([[5]])], 1))[1], atol=1e-4, rtol=0)

    def run_verify_equals_decode(self, quantization, dtype, device, atol):
        decoder = self.decoder(quantization, dtype, device)
        ids = self.ids[0, 5:9].tolist()
        cache, _ = self.prefill(decoder, self.ids[:, :5], device)
        verify = DecodeSession(decoder, cache).step(ids)
        self.assertEqual(cache.length, 9)
        cache, _ = self.prefill(decoder, self.ids[:, :5], device)
        session = DecodeSession(decoder, cache)
        singles = torch.cat([session.step([t]) for t in ids])
        torch.testing.assert_close(verify, singles, atol=atol, rtol=0)
        # Rewinding to slot 6 and appending reproduces a fresh session's step at that slot.
        session.rewind(6)
        again = session.step([77])
        fresh_cache, _ = self.prefill(decoder, self.ids[:, :5], device)
        fresh = DecodeSession(decoder, fresh_cache)
        fresh.step(ids[:1])
        torch.testing.assert_close(again, fresh.step([77]), atol=atol, rtol=0)

    def test_verify_equals_decode(self):
        self.run_verify_equals_decode("f16", torch.float32, "cpu", 1e-4)

    @unittest.skipUnless(CUDA, "CUDA required")
    def test_cuda_decode_and_verify(self):
        self.run_decode_equals_prefill("q8_0", torch.float16, "cuda", 5e-2)
        self.run_verify_equals_decode("q8_0", torch.float16, "cuda", 5e-2)

    @unittest.skipUnless(CUDA, "CUDA required")
    def test_step_waits_on_blocking_event(self):
        decoder = self.decoder("q8_0", torch.float16, "cuda")
        cache, _ = self.prefill(decoder, self.ids[:, :5], "cuda")
        with mock.patch.object(torch.cuda, "Event", wraps=torch.cuda.Event) as event:
            session = DecodeSession(decoder, cache)
        event.assert_called_once_with(blocking=True)
        with mock.patch.object(session.done, "synchronize", wraps=session.done.synchronize) as wait:
            session.step([3])
            session.rewind(5)
            session.step([3])
        self.assertEqual(wait.call_count, 2)

    @unittest.skipUnless(CUDA, "CUDA required")
    def test_graph_replay(self):
        decoder = self.decoder("q8_0", torch.float16, "cuda")
        cache, _ = self.prefill(decoder, self.ids[:, :5], "cuda")
        first = DecodeSession(decoder, cache)
        rows = first.step([3])
        self.assertGreater(first.capture_seconds, 0)
        first.rewind(5)
        second = DecodeSession(decoder, cache)
        replay = second.step([3])
        self.assertEqual(second.capture_seconds, 0)
        self.assertEqual(len(cache.graphs), 1)
        torch.testing.assert_close(replay, rows, atol=5e-2, rtol=0)
        eager = decoder.forward_rows(torch.tensor([3], device="cuda"), cache.__class__(**{**cache.__dict__, "length": 5}), 512)
        torch.testing.assert_close(replay, eager, atol=5e-2, rtol=0)

    @unittest.skipUnless(CUDA, "CUDA required")
    def test_kernel_failure_in_warmup_falls_back_and_keeps_buffers_aligned(self):
        decoder = self.decoder("q4_0", torch.float16, "cuda")
        cache, _ = self.prefill(decoder, self.ids[:, :5], "cuda")
        eager = decoder.forward_rows(torch.tensor([3], device="cuda"), cache.__class__(**{**cache.__dict__, "length": 5}), 512)
        kernel = quant.soa_gemv_rows

        def fail_q4_0_once(inp, qbytes, scales, out, qtype, bias):
            if qtype == "Q4_0" and not failures:
                failures.append(1)
                raise RuntimeError("compilation failed")
            return kernel(inp, qbytes, scales, out, qtype, bias)

        failures = []
        self.addCleanup(quant._UNSUPPORTED_KERNELS.clear)
        with mock.patch.object(quant, "soa_gemv_rows", fail_q4_0_once), self.assertLogs(quant._LOG, "WARNING"):
            rows = DecodeSession(decoder, cache).step([3])
        self.assertEqual(failures, [1])
        self.assertEqual(len(cache.io[1]), 1)
        torch.testing.assert_close(rows, eager, atol=5e-2, rtol=0)

    @unittest.skipUnless(CUDA, "CUDA required")
    def test_graphs_share_static_buffers(self):
        decoder = self.decoder("q8_0", torch.float16, "cuda")
        ids = torch.randint(0, 1000, (1, 530), generator=torch.Generator().manual_seed(3))
        cache, _ = self.prefill(decoder, ids[:, :480], "cuda", capacity=2048)
        session = DecodeSession(decoder, cache)
        # Spans 512 and 1024 with one row (decode) and 16 rows (verify), driven by steps that cross the span boundary.
        for tokens in ([3], [4] * 3, [5] * 16, [6] * 10, [7], [8] * 16, [9]):
            length = cache.length
            rows = session.step(tokens)
            cache.length = length
            eager = decoder.forward_rows(torch.tensor(tokens, device="cuda"), cache, 1024)
            torch.testing.assert_close(rows, eager, atol=5e-2, rtol=0)
            cache.length = length + len(tokens)
        self.assertEqual({(span, rows) for span, rows in cache.graphs}, {(512, 1), (512, 16), (1024, 1), (1024, 16)})
        self.assertEqual(set(cache.io), {1, 16})
        for rows, buffers in cache.io.items():
            self.assertTrue(all(inp.shape[0] == rows and out.shape[0] == rows for inp, out in buffers))
        pointers = {rows: [(inp.data_ptr(), out.data_ptr()) for inp, out in buffers] for rows, buffers in cache.io.items()}
        session.rewind(480)
        session.step([3])
        session.step([4] * 16)
        self.assertEqual(pointers, {rows: [(inp.data_ptr(), out.data_ptr()) for inp, out in buffers] for rows, buffers in cache.io.items()})
        self.assertGreater(decoder.session_bytes(2048), 0)

    @unittest.skipUnless(CUDA, "CUDA required")
    def test_graph_capture_and_release_are_silent(self):
        # PyTorch prints the freeAsync warning from C++ under cudaMallocAsync, so run a child process with that allocator.
        code = """
import sys, warnings
sys.path.insert(0, %r)
import torch
from test_decoder import DecoderTests
from introsvg_node.qwen2_decoder import DecodeSession
test = DecoderTests()
DecoderTests.setUpClass()
decoder = test.decoder("q8_0", torch.float16, "cuda")
cache, _ = test.prefill(decoder, torch.randint(0, 1000, (1, 480)), "cuda", capacity=2048)
session = DecodeSession(decoder, cache)
with warnings.catch_warnings():
    warnings.simplefilter("error")
    for tokens in ([3], [4] * 16, [5] * 16, [1]):
        session.step(tokens)
    assert {key[0] for key in cache.graphs} == {512, 1024} and {key[1] for key in cache.graphs} == {1, 16}
    cache.graphs.clear()
print(torch.cuda.get_allocator_backend())
""" % str(Path(__file__).parent)
        env = {**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "backend:cudaMallocAsync"}
        done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, cwd=Path(__file__).parent, timeout=300)
        self.assertEqual(done.returncode, 0, done.stderr[-2000:])
        if done.stdout.strip().splitlines()[-1] != "cudaMallocAsync":
            self.skipTest("cudaMallocAsync is not available")
        self.assertNotIn("freeAsync", done.stderr)

    @unittest.skipUnless(CUDA, "CUDA required")
    def test_graphs_survive_reprefill(self):
        decoder = self.decoder("q8_0", torch.float16, "cuda")
        cache, _ = self.prefill(decoder, self.ids[:, :5], "cuda")
        session = DecodeSession(decoder, cache)
        session.step([3])
        table = cache.rope.data_ptr()
        image = torch.tensor([[1, 2, VISION_START] + [IMAGE_PAD] * 4 + [VISION_END, 5]], device="cuda")
        grid = torch.tensor([[1, 4, 4]])
        image_positions, _ = self.reference.model.get_rope_index(image.cpu(), image_grid_thw=grid, mm_token_type_ids=(image.cpu() == IMAGE_PAD).long(), attention_mask=torch.ones_like(image.cpu()))
        for embeddings, positions in ((decoder.embed(self.ids[:, :8].cuda()), self.text_positions(8, "cuda")),
                                      (decoder.embed(image), image_positions[:, 0].cuda())):
            decoder.prefill(embeddings, positions, cache)
            self.assertEqual(cache.rope.data_ptr(), table)
            replay = session.step([7])
            self.assertEqual(len(cache.graphs), 1)
            cache.length -= 1
            eager = decoder.forward_rows(torch.tensor([7], device="cuda"), cache, 512)
            torch.testing.assert_close(replay, eager, atol=5e-2, rtol=0)
            fresh = decoder.new_cache(1024)
            decoder.prefill(embeddings, positions, fresh)
            torch.testing.assert_close(replay, DecodeSession(decoder, fresh).step([7]), atol=5e-2, rtol=0)
            cache.length += 1


if __name__ == "__main__":
    unittest.main()
