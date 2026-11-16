"""CUDA/Triton kernel tests: multi-row GEMV, dequantization and the linear dispatch."""
import unittest
from unittest import mock

import torch
import torch.nn.functional as F

import package_alias  # noqa: F401
from introsvg_node import quantization as quant

N, K = 96, 256
NEEDS_TRITON = unittest.skipUnless(torch.cuda.is_available() and quant.triton is not None, "CUDA and Triton required")


def random_soa(qtype, n=N, k=K, seed=0):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    blocks = n * (k // 32)
    q = torch.randint(0, 256, (blocks * (quant.BLOCK_BYTES[qtype] - 2),), dtype=torch.uint8, device="cuda", generator=generator)
    scales = (torch.rand(blocks, device="cuda", generator=generator) * 0.04 + 0.002).half()
    return q, scales


def linear_module(qtype, q, scales, bias):
    layer = quant.GGUFLinear(K, N)
    layer.weight = torch.nn.Parameter(q, requires_grad=False)
    layer.weight_scales = torch.nn.Parameter(scales.view(torch.uint8), requires_grad=False)
    layer.bias = None if bias is None else torch.nn.Parameter(bias, requires_grad=False)
    layer._specs["weight"] = quant.PackedSpec(qtype, (N, K))
    if bias is not None:
        layer._specs["bias"] = quant.PackedSpec("F16", (N,))
    return layer


@NEEDS_TRITON
class KernelTests(unittest.TestCase):
    def test_dequantize_matches_reference(self):
        for qtype in ("Q4_0", "Q5_0", "Q8_0"):
            q, scales = random_soa(qtype)
            out = torch.empty((N, K), dtype=torch.float16, device="cuda")
            result = quant.soa_dequantize(q, scales, qtype, (N, K), out)
            expected = quant.dequantize_soa(q, scales.view(torch.uint8), qtype, (N, K), torch.float32).half()
            self.assertIs(result, out)
            self.assertTrue(torch.equal(out, expected), qtype)

    def test_linear_matches_dequantized_reference(self):
        for qtype in ("Q4_0", "Q5_0", "Q8_0"):
            q, scales = random_soa(qtype)
            weight = quant.dequantize_soa(q, scales.view(torch.uint8), qtype, (N, K), torch.float32).half()
            for use_bias in (False, True):
                bias = torch.randn(N, dtype=torch.float16, device="cuda") if use_bias else None
                layer = linear_module(qtype, q, scales, bias)
                for rows in (1, 2, 3, 8, 16, 17, 64):
                    x = torch.randn(rows, K, dtype=torch.float16, device="cuda")
                    with self.subTest(qtype=qtype, bias=use_bias, rows=rows):
                        torch.testing.assert_close(layer(x), F.linear(x, weight, bias), atol=2e-2, rtol=2e-2)

    def test_rows_gemv_every_row_count(self):
        for qtype in ("Q4_0", "Q5_0", "Q8_0"):
            q, scales = random_soa(qtype)
            weight = quant.dequantize_soa(q, scales.view(torch.uint8), qtype, (N, K), torch.float32).half()
            for rows in range(1, 17):
                x = torch.randn(rows, K, dtype=torch.float16, device="cuda")
                out = torch.empty((rows, N), dtype=torch.float16, device="cuda")
                quant.soa_gemv_rows(x, q, scales, out, qtype)
                with self.subTest(qtype=qtype, rows=rows):
                    torch.testing.assert_close(out, F.linear(x, weight), atol=2e-2, rtol=2e-2)

    def test_rows_gemv_decodes_exactly(self):
        # One-hot rows select single weight columns: each output must equal the dequantized weight bit for bit.
        for qtype in ("Q4_0", "Q5_0", "Q8_0"):
            q, scales = random_soa(qtype)
            weight = quant.dequantize_soa(q, scales.view(torch.uint8), qtype, (N, K), torch.float32).half()
            for first in range(0, K, 16):
                x = torch.zeros(16, K, dtype=torch.float16, device="cuda")
                x[torch.arange(16), first + torch.arange(16)] = 1
                out = torch.empty((16, N), dtype=torch.float16, device="cuda")
                quant.soa_gemv_rows(x, q, scales, out, qtype)
                with self.subTest(qtype=qtype, first=first):
                    self.assertTrue(torch.equal(out, weight[:, first:first + 16].T), qtype)

    @unittest.skipIf(torch.version.hip, "the compiled PTX exists on NVIDIA only")
    def test_rows_gemv_decodes_exactly_without_ptx(self):
        # The HIP decode of Q5_0, checked on the compiled kernel to contain no PTX unpack.
        kernel, compiled = quant._soa_gemv_rows, []

        class Recorder:
            def __getitem__(self, grid):
                def launch(*args, **kwargs):
                    compiled.append(kernel[grid](*args, **kwargs))
                return launch

        self.addCleanup(lambda: (setattr(quant, "_Q5_PTX", True), setattr(quant, "_soa_gemv_rows", kernel)))
        quant._Q5_PTX = False
        quant._soa_gemv_rows = Recorder()
        q, scales = random_soa("Q5_0")
        weight = quant.dequantize_soa(q, scales.view(torch.uint8), "Q5_0", (N, K), torch.float32).half()
        for first in range(0, K, 16):
            x = torch.zeros(16, K, dtype=torch.float16, device="cuda")
            x[torch.arange(16), first + torch.arange(16)] = 1
            out = torch.empty((16, N), dtype=torch.float16, device="cuda")
            quant.soa_gemv_rows(x, q, scales, out, "Q5_0")
            self.assertTrue(torch.equal(out, weight[:, first:first + 16].T), first)
        self.assertNotIn("0x64646464", compiled[0].asm["ptx"])
        quant._Q5_PTX = True
        quant.soa_gemv_rows(x, q, scales, out, "Q5_0")
        self.assertIn("0x64646464", compiled[-1].asm["ptx"])

    def test_q5_misaligned_payload_works_beyond_16_rows(self):
        q, scales = random_soa("Q5_0")
        weight = quant.dequantize_soa(q, scales.view(torch.uint8), "Q5_0", (N, K), torch.float32).half()
        shifted = torch.cat([torch.zeros(1, dtype=torch.uint8, device="cuda"), q])[1:]
        x = torch.randn(17, K, dtype=torch.float16, device="cuda")
        torch.testing.assert_close(linear_module("Q5_0", shifted, scales, None)(x), F.linear(x, weight), atol=2e-2, rtol=2e-2)

    def test_q5_misaligned_payload_raises(self):
        q, scales = random_soa("Q5_0")
        shifted = torch.cat([torch.zeros(1, dtype=torch.uint8, device="cuda"), q])[1:]
        x = torch.randn(3, K, dtype=torch.float16, device="cuda")
        out = torch.empty((3, N), dtype=torch.float16, device="cuda")
        with self.assertRaisesRegex(ValueError, "Q5_0 payload of shape"):
            quant.soa_gemv_rows(x, shifted, scales, out, "Q5_0")

    def test_rows_gemv_odd_shapes(self):
        # 9 blocks per row (tile tail) and more than 1024 output rows (wide tile).
        for qtype in ("Q4_0", "Q5_0", "Q8_0"):
            for n, k in ((40, 288), (1100, 96)):
                q, scales = random_soa(qtype, n, k)
                weight = quant.dequantize_soa(q, scales.view(torch.uint8), qtype, (n, k), torch.float32).half()
                bias = torch.randn(n, dtype=torch.float16, device="cuda")
                for rows in (1, 5, 16):
                    x = torch.randn(rows, k, dtype=torch.float16, device="cuda")
                    out = torch.empty((rows, n), dtype=torch.float16, device="cuda")
                    quant.soa_gemv_rows(x, q, scales, out, qtype, bias)
                    with self.subTest(qtype=qtype, n=n, k=k, rows=rows):
                        torch.testing.assert_close(out, F.linear(x, weight, bias), atol=2e-2, rtol=2e-2)

    def test_large_weight_dequantizes_in_chunks(self):
        old = quant.DEQUANT_CHUNK_ELEMENTS
        quant.DEQUANT_CHUNK_ELEMENTS = 40 * K
        try:
            q, scales = random_soa("Q8_0")
            weight = quant.dequantize_soa(q, scales.view(torch.uint8), "Q8_0", (N, K), torch.float32).half()
            x = torch.randn(20, K, dtype=torch.float16, device="cuda")
            torch.testing.assert_close(linear_module("Q8_0", q, scales, None)(x), F.linear(x, weight), atol=2e-2, rtol=2e-2)
        finally:
            quant.DEQUANT_CHUNK_ELEMENTS = old

    def test_kernel_failure_disables_the_kernel_for_its_device_and_type(self):
        q, scales = random_soa("Q8_0")
        weight = quant.dequantize_soa(q, scales.view(torch.uint8), "Q8_0", (N, K), torch.float32).half()
        layer = linear_module("Q8_0", q, scales, None)
        x = torch.randn(4, K, dtype=torch.float16, device="cuda")
        calls = []

        def broken(*args, **kwargs):
            calls.append(1)
            raise RuntimeError("compilation failed")

        self.addCleanup(quant._UNSUPPORTED_KERNELS.clear)
        with mock.patch.object(quant, "soa_gemv_rows", broken), self.assertLogs(quant._LOG, "WARNING"):
            torch.testing.assert_close(layer(x), F.linear(x, weight), atol=2e-2, rtol=2e-2)
            torch.testing.assert_close(layer(x), F.linear(x, weight), atol=2e-2, rtol=2e-2)
        self.assertEqual(len(calls), 1)
        self.assertEqual(quant._UNSUPPORTED_KERNELS, {(x.device, "Q8_0")})

    def test_out_of_memory_and_value_errors_propagate_from_the_kernel(self):
        q, scales = random_soa("Q4_0")
        layer = linear_module("Q4_0", q, scales, None)
        x = torch.randn(4, K, dtype=torch.float16, device="cuda")
        self.addCleanup(quant._UNSUPPORTED_KERNELS.clear)
        for error in (torch.OutOfMemoryError("out of memory"), ValueError("bad shape")):
            with mock.patch.object(quant, "soa_gemv_rows", side_effect=error), self.assertRaises(type(error)):
                layer(x)
        self.assertEqual(quant._UNSUPPORTED_KERNELS, set())

    def test_rows_gemv_in_cuda_graph(self):
        q, scales = random_soa("Q4_0")
        layer = linear_module("Q4_0", q, scales, None)
        x = torch.randn(8, K, dtype=torch.float16, device="cuda")
        buffers = []
        with quant.graph_io(buffers):
            layer(x)
        graph = torch.cuda.CUDAGraph()
        with quant.graph_io(buffers), torch.cuda.graph(graph):
            result = layer(x)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(result, layer(x), atol=2e-2, rtol=2e-2)


if __name__ == "__main__":
    unittest.main()
