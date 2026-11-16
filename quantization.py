"""IntroSVG GGUF ops: compact SoA parameters and low-batch Triton GEMV."""
from __future__ import annotations
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import logging
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SUPPORTED_TYPES = {"Q4_0", "Q5_0", "Q8_0", "F16", "F32"}
# Bytes of one GGUF block of 32 weights: an FP16 scale followed by the payload.
BLOCK_BYTES = {"Q4_0": 18, "Q5_0": 22, "Q8_0": 34}
_LOG = logging.getLogger(__name__)
_UNSUPPORTED_KERNELS = set()

def type_name(tensor):
    return tensor.tensor_type.name

def tensor_shape(tensor):
    return tuple(int(x) for x in reversed(tensor.shape))

def tensor_payload(tensor):
    qtype = type_name(tensor)
    if qtype not in SUPPORTED_TYPES:
        raise ValueError(f"Unsupported GGUF type {qtype}: {tensor.name}")
    data = np.array(tensor.data, copy=True)
    if qtype in BLOCK_BYTES:
        data = data.view(np.uint8).reshape(-1)
    return torch.from_numpy(data)


def _block_values(payload, qtype):
    """Integer weights [blocks, 32] of GGUF block payloads, separate from the scales."""
    if qtype == "Q8_0":
        return payload.contiguous().view(torch.int8).float()
    if qtype == "Q4_0":
        return torch.cat((payload & 15, payload >> 4), -1).float() - 8
    # Q5_0: 4 bytes holding the fifth bit of each of the 32 weights, read as one
    # little-endian int32, then the Q4_0-style nibbles. Viewing bytes as int32
    # needs contiguous memory starting at a multiple of 4 bytes, and blocks are
    # 20 or 22 bytes apart, so those 4 bytes of each block are copied first.
    high = payload[:, :4].clone(memory_format=torch.contiguous_format).view(torch.int32)
    bits = (high >> torch.arange(32, device=payload.device, dtype=torch.int32)) & 1
    low = payload[:, 4:]
    return (torch.cat((low & 15, low >> 4), -1).int() | (bits << 4)).float() - 16


def dequantize(data, qtype, shape, dtype):
    if qtype in ("F16", "F32"):
        return data.reshape(shape).to(dtype)
    if qtype not in BLOCK_BYTES:
        raise ValueError(f"Unsupported GGUF quantization: {qtype}")
    block_bytes = BLOCK_BYTES[qtype]
    n = 1
    for dim in shape:
        n *= dim
    if n % 32 or data.numel() != n // 32 * block_bytes:
        raise ValueError(f"Invalid {qtype} payload for shape {shape}")
    blocks = data.contiguous().view(torch.uint8).reshape(-1, block_bytes)
    scales = blocks[:, :2].contiguous().view(torch.float16).float().reshape(-1, 1)
    return (_block_values(blocks[:, 2:], qtype) * scales).reshape(shape).to(dtype)

def pack_gguf_soa(tensor):
    """Reorder a GGUF Q4_0/Q5_0/Q8_0 matrix on CPU into its final Q/scales layout.

    The parameters hold the final payload and the FP16 scale bits; ComfyUI's
    ordinary CPU/GPU offloading transfers exactly these tensors.
    """
    qtype = type_name(tensor)
    shape = tensor_shape(tensor)
    if qtype not in BLOCK_BYTES or len(shape) != 2 or shape[1] % 32:
        raise ValueError(f"Expected rank-2 GGUF Q4_0/Q5_0/Q8_0 matrix: {tensor.name}")
    block_bytes = BLOCK_BYTES[qtype]
    blocks = shape[0] * (shape[1] // 32)
    src = torch.from_numpy(np.asarray(tensor.data).view(np.uint8))
    if src.numel() != blocks * block_bytes:
        raise ValueError(f"Invalid packed GGUF payload: {tensor.name}")
    raw = src.reshape(blocks, block_bytes)
    # Store scale as raw bytes to preserve FP16 bits even if ComfyUI calls
    # model.to(dtype=float32/bfloat16). Torch's multithreaded copies run ~2x
    # faster than NumPy's here; with even block sizes each leading FP16 scale
    # is one int16 column.
    q = raw[:, 2:].clone(memory_format=torch.contiguous_format).reshape(-1)
    scales = raw.view(torch.int16)[:, 0].clone(memory_format=torch.contiguous_format).view(torch.uint8)
    return q, scales


def dequantize_soa(qbytes, scale_bits, qtype, shape, dtype):
    """Decode the Q/scales representation on CPU or GPU."""
    if qtype not in BLOCK_BYTES:
        raise ValueError(f"Unsupported SoA quantization: {qtype}")
    n = 1
    for dim in shape:
        n *= dim
    width = BLOCK_BYTES[qtype] - 2
    if (n % 32 or qbytes.numel() != (n // 32) * width or
            scale_bits.numel() != (n // 32) * 2):
        raise ValueError(f"Invalid SoA payload for shape {shape}")
    scale = scale_bits.contiguous().view(torch.float16).float().reshape(-1, 1)
    return (_block_values(qbytes.reshape(-1, width), qtype) * scale).reshape(shape).to(dtype)


@dataclass(frozen=True)
class PackedSpec:
    qtype: str
    shape: tuple[int, ...]

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None

# Triton kernel code of each SoA quantization.
_QTYPE_CODE = {"Q4_0": 4, "Q5_0": 5, "Q8_0": 8}
# Multi-row GEMV launch parameters (bn, blocks, num_warps) per quantization and
# rows, from `docs/benchmarks/kernels_qwen7b.py --sweep-rows` on an RTX 3060.
SOA_GEMV_ROWS_CONFIG = {
    **{("Q4_0", rows): (32, 16, 2) for rows in range(1, 9)},
    **{("Q4_0", rows): (64, 8, 4) for rows in range(9, 17)},
    **{("Q5_0", rows): (32, 16, 2) for rows in range(1, 9)},
    **{("Q5_0", rows): (128, 8, 4) for rows in range(9, 17)},
    **{("Q8_0", rows): (64, 8, 4) for rows in range(1, 17)},
}
# Elements of the FP16 dequantization chunk at most; larger weights are
# dequantized and multiplied in chunks of rows.
DEQUANT_CHUNK_ELEMENTS = 1 << 27

if triton is not None:
    # The PTX decode of Q5_0 is for NVIDIA; HIP decodes with Triton operations.
    _Q5_PTX = torch.version.hip is None
    # Q5_0 bytes to scaled FP16: four elements per call. `$4` holds four payload bytes (two nibbles each),
    # `$5` four bytes with the fifth bits at bits 4 (low nibble) and 5 (high nibble), `$6`/`$7` four FP16 scales.
    # A nibble with its fifth bit is ORed into the mantissa of 1024.0, then 1040 is subtracted (value - 16).
    _Q5_UNPACK = tl.constexpr("""
    {
    .reg .b32 m, k, v, w, t;
    mov.b32 m, 0x0F0F0F0F;
    mov.b32 k, 0x64106410;
    and.b32 v, $4, m;
    and.b32 t, $5, 0x10101010;
    or.b32 v, v, t;
    shr.b32 w, $4, 4;
    and.b32 w, w, m;
    shr.b32 t, $5, 1;
    and.b32 t, t, 0x10101010;
    or.b32 w, w, t;
    prmt.b32 $0, v, 0x64646464, 0x5140;
    prmt.b32 $1, v, 0x64646464, 0x5342;
    prmt.b32 $2, w, 0x64646464, 0x5140;
    prmt.b32 $3, w, 0x64646464, 0x5342;
    sub.rn.f16x2 $0, $0, k;
    sub.rn.f16x2 $1, $1, k;
    sub.rn.f16x2 $2, $2, k;
    sub.rn.f16x2 $3, $3, k;
    mul.rn.f16x2 $0, $0, $6;
    mul.rn.f16x2 $1, $1, $7;
    mul.rn.f16x2 $2, $2, $6;
    mul.rn.f16x2 $3, $3, $7;
    }
    """)

    @triton.jit
    def _soa_gemv_rows(X, Q, S, Y, B, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                       QTYPE: tl.constexpr, BN: tl.constexpr, BLOCKS: tl.constexpr,
                       MP: tl.constexpr, HAS_BIAS: tl.constexpr, PTX: tl.constexpr):
        """Each program dequantizes a (BN, BLOCKS * 32) weight tile once and multiplies it with all M rows."""
        rows = tl.program_id(0) * BN + tl.arange(0, BN)
        blocks = tl.arange(0, BLOCKS)
        half: tl.constexpr = 32 if QTYPE == 8 else 16
        ks = tl.arange(0, BLOCKS * half)
        ms = tl.arange(0, MP)
        width: tl.constexpr = 32 if QTYPE == 8 else (20 if QTYPE == 5 else 16)
        nibbles: tl.constexpr = 4 if QTYPE == 5 else 0
        nblocks: tl.constexpr = K // 32
        words = Q.to(tl.pointer_type(tl.int32))
        accum = tl.full((BN, MP), 0., tl.float32)
        for tile in range(tl.cdiv(nblocks, BLOCKS)):
            bid = tile * BLOCKS + blocks
            bmask = (rows[:, None] < N) & (bid < nblocks)[None, :]
            kb = tile * BLOCKS * half + ks
            xblock = kb // half
            xmask = (xblock[:, None] < nblocks) & (ms[None, :] < M)
            # The payload tile is loaded as one 2D tensor of bytes; the block scale is
            # broadcast over its bytes.
            raw = tl.load(Q + rows[:, None] * (nblocks * width) + (xblock * width + nibbles + kb % half)[None, :],
                          (rows[:, None] < N) & (xblock < nblocks)[None, :], other=0)
            scale = tl.reshape(tl.broadcast_to(tl.load(S + rows[:, None] * nblocks + bid[None, :], bmask, other=0)[:, :, None],
                                               (BN, BLOCKS, half)), (BN, BLOCKS * half))
            xptr = X + ms[None, :] * K + (xblock * 32 + kb % half)[:, None]
            if QTYPE == 8:
                weights = raw.to(tl.int8, bitcast=True).to(tl.float16) * scale
                accum = tl.dot(weights, tl.load(xptr, xmask, other=0), accum)
            else:
                if QTYPE == 5:
                    # The fifth bits are spread to bit 4 (low nibbles) and bit 5 (high nibbles) of one byte per element.
                    fifth = tl.load(words + (rows[:, None] * nblocks + bid[None, :]) * 5, bmask, other=0)[:, :, None]
                    shift = (tl.arange(0, 4) * 4)[None, None, :]
                    spread = ((((fifth >> shift) & 15) * 0x02040810) & 0x10101010) | (((((fifth >> (shift + 16)) & 15) * 0x02040810) & 0x10101010) << 1)
                    spread = tl.join(tl.join((spread & 255).to(tl.uint8), ((spread >> 16) & 255).to(tl.uint8)),
                                     tl.join(((spread >> 8) & 255).to(tl.uint8), (spread >> 24).to(tl.uint8)))
                    spread = tl.reshape(spread, (BN, BLOCKS * half))
                    if PTX:
                        qlo, qhi = tl.inline_asm_elementwise(
                            _Q5_UNPACK, "=r,=r,=r,=r,r,r,r,r", [raw, spread, scale],
                            (tl.float16, tl.float16), is_pure=True, pack=4)
                    else:
                        qlo = (((raw & 15) | (spread & 16)).to(tl.uint16) | 0x6400).to(tl.float16, bitcast=True) - 1040.
                        qhi = (((raw >> 4) | ((spread >> 1) & 16)).to(tl.uint16) | 0x6400).to(tl.float16, bitcast=True) - 1040.
                        qlo = qlo * scale
                        qhi = qhi * scale
                else:
                    qlo = ((raw & 15).to(tl.uint16) | 0x6400).to(tl.float16, bitcast=True) - 1032.
                    qhi = ((raw >> 4).to(tl.uint16) | 0x6400).to(tl.float16, bitcast=True) - 1032.
                    qlo = qlo * scale
                    qhi = qhi * scale
                accum = tl.dot(qlo, tl.load(xptr, xmask, other=0), accum)
                accum = tl.dot(qhi, tl.load(xptr + 16, xmask, other=0), accum)
        if HAS_BIAS:
            accum += tl.load(B + rows, rows < N, other=0).to(tl.float32)[:, None]
        tl.store(Y + ms[None, :] * N + rows[:, None], accum,
                 (rows[:, None] < N) & (ms[None, :] < M))

    @triton.jit
    def _soa_dequantize(Q, S, Y, N: tl.constexpr, K: tl.constexpr, QTYPE: tl.constexpr,
                        BN: tl.constexpr, BLOCKS: tl.constexpr):
        """Write the FP16 values of a (BN, BLOCKS) tile of weight blocks."""
        rows = tl.program_id(0) * BN + tl.arange(0, BN)
        bid = tl.program_id(1) * BLOCKS + tl.arange(0, BLOCKS)
        half: tl.constexpr = 32 if QTYPE == 8 else 16
        lanes = tl.arange(0, half)
        width: tl.constexpr = 32 if QTYPE == 8 else (20 if QTYPE == 5 else 16)
        nibbles: tl.constexpr = 4 if QTYPE == 5 else 0
        nblocks: tl.constexpr = K // 32
        mask = (rows[:, None] < N) & (bid[None, :] < nblocks)
        bid_flat = rows[:, None] * nblocks + bid[None, :]
        scale = tl.load(S + bid_flat, mask, other=0).to(tl.float32)[:, :, None]
        raw = tl.load(Q + bid_flat[:, :, None] * width + nibbles + lanes[None, None, :],
                      mask[:, :, None], other=0)
        out = Y + bid_flat[:, :, None] * 32 + lanes[None, None, :]
        if QTYPE == 8:
            tl.store(out, (raw.to(tl.int8, bitcast=True).to(tl.float32) * scale).to(tl.float16), mask[:, :, None])
        else:
            if QTYPE == 5:
                shifts = tl.arange(0, 4)
                fifth = tl.sum(tl.load(Q + bid_flat[:, :, None] * width + shifts[None, None, :],
                                       mask[:, :, None], other=0).to(tl.int32) << (shifts * 8)[None, None, :], 2)
                qlo = ((raw & 15).to(tl.int32) | (((fifth[:, :, None] >> lanes[None, None, :]) & 1) << 4)).to(tl.float32) - 16.
                qhi = ((raw >> 4).to(tl.int32) | (((fifth[:, :, None] >> (lanes[None, None, :] + 16)) & 1) << 4)).to(tl.float32) - 16.
            else:
                qlo = (raw & 15).to(tl.float32) - 8.
                qhi = (raw >> 4).to(tl.float32) - 8.
            tl.store(out, (qlo * scale).to(tl.float16), mask[:, :, None])
            tl.store(out + 16, (qhi * scale).to(tl.float16), mask[:, :, None])


def _check_soa(x, q, s, out, qtype, n, k):
    if qtype not in ("Q4_0", "Q5_0", "Q8_0"):
        raise ValueError(f"Unsupported GGUF type: {qtype}")
    if k <= 0 or k % 32 or n <= 0:
        raise ValueError("Invalid matrix dimensions")
    if x.device.type != "cuda" or q.device != x.device or s.device != x.device or out.device != x.device:
        raise ValueError("Expected CUDA tensors on the same device")
    if x.dtype != torch.float16 or out.dtype != torch.float16 or q.dtype != torch.uint8 or s.dtype != torch.float16:
        raise ValueError("Expected FP16 input/output/scales and uint8 Q bytes")
    if not (x.is_contiguous() and q.is_contiguous() and s.is_contiguous() and out.is_contiguous()):
        raise ValueError("Expected contiguous tensors")
    total = n * (k // 32)
    if q.numel() != total * (BLOCK_BYTES[qtype] - 2) or s.numel() != total:
        raise ValueError("Incorrect repacked weight size")


def _check_q5_alignment(q, qtype, n, k):
    # The Q5_0 kernel reads the fifth bits as int32 words.
    if qtype == "Q5_0" and q.data_ptr() % 4:
        raise ValueError(f"Q5_0 payload of shape ({n}, {k}) must be 4-byte aligned")


def soa_gemv_rows(x: torch.Tensor, q: torch.Tensor, s: torch.Tensor,
                  out: torch.Tensor, qtype: str, bias=None, *,
                  bn=None, blocks=None, num_warps=None) -> None:
    """GEMV for 1 to 16 rows on the Q/scale layout; each weight block is read once."""
    if x.ndim != 2 or out.ndim != 2 or not 1 <= x.shape[0] <= 16:
        raise ValueError("SoA rows kernel requires rank-2 input with 1 <= M <= 16")
    m, k = x.shape
    n = out.shape[1]
    if out.shape[0] != m:
        raise ValueError("Invalid matrix dimensions")
    _check_soa(x, q, s, out, qtype, n, k)
    _check_q5_alignment(q, qtype, n, k)
    if bias is not None and (bias.shape != (n,) or bias.device != x.device or
                             bias.dtype != torch.float16 or not bias.is_contiguous()):
        raise ValueError("SoA GEMV bias must be contiguous FP16 on input device")
    default = SOA_GEMV_ROWS_CONFIG[qtype, m]
    # Narrow outputs (k/v projections) need small tiles to keep every SM busy.
    bn = bn or (min(default[0], 32) if n <= 1024 else default[0])
    blocks, num_warps = blocks or default[1], num_warps or default[2]
    if bn not in (16, 32, 64, 128) or blocks not in (1, 2, 4, 8, 16, 32):
        raise ValueError("Unsupported GEMV tile size")
    _soa_gemv_rows[(triton.cdiv(n, bn),)](
        x, q, s, out, bias if bias is not None else q,
        m, n, k, _QTYPE_CODE[qtype], bn, blocks, 8 if m <= 8 else 16,
        bias is not None, _Q5_PTX, num_warps=num_warps)


def soa_dequantize(q: torch.Tensor, s: torch.Tensor, qtype: str,
                   shape: tuple[int, int], out: torch.Tensor,
                   *, bn: int = 4, blocks: int = 16, num_warps: int = 2) -> torch.Tensor:
    """Write the FP16 values of a Q/scale matrix into `out` [N, K]."""
    n, k = shape
    if out.shape != (n, k):
        raise ValueError("Output shape does not match the weight shape")
    _check_soa(out, q, s, out, qtype, n, k)
    _soa_dequantize[(triton.cdiv(n, bn), triton.cdiv(k // 32, blocks))](
        q, s, out, n, k, _QTYPE_CODE[qtype], bn, blocks, num_warps=num_warps)
    return out


def _linear_dequantized(x, q, s, qtype, n, k, bias):
    """Dequantize the weight in row chunks into one FP16 buffer and run F.linear."""
    chunk = max(1, min(n, DEQUANT_CHUNK_ELEMENTS // k))
    scratch = torch.empty(chunk * k, dtype=torch.float16, device=x.device)
    width, per_row = BLOCK_BYTES[qtype] - 2, k // 32
    outputs = []
    for start in range(0, n, chunk):
        count = min(chunk, n - start)
        weight = soa_dequantize(q[start * per_row * width:(start + count) * per_row * width],
                                s[start * per_row:(start + count) * per_row],
                                qtype, (count, k), scratch[:count * k].view(count, k))
        outputs.append(F.linear(x, weight, bias[start:start + count] if bias is not None else None))
    return outputs[0] if len(outputs) == 1 else torch.cat(outputs, -1)


# Triton checks pointer arguments with cuPointerGetAttribute, which rejects
# memory allocated while a CUDA graph is captured with cudaMallocAsync (the
# ComfyUI default). A decode graph therefore runs its Triton linears on
# input/output buffers allocated by the eager warm-up of the same step.
_GRAPH_IO = ContextVar("introsvg_graph_io", default=None)


@contextmanager
def graph_io(buffers):
    """Warm-up (empty `buffers`) allocates static Triton input/output pairs in
    call order; the CUDA graph capture that follows reuses them."""
    token = _GRAPH_IO.set((buffers, not buffers, [0]))
    try:
        yield
    finally:
        _GRAPH_IO.reset(token)


def _fused_soa_linear(x, qbytes, scale_bits, spec, bias):
    """Dispatch the multi-row GEMV (1-16 rows) and dequantization + cuBLAS (more)."""
    if (triton is None or x.device.type != "cuda" or (x.device, spec.qtype) in _UNSUPPORTED_KERNELS or
            os.getenv("INTROSVG_DISABLE_TRITON") == "1" or
            x.dtype != torch.float16 or x.numel() % spec.shape[1]):
        return None
    nrows = x.numel() // spec.shape[1]
    n, k = spec.shape
    if nrows <= 16:
        _check_q5_alignment(qbytes, spec.qtype, n, k)
    io = _GRAPH_IO.get()
    claimed = False
    try:
        inp = x.reshape(nrows, k).contiguous()
        scales = scale_bits.view(torch.float16)
        if nrows > 16:
            return _linear_dequantized(inp, qbytes, scales, spec.qtype, n, k, bias).reshape(*x.shape[:-1], n)
        if io is None:
            out = torch.empty((nrows, n), dtype=inp.dtype, device=inp.device)
        else:
            buffers, warmup, calls = io
            if warmup:
                buffers.append((torch.empty((nrows, k), dtype=inp.dtype, device=inp.device),
                                torch.empty((nrows, n), dtype=inp.dtype, device=inp.device)))
            static_in, out = buffers[calls[0]]
            calls[0] += 1
            claimed = True
            inp = static_in.copy_(inp)
        soa_gemv_rows(inp, qbytes, scales, out, spec.qtype, bias)
        return out.reshape(*x.shape[:-1], n)
    except (torch.OutOfMemoryError, ValueError):
        raise
    except Exception as exc:
        if claimed and io[1]:
            io[0].pop()
            io[2][0] -= 1
        _UNSUPPORTED_KERNELS.add((x.device, spec.qtype))
        _LOG.warning("IntroSVG SoA %s kernel failed on %s; using tiled fallback: %s", spec.qtype, x.device, exc)
        return None


class GGUFWeights(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_parameter("weight", None)
        self.register_parameter("weight_scales", None)
        self.register_parameter("bias", None)
        self._specs: dict[str, PackedSpec] = {}

    def set_gguf(self, key, tensor):
        if key not in ("weight", "bias"):
            raise ValueError("Unexpected weight key")
        self.register_parameter(key, nn.Parameter(tensor_payload(tensor), requires_grad=False))
        self._specs[key] = PackedSpec(type_name(tensor), tensor_shape(tensor))

    def _set_soa_weight(self, tensor):
        if type_name(tensor) not in BLOCK_BYTES:
            return False
        q, scales = pack_gguf_soa(tensor)
        self.register_parameter(
            "weight", nn.Parameter(q, requires_grad=False))
        self.register_parameter(
            "weight_scales", nn.Parameter(scales, requires_grad=False))
        self._specs["weight"] = PackedSpec(type_name(tensor), tensor_shape(tensor))
        return True

    def unpack(self, key, dtype, device):
        data = getattr(self, key)
        if data is None:
            return None
        spec = self._specs[key]
        if key == "weight" and self.weight_scales is not None:
            return dequantize_soa(
                data.to(device=device), self.weight_scales.to(device=device),
                spec.qtype, spec.shape, dtype)
        return dequantize(data.to(device=device), spec.qtype, spec.shape, dtype)

class GGUFLinear(GGUFWeights):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features

    def set_gguf(self, key, tensor):
        if key == "weight" and self._set_soa_weight(tensor):
            return
        super().set_gguf(key, tensor)

    def forward(self, x):
        spec = self._specs["weight"]
        bias = self.unpack("bias", x.dtype, x.device)
        if spec.qtype in BLOCK_BYTES:
            fused = _fused_soa_linear(
                x, self.weight, self.weight_scales, spec, bias)
            if fused is not None:
                return fused
            # CPU, non-FP16 inputs or missing Triton: dequantize small
            # slices of the resident SoA layout directly.
            rows_per_chunk = max(1, 2 * 1024 * 1024 // self.in_features)
            x_flat = x.reshape(-1, self.in_features)
            blocks_per_row = self.in_features // 32
            payload = self.weight.reshape(
                self.out_features, blocks_per_row * (BLOCK_BYTES[spec.qtype] - 2))
            scales = self.weight_scales.reshape(
                self.out_features, blocks_per_row * 2)
            outputs = []
            for start in range(0, self.out_features, rows_per_chunk):
                end = min(start + rows_per_chunk, self.out_features)
                weights = dequantize_soa(
                    payload[start:end].flatten(), scales[start:end].flatten(),
                    spec.qtype, (end - start, self.in_features), x.dtype)
                outputs.append(F.linear(
                    x_flat, weights, bias[start:end] if bias is not None else None))
            return torch.cat(outputs, -1).reshape(*x.shape[:-1], self.out_features)
        return F.linear(x, self.unpack("weight", x.dtype, x.device), bias)

class GGUFEmbedding(GGUFWeights):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.num_embeddings, self.embedding_dim = num_embeddings, embedding_dim
        self.output_dtype = torch.float32

    def set_gguf(self, key, tensor):
        if key == "weight" and self._set_soa_weight(tensor):
            return
        super().set_gguf(key, tensor)

    def forward(self, ids):
        spec = self._specs["weight"]
        if spec.qtype in ("F16", "F32"):
            return F.embedding(ids, self.unpack("weight", self.output_dtype, ids.device))
        blocks_per_row = spec.shape[-1] // 32
        stride = blocks_per_row * (BLOCK_BYTES[spec.qtype] - 2)
        flat_ids = ids.reshape(-1)
        if flat_ids.numel() <= 16:
            # Token-by-token decoding indexes the rows directly, which keeps its steps synchronization-free.
            selected = flat_ids
            inverse = None
        else:
            selected, inverse = torch.unique(flat_ids, return_inverse=True)
        q = self.weight.reshape(spec.shape[0], stride).index_select(0, selected)
        scales = self.weight_scales.reshape(
            spec.shape[0], blocks_per_row * 2).index_select(0, selected)
        decoded = dequantize_soa(
            q.flatten(), scales.flatten(), spec.qtype,
            (selected.numel(), spec.shape[-1]), self.output_dtype)
        if inverse is not None:
            decoded = decoded.index_select(0, inverse)
        return decoded.reshape(*ids.shape, spec.shape[-1])
