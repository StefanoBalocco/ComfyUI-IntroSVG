"""Qwen2 decoder of IntroSVG: KV cache, prefill, and CUDA-graph decode/verify steps."""
from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt
from time import perf_counter

import torch
import torch.nn.functional as F
from torch import nn

from comfy.ldm.modules.attention import AttentionTensorContainer, ComfyAttention, optimized_attention_for_device
from comfy.text_encoders.llama import apply_rope, precompute_freqs_cis, rope_matrix

from .quantization import GGUFEmbedding, GGUFLinear, dequantize, graph_io, tensor_payload, tensor_shape, type_name

MIN_SPAN = 512
MAX_GRAPH_ROWS = 16
GRAPH_ROWS = (1, MAX_GRAPH_ROWS)


def _make_causal_mask(q, k):
    qlen, klen = q.shape[-2], k.shape[-2]
    return torch.full((qlen, klen), torch.finfo(q.dtype).min, dtype=q.dtype, device=q.device).triu_(klen - qlen + 1)[None, None]


def comfy_attention(q, k, v, preferred_attention):
    """Causal attention of [B,H,T,D] queries over [B,Hkv,S,D] keys/values through ComfyUI; returns [B,T,H,D]."""
    mask = _make_causal_mask(q, k)
    optimized_attention = optimized_attention_for_device(q.device, mask=True)
    heads, head_dim, enable_gqa = q.shape[1], q.shape[-1], q.shape[1] != k.shape[1]
    # SageAttention's INT8 Q/K quantization corrupts the decoder logits, so opt out of low precision attention.
    output = optimized_attention(
        AttentionTensorContainer(q), AttentionTensorContainer(k), AttentionTensorContainer(v), heads=heads, mask=mask,
        skip_reshape=True, skip_output_reshape=True, enable_gqa=enable_gqa, scale=1.0 / sqrt(head_dim),
        low_precision_attention=False, preferred_attention=preferred_attention)
    return output.transpose(1, 2).contiguous()


class TensorStore:
    """Build GGUF layers from the tensors of an opened GGUF file, by name."""

    def __init__(self, tensors):
        self.tensors = tensors

    def get(self, name):
        try:
            return self.tensors[name]
        except KeyError as exc:
            raise ValueError(f"Missing IntroSVG tensor: {name}") from exc

    def dense(self, name, dtype=torch.float32):
        tensor = self.get(name)
        return dequantize(tensor_payload(tensor), type_name(tensor), tensor_shape(tensor), dtype)

    def linear(self, name, in_features, out_features, bias=True):
        result = GGUFLinear(in_features, out_features)
        weight = self.get(name + ".weight")
        if tensor_shape(weight) != (out_features, in_features):
            raise ValueError(f"Unexpected shape for {name}.weight: {tensor_shape(weight)}, expected {(out_features, in_features)}")
        result.set_gguf("weight", weight)
        if bias:
            result.set_gguf("bias", self.get(name + ".bias"))
        return result

    def embedding(self, name, num_embeddings, embedding_dim):
        result = GGUFEmbedding(num_embeddings, embedding_dim)
        weight = self.get(name + ".weight")
        if tensor_shape(weight) != (num_embeddings, embedding_dim):
            raise ValueError(f"Unexpected shape for {name}.weight: {tensor_shape(weight)}, expected {(num_embeddings, embedding_dim)}")
        result.set_gguf("weight", weight)
        return result

    def norm(self, name, size, eps):
        layer = nn.RMSNorm(size, eps=eps)
        weight = self.dense(name + ".weight")
        if tuple(layer.weight.shape) != tuple(weight.shape):
            raise ValueError(f"Invalid norm shape: {name}")
        layer.weight = nn.Parameter(weight, requires_grad=False)
        return layer


@dataclass
class KVCache:
    """Keys/values [layers, 1, kv_heads, capacity, head_dim] of one sequence, filled up to `length`.

    Generated tokens at slot s have position s + offset. `rope` is the rotation table for positions
    [0, capacity), `graphs` the CUDA graphs captured over this cache, and `io` and `logits` the static Triton buffers and the output rows they share, per row count."""

    keys: torch.Tensor
    values: torch.Tensor
    length: int = 0
    offset: int = 0
    rope: torch.Tensor = field(default=None, repr=False)
    graphs: dict = field(default_factory=dict, repr=False)
    io: dict = field(default_factory=dict, repr=False)
    logits: dict = field(default_factory=dict, repr=False)
    pool: object = field(default=None, repr=False)


class Qwen2Block(nn.Module):
    def __init__(self, store, name, config):
        super().__init__()
        hidden, heads, kv_heads = config["hidden_size"], config["num_attention_heads"], config["num_key_value_heads"]
        eps, inner = config["rms_norm_eps"], config["intermediate_size"]
        self.heads, self.kv_heads, self.head_dim = heads, kv_heads, hidden // heads
        self.comfy_attention = ComfyAttention()
        self.input_layernorm = store.norm(name + ".input_layernorm", hidden, eps)
        self.q_proj = store.linear(name + ".self_attn.q_proj", hidden, hidden)
        self.k_proj = store.linear(name + ".self_attn.k_proj", hidden, kv_heads * self.head_dim)
        self.v_proj = store.linear(name + ".self_attn.v_proj", hidden, kv_heads * self.head_dim)
        self.o_proj = store.linear(name + ".self_attn.o_proj", hidden, hidden, bias=False)
        self.post_attention_layernorm = store.norm(name + ".post_attention_layernorm", hidden, eps)
        self.gate_proj = store.linear(name + ".mlp.gate_proj", hidden, inner, bias=False)
        self.up_proj = store.linear(name + ".mlp.up_proj", hidden, inner, bias=False)
        self.down_proj = store.linear(name + ".mlp.down_proj", inner, hidden, bias=False)

    def forward(self, x, rope, keys, values, slots=None, bias=None):
        """Prefill (slots None) fills cache slots from 0 and attends causally through ComfyUI.
        Rows write their slots and attend the cache span of `keys`, with `bias` [rows, span] hiding later slots."""
        length, hidden = x.shape[1], x.shape[2]
        h = self.input_layernorm(x)
        q = self.q_proj(h).view(1, length, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).view(1, length, self.kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(1, length, self.kv_heads, self.head_dim).transpose(1, 2)
        q, k = apply_rope(q, k, rope)
        if slots is None:
            keys[:, :, :length] = k
            values[:, :, :length] = v
            out = comfy_attention(q, k, v, self.comfy_attention).reshape(1, length, hidden)
        else:
            keys.index_copy_(2, slots, k)
            values.index_copy_(2, slots, v)
            # Each key/value head serves a contiguous group of query heads.
            group = self.heads // self.kv_heads
            # Scale before the product: Qwen2's large q/k biases can overflow FP16 scores.
            q = (q * self.head_dim ** -0.5).reshape(self.kv_heads, group * length, self.head_dim)
            scores = torch.bmm(q, keys.flatten(0, 1).transpose(1, 2)).view(self.kv_heads, group, length, -1)
            scores = scores.float().add_(bias)
            out = torch.bmm(torch.softmax(scores, -1).to(x.dtype).view(self.kv_heads, group * length, -1), values.flatten(0, 1))
            out = out.view(self.heads, length, self.head_dim).transpose(0, 1).reshape(1, length, hidden)
        x = x + self.o_proj(out)
        h = self.post_attention_layernorm(x)
        return x + self.down_proj(F.silu(self.gate_proj(h)).mul_(self.up_proj(h)))


class Qwen2Decoder(nn.Module):
    """Qwen2.5-VL language model over GGUF tensors, for token embeddings in and logits out."""

    def __init__(self, store, config):
        super().__init__()
        hidden, heads = config["hidden_size"], config["num_attention_heads"]
        vocab = config["vocab_size"]
        self.kv_heads, self.head_dim = config["num_key_value_heads"], hidden // heads
        self.theta = config["rope_theta"]
        self.mrope_section = config["rope_scaling"]["mrope_section"]
        self.embed_tokens = store.embedding("model.embed_tokens", vocab, hidden)
        self.layers = nn.ModuleList([Qwen2Block(store, f"model.layers.{i}", config) for i in range(config["num_hidden_layers"])])
        self.norm = store.norm("model.norm", hidden, config["rms_norm_eps"])
        self.lm_head = store.linear("lm_head", hidden, vocab, bias=False)

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse)
        self.embed_tokens.output_dtype = self.norm.weight.dtype
        return self

    def embed(self, ids):
        return self.embed_tokens(ids)

    def new_cache(self, capacity):
        weight = self.norm.weight
        shape = (len(self.layers), 1, self.kv_heads, capacity, self.head_dim)
        # Zero-initialized: decode attends whole spans with masked tails, and stale NaN/Inf bits would survive the mask.
        # The rotation table for positions [0, capacity) is fixed per cache, so captured graphs keep reading the same buffer.
        positions = torch.arange(capacity, device=weight.device)[None]
        rope = rope_matrix(precompute_freqs_cis(self.head_dim, positions, self.theta, rope_dims=self.mrope_section, device=weight.device))
        return KVCache(torch.zeros(shape, dtype=weight.dtype, device=weight.device), torch.zeros(shape, dtype=weight.dtype, device=weight.device), rope=rope)

    def session_bytes(self, capacity):
        """Memory of a KV cache of `capacity` slots plus the static Triton buffers of the decode graphs (FP16 inputs and outputs of every linear layer)."""
        cache = 2 * len(self.layers) * self.kv_heads * capacity * self.head_dim * self.norm.weight.element_size()
        io = sum((m.in_features + m.out_features) * 2 for m in self.modules() if isinstance(m, GGUFLinear)) * sum(GRAPH_ROWS)
        return cache + io + self.lm_head.out_features * 4 * sum(GRAPH_ROWS)

    def _logits(self, x):
        return self.lm_head(self.norm(x)).float()

    def prefill(self, embeddings, positions, cache):
        """Encode a prompt at M-RoPE `positions` [3, T] from an empty cache; returns the logits of its last position."""
        length = embeddings.shape[1]
        cache.length = length
        cache.offset = int(positions.max()) + 1 - length
        rope = precompute_freqs_cis(self.head_dim, positions.to(embeddings.device), self.theta, rope_dims=self.mrope_section, device=embeddings.device)
        x = embeddings
        for layer, keys, values in zip(self.layers, cache.keys, cache.values):
            x = layer(x, rope, keys, values)
        return self._logits(x[0, -1])

    def _rows(self, tokens, slots, positions, cache, span):
        """Logits of the rows `tokens` written to cache slots `slots` (device tensors); each row attends slots up to its own."""
        bias = torch.zeros((tokens.shape[0], span), dtype=torch.float32, device=tokens.device)
        bias.masked_fill_(torch.arange(span, device=tokens.device)[None] > slots[:, None], float("-inf"))
        rope = cache.rope.index_select(2, positions)
        x = self.embed(tokens)[None]
        for layer, keys, values in zip(self.layers, cache.keys, cache.values):
            x = layer(x, rope, keys[:, :, :span], values[:, :, :span], slots, bias)
        return self._logits(x[0])

    def forward_rows(self, tokens, cache, span):
        """Logits [R, V] of R new tokens at slots length .. length + R - 1 attending the first `span` slots.
        Rows are causal among themselves; `cache.length` stays."""
        slots = torch.arange(cache.length, cache.length + tokens.shape[0], device=tokens.device)
        return self._rows(tokens, slots, slots + cache.offset, cache, span)


class DecodeSession:
    """Token appends to a KVCache: eager on CPU, CUDA graphs per (power-of-two span, row count) on CUDA.

    Graphs never run concurrently, so all spans with one row count share the static Triton buffers of the first graph."""

    def __init__(self, decoder, cache):
        self.decoder, self.cache = decoder, cache
        self.capture_seconds = 0.0
        self.done = torch.cuda.Event(blocking=True) if cache.keys.device.type == "cuda" else None

    def rewind(self, length):
        self.cache.length = length

    def step(self, tokens):
        """Append `tokens` to the cache; returns their logits [len(tokens), V]."""
        cache, rows = self.cache, len(tokens)
        capacity = cache.keys.shape[3]
        if cache.length + rows > capacity:
            raise ValueError("KV cache is full")
        device = cache.keys.device
        if device.type != "cuda":
            span = min(capacity, max(MIN_SPAN, 1 << (cache.length + rows - 1).bit_length()))
            logits = self.decoder.forward_rows(torch.tensor(tokens, dtype=torch.long, device=device), cache, span)
        else:
            # Decode runs one row and every verify pads to MAX_GRAPH_ROWS; padded rows write slots that later steps overwrite.
            padded = rows if rows > MAX_GRAPH_ROWS or cache.length + MAX_GRAPH_ROWS > capacity else min(size for size in GRAPH_ROWS if size >= rows)
            span = min(capacity, max(MIN_SPAN, 1 << (cache.length + padded - 1).bit_length()))
            ids = torch.tensor(tokens + [0] * (padded - rows), dtype=torch.long, device=device)
            entry = cache.graphs.get((span, padded))
            ramp = torch.arange(padded, device=device)
            if entry is None:
                if padded not in cache.logits:
                    cache.logits[padded] = torch.empty((padded, self.decoder.lm_head.out_features), dtype=torch.float32, device=device)
                entry = self._capture(ids, ramp + cache.length, span, cache.io.setdefault(padded, []), cache.logits[padded])
                cache.graphs[span, padded] = entry
            graph, static_tokens, slots, positions, output = entry
            static_tokens.copy_(ids)
            torch.add(ramp, cache.length, out=slots)
            torch.add(slots, cache.offset, out=positions)
            graph.replay()
            logits = output[:rows].clone()
            self.done.record()
            self.done.synchronize()  # the host thread sleeps until the step completes
        cache.length += rows
        return logits

    def _capture(self, tokens, slots, span, buffers, output):
        started = perf_counter()
        cache, decoder = self.cache, self.decoder
        positions = slots + cache.offset
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream), graph_io(buffers):
            decoder._rows(tokens, slots, positions, cache, span)  # compiles kernels and allocates `buffers` for the first span of a row count; rewrites only the slots of this step
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        if cache.pool is None:
            cache.pool = torch.cuda.graph_pool_handle()
        # Graphs of one cache share a memory pool and the static `output` of their row count: each replay's output is cloned before the next replay.
        with torch.cuda.graph(graph, pool=cache.pool, capture_error_mode="thread_local"), graph_io(buffers):
            output.copy_(decoder._rows(tokens, slots, positions, cache, span))
        self.capture_seconds += perf_counter() - started
        return graph, tokens, slots, positions, output
