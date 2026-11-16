"""Logit processing, prompt-lookup drafting, draft acceptance and the generation loop."""
from dataclasses import dataclass
from time import perf_counter
from typing import Callable

import torch


@dataclass(frozen=True)
class SamplingParams:
    temperature: float
    top_p: float = 1.0
    top_k: int = 0
    repetition_penalty: float = 1.05


@dataclass
class GenerationResult:
    tokens: list[int]
    steps: int
    drafted: int
    accepted: int
    seconds: float
    stop_reason: str


def process_logits(logits, seen, drafted, params):
    """Repetition penalty, temperature, top-k and top-p on every row.

    Row i penalizes `seen` plus `drafted[:i]`. Returns the penalized logits for greedy decoding and probabilities for sampling.
    """
    rows = logits.shape[0]
    penalized = seen.expand(rows, -1).clone()
    for i, token in enumerate(drafted[:rows - 1]):
        penalized[i + 1:, token] = True
    logits = torch.where(penalized, torch.where(logits > 0, logits / params.repetition_penalty, logits * params.repetition_penalty), logits)
    if params.temperature == 0:
        return logits
    logits = logits / params.temperature
    if params.top_k > 0 and params.top_k < logits.shape[-1]:
        logits = logits.masked_fill(logits < logits.topk(params.top_k, dim=-1).values[:, -1:], float("-inf"))
    probs = logits.softmax(dim=-1)
    if params.top_p < 1.0:
        sorted_probs, order = probs.sort(dim=-1, descending=True)
        drop = sorted_probs.cumsum(dim=-1) - sorted_probs >= params.top_p
        probs = probs.masked_fill(drop.scatter(-1, order, drop), 0.0)
        probs = probs / probs.sum(dim=-1, keepdim=True)
    return probs


def propose(history, max_tokens, max_ngram=4, min_ngram=2):
    """Draft up to `max_tokens` tokens that followed the latest earlier occurrence of the longest matching suffix of at least `min_ngram` tokens."""
    last, end = history[-1], len(history) - 1
    best_length, best_start = 0, -1
    for start in range(end - 1, -1, -1):
        if history[start] != last:
            continue
        length = 1
        while length < max_ngram and start >= length and history[start - length] == history[end - length]:
            length += 1
        if length > best_length:
            best_length, best_start = length, start
            if length == max_ngram:
                break
    if best_length < min_ngram:
        return []
    return history[best_start + 1:best_start + 1 + max_tokens]


def accept_greedy(processed, drafted):
    """Keep drafted tokens while they equal the row argmax, then append the argmax of the first mismatching row (or the bonus row)."""
    best = processed.argmax(dim=-1).tolist()
    out = []
    for i, token in enumerate(drafted):
        out.append(best[i])
        if best[i] != token:
            return out
    out.append(best[len(drafted)])
    return out


def accept_sampled(probs, drafted, generator):
    """Speculative sampling: accept d with probability p(d), else sample from p without d; sample a bonus token after a full acceptance."""
    k = len(drafted)
    if k:
        index = torch.tensor(drafted, device=probs.device)
        p_draft = probs[torch.arange(k, device=probs.device), index]
        uniform = torch.rand(k, generator=generator, device=probs.device)
        accepted = (uniform < p_draft).tolist()
    else:
        accepted = []
    out = []
    for i, token in enumerate(drafted):
        if not accepted[i]:
            residual = probs[i].clone()
            residual[token] = 0.0
            out.append(torch.multinomial(residual, 1, generator=generator).item())
            return out
        out.append(token)
    out.append(torch.multinomial(probs[k], 1, generator=generator).item())
    return out


LOOP_WINDOW = 64
LOOP_REPEATS = 3


def generate(session, first_logits, prompt_ids, params, max_new_tokens, stop_ids, generator, speculative, max_draft=15, loop_guard=True, on_tokens: Callable[[int], None] | None = None):
    """Generate up to `max_new_tokens` tokens after a prefill; `first_logits` is the distribution of the first token.

    With `speculative`, tokens drafted from the history are verified together in one session step. With `loop_guard`,
    generation stops once the last 64 generated tokens have occurred 3 times among the generated tokens.
    The result's `stop_reason` is "stop_token", "max_new_tokens" or "loop".
    """
    started = perf_counter()
    accept = accept_greedy if params.temperature == 0 else lambda rows, drafted: accept_sampled(rows, drafted, generator)
    seen = torch.zeros(first_logits.shape[-1], dtype=torch.bool, device=first_logits.device)
    seen[prompt_ids] = True
    history = list(prompt_ids)
    tokens = []
    steps = drafted_total = accepted_total = 0
    window_counts = {}
    stop_reason = "max_new_tokens"

    def emit(new_tokens):
        """Append tokens up to the first stop id, the limit or a loop; returns True when generation is over."""
        nonlocal stop_reason
        for token in new_tokens:
            if token in stop_ids:
                stop_reason = "stop_token"
                return True
            if len(tokens) == max_new_tokens:
                return True
            tokens.append(token)
            history.append(token)
            seen[token] = True
            if on_tokens is not None:
                on_tokens(token)
            if loop_guard and len(tokens) >= LOOP_WINDOW:
                window = hash(tuple(tokens[-LOOP_WINDOW:]))
                window_counts[window] = window_counts.get(window, 0) + 1
                if window_counts[window] >= LOOP_REPEATS:
                    stop_reason = "loop"
                    return True
        return len(tokens) == max_new_tokens

    done = max_new_tokens == 0 or emit(accept(process_logits(first_logits[None], seen, [], params), []))
    while not done:
        remaining = max_new_tokens - len(tokens)
        draft = propose(history, min(max_draft, remaining - 1)) if speculative and remaining > 1 else []
        length = session.cache.length
        rows = process_logits(session.step([tokens[-1]] + draft), seen, draft, params)
        new_tokens = accept(rows, draft)
        session.rewind(length + len(new_tokens))
        steps += 1
        drafted_total += len(draft)
        accepted_total += len(new_tokens) - 1
        done = emit(new_tokens)
    return GenerationResult(tokens, steps, drafted_total, accepted_total, perf_counter() - started, stop_reason)
