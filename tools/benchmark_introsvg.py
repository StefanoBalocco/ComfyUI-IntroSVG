"""Benchmark the IntroSVG loop on one GGUF: per-phase speed, speculation, looping, validity, VRAM and graph capture.

Usage: benchmark_introsvg.py --gguf FILE --prompts test.csv [--count 20] [--seed 0] [--no-speculative] [--out result.json]
       [--refine-temperature T] [--critique-temperature T] [--max-draft N] [--profile] [--compare-plain]
--profile traces the draft phase of the first prompt (kernel time against wall time per decode step, top Python costs).
--compare-plain runs every greedy phase with plain and with speculative decoding and reports each divergence with its top-2 logit gap.
"""
import argparse
import csv
import cProfile
import functools
import json
import pstats
import statistics
import sys
import types
from dataclasses import replace
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT.parents[1]))
import cuda_malloc  # noqa: E402,F401  ComfyUI's default CUDA allocator, selected before torch loads

import torch  # noqa: E402

if "introsvg_node" not in sys.modules:
    package = types.ModuleType("introsvg_node")
    package.__path__ = [str(ROOT)]
    sys.modules["introsvg_node"] = package

from introsvg_node import decoding, introsvg_model  # noqa: E402
from introsvg_node.gguf_support import inspect_introsvg_gguf  # noqa: E402
from introsvg_node.introsvg import DRAFT_TEMPLATE, IMAGE_TOKENS, LoopParams, chat_ids, run_introsvg  # noqa: E402
from introsvg_node.introsvg_model import IntroSVGModel, session_capacity  # noqa: E402

LOOP_WINDOW = 64
LOOP_REPEATS = 3


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def has_loop(tokens):
    """True when the last LOOP_WINDOW tokens occur at least LOOP_REPEATS times in the output."""
    if len(tokens) < LOOP_WINDOW * LOOP_REPEATS:
        return False
    tail = tokens[-LOOP_WINDOW:]
    count = 0
    for start in range(len(tokens) - LOOP_WINDOW + 1):
        if tokens[start:start + LOOP_WINDOW] == tail:
            count += 1
    return count >= LOOP_REPEATS


def phase_type(name):
    return name.split()[0]


class Recorder:
    """Wraps the model once to record every phase, optionally replaying greedy phases with plain decoding.

    `progress` is the callback of run_introsvg; it names the phase whose generation follows.
    """

    def __init__(self, model, max_draft, compare_plain):
        self.model, self.compare_plain = model, compare_plain
        self.phases = []
        self.name = None
        self.gaps = None
        self.original = model.generate
        self.prefill = 0.0
        original_prefill = model.decoder.prefill

        def prefill(embeddings, positions, cache):
            sync()
            started = perf_counter()
            logits = original_prefill(embeddings, positions, cache)
            sync()
            self.prefill = perf_counter() - started
            return logits

        model.decoder.prefill = prefill
        introsvg_model.generate = functools.partial(decoding.generate, max_draft=max_draft)
        if compare_plain:
            original_process = decoding.process_logits

            def process(logits, seen, drafted, params):
                out = original_process(logits, seen, drafted, params)
                if self.gaps is not None and params.temperature == 0:
                    top = out.topk(2, dim=-1).values
                    self.gaps.extend((top[:, 0] - top[:, 1]).tolist())
                return out

            decoding.process_logits = process
        model.generate = self.generate

    def progress(self, name):
        self.name = name

    def generate(self, prompt_ids, image, sampling, max_new_tokens, generator, speculative, loop_guard=True):
        text, result = self.original(prompt_ids, image, sampling, max_new_tokens, generator, speculative, loop_guard)
        prefill = self.prefill
        record = {"phase": self.name, "prompt_tokens": len(prompt_ids), "prefill_seconds": prefill, "tokens": len(result.tokens), "decode_seconds": result.seconds,
                  "tokens_per_second": len(result.tokens) / result.seconds if result.seconds > 0 else 0.0,
                  "steps": result.steps, "drafted": result.drafted, "accepted": result.accepted,
                  "looped": has_loop(result.tokens), "closed": "</svg>" in text, "temperature": sampling.temperature}
        if self.compare_plain and speculative and sampling.temperature == 0:
            self.gaps = []
            plain_text, plain = self.original(prompt_ids, image, sampling, max_new_tokens, generator, False, loop_guard)
            gaps, self.gaps = self.gaps, None
            record["plain_tokens_per_second"] = len(plain.tokens) / plain.seconds if plain.seconds > 0 else 0.0
            record["plain_seconds"] = plain.seconds
            index = next((i for i, (a, b) in enumerate(zip(result.tokens, plain.tokens)) if a != b), None)
            if index is None and len(result.tokens) != len(plain.tokens):
                index = min(len(result.tokens), len(plain.tokens))
            if index is not None:
                gap = gaps[index] if index < len(gaps) else None
                record["divergence"] = {"index": index, "gap": gap, "speculative_token": result.tokens[index] if index < len(result.tokens) else None,
                                        "plain_token": plain.tokens[index] if index < len(plain.tokens) else None}
        self.phases.append((record, text, result))
        return text, result


def summarize(rows):
    """Mean, median and weighted tokens/s of the phase records in `rows`."""
    if not rows:
        return {}
    rate = [r["tokens_per_second"] for r in rows]
    decode = sum(r["decode_seconds"] for r in rows)
    drafted = sum(r["drafted"] for r in rows)
    return {"count": len(rows), "mean_tokens": statistics.mean(r["tokens"] for r in rows),
            "mean_prompt_tokens": statistics.mean(r["prompt_tokens"] for r in rows),
            "mean_prefill_seconds": statistics.mean(r["prefill_seconds"] for r in rows),
            "mean_decode_seconds": decode / len(rows), "median_decode_seconds": statistics.median(r["decode_seconds"] for r in rows),
            "mean_tokens_per_second": statistics.mean(rate), "median_tokens_per_second": statistics.median(rate),
            "weighted_tokens_per_second": sum(r["tokens"] for r in rows) / decode if decode else 0.0,
            "tokens_per_step": sum(r["tokens"] for r in rows) / max(1, sum(r["steps"] for r in rows)),
            "acceptance": sum(r["accepted"] for r in rows) / drafted if drafted else None,
            "loop_rate": sum(r["looped"] for r in rows) / len(rows), "closed_rate": sum(r["closed"] for r in rows) / len(rows)}


def totals(prompts):
    rows = [p for r in prompts for p in r["phases"]]
    out = {"phases": {}, "prompts": len(prompts)}
    for kind in ("draft", "critique", "refine"):
        out["phases"][kind] = summarize([p for p in rows if phase_type(p["phase"]) == kind])
    out["phases"]["all"] = summarize(rows)
    out["valid_draft_rate"] = statistics.mean(r["valid_draft"] for r in prompts)
    out["valid_final_rate"] = statistics.mean(r["valid_final"] for r in prompts)
    out["mean_rounds"] = statistics.mean(r["rounds"] for r in prompts)
    out["mean_total_seconds"] = statistics.mean(r["total_seconds"] for r in prompts)
    out["median_total_seconds"] = statistics.median(r["total_seconds"] for r in prompts)
    out["max_peak_vram_gib"] = max(r["peak_vram_gib"] for r in prompts)
    out["mean_capture_seconds"] = statistics.mean(r["capture_seconds"] for r in prompts)
    return out


def print_phase(record):
    acceptance = f"{record['accepted']}/{record['drafted']}"
    print(f"  {record['phase']:<11} prompt {record['prompt_tokens']:>5} prefill {record['prefill_seconds']:.2f}s | {record['tokens']:>5} tokens, decode {record['decode_seconds']:.1f}s, "
          f"{record['tokens_per_second']:.1f} tok/s, steps {record['steps']}, accepted {acceptance}, looped {record['looped']}, closed {record['closed']}, score {record.get('score')}", flush=True)


def run_prompt(model, prompt, params, recorder):
    recorder.phases.clear()
    error = None
    result = None
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = perf_counter()
    with torch.inference_mode(), model.session(session_capacity(params.max_new_tokens)):
        try:
            result = run_introsvg(model, prompt, params, recorder.progress)
        except ValueError as e:
            error = str(e)
        except torch.OutOfMemoryError as e:
            error = f"out of memory in phase {len(recorder.phases)} after {len(recorder.phases)} completed phases (cache length {model.decode_session.cache.length}, {len(model.decode_session.cache.graphs)} graphs, {torch.cuda.memory_allocated() / 2**30:.2f} GiB allocated)"
    total = perf_counter() - started
    phases = []
    scores = {}
    if result is not None:
        scores = {p["phase"]: p.get("score") for p in result.phases}
    for record, _, _ in recorder.phases:
        record = {**record, "score": scores.get(record["phase"])}
        phases.append(record)
        print_phase(record)
    peak = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
    out = {"prompt": prompt, "phases": phases, "valid_draft": bool(phases) and phases[0]["closed"] and not (error or "").startswith("The draft"), "valid_final": result is not None,
           "error": error, "rounds": sum(phase_type(p["phase"]) == "critique" for p in phases), "scores": result.scores if result else [],
           "total_seconds": total, "peak_vram_gib": peak, "capture_seconds": model.capture_seconds,
           "divergences": [{"phase": p["phase"], **p["divergence"]} for p in phases if "divergence" in p]}
    print(f"  -> valid final {out['valid_final']}, rounds {out['rounds']}, scores {out['scores']}, total {total:.1f}s, peak {peak:.2f} GiB, capture {model.capture_seconds:.1f}s" + (f", error: {error}" if error else ""), flush=True)
    return out


def profile_draft(model, prompt, params, max_draft, steps_tokens, out):
    """Profile a draft of `steps_tokens` tokens after a warm-up draft that captures the graphs: kernel time and CPU costs per decode step."""
    from torch.profiler import ProfilerActivity, profile
    sampling = decoding.SamplingParams(params.temperature, 1.0, 0, params.repetition_penalty)
    ids = chat_ids(model.tokenizer, DRAFT_TEMPLATE.format(origin_prompt=prompt), 0)
    run = lambda n, seed: model.generate(ids, None, sampling, n, torch.Generator(device=model.device).manual_seed(seed), params.speculative)
    Recorder(model, max_draft, False)
    report = {}
    with torch.inference_mode(), model.session(session_capacity(params.max_new_tokens)) as session:
        run(steps_tokens, 0)
        sync()
        run(steps_tokens, 0)
        sync()
        started = perf_counter()
        text, result = run(steps_tokens, 0)
        sync()
        wall = perf_counter() - started
        report["unprofiled"] = {"tokens": len(result.tokens), "steps": result.steps, "decode_seconds": result.seconds, "wall_ms_per_step": 1000 * result.seconds / result.steps,
                                "tokens_per_second": len(result.tokens) / result.seconds, "drafted": result.drafted, "accepted": result.accepted}
        # Pure graph replay time of the graphs this draft used: back-to-back replays.
        replay = {}
        for (span, rows), entry in sorted(session.cache.graphs.items()):
            graph = entry[0]
            for _ in range(5):
                graph.replay()
            sync()
            t = perf_counter()
            for _ in range(100):
                graph.replay()
            sync()
            replay[f"span{span}_rows{rows}"] = 1000 * (perf_counter() - t) / 100
        report["graph_replay_ms"] = replay
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            text, result = run(steps_tokens, 0)
            sync()
        events = prof.key_averages()
        kernel_us = sum(e.self_device_time_total for e in events)
        cpu_ops = sorted(events, key=lambda e: e.self_cpu_time_total, reverse=True)[:15]
        kernels = sorted((e for e in events if e.self_device_time_total > 0), key=lambda e: e.self_device_time_total, reverse=True)[:15]
        report["profiled"] = {"steps": result.steps, "kernel_ms_per_step": kernel_us / 1000 / result.steps}
        report["top_kernels"] = [{"name": e.key[:90], "device_ms_per_step": e.self_device_time_total / 1000 / result.steps, "calls_per_step": e.count / result.steps} for e in kernels]
        report["top_cpu_ops"] = [{"name": e.key[:90], "cpu_ms_per_step": e.self_cpu_time_total / 1000 / result.steps, "calls_per_step": e.count / result.steps} for e in cpu_ops]
        profiler = cProfile.Profile()
        profiler.enable()
        text, result = run(steps_tokens, 0)
        sync()
        profiler.disable()
        stats = pstats.Stats(profiler)
        rows = sorted(stats.stats.items(), key=lambda item: item[1][2], reverse=True)[:20]
        report["top_python_tottime"] = [{"function": f"{Path(k[0]).name}:{k[1]}({k[2]})", "ms_per_step": 1000 * v[2] / result.steps, "calls_per_step": v[0] / result.steps} for k, v in rows]
    print(json.dumps(report, indent=1))
    Path(out).write_text(json.dumps(report, indent=1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gguf", required=True)
    parser.add_argument("--prompts", required=True, help="CSV with a `desc` column")
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-speculative", action="store_true")
    parser.add_argument("--refine-temperature", type=float, default=0.0)
    parser.add_argument("--critique-temperature", type=float, default=0.0)
    parser.add_argument("--max-draft", type=int, default=15)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-tokens", type=int, default=300)
    parser.add_argument("--compare-plain", action="store_true")
    parser.add_argument("--out", default="benchmark.json")
    args = parser.parse_args()

    with open(args.prompts, newline="", encoding="utf-8") as f:
        prompts = [row["desc"] for row in csv.DictReader(f)][:args.count]
    params = LoopParams(critic_rounds=args.rounds, refine_temperature=args.refine_temperature, critique_temperature=args.critique_temperature, speculative=not args.no_speculative, seed=args.seed)

    started = perf_counter()
    model = IntroSVGModel(inspect_introsvg_gguf(Path(args.gguf)))
    print(f"loaded {args.gguf} in {perf_counter() - started:.1f}s on {model.device}", flush=True)

    if args.profile:
        profile_draft(model, prompts[0], params, args.max_draft, args.profile_tokens, args.out)
        return

    recorder = Recorder(model, args.max_draft, args.compare_plain)
    results = []
    for index, prompt in enumerate(prompts):
        print(f"[{index + 1}/{len(prompts)}] {prompt}", flush=True)
        results.append(run_prompt(model, prompt, replace(params, seed=args.seed + index), recorder))
        Path(args.out).write_text(json.dumps({"args": vars(args), "prompts": results, "totals": totals(results)}, indent=1))
    summary = totals(results)
    print(json.dumps(summary, indent=1))
    if args.compare_plain:
        divergences = [d for r in results for d in r["divergences"]]
        print(f"divergences: {len(divergences)}")
        for d in divergences:
            print(d)
        print("max gap:", max((d["gap"] for d in divergences if d["gap"] is not None), default=None))


if __name__ == "__main__":
    main()
