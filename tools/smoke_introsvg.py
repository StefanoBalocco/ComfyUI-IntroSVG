"""Run the full IntroSVG loop on one GGUF and save every phase's prompt, output and the critic's render.

Usage: smoke_introsvg.py GGUF OUTPUT_DIR [--prompt TEXT] [--seed N] [--rounds N]
Checks that every logits tensor is finite and prints per-phase tokens, speed, speculation, scores, graph capture time and peak VRAM.
"""
import argparse
import json
import sys
import types
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT.parents[1]))
import cuda_malloc  # noqa: E402,F401  ComfyUI's default CUDA allocator, selected before torch loads

import torch  # noqa: E402

package = types.ModuleType("introsvg_node")
package.__path__ = [str(ROOT)]
sys.modules["introsvg_node"] = package

from introsvg_node.gguf_support import inspect_introsvg_gguf  # noqa: E402
from introsvg_node.introsvg import LoopParams, run_introsvg  # noqa: E402
from introsvg_node.introsvg_model import IntroSVGModel, session_capacity  # noqa: E402
from introsvg_node.qwen2_decoder import DecodeSession  # noqa: E402


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("gguf")
    parser.add_argument("output")
    parser.add_argument("--prompt", default="a red fox sitting, flat icon")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    started = perf_counter()
    model = IntroSVGModel(inspect_introsvg_gguf(Path(args.gguf)))
    print(f"loaded in {perf_counter() - started:.1f}s on {model.device}")

    record = {}
    original_prefill, original_step, original_generate = model.decoder.prefill, DecodeSession.step, model.generate

    def check(logits, where):
        if not torch.isfinite(logits).all():
            raise SystemExit(f"BLOCKED: non-finite logits in {where}")

    def prefill(embeddings, positions, cache):
        sync()
        t = perf_counter()
        logits = original_prefill(embeddings, positions, cache)
        sync()
        record["prefill"] = perf_counter() - t
        check(logits, f"prefill of {record['name']}")
        return logits

    def step(self, tokens):
        logits = original_step(self, tokens)
        check(logits, f"decode step of {record['name']} at length {self.cache.length}")
        return logits

    def generate(prompt_ids, image, sampling, max_new_tokens, generator, speculative, loop_guard=True):
        record["name"] = f"phase {len(record['phases'])}"
        record["prompt_ids"] = list(prompt_ids)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        text, result = original_generate(prompt_ids, image, sampling, max_new_tokens, generator, speculative, loop_guard)
        peak = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
        index = len(record["phases"])
        record["phases"].append({"prompt_tokens": len(prompt_ids), "prefill_seconds": record["prefill"], "peak_vram_gib": peak})
        (out / f"phase{index}_prompt_ids.json").write_text(json.dumps(prompt_ids))
        (out / f"phase{index}_output_ids.json").write_text(json.dumps(result.tokens))
        (out / f"phase{index}_text.txt").write_text(text, encoding="utf-8")
        if image is not None:
            image.save(out / f"phase{index}_critic_render.png")
        return text, result

    record["phases"] = []
    model.decoder.prefill = prefill
    DecodeSession.step = step
    model.generate = generate

    params = LoopParams(critic_rounds=args.rounds, seed=args.seed)
    started = perf_counter()
    with torch.inference_mode(), model.session(session_capacity(params.max_new_tokens)) as session:
        try:
            result = run_introsvg(model, args.prompt, params, lambda name: print(f"-- {name}", flush=True))
        except torch.OutOfMemoryError:
            print(f"out of memory with {len(session.cache.graphs)} captured graphs, cache length {session.cache.length}, {torch.cuda.memory_allocated() / 2**30:.2f} GiB allocated")
            raise
    total = perf_counter() - started

    for info, phase in zip(record["phases"], result.phases):
        print(f"{phase['phase']}: prompt {info['prompt_tokens']} tokens, prefill {info['prefill_seconds']:.2f}s, generated {phase['tokens']} tokens, "
              f"{phase['tokens_per_second']:.1f} tokens/s, drafted {phase['drafted']}, accepted {phase['accepted']}, score {phase.get('score')}, peak VRAM {info['peak_vram_gib']:.2f} GiB")
    print(f"scores {result.scores}, graph capture {model.capture_seconds:.2f}s, total {total:.1f}s")
    (out / "final.svg").write_text(result.svg, encoding="utf-8")
    (out / "summary.json").write_text(json.dumps({"scores": result.scores, "phases": result.phases, "details": record["phases"], "capture_seconds": model.capture_seconds, "total_seconds": total}, indent=1))
    print("all logits finite")


if __name__ == "__main__":
    main()
