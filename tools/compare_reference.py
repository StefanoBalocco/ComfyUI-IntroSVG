"""Compare the native IntroSVG runtime with Transformers on teacher-forced sequences.

Usage:
  compare_reference.py reference --reference-dir HF_CHECKPOINT --sequences DIR --work DIR
  compare_reference.py compare --gguf FILE [FILE ...] --sequences DIR --work DIR

`--sequences` holds phase{N}_prompt_ids.json and phase{N}_output_ids.json (token ids of the prompt and of the
generated answer, as saved by smoke_introsvg.py) and phase{N}_critic_render.png for the phase with an image.
`reference` runs Transformers' Qwen2_5_VLForConditionalGeneration in FP32 on the CPU over prompt + output and saves,
for the positions that predict the output tokens, the log-softmax in FP16, the greedy choices (plain and with the
loop's repetition penalty) and the image embedding. `compare` loads each GGUF (F16 on the CPU in FP32, other types on
the GPU in FP16) and prints per sequence the top-1 agreement, the mean KL(reference || native) and the image-embedding
relative error, split where the output starts repeating (one period after the first repeated span).
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

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

package = types.ModuleType("introsvg_node")
package.__path__ = [str(ROOT)]
sys.modules["introsvg_node"] = package

import comfy.model_management  # noqa: E402
from comfy.text_encoders.llama import precompute_freqs_cis  # noqa: E402
from comfy.text_encoders.qwen_vl import qwen2vl_mrope_position_ids  # noqa: E402

from introsvg_node.gguf_support import inspect_introsvg_gguf  # noqa: E402
from introsvg_node.introsvg import IMAGE_PAD  # noqa: E402
from introsvg_node.introsvg_model import IntroSVGModel  # noqa: E402

REPETITION_PENALTY = 1.05
CHUNK = 64


def load_sequences(directory, max_output, only=None):
    """Return {name: (prompt ids, output ids, render path or None)} for the phases in `directory`."""
    sequences = {}
    for prompt_file in sorted(Path(directory).glob("phase*_prompt_ids.json")):
        name = prompt_file.name.removesuffix("_prompt_ids.json")
        if only and name not in only:
            continue
        render = Path(directory) / f"{name}_critic_render.png"
        sequences[name] = (json.loads(prompt_file.read_text()), json.loads((Path(directory) / f"{name}_output_ids.json").read_text())[:max_output], render if render.exists() else None)
    return sequences


def first_repeat(tokens, min_period=20):
    """(start, period) of the earliest span of at least `min_period` tokens that is followed by an identical copy, or None."""
    tokens = np.asarray(tokens)
    best = None
    for period in range(min_period, len(tokens) // 2 + 1):
        equal = np.concatenate([[False], tokens[:-period] == tokens[period:], [False]]).astype(np.int8)
        edges = np.flatnonzero(np.diff(equal))
        for start, end in zip(edges[::2], edges[1::2]):
            if end - start >= period and (best is None or start < best[0]):
                best = (int(start), period)
    return best


def position_choices(logits, prompt, output, first):
    """Plain argmax and repetition-penalized argmax of logits [R, V] at output positions first .. first + R - 1."""
    rows = logits.shape[0]
    base = torch.zeros(logits.shape[1], dtype=torch.bool, device=logits.device)
    base[prompt] = True
    base[output[:first]] = True
    seen = torch.empty(rows, logits.shape[1], dtype=torch.bool, device=logits.device)
    for row in range(rows):
        seen[row] = base
        base[output[first + row]] = True
    penalized = torch.where(seen, torch.where(logits > 0, logits / REPETITION_PENALTY, logits * REPETITION_PENALTY), logits)
    return logits.argmax(-1), penalized.argmax(-1)


def reference(args):
    from transformers import Qwen2_5_VLForConditionalGeneration
    from transformers.models.qwen2_vl.image_processing_pil_qwen2_vl import Qwen2VLImageProcessorPil

    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.reference_dir, torch_dtype=torch.float32, low_cpu_mem_usage=True).eval().requires_grad_(False)
    print(f"loaded in {perf_counter() - started:.0f}s", flush=True)
    processor = Qwen2VLImageProcessorPil.from_pretrained(args.reference_dir)
    for name, (prompt, output, render) in load_sequences(args.sequences, args.max_output, args.phases).items():
        started = perf_counter()
        ids = torch.tensor([prompt + output])
        inputs = {"input_ids": ids, "attention_mask": torch.ones_like(ids), "mm_token_type_ids": (ids == IMAGE_PAD).int(), "use_cache": False}
        if render is not None:
            image = processor(Image.open(render).convert("RGB"), return_tensors="pt")
            inputs["pixel_values"], inputs["image_grid_thw"] = image["pixel_values"], image["image_grid_thw"]
            features = model.model.get_image_features(inputs["pixel_values"], inputs["image_grid_thw"])
            features = getattr(features, "pooler_output", features)
            features = torch.cat(list(features)) if isinstance(features, (list, tuple)) else features
            np.save(work / f"ref_{name}_image.npy", features.float().numpy())
        hidden = model.model(**inputs).last_hidden_state[0]
        first = len(prompt) - 1
        logprobs, plain, penalized = [], [], []
        for start in range(first, first + len(output), CHUNK):
            end = min(start + CHUNK, first + len(output))
            logits = model.lm_head(hidden[start:end]).float()
            logprobs.append(logits.log_softmax(-1).half().numpy())
            choices = position_choices(logits, prompt, output, start - first)
            plain.append(choices[0].numpy())
            penalized.append(choices[1].numpy())
        np.save(work / f"ref_{name}_logprobs.npy", np.concatenate(logprobs))
        np.save(work / f"ref_{name}_greedy.npy", np.stack([np.concatenate(plain), np.concatenate(penalized)]))
        print(f"{name}: {len(prompt)} prompt + {len(output)} output tokens in {perf_counter() - started:.0f}s", flush=True)


def native_logits(model, prompt, output, render):
    """Yield (first output index, log-softmax of the native logits, plain and penalized choices) per chunk; also return the image embeddings."""
    decoder = model.decoder
    ids = torch.tensor([prompt + output], device=model.device)
    length = ids.shape[1]
    embeddings = decoder.embed(ids)
    embedding = None
    if render is None:
        positions = torch.arange(length, device=model.device).expand(3, -1)
    else:
        embedding, grid = model.vision.encode(Image.open(render))
        start = prompt.index(IMAGE_PAD)
        embeddings[0, start:start + embedding.shape[0]] = embedding
        positions = qwen2vl_mrope_position_ids([{"type": "image", "index": start, "size": embedding.shape[0], "extra": grid}], length, model.device).long()
    cache = decoder.new_cache(length)
    rope = precompute_freqs_cis(decoder.head_dim, positions, decoder.theta, rope_dims=decoder.mrope_section, device=model.device)
    x = embeddings
    for layer, keys, values in zip(decoder.layers, cache.keys, cache.values):
        x = layer(x, rope, keys, values)
    first = len(prompt) - 1
    for start in range(first, first + len(output), CHUNK):
        logits = decoder._logits(x[0, start:min(start + CHUNK, first + len(output))])
        yield start - first, logits.log_softmax(-1), position_choices(logits, prompt, output, start - first)
    return embedding


def compare_file(path, sequences, work):
    started = perf_counter()
    reference = inspect_introsvg_gguf(path)
    device = "cpu" if "f16" in Path(path).name.lower() else "cuda"
    model = IntroSVGModel(reference, device)
    comfy.model_management.load_models_gpu([model.decoder_patcher, model.vision_patcher], force_full_load=True)
    print(f"\n{Path(path).name} on {device}, loaded in {perf_counter() - started:.0f}s")
    for name, (prompt, output, render) in sequences.items():
        started = perf_counter()
        ref_logprobs = np.load(work / f"ref_{name}_logprobs.npy")
        ref_choices = np.load(work / f"ref_{name}_greedy.npy")
        count = len(output)
        kl, plain, penalized = [], [], []
        generator = native_logits(model, prompt, output, render)
        embedding = None
        with torch.inference_mode():
            while True:
                try:
                    first, logprobs, choices = next(generator)
                except StopIteration as done:
                    embedding = done.value
                    break
                end = first + logprobs.shape[0]
                ref = torch.from_numpy(ref_logprobs[first:end]).to(logprobs.device).float()
                kl.append((ref.exp() * (ref - logprobs)).sum(-1).cpu())
                plain.append(choices[0].cpu())
                penalized.append(choices[1].cpu())
        kl, plain, penalized = torch.cat(kl).numpy(), torch.cat(plain).numpy(), torch.cat(penalized).numpy()
        actual = np.asarray(output)
        repeat = first_repeat(output)
        top1 = plain == ref_choices[0]
        agree = penalized == ref_choices[1]
        row = {"sequence": name, "output_tokens": count, "top1": float(top1.mean()), "top1_penalized": float(agree.mean()), "kl": float(kl.mean()),
               "ref_matches_output": float((ref_choices[1] == actual).mean()), "native_matches_output": float((penalized == actual).mean()), "first_repeated_span": repeat}
        if repeat is not None:
            # The first copy still holds fresh content; the loop proper starts one period after the span.
            onset = repeat[0] + repeat[1]
            row.update({"loop_onset": onset, "top1_before_loop": float(top1[:onset].mean()), "top1_in_loop": float(top1[onset:].mean()),
                        "kl_before_loop": float(kl[:onset].mean()), "kl_in_loop": float(kl[onset:].mean()),
                        "top1_penalized_before_loop": float(agree[:onset].mean()), "top1_penalized_in_loop": float(agree[onset:].mean()),
                        "native_equals_ref_at_loop_onset": bool(penalized[onset] == ref_choices[1][onset]), "output_equals_ref_at_loop_onset": bool(actual[onset] == ref_choices[1][onset])})
        if embedding is not None:
            ref_embedding = torch.from_numpy(np.load(work / f"ref_{name}_image.npy"))
            row["image_relative_error"] = float((embedding.float().cpu() - ref_embedding).norm() / ref_embedding.norm())
        (work / f"compare_{Path(path).stem}_{name}.json").write_text(json.dumps(row, indent=1))
        print(json.dumps(row), f"({perf_counter() - started:.0f}s)", flush=True)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["reference", "compare"])
    parser.add_argument("--reference-dir")
    parser.add_argument("--gguf", nargs="+")
    parser.add_argument("--sequences", required=True)
    parser.add_argument("--work", required=True, help="directory of the saved per-file results")
    parser.add_argument("--phases", nargs="+", help="sequence names to run, e.g. phase1 (default: all)")
    parser.add_argument("--max-output", type=int, default=1500, help="output tokens kept per sequence")
    args = parser.parse_args()
    if args.step == "reference":
        reference(args)
    else:
        sequences = load_sequences(args.sequences, args.max_output, args.phases)
        for path in args.gguf:
            compare_file(path, sequences, Path(args.work))


if __name__ == "__main__":
    main()
