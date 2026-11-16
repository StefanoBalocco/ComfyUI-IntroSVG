# ComfyUI-IntroSVG

Text to SVG for ComfyUI with [IntroSVG](https://github.com/gitcat-404/IntroSVG), a Qwen2.5-VL-7B fine-tune that drafts an SVG from a text prompt, critiques a render of its own drawing and refines it. The nodes run the model from quantized GGUF files with their own decoder: CUDA-graph decode steps, Triton kernels for Q4_0, Q5_0 and Q8_0 weights and speculative decoding from the prompt and the generated text. The Q4_0 file runs the complete loop in about 7 GiB of VRAM.

## Install

1. Clone this repository into `ComfyUI/custom_nodes/` and install the requirements with the Python interpreter of ComfyUI:

   ```
   cd ComfyUI/custom_nodes
   git clone https://github.com/StefanoBalocco/ComfyUI-IntroSVG.git
   python -m pip install -r ComfyUI-IntroSVG/requirements.txt
   ```
2. CairoSVG draws the renders for the critic and needs the Cairo system library: `sudo apt install libcairo2` on Debian and Ubuntu, `brew install cairo` on macOS.
3. Triton is optional. With it, CUDA GPUs decode with the fused kernels; the environment variable `INTROSVG_DISABLE_TRITON=1` selects the PyTorch path.

## Models

Ready-made Q4_0, Q5_0 and Q8_0 files are on Hugging Face: [StefanoBalocco/IntroSVG-Qwen2.5-VL-7B-GGUF](https://huggingface.co/StefanoBalocco/IntroSVG-Qwen2.5-VL-7B-GGUF).

To create the files yourself, convert the checkpoint [gitcat404/IntroSVG-Qwen2.5-VL-7B](https://huggingface.co/gitcat404/IntroSVG-Qwen2.5-VL-7B) into one GGUF file per quantization (the converter needs `safetensors`, `torch`, `numpy` and `gguf`; it runs on the CPU, one tensor at a time):

```
python tools/introsvg_safetensors_to_gguf.py --model-dir /path/to/IntroSVG-Qwen2.5-VL-7B --output ComfyUI/models/introsvg/introsvg-qwen2.5-vl-7b-q4_0.gguf --quantization q4_0
```

`--quantization` is `f16`, `q8_0`, `q5_0` or `q4_0`; the files take 16.6, 9.5, 7.0 and 6.0 GB. Decoder matrices take the chosen quantization, the token embedding takes Q8_0 in the Q5_0 and Q4_0 files, the output head takes Q8_0 in the Q5_0 file and Q5_0 in the Q4_0 file, the vision tower is F16 and biases and norms are F32. Each file embeds the model config and the tokenizer, so the GGUF is the only file the nodes read.

The nodes find the files in the `introsvg` model folder, `ComfyUI/models/introsvg/`. Register another location in `extra_model_paths.yaml`:

```yaml
introsvg_models:
    base_path: /path/to/models
    introsvg: introsvg
```

## Nodes

Both nodes are in the category `IntroSVG`.

**Load IntroSVG GGUF** (`LoadIntroSVGGGUF`) loads a file of the `introsvg` folder and outputs an `INTROSVG` model. The node verifies that the file is an IntroSVG GGUF with all its tensors.

**IntroSVG Text to SVG** (`IntroSVGTextToSVG`) runs the loop and outputs an `SVG` that the Save SVG and preview nodes accept.

| Input | Default | Meaning |
|---|---|---|
| `model` | | The `INTROSVG` model. |
| `prompt` | `a red fox sitting, flat icon` | Description of the icon. |
| `seed` | 0 | Seed of the sampling generator. |
| `critic_rounds` | 3 (0 to 10) | Critique and refine rounds after the draft; 0 returns the draft. |
| `score_threshold` | 9.5 | The loop stops when a critique scores at least this; above 10 every round runs. |
| `temperature` | 0.5 | Sampling temperature of the draft. |
| `max_new_tokens` | 6000 | Token limit of the draft and of each refinement. |
| `speculative` | on | Verify tokens drafted from the prompt and the generated text together in one step. The output equals that of plain decoding. |
| `refine_temperature` (advanced) | 0.0 | Sampling temperature of the refinements; 0 is greedy. |
| `repetition_penalty` (advanced) | 1.05 | Penalty over the prompt and the generated tokens. |
| `critique_temperature` (advanced) | 0.0 | Sampling temperature of the critiques; 0 is greedy. |
| `loop_guard` (advanced) | on | Stop a generation when its last 64 tokens have occurred 3 times. |

The progress bar counts the tokens of the running phase; interrupting the workflow ends the generation at the next token.

## The loop

1. **Draft.** The model writes an SVG for the prompt. The node cuts a draft that stops before `</svg>` (loop guard or token limit) after the last complete element of its root and closes it. When a draft fails to parse or render, the node draws a new one with seed + 1, then seed + 2: 3 attempts in all (1 when `temperature` is 0). It logs the stop reason of each failed attempt and raises an error naming the attempts when the last one fails.
2. **Critique.** The node renders the SVG at 200 x 200 on white, and the model rates the picture against the prompt in a JSON object with a `score` from 0 to 10, a `critique` and `suggestions`. The loop ends when the score reaches `score_threshold`.
3. **Refine.** The model receives the prompt, the SVG, the critique and the suggestions and writes the next SVG, which the node renders for the next critique.

After `critic_rounds` rounds the loop returns the last valid SVG. It returns it earlier, with a warning, when a critique is not a complete JSON object, a refinement loops, stops before `</svg>` or fails to render, or a critique or refine prompt exceeds the KV cache. The KV cache holds `min(16384, 2 * max_new_tokens + 4096)` tokens for all phases of one node run; a phase whose prompt plus `max_new_tokens` exceeds it generates up to the capacity.

The host waits for each decode or verify step on a blocking CUDA event, so a generation keeps about 7-13% of one CPU core busy. During one run the decoder, the vision tower and the CUDA graphs stay loaded; the node releases the KV cache and the CUDA graphs when the run ends, and ComfyUI keeps the weights loaded until it needs the memory. SVGs are parsed with an lxml parser that blocks network access and leaves entities unresolved, and rendered with CairoSVG in safe mode, so a render draws only the content of the SVG itself.

## Fidelity

The native runtime and the reference, Transformers 5.22 `Qwen2_5_VLForConditionalGeneration` in FP32 on the CPU, process the same teacher-forced sequences of one smoke run: a draft (37 prompt + 1436 output tokens), a critique with its 200 x 200 render (289 + 254), and the first 1500 output tokens of a refine answer (1770 prompt tokens). In every column, top-1 is the plain argmax agreement with the reference and KL the mean KL(reference || native) over the output positions.

The refine answer repeats one 127-token path from output index 322 (first repeated span); from index 449 on, every token copies earlier output. These 1051 positions are 33% of the 3190 output positions and are the easiest to predict, so read the "pre-loop" columns (draft, critique and refine positions 0-448: 2139 positions) for fresh content.

| File | Device | Top-1 draft / critique / refine | Top-1 all | Top-1 pre-loop / in-loop | KL all | KL pre-loop / in-loop | Image embedding rel. error |
|---|---|---|---|---|---|---|---|
| F16 | CPU, FP32 | 100 / 100 / 100% | 100% | 100% / 100% | <1e-5 | <1e-5 / <1e-5 | 1.1e-5 |
| Q8_0 | GPU, FP16 | 99.93 / 100 / 100% | 99.97% | 99.95% / 100% | 2.1e-5 | 2.7e-5 / 7.9e-6 | 0.32% |
| Q5_0 | GPU, FP16 | 99.03 / 98.82 / 99.93% | 99.44% | 99.16% / 100% | 1.1e-3 | 1.4e-3 / 5.2e-4 | 0.32% |
| Q4_0 | GPU, FP16 | 97.98 / 98.43 / 99.73% | 98.84% | 98.36% / 99.81% | 4.5e-3 | 5.5e-3 / 2.4e-3 | 0.32% |

All targets hold on the pre-loop figures as well (Q8_0 >= 99%, Q4_0 >= 97%, image error <= 1%, F16 on the CPU >= 99.9%). At the loop onset (index 449) the native and the reference repetition-penalized greedy choices agree for every file. The vision tower computes in FP32 over FP16 weights; the image error comes from the FP16 weights and output and from GPU kernels, and is the same for every quantization because the vision weights are F16 in every file.

Reproduce with `tools/compare_reference.py reference --reference-dir <HF checkpoint> --sequences <dir> --work <dir>`, then `tools/compare_reference.py compare --gguf <files> --sequences <dir> --work <dir>`; the command saves results in the work directory per file and phase as `compare_<file>_<phase>.json`.

## Speed

Measured on an RTX 3060 12 GB with CUDA: 20 prompts from IntroSVG's `test.csv`, seed 0, the node defaults (3 critic rounds, threshold 9.5, speculative decoding on). Token rates are total tokens over total decode seconds per phase type.

| File | Valid final SVG | Seconds per loop (mean / median) | Tokens/s draft | Tokens/s critique | Tokens/s refine | Mean rounds | Peak VRAM |
|---|---|---|---|---|---|---|---|
| Q8_0 | 20/20 | 82.5 / 58.2 | 75.9 | 43.6 | 75.6 | 2.30 | 10.7 GiB |
| Q5_0 | 20/20 | 66.6 / 51.0 | 80.6 | 54.8 | 86.6 | 1.85 | 8.3 GiB |
| Q4_0 | 20/20 | 56.4 / 41.8 | 102.2 | 71.8 | 93.0 | 1.80 | 7.35 GiB |

First drafts were valid for 70% (Q8_0), 65% (Q5_0) and 75% (Q4_0) of the prompts; draft recovery and the retries with seed + 1 and seed + 2 gave every prompt a valid final SVG. Loop times follow the number of generated tokens; some greedy refines repeat until the loop guard stops them. Graph capture takes about 1.0-1.4 s per node call.

Plain and speculative decoding on the same 3 prompts (tokens/s):

| File | Decoding | Draft | Critique | Refine |
|---|---|---|---|---|
| Q8_0 | plain | 39.0 | 40.2 | 36.6 |
| Q8_0 | speculative | 69.7 | 44.6 | 53.4 |
| Q4_0 | plain | 63.3 | 67.8 | 59.8 |
| Q4_0 | speculative | 112.3 | 73.4 | 65.7 |

Plain and speculative greedy phases of Q8_0 (3 prompts) diverged once, at a top-2 logit gap of 0.03 (an FP16 near-tie). One decode step of the 7B model with a CUDA graph (197 linear layers) takes 12.9 ms for 1 token and 13.9 ms for a 16-token verification with Q4_0, 17.2 / 17.8 ms with Q5_0 and 22.9 / 23.7 ms with Q8_0. In the profile of a Q4_0 draft a step takes 16.6 ms wall time, with 17.2 ms of kernel time under the profiler, 81% of it in the linear-layer kernel.

Reproduce with `tools/benchmark_introsvg.py --gguf <file> --prompts <IntroSVG test.csv>`; `--compare-plain` also runs every greedy phase with plain decoding and reports each divergence with its top-2 logit gap, and `--profile` traces the decode steps.

## Tested hardware

NVIDIA RTX 3060 12 GB, CUDA, Python 3.11, PyTorch 2.15 with CUDA 13.2, Triton 3.9. The tests (`python -m unittest discover -s tests`) also run the decoder on the CPU in FP32; they read the Hugging Face checkpoint directory named by `INTROSVG_CHECKPOINT`, and Transformers 5.22 serves as their reference.

## License

The code is BSD-3-Clause (`LICENSE`). The prompt templates in `introsvg.py` are copied from the IntroSVG inference script (https://github.com/gitcat-404/IntroSVG), released by the IntroSVG authors under the Apache License 2.0 (`LICENSE-APACHE`). The IntroSVG weights are released under the Apache License 2.0 as well, and GGUF files converted from them keep it.
