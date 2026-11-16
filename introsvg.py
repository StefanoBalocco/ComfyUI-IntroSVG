"""IntroSVG prompts, chat format, SVG and critique parsing, and the draft, critique and refine loop."""
import io
import json
import logging
import math
import re
from dataclasses import dataclass

import cairosvg
import torch
from lxml import etree
from PIL import Image

from .decoding import SamplingParams

log = logging.getLogger(__name__)

# The prompt templates below are copied from the IntroSVG inference script (https://github.com/gitcat-404/IntroSVG),
# released by the IntroSVG authors under the Apache-2.0 license (LICENSE-APACHE);
# "{origin_prompt}", "{svg}", "{critique}" and "{suggestions}" are the template fields.
SYSTEM_PROMPT = "You are a helpful assistant."

DRAFT_TEMPLATE = "Please generate an SVG icon that meets the following description: {origin_prompt}"

CRITIC_TEMPLATE = """You are a professional SVG design critic. Please analyze the input AI-generated SVG draft<image> according to the "Original Design Prompt".

**Original Design Prompt**: "{origin_prompt}"

Your task is to output a structured critique report, strictly following the JSON format below. The JSON object must contain three fields:
1. "score": A float between 0.0 and 10.0, for a comprehensive evaluation of the draft's quality and alignment.
2. "critique": A comprehensive evaluation explaining the extent to which the draft meets the original prompt, and pointing out its main differences and shortcomings in aesthetics, color, and geometric construction compared to an ideal version.
3. "suggestions": A single string. If the draft's quality is high, this string should be an affirmative statement, such as "The overall quality is excellent, no changes are needed." Otherwise, it should contain 2 to 4 specific, actionable SVG modification suggestions, presented as a numbered list within this string (e.g., "1. Suggestion A\n2. Suggestion B")."""

REFINE_TEMPLATE = """Please analyze all the information provided below and generate a final, high-quality SVG code. The original design goal was: "{origin_prompt}"; a draft SVG code that needs improvement is as follows: ```xml
{svg}
```; an expert critique of this draft is: "{critique}"; and the specific modification suggestions are: {suggestions}. Please strictly follow these suggestions to improve the draft and directly output the final, refined SVG code."""

IM_START, IM_END = 151644, 151645
VISION_START, VISION_END, IMAGE_PAD = 151652, 151653, 151655
RENDER_SIZE = 200
IMAGE_TOKENS = 49
CRITIQUE_MAX_NEW_TOKENS = 2048
DRAFT_ATTEMPTS = 3


class PromptTooLong(ValueError):
    """The prompt does not fit the KV cache."""


def encode_plain(tokenizer, text):
    """Encode text so that special-token strings inside it stay plain text."""
    ids = []
    start = 0
    markers = sorted({token.content for token in tokenizer.get_added_tokens_decoder().values()}, key=len, reverse=True)
    pattern = re.compile("|".join(map(re.escape, markers)))
    for match in pattern.finditer(text):
        cut = match.start() + 1
        ids += tokenizer.encode(text[start:cut], add_special_tokens=False).ids
        start = cut
    return ids + tokenizer.encode(text[start:], add_special_tokens=False).ids


def chat_ids(tokenizer, user_text, image_tokens=0):
    """Token ids of the `qwen2_vl` chat template for one user message, ending at the assistant header.

    With `image_tokens` > 0 the first `<image>` in `user_text` becomes the vision placeholder tokens.
    """
    if image_tokens:
        before, marker, after = user_text.partition("<image>")
        if not marker:
            raise ValueError("user text has no <image> marker")
        user = encode_plain(tokenizer, before) + [VISION_START] + [IMAGE_PAD] * image_tokens + [VISION_END] + encode_plain(tokenizer, after)
    else:
        user = encode_plain(tokenizer, user_text)
    newline = encode_plain(tokenizer, "\n")
    return ([IM_START] + encode_plain(tokenizer, "system\n" + SYSTEM_PROMPT) + [IM_END] + newline
            + [IM_START] + encode_plain(tokenizer, "user\n") + user + [IM_END] + newline
            + [IM_START] + encode_plain(tokenizer, "assistant\n"))


def extract_svg(text):
    """The text from the first `<svg` to the last `</svg>`, or None when the element does not close."""
    start = text.find("<svg")
    end = text.rfind("</svg>")
    if start < 0 or end < start:
        return None
    return text[start:end + len("</svg>")]


TAG = re.compile(r"<(/?)([A-Za-z_][^\s/>]*)((?:\"[^\"]*\"|'[^']*'|[^>\"'])*)>")


def recover_svg(text):
    """Cut an unclosed SVG after the last complete child of its root element and close it; None when no such SVG renders."""
    start = text.find("<svg")
    if start < 0:
        return None
    depth, cuts = 0, []
    for match in TAG.finditer(text, start):
        closing, selfclosing = match.group(1), match.group(3).endswith("/")
        if closing:
            depth -= 1
        elif not selfclosing:
            depth += 1
        if depth == 1 and (closing or selfclosing) and match.group(2) != "svg":
            cuts.append(match.end())
        elif depth < 1:
            break
    for cut in reversed(cuts):
        svg = text[start:cut] + "</svg>"
        try:
            render_png(svg)
        except Exception:
            continue
        return svg
    return None


@dataclass
class Critique:
    score: float
    critique: str
    suggestions: str


def extract_critique(text):
    """Parse the critic's JSON object (plain or in a ```json fence); None when it is not a complete critique."""
    match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        block = match.group(1)
    else:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end < start:
            return None
        block = text[start:end + 1]
    try:
        data = json.loads(block)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    score = data.get("score")
    if isinstance(score, str):
        try:
            score = float(score)
        except ValueError:
            return None
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
        return None
    critique, suggestions = data.get("critique"), data.get("suggestions")
    if not (isinstance(critique, str) and critique and isinstance(suggestions, str) and suggestions):
        return None
    return Critique(float(score), critique, suggestions)


def render_png(svg, size=RENDER_SIZE):
    """Render an SVG on white at size x size: parse with an lxml parser that blocks network access and leaves entities unresolved, and render with CairoSVG in safe mode."""
    parser = etree.XMLParser(no_network=True, resolve_entities=False)
    root = etree.fromstring(svg.encode("utf-8"), parser)
    png = cairosvg.svg2png(bytestring=etree.tostring(root), output_width=size, output_height=size, background_color="white", unsafe=False)
    return Image.open(io.BytesIO(png)).convert("RGB")


@dataclass
class LoopParams:
    critic_rounds: int = 3
    score_threshold: float = 9.5
    temperature: float = 0.5
    refine_temperature: float = 0.0
    repetition_penalty: float = 1.05
    max_new_tokens: int = 6000
    speculative: bool = True
    seed: int = 0
    critique_temperature: float = 0.0
    loop_guard: bool = True


@dataclass
class LoopResult:
    svg: str
    scores: list[float]
    phases: list[dict]


def run_introsvg(model, prompt, params, progress=None):
    """Draft an SVG for `prompt`, then critique its render and refine it for up to `params.critic_rounds` rounds.

    `model.tokenizer` is a tokenizers.Tokenizer, `model.device` the device of the sampling generator, and
    `model.generate(prompt_ids, image, sampling, max_new_tokens, generator, speculative, loop_guard)` returns the generated text and a
    GenerationResult. A draft without `</svg>` is cut after its last complete element; a draft that is invalid after recovery is drawn again with seed + 1 and seed + 2 (a greedy draft is drawn once), then raises ValueError; a critique or refinement that is invalid, loops, does not close or whose prompt does not fit the KV cache (PromptTooLong) ends the loop with the last valid SVG.
    """
    generator = current = torch.Generator(device=model.device).manual_seed(params.seed)
    phases = []

    def generate(name, user_text, image, temperature, max_new_tokens):
        if progress:
            progress(name)
        ids = chat_ids(model.tokenizer, user_text, IMAGE_TOKENS if image is not None else 0)
        sampling = SamplingParams(temperature, 1.0, 0, params.repetition_penalty)
        text, result = model.generate(ids, image, sampling, max_new_tokens, current, params.speculative, params.loop_guard)
        count = len(result.tokens)
        rate = count / result.seconds if result.seconds > 0 else 0.0
        phases.append({"phase": name, "tokens": count, "seconds": result.seconds, "tokens_per_second": rate, "drafted": result.drafted, "accepted": result.accepted})
        log.info("IntroSVG %s: %d tokens, %.1f tokens/s, drafted %d, accepted %d", name, count, rate, result.drafted, result.accepted)
        return text, result

    failures = []
    attempts = DRAFT_ATTEMPTS if params.temperature > 0 else 1
    for attempt in range(attempts):
        seed = (params.seed + attempt) % 2**64
        name = "draft" if attempt == 0 else f"draft retry {attempt}"
        current = generator if attempt == 0 else torch.Generator(device=model.device).manual_seed(seed)
        text, result = generate(name, DRAFT_TEMPLATE.format(origin_prompt=prompt), None, params.temperature, params.max_new_tokens)
        svg = extract_svg(text)
        if svg is None and "<svg" in text:
            svg = recover_svg(text)
            if svg is not None:
                log.warning("IntroSVG draft: no closing </svg> (stop reason: %s); keeping the SVG up to its last complete element", result.stop_reason)
        if svg is None:
            failure = "never closes </svg> and has no complete element to keep" if "<svg" in text else "contains no <svg> element"
        else:
            try:
                image = render_png(svg)
                break
            except Exception as error:
                failure = f"does not render: {error}"
        failures.append(f"seed {seed}: {failure} (stop reason: {result.stop_reason})")
        log.warning("IntroSVG draft attempt %d (seed %d) is not valid: %s (stop reason: %s)", attempt + 1, seed, failure, result.stop_reason)
    else:
        raise ValueError(f"The draft failed in {attempts} attempt{'s' if attempts > 1 else ''}: " + "; ".join(failures))
    current = generator

    scores = []
    for round_number in range(1, params.critic_rounds + 1):
        name = f"critique {round_number}"
        try:
            critique_text, _ = generate(name, CRITIC_TEMPLATE.format(origin_prompt=prompt), image, params.critique_temperature, CRITIQUE_MAX_NEW_TOKENS)
        except PromptTooLong as error:
            log.warning("IntroSVG round %d: %s; keeping the last valid SVG", round_number, error)
            break
        critique = extract_critique(critique_text)
        if critique is None:
            log.warning("IntroSVG round %d: the critique is not valid; keeping the last valid SVG", round_number)
            break
        scores.append(critique.score)
        phases[-1]["score"] = critique.score
        log.info("IntroSVG round %d score: %s", round_number, critique.score)
        if critique.score >= params.score_threshold:
            break
        try:
            refine_text, refine_result = generate(f"refine {round_number}", REFINE_TEMPLATE.format(origin_prompt=prompt, svg=svg, critique=critique.critique, suggestions=critique.suggestions), None, params.refine_temperature, params.max_new_tokens)
        except PromptTooLong as error:
            log.warning("IntroSVG round %d: %s; keeping the last valid SVG", round_number, error)
            break
        refined = extract_svg(refine_text)
        if refine_result.stop_reason == "loop" or refined is None:
            log.warning("IntroSVG round %d: the refined SVG is not complete (stop reason: %s); keeping the last valid SVG", round_number, refine_result.stop_reason)
            break
        try:
            image = render_png(refined)
        except Exception as error:
            log.warning("IntroSVG round %d: the refined SVG does not render (%s); keeping the last valid SVG", round_number, error)
            break
        svg = refined
    return LoopResult(svg, scores, phases)
