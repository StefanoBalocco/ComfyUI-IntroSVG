"""IntroSVG ComfyUI nodes: GGUF loader and text to SVG."""
import logging
from io import BytesIO
from pathlib import Path

import folder_paths
from comfy_api.latest import ComfyExtension, io

from .gguf_support import inspect_introsvg_gguf
from .introsvg import LoopParams, run_introsvg
from .introsvg_model import IntroSVGModel, session_capacity

log = logging.getLogger(__name__)

# extra_model_paths.yaml has already been loaded when custom nodes are imported; ComfyUI/models/introsvg is the fallback.
folder_paths.add_model_folder_path("introsvg", str(Path(folder_paths.models_dir) / "introsvg"))
folder_paths.folder_names_and_paths["introsvg"][1].add(".gguf")

IntroSVG = io.Custom("INTROSVG")


class LoadIntroSVGGGUF(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LoadIntroSVGGGUF",
            display_name="Load IntroSVG GGUF",
            category="IntroSVG",
            inputs=[io.Combo.Input("gguf_name", options=folder_paths.get_filename_list("introsvg"))],
            outputs=[IntroSVG.Output()],
        )

    @classmethod
    def execute(cls, gguf_name):
        path = folder_paths.get_full_path("introsvg", gguf_name)
        if path is None:
            raise FileNotFoundError(f"No GGUF file {gguf_name!r} in the introsvg model folders: {folder_paths.get_folder_paths('introsvg')}")
        return io.NodeOutput(IntroSVGModel(inspect_introsvg_gguf(Path(path))))


class IntroSVGTextToSVG(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="IntroSVGTextToSVG",
            display_name="IntroSVG Text to SVG",
            category="IntroSVG",
            inputs=[
                IntroSVG.Input("model"),
                io.String.Input("prompt", multiline=True, default="a red fox sitting, flat icon"),
                io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff, control_after_generate=True),
                io.Int.Input("critic_rounds", default=3, min=0, max=10, tooltip="Critique and refine rounds after the draft."),
                io.Float.Input("score_threshold", default=9.5, min=0.0, max=100.0, step=0.1, tooltip="The loop stops when a critique scores at least this; above 10 every round runs."),
                io.Float.Input("temperature", default=0.5, min=0.0, max=2.0, step=0.05, tooltip="Sampling temperature of the draft."),
                io.Int.Input("max_new_tokens", default=6000, min=64, max=16000, tooltip="Token limit of the draft and of each refinement."),
                io.Boolean.Input("speculative", default=True, tooltip="Verify tokens drafted from the prompt and the generated text together in one step."),
                io.Float.Input("refine_temperature", default=0.0, min=0.0, max=2.0, step=0.05, advanced=True, tooltip="Sampling temperature of the refinements; 0 is greedy."),
                io.Float.Input("repetition_penalty", default=1.05, min=1.0, max=2.0, step=0.01, advanced=True),
                io.Float.Input("critique_temperature", default=0.0, min=0.0, max=2.0, step=0.05, advanced=True, tooltip="Sampling temperature of the critiques; 0 is greedy."),
                io.Boolean.Input("loop_guard", default=True, advanced=True, tooltip="Stop a generation when its last 64 tokens have occurred 3 times."),
            ],
            outputs=[io.SVG.Output()],
        )

    @classmethod
    def execute(cls, model, prompt, seed, critic_rounds, score_threshold, temperature, max_new_tokens, speculative, refine_temperature=0.0, repetition_penalty=1.05, critique_temperature=0.0, loop_guard=True):
        params = LoopParams(critic_rounds, score_threshold, temperature, refine_temperature, repetition_penalty, max_new_tokens, speculative, seed, critique_temperature, loop_guard)
        with model.session(session_capacity(max_new_tokens)):
            result = run_introsvg(model, prompt, params)
        log.info("IntroSVG: scores %s, graph capture %.2fs", result.scores, model.capture_seconds)
        return io.NodeOutput(io.SVG.Type([BytesIO(result.svg.encode("utf-8"))]))


class IntroSVGExtension(ComfyExtension):
    async def get_node_list(self):
        return [LoadIntroSVGGGUF, IntroSVGTextToSVG]


async def comfy_entrypoint():
    return IntroSVGExtension()
