"""The benchmark tool's recorder installed once on a scripted model: phase names after a draft retry, one record per phase."""
import importlib.util
import logging
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import package_alias  # noqa: F401

from test_introsvg import SVG, FakeModel, critique
from introsvg_node import introsvg_model
from introsvg_node.introsvg import LoopParams, run_introsvg

SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "benchmark_introsvg.py"
spec = importlib.util.spec_from_file_location("_introsvg_benchmark_test", SCRIPT)
benchmark = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = benchmark
spec.loader.exec_module(benchmark)


class RecorderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        FakeModel([])

    def test_phases_follow_the_loop_across_prompts(self):
        replies = ["hello", SVG % 10, critique(5), SVG % 20, critique(9.9)]
        model = FakeModel(replies * 2)
        model.decoder = SimpleNamespace(prefill=lambda embeddings, positions, cache: None)
        with mock.patch.object(introsvg_model, "generate"):
            recorder = benchmark.Recorder(model, 15, False)
            for prompt in ("a cat", "a dog"):
                recorder.phases.clear()
                with self.assertLogs("introsvg_node.introsvg", logging.WARNING):
                    result = run_introsvg(model, prompt, LoopParams(), recorder.progress)
                names = [record["phase"] for record, _, _ in recorder.phases]
                self.assertEqual(names, ["draft", "draft retry 1", "critique 1", "refine 1", "critique 2"])
                self.assertEqual(names, [phase["phase"] for phase in result.phases])
                scores = {phase["phase"]: phase.get("score") for phase in result.phases}
                self.assertEqual(scores, {"draft": None, "draft retry 1": None, "critique 1": 5, "refine 1": None, "critique 2": 9.9})
        self.assertEqual(len(model.calls), 2 * len(replies))


if __name__ == "__main__":
    unittest.main()
