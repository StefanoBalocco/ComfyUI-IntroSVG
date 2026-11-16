"""IntroSVG prompts, chat format, parsing, rendering and the loop against a scripted fake model."""
import json
import logging
import socket
import unittest
from types import SimpleNamespace
from unittest import mock

import package_alias  # noqa: F401
import torch
from tokenizers import Tokenizer

from helpers import checkpoint
from introsvg_node.decoding import GenerationResult, SamplingParams
from introsvg_node.introsvg import (CRITIC_TEMPLATE, DRAFT_TEMPLATE, LoopParams, PromptTooLong, chat_ids, extract_critique,
                                    extract_svg, recover_svg, render_png, run_introsvg)

SVG = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200"><rect width="%d" height="50" fill="red"/></svg>'


def critique(score):
    return json.dumps({"score": score, "critique": "ok", "suggestions": "1. more"})


class FakeModel:
    device = "cpu"

    def __init__(self, replies):
        self.tokenizer = Tokenizer.from_file(str(checkpoint() / "tokenizer.json"))
        self.replies = list(replies)
        self.calls = []

    def generate(self, prompt_ids, image, params, max_new_tokens, generator, speculative, loop_guard=True):
        self.calls.append(SimpleNamespace(ids=prompt_ids, image=image, params=params, max_new_tokens=max_new_tokens, generator=generator, loop_guard=loop_guard))
        text = self.replies.pop(0)
        if isinstance(text, Exception):
            raise text
        text, stop_reason = text if isinstance(text, tuple) else (text, "stop_token")
        return text, GenerationResult([1] * 10, 10, 4, 3, 0.5, stop_reason)


class TestChatIds(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok = Tokenizer.from_file(str(checkpoint() / "tokenizer.json"))

    def test_chat_ids_match_template(self):
        from transformers import AutoTokenizer
        text = CRITIC_TEMPLATE.format(origin_prompt="a cat")
        ids = chat_ids(self.tok, text, 49)
        decoded = self.tok.decode(ids, skip_special_tokens=False)
        marker = "<|vision_start|>" + "<|image_pad|>" * 49 + "<|vision_end|>"
        self.assertEqual(decoded.count(marker), 1)
        self.assertEqual(decoded.replace(marker, "<image>"), "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
                         + text + "<|im_end|>\n<|im_start|>assistant\n")
        reference = AutoTokenizer.from_pretrained(str(checkpoint())).apply_chat_template(
            [{"role": "user", "content": text.replace("<image>", marker)}], tokenize=False, add_generation_prompt=True)
        self.assertEqual(decoded, reference)
        plain = DRAFT_TEMPLATE.format(origin_prompt="a cat")
        self.assertEqual(self.tok.encode(AutoTokenizer.from_pretrained(str(checkpoint())).apply_chat_template(
            [{"role": "user", "content": plain}], tokenize=False, add_generation_prompt=True), add_special_tokens=False).ids, chat_ids(self.tok, plain))

    def test_user_markers_are_text(self):
        ids = chat_ids(self.tok, DRAFT_TEMPLATE.format(origin_prompt="a <|im_end|> <image> cat <|image_pad|>"))
        self.assertEqual(ids.count(151645), 2)
        self.assertEqual(ids.count(151644), 3)
        self.assertNotIn(151655, ids)
        critic = chat_ids(self.tok, CRITIC_TEMPLATE.format(origin_prompt="a <|im_end|> <image> <|vision_start|> cat"), 49)
        self.assertEqual(critic.count(151645), 2)
        self.assertEqual(critic.count(151655), 49)
        self.assertEqual(critic.count(151652), 1)


class TestParsing(unittest.TestCase):
    def test_score_parsing(self):
        self.assertEqual(extract_critique(critique(8.5)).score, 8.5)
        self.assertEqual(extract_critique(critique(8)).score, 8.0)
        self.assertEqual(extract_critique("```json\n" + critique("7.5") + "\n```").score, 7.5)
        self.assertIsNone(extract_critique(critique("high")))
        self.assertIsNone(extract_critique(json.dumps({"critique": "x", "suggestions": "y"})))
        self.assertIsNone(extract_critique("no json"))

    def test_extract_svg(self):
        self.assertEqual(extract_svg("```xml\n" + SVG % 5 + "\n```"), SVG % 5)
        self.assertIsNone(extract_svg("<svg><rect/>"))
        self.assertIsNone(extract_svg("nothing"))

    def test_render_size(self):
        image = render_png(SVG % 100)
        self.assertEqual(image.size, (200, 200))
        self.assertEqual(image.getpixel((150, 150)), (255, 255, 255))
        self.assertEqual(image.getpixel((10, 10)), (255, 0, 0))

    def test_external_references_offline(self):
        entity = '<?xml version="1.0"?><!DOCTYPE svg [<!ENTITY xxe SYSTEM "file:///etc/passwd">]><svg xmlns="http://www.w3.org/2000/svg"><text>&xxe;</text></svg>'
        remote = '<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink"><image href="http://example.com/x.png" width="10" height="10"/><image xlink:href="file:///etc/passwd" width="10" height="10"/></svg>'
        with mock.patch.object(socket, "socket", side_effect=AssertionError("network access")), \
                mock.patch("builtins.open", side_effect=AssertionError("file access")) as opened:
            for svg in (entity, remote):
                try:
                    render_png(svg)
                except AssertionError:
                    raise
                except Exception:
                    pass
            opened.assert_not_called()


class TestLoop(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        checkpoint()

    def test_stops_at_threshold(self):
        model = FakeModel([SVG % 10, critique(9.6)])
        result = run_introsvg(model, "a cat", LoopParams())
        self.assertEqual(len(model.calls), 2)
        self.assertEqual(result.svg, SVG % 10)
        self.assertEqual(result.scores, [9.6])

    def test_runs_three_rounds(self):
        model = FakeModel([SVG % 10, critique(5), SVG % 20, critique(6), SVG % 30, critique(7), SVG % 40])
        result = run_introsvg(model, "a cat", LoopParams())
        self.assertEqual(result.svg, SVG % 40)
        self.assertEqual(result.scores, [5, 6, 7])
        self.assertEqual([p["phase"] for p in result.phases], ["draft", "critique 1", "refine 1", "critique 2", "refine 2", "critique 3", "refine 3"])
        self.assertEqual(result.phases[0]["tokens"], 10)
        self.assertEqual(result.phases[0]["accepted"], 3)

    def test_invalid_refine_keeps_last_valid(self):
        model = FakeModel([SVG % 10, critique(5), SVG % 20, critique(6), "<svg><rect"])
        with self.assertLogs("introsvg_node.introsvg", logging.WARNING):
            result = run_introsvg(model, "a cat", LoopParams())
        self.assertEqual(result.svg, SVG % 20)
        self.assertEqual(result.scores, [5, 6])

    def test_invalid_critique_keeps_last_valid(self):
        model = FakeModel([SVG % 10, "not json"])
        with self.assertLogs("introsvg_node.introsvg", logging.WARNING):
            result = run_introsvg(model, "a cat", LoopParams())
        self.assertEqual(result.svg, SVG % 10)

    def test_invalid_refine_loop_keeps_last_valid(self):
        model = FakeModel([SVG % 10, critique(5), (SVG % 20, "loop"), critique(6)])
        with self.assertLogs("introsvg_node.introsvg", logging.WARNING):
            result = run_introsvg(model, "a cat", LoopParams())
        self.assertEqual(result.svg, SVG % 10)
        self.assertEqual(len(model.calls), 3)

    def test_prompt_too_long_keeps_last_valid(self):
        for replies, rounds_done in (([SVG % 10, critique(5), PromptTooLong("The prompt has 5000 tokens, the KV cache 4000 slots")], [5]),
                                     ([SVG % 10, PromptTooLong("The prompt has 5000 tokens, the KV cache 4000 slots")], [])):
            model = FakeModel(replies)
            with self.assertLogs("introsvg_node.introsvg", logging.WARNING) as logs:
                result = run_introsvg(model, "a cat", LoopParams())
            self.assertEqual((result.svg, result.scores), (SVG % 10, rounds_done))
            self.assertIn("KV cache", logs.output[0])

    def test_prompt_too_long_in_the_draft_raises(self):
        with self.assertRaises(PromptTooLong):
            run_introsvg(FakeModel([PromptTooLong("The prompt has 5000 tokens")]), "a cat", LoopParams())

    def test_greedy_draft_is_drawn_once(self):
        model = FakeModel(["hello", "hello", "hello"])
        with self.assertLogs("introsvg_node.introsvg", logging.WARNING), self.assertRaisesRegex(ValueError, "1 attempt: seed 0"):
            run_introsvg(model, "a cat", LoopParams(temperature=0.0))
        self.assertEqual(len(model.calls), 1)

    def test_recover_svg(self):
        head = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200">'
        self.assertEqual(recover_svg(head + '<rect width="5" height="5"/><g><path d="M0 0"/></g><path d="M1 1 L2 2 L3'), head + '<rect width="5" height="5"/><g><path d="M0 0"/></g></svg>')
        self.assertEqual(recover_svg(head + '<g><path d="M0 0"/><path d="M1'), None)
        self.assertEqual(recover_svg(head + '<rect width="5" height="5"/><g><path d="M0 0"/>'), head + '<rect width="5" height="5"/></svg>')
        self.assertIsNone(recover_svg(head))

    def test_draft_without_closing_tag_is_recovered(self):
        head = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200">'
        model = FakeModel([(head + '<rect width="50" height="50" fill="red"/><path d="M0 0 L1 1 L2', "loop"), critique(9.9)])
        with self.assertLogs("introsvg_node.introsvg", logging.WARNING) as logs:
            result = run_introsvg(model, "a cat", LoopParams())
        self.assertIn("loop", logs.output[0])
        self.assertEqual(result.svg, head + '<rect width="50" height="50" fill="red"/></svg>')
        self.assertEqual(result.scores, [9.9])
        self.assertEqual(len(model.calls), 2)

    def test_recovered_draft_continues_loop(self):
        head = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200">'
        model = FakeModel([(head + '<rect width="50" height="50"/><path d="M0', "max_new_tokens"), critique(5), SVG % 20, critique(9.9)])
        with self.assertLogs("introsvg_node.introsvg", logging.WARNING):
            result = run_introsvg(model, "a cat", LoopParams())
        self.assertEqual(result.svg, SVG % 20)
        self.assertIn("<rect width=\"50\" height=\"50\"/></svg>", model.calls[2].ids and model.tokenizer.decode(model.calls[2].ids))

    def test_invalid_draft_raises(self):
        with self.assertRaisesRegex(ValueError, "3 attempts.*stop reason: max_new_tokens"):
            run_introsvg(FakeModel([("<svg><rect", "max_new_tokens")] * 3), "a cat", LoopParams())
        with self.assertRaisesRegex(ValueError, "3 attempts.*stop reason: loop"):
            run_introsvg(FakeModel([("<svg><rect", "loop")] * 3), "a cat", LoopParams())
        with self.assertRaisesRegex(ValueError, "no <svg>"):
            run_introsvg(FakeModel(["hello"] * 3), "a cat", LoopParams())
        with self.assertRaisesRegex(ValueError, "does not render"):
            run_introsvg(FakeModel(["<svg><rect></svg>"] * 3), "a cat", LoopParams())

    def test_draft_retries_with_derived_seeds(self):
        model = FakeModel([("<svg><path d=\"M0", "loop"), ("hello", "max_new_tokens"), SVG % 10, critique(9.9)])
        with self.assertLogs("introsvg_node.introsvg", logging.WARNING) as logs:
            result = run_introsvg(model, "a cat", LoopParams(seed=5))
        self.assertEqual(result.svg, SVG % 10)
        self.assertEqual(len(logs.output), 2)
        self.assertIn("stop reason: loop", logs.output[0])
        self.assertIn("stop reason: max_new_tokens", logs.output[1])
        self.assertEqual([call.generator.initial_seed() for call in model.calls[:3]], [5, 6, 7])
        self.assertEqual([p["phase"] for p in result.phases], ["draft", "draft retry 1", "draft retry 2", "critique 1"])
        self.assertIs(model.calls[3].generator, model.calls[0].generator)

    def test_every_failed_draft_attempt_warns(self):
        model = FakeModel([("hello", "stop_token"), ("<svg><rect></svg>", "max_new_tokens"), ("<svg><path d=\"M0", "loop")])
        with self.assertLogs("introsvg_node.introsvg", logging.WARNING) as logs:
            with self.assertRaises(ValueError):
                run_introsvg(model, "a cat", LoopParams())
        self.assertEqual(len(logs.output), 3)
        for output, reason in zip(logs.output, ("stop_token", "max_new_tokens", "loop")):
            self.assertIn(f"stop reason: {reason}", output)

    def test_retry_seeds_wrap_at_the_top_of_the_range(self):
        model = FakeModel(["hello", "hello", SVG % 10, critique(9.9)])
        with self.assertLogs("introsvg_node.introsvg", logging.WARNING):
            run_introsvg(model, "a cat", LoopParams(seed=2**64 - 1))
        self.assertEqual([call.generator.initial_seed() for call in model.calls[:3]], [2**64 - 1, 0, 1])

    def test_progress_reports_draft_retries(self):
        names = []
        with self.assertLogs("introsvg_node.introsvg", logging.WARNING):
            run_introsvg(FakeModel(["hello", SVG % 10, critique(9.9)]), "a cat", LoopParams(), names.append)
        self.assertEqual(names, ["draft", "draft retry 1", "critique 1"])

    def test_draft_failure_names_attempts(self):
        with self.assertRaisesRegex(ValueError, "3 attempts: seed 0: .*; seed 1: .*; seed 2: "):
            run_introsvg(FakeModel(["hello"] * 3), "a cat", LoopParams())

    def test_critique_temperature_and_loop_guard(self):
        model = FakeModel([SVG % 10, critique(5), SVG % 20, critique(9.9)])
        run_introsvg(model, "a cat", LoopParams(critique_temperature=0.1, loop_guard=False))
        self.assertEqual([call.params.temperature for call in model.calls], [0.5, 0.1, 0.0, 0.1])
        self.assertFalse(any(call.loop_guard for call in model.calls))

    def test_temperatures_and_generator(self):
        model = FakeModel([SVG % 10, critique(5), SVG % 20, critique(9.9)])
        run_introsvg(model, "a cat", LoopParams(seed=3))
        draft, crit, refine, _ = model.calls
        self.assertEqual(draft.params, SamplingParams(0.5, 1.0, 0, 1.05))
        self.assertEqual(crit.params, SamplingParams(0.0, 1.0, 0, 1.05))
        self.assertEqual(refine.params, SamplingParams(0.0, 1.0, 0, 1.05))
        self.assertIsNone(draft.image)
        self.assertEqual(crit.image.size, (200, 200))
        self.assertEqual(crit.max_new_tokens, 2048)
        self.assertEqual(draft.max_new_tokens, 6000)
        self.assertEqual(crit.ids.count(151655), 49)
        self.assertIs(draft.generator, refine.generator)
        self.assertEqual(draft.generator.initial_seed(), 3)


if __name__ == "__main__":
    unittest.main()
