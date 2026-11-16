"""Logit processing, prompt lookup, speculative acceptance and the generation loop against a fake session."""
import unittest
from types import SimpleNamespace

import package_alias  # noqa: F401
import torch
from scipy.stats import chisquare

from introsvg_node.decoding import SamplingParams, accept_greedy, accept_sampled, generate, process_logits, propose

V = 50


class FakeSession:
    """Logits are a fixed function of (previous token, position); the previous token dominates."""

    def __init__(self, prompt_length):
        table = torch.Generator().manual_seed(7)
        self.by_token = torch.randn(V, V, generator=table) * 3
        self.by_position = torch.randn(5, V, generator=table) * 0.2
        self.tokens = [0] * prompt_length
        self.cache = SimpleNamespace(length=prompt_length)

    def logits_after(self, token, position):
        return self.by_token[token] + self.by_position[position % 5]

    def step(self, tokens):
        rows = []
        for i, token in enumerate(tokens):
            position = self.cache.length + i
            del self.tokens[position:]
            self.tokens.append(token)
            rows.append(self.logits_after(token, position))
        self.cache.length += len(tokens)
        return torch.stack(rows)

    def rewind(self, length):
        self.cache.length = length


def run(prompt, params, speculative, max_new=300, stop=frozenset(), seed=0, max_draft=15, loop_guard=True):
    session = FakeSession(len(prompt))
    first = session.logits_after(prompt[-1], len(prompt) - 1)
    generator = torch.Generator().manual_seed(seed)
    return generate(session, first, prompt, params, max_new, stop, generator, speculative, max_draft, loop_guard)


class DecodingTest(unittest.TestCase):
    def test_speculative_greedy_identical(self):
        prompt = [3, 8, 1, 9, 4, 6] * 5 + [3]
        params = SamplingParams(temperature=0.0)
        plain = run(prompt, params, False, loop_guard=False)
        fast = run(prompt, params, True, loop_guard=False)
        self.assertEqual(plain.tokens, fast.tokens)
        self.assertEqual(len(fast.tokens), 300)
        self.assertGreater(fast.accepted, 0)
        self.assertLess(fast.steps, plain.steps)

    def test_speculative_sampling_distribution(self):
        generator = torch.Generator().manual_seed(1)
        params = SamplingParams(temperature=1.0, repetition_penalty=1.0)
        v = 5
        by_token = torch.randn(v, v, generator=generator)
        seen = torch.zeros(v, dtype=torch.bool)
        t, draft = 2, [1, 3]

        def probs_after(token):
            return process_logits(by_token[token][None], seen, [], params)[0]

        expected = torch.zeros(v, v, dtype=torch.float64)
        for a in range(v):
            expected[a] = probs_after(t)[a].double() * probs_after(a).double()
        draws = 20000
        counts = torch.zeros(v, v)
        rows = process_logits(by_token[[t, draft[0], draft[1]]], seen, [], params)
        for _ in range(draws):
            out = accept_sampled(rows, draft, generator)
            first = out[0]
            if len(out) > 1:
                second = out[1]
            else:
                second = torch.multinomial(probs_after(first), 1, generator=generator).item()
            counts[first, second] += 1
        observed, mass = counts.flatten().numpy(), expected.flatten().numpy()
        keep = mass * draws >= 5
        observed = list(observed[keep]) + [observed[~keep].sum()]
        mass = list(mass[keep]) + [mass[~keep].sum()]
        if mass[-1] == 0:
            observed, mass = observed[:-1], mass[:-1]
        scale = sum(observed) / sum(mass)
        self.assertGreater(chisquare(observed, [m * scale for m in mass]).pvalue, 0.01)

    def test_speculative_sampling_with_repetition_penalty_through_generate(self):
        # Three generated tokens over a 6-token vocabulary; after a first token 4 the repeated prompt drafts [1].
        prompt = [1, 2, 3, 4, 1, 2, 3]
        params = SamplingParams(temperature=1.0, repetition_penalty=1.3)
        draws = 6000

        class SmallSession(FakeSession):
            def __init__(self, prompt_length):
                super().__init__(prompt_length)
                self.by_token[:, 6:] = -1e9

        def run_small(speculative, seed):
            session = SmallSession(len(prompt))
            first = session.logits_after(prompt[-1], len(prompt) - 1)
            return generate(session, first, prompt, params, 3, frozenset(), torch.Generator().manual_seed(seed), speculative)

        reference = SmallSession(len(prompt))
        seen = torch.zeros(V, dtype=torch.bool)
        seen[prompt] = True
        first = process_logits(reference.logits_after(prompt[-1], len(prompt) - 1)[None], seen, [], params)[0]
        expected = torch.zeros(6, 6, dtype=torch.float64)
        for a in range(6):
            after = seen.clone()
            after[a] = True
            expected[a] = first[a].double() * process_logits(reference.logits_after(a, len(prompt))[None], after, [], params)[0, :6].double()
        for speculative in (False, True):
            counts = torch.zeros(6, 6)
            drafted = 0
            for seed in range(draws):
                result = run_small(speculative, seed)
                counts[result.tokens[0], result.tokens[1]] += 1
                drafted += result.drafted
            self.assertEqual(drafted > 0, speculative)
            observed, mass = counts.flatten().numpy(), expected.flatten().numpy()
            keep = mass * draws >= 5
            observed = list(observed[keep]) + [observed[~keep].sum()]
            mass = list(mass[keep]) + [mass[~keep].sum()]
            scale = sum(observed) / sum(mass)
            self.assertGreater(chisquare(observed, [m * scale for m in mass]).pvalue, 0.01, f"speculative={speculative}")

    def test_penalty_counts_drafted_tokens(self):
        logits = torch.ones(3, 10)
        logits[:, 6] = -1.0
        seen = torch.zeros(10, dtype=torch.bool)
        seen[5] = True
        params = SamplingParams(temperature=0.0, repetition_penalty=2.0)
        out = process_logits(logits, seen, [7, 6], params)
        self.assertEqual(out[0, 7].item(), 1.0)
        self.assertEqual(out[0, 6].item(), -1.0)
        self.assertEqual(out[1, 7].item(), 0.5)
        self.assertEqual(out[1, 6].item(), -1.0)
        self.assertEqual(out[2, 7].item(), 0.5)
        self.assertEqual(out[2, 6].item(), -2.0)
        self.assertTrue((out[:, 5] == 0.5).all())

    def test_top_k_top_p(self):
        logits = torch.log(torch.tensor([[0.5, 0.3, 0.15, 0.05]]))
        seen = torch.zeros(4, dtype=torch.bool)
        k2 = process_logits(logits, seen, [], SamplingParams(1.0, top_k=2, repetition_penalty=1.0))
        self.assertTrue(torch.allclose(k2, torch.tensor([[0.625, 0.375, 0.0, 0.0]])))
        p = process_logits(logits, seen, [], SamplingParams(1.0, top_p=0.7, repetition_penalty=1.0))
        self.assertTrue(torch.allclose(p, torch.tensor([[0.625, 0.375, 0.0, 0.0]])))

    def test_propose(self):
        self.assertEqual(propose([1, 2, 3, 4, 9, 1, 2, 3], 3), [4, 9, 1])
        self.assertEqual(propose([1, 2, 3, 4], 3), [])
        self.assertEqual(propose([1, 2, 3, 4, 9, 1, 2, 3], 2), [4, 9])
        self.assertEqual(propose([1, 2, 3, 4, 9, 7, 3], 3), [])
        self.assertEqual(propose([1, 2, 3, 4, 9, 7, 3], 3, min_ngram=1), [4, 9, 7])
        self.assertEqual(propose([5, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5], 3), [6, 7, 8])

    def test_accept_greedy(self):
        processed = torch.full((4, 6), -1.0)
        for row, token in enumerate([2, 3, 5, 1]):
            processed[row, token] = 1.0
        self.assertEqual(accept_greedy(processed, [2, 3, 5]), [2, 3, 5, 1])
        self.assertEqual(accept_greedy(processed, [2, 4, 5]), [2, 3])
        self.assertEqual(accept_greedy(processed[:1], []), [2])

    def test_stop_and_limit(self):
        prompt = [3, 8, 1, 9, 4, 6] * 5 + [3]
        params = SamplingParams(temperature=0.0)
        free = run(prompt, params, False, max_new=40).tokens
        stop_id = free[12]
        cut = free.index(stop_id)
        for speculative in (False, True):
            result = run(prompt, params, speculative, max_new=40, stop=frozenset({stop_id}))
            self.assertEqual(result.tokens, free[:cut])
            for limit in (1, 5, 17):
                result = run(prompt, SamplingParams(0.7), speculative, max_new=limit)
                self.assertLessEqual(len(result.tokens), limit)
                self.assertEqual(len(result.tokens), limit)
        self.assertEqual(run(prompt, params, True, max_new=0).tokens, [])

    def test_stop_reasons(self):
        prompt = [3, 8, 1, 9, 4, 6] * 5 + [3]
        params = SamplingParams(temperature=0.0)
        free = run(prompt, params, False, max_new=40, loop_guard=False)
        self.assertEqual(free.stop_reason, "max_new_tokens")
        self.assertEqual(run(prompt, params, True, max_new=0).stop_reason, "max_new_tokens")
        self.assertEqual(run(prompt, params, False, max_new=40, stop=frozenset({free.tokens[5]}), loop_guard=False).stop_reason, "stop_token")

    def test_loop_guard(self):
        # Greedy decoding on the fake table falls into a cycle that repeats one 64-token window.
        prompt = [3, 8, 1, 9, 4, 6] * 5 + [3]
        params = SamplingParams(temperature=0.0, repetition_penalty=1.0)
        free = run(prompt, params, False, max_new=2000, loop_guard=False)
        self.assertEqual(free.stop_reason, "max_new_tokens")
        for speculative in (False, True):
            guarded = run(prompt, params, speculative, max_new=2000)
            self.assertEqual(guarded.stop_reason, "loop")
            self.assertLess(len(guarded.tokens), 2000)
            self.assertEqual(guarded.tokens, free.tokens[:len(guarded.tokens)])
            tail = guarded.tokens[-64:]
            occurrences = sum(guarded.tokens[i:i + 64] == tail for i in range(len(guarded.tokens) - 63))
            self.assertEqual(occurrences, 3)

    def test_loop_guard_ignores_long_non_repeating(self):
        prompt = [3]
        params = SamplingParams(temperature=1.0, repetition_penalty=1.0)
        session = FakeSession(1)
        session.by_token = torch.zeros(V, V)  # uniform sampling: 64-token windows never repeat three times
        result = generate(session, session.by_token[3], prompt, params, 1000, frozenset(), torch.Generator().manual_seed(0), False)
        self.assertEqual((len(result.tokens), result.stop_reason), (1000, "max_new_tokens"))

    def test_on_tokens_and_seed(self):
        prompt = [3, 8, 1, 9, 4, 6] * 3 + [3]
        params = SamplingParams(temperature=0.8, top_p=0.9)
        seen = []
        session = FakeSession(len(prompt))
        first = session.logits_after(prompt[-1], len(prompt) - 1)
        result = generate(session, first, prompt, params, 50, frozenset(), torch.Generator().manual_seed(4), True, on_tokens=seen.append)
        self.assertEqual(seen, result.tokens)
        again = run(prompt, params, True, max_new=50, seed=4)
        self.assertEqual(again.tokens, result.tokens)


if __name__ == "__main__":
    unittest.main()
