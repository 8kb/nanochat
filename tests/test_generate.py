"""
Test that nanochat.engine.generate_naive (no KV cache, recomputes the full forward every step)
and Engine.generate (KV cache, prefill + decode) agree exactly at temperature=0. This exercises
KV cache allocation via kv_cache_spec(), the RoPE position offset from kv_cache.get_pos(), and
(for GPT) the smear decode path (kv_cache.state["prev_embedding"]) together, for both a
full-context and a sliding-window attention pattern, and for every registered architecture.

python -m pytest tests/test_generate.py -v
"""

import pytest

from nanochat.engine import Engine, generate_naive
from tests.conftest import build_tiny_model


class _FakeTokenizer:
    """Minimal tokenizer stub for Engine.generate's tool-use bookkeeping. Special-token ids are
    placed at vocab_size and above, so an untrained tiny model's logits (always < vocab_size)
    can never produce one -- this keeps the test deterministic and independent of random init,
    since the tool-use / early-stop machinery in Engine.generate is guaranteed to never trigger."""
    def __init__(self, vocab_size):
        base = vocab_size
        self._specials = {
            "<|python_start|>": base + 0,
            "<|python_end|>": base + 1,
            "<|output_start|>": base + 2,
            "<|output_end|>": base + 3,
            "<|assistant_end|>": base + 4,
        }
        self._bos = base + 5

    def encode_special(self, s):
        return self._specials[s]

    def get_bos_token_id(self):
        return self._bos

    def decode(self, ids):
        return "".join(str(i) for i in ids)


@pytest.mark.parametrize("arch", ["gpt", "llama"])
@pytest.mark.parametrize("window_pattern", ["L", "SSSL"])
def test_generate_naive_matches_engine_generate_at_temperature_zero(window_pattern, arch):
    model = build_tiny_model(arch, window_pattern=window_pattern)
    tokenizer = _FakeTokenizer(model.config.vocab_size)
    prompt = [1, 2, 3, 4]

    naive_tokens = list(generate_naive(model, prompt, max_tokens=8, temperature=0.0))

    engine = Engine(model, tokenizer)
    results, masks = engine.generate_batch(prompt, num_samples=1, max_tokens=8, temperature=0.0)
    engine_tokens = results[0][len(prompt):]

    assert naive_tokens == engine_tokens
