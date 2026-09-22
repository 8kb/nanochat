"""
Test that nanochat.engine.generate_naive (no KV cache, recomputes the full forward every step)
and Engine.generate (KV cache, prefill + decode) agree exactly at temperature=0. This exercises
KV cache allocation via ModelManager.new_kv_cache(), the RoPE position offset from
kv_cache.get_pos(), and (for gpt) the smear decode path (kv_cache.state["prev_embedding"])
together, for both a full-context and a sliding-window attention pattern, and for every preset.

python -m pytest tests/test_generate.py -v
"""

import pytest
import torch

from modelcore import ModelManager
from nanochat.architectures import presets
from nanochat.engine import Engine, generate_naive


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


def _build(preset, window_pattern, manager, **extra):
    kwargs = dict(depth=4, aspect_ratio=16, head_dim=32, max_seq_len=32, vocab_size=128, window_pattern=window_pattern)
    kwargs.update(extra)
    config = presets.expand(preset, **kwargs)
    return manager.create_model(config, device=torch.device("cpu"), seed=0)


@pytest.mark.parametrize("preset,extra", [
    ("gpt", {}), ("llama", {}), ("llama_kvshare", {"kv_share_frac": 0.5}), ("llama_kvshare_win", {"kv_share_frac": 0.5}),
])
@pytest.mark.parametrize("window_pattern", ["L", "SSSL"])
def test_generate_naive_matches_engine_generate_at_temperature_zero(window_pattern, preset, extra):
    manager = ModelManager()
    model = _build(preset, window_pattern, manager, **extra)
    tokenizer = _FakeTokenizer(model.config.vocab_size)
    prompt = [1, 2, 3, 4]

    naive_tokens = list(generate_naive(model, prompt, max_tokens=8, temperature=0.0))

    engine = Engine(model, tokenizer, manager=manager)
    results, masks = engine.generate_batch(prompt, num_samples=1, max_tokens=8, temperature=0.0)
    engine_tokens = results[0][len(prompt):]

    assert naive_tokens == engine_tokens


# -----------------------------------------------------------------------------
# Engine.generate_batch_multi: several different prompts decoded in one batch. Each group must
# equal what that prompt alone produces at temperature 0 -- checked against generate_naive, which
# shares no code with the KV cache or the ragged decode path.

def _build_awake(preset, window_pattern, manager, **extra):
    """_build, with the zero-initialised parameters given small random values. Init zeroes every
    attention/MLP output projection and the smear lambda_, so a fresh model's logits ignore
    attention, RoPE, the window and the smear state entirely -- a test of those passes whatever the
    code does (the existing test above cannot see them for that reason)."""
    model = _build(preset, window_pattern, manager, **extra)
    g = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for p in model.parameters():
            if p.numel() and p.abs().sum() == 0:
                p.copy_(torch.randn(p.shape, generator=g) * 0.1)
    return model


# lengths straddle the sliding window ("SSSL" windows are shorter than the longest prompts)
MULTI_PROMPTS = [[1, 2], [3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15, 16], [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]]


@pytest.mark.parametrize("preset,extra", [
    ("gpt", {}), ("llama", {}), ("llama_kvshare", {"kv_share_frac": 0.5}), ("llama_kvshare_win", {"kv_share_frac": 0.5}),
])
@pytest.mark.parametrize("window_pattern", ["L", "SSSL"])
def test_generate_batch_multi_matches_generate_naive_for_every_prompt(window_pattern, preset, extra):
    manager = ModelManager()
    model = _build_awake(preset, window_pattern, manager, **extra)
    engine = Engine(model, _FakeTokenizer(model.config.vocab_size), manager=manager)

    results, masks = engine.generate_batch_multi(MULTI_PROMPTS, num_samples=1, max_tokens=8, temperature=0.0)

    assert len(results) == len(MULTI_PROMPTS)
    for prompt, group in zip(MULTI_PROMPTS, results):
        assert group[0][:len(prompt)] == prompt
        assert group[0][len(prompt):] == list(generate_naive(model, prompt, max_tokens=8, temperature=0.0))


def test_generate_batch_multi_of_one_prompt_is_generate_batch_wrapped():
    manager = ModelManager()
    model = _build_awake("gpt", "SSSL", manager)
    engine = Engine(model, _FakeTokenizer(model.config.vocab_size), manager=manager)
    prompt = MULTI_PROMPTS[2]
    single = engine.generate_batch(prompt, num_samples=2, max_tokens=6, temperature=0.0)
    multi = engine.generate_batch_multi([prompt], num_samples=2, max_tokens=6, temperature=0.0)
    assert multi == ([single[0]], [single[1]])


def test_single_prompt_generate_batch_is_unchanged_by_multi_prompt_support():
    """generate_batch (what chat_rl and every existing caller use) must not know multi exists."""
    manager = ModelManager()
    model = _build_awake("llama_kvshare_win", "SSSL", manager, kv_share_frac=0.5)
    engine = Engine(model, _FakeTokenizer(model.config.vocab_size), manager=manager)
    prompt = MULTI_PROMPTS[3]
    results, _ = engine.generate_batch(prompt, num_samples=1, max_tokens=8, temperature=0.0)
    assert results[0][len(prompt):] == list(generate_naive(model, prompt, max_tokens=8, temperature=0.0))
