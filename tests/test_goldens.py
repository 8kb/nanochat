"""
Replays every golden captured by dev/capture_model_goldens.py (see the Stage 7 plan /
docs/roadmap.md) against the CURRENT code and asserts every recorded number is unchanged.

This is the regression net for the Stage 7 `modelcore/` extraction: it must keep passing,
unmodified in its assertions, through every commit of that refactor. Real-checkpoint cases skip
automatically on a machine without ~/.cache/nanochat populated; the synthetic tiny_* cases (a
committed state dict per architecture/preset under tests/goldens/tiny/) always run.

python -m pytest tests/test_goldens.py -v
"""
import json
import os

import pytest
import torch

from nanochat import checkpoint_manager
from nanochat.checkpoint_manager import load_model_from_dir
from nanochat.common import get_base_dir
from nanochat.engine import Engine, generate_naive

from dev.capture_model_goldens import (
    GOLDENS_DIR, TINY_DIR, DEVICE, GEN_TOKENS, PROMPT_TEXT,
    _FakeTokenizer, accounting, fixed_logits_hash, optimizer_digest_from_dir, optimizer_layout,
    state_dict_fingerprint,
)

# out_name -> subdir under get_base_dir() ("base_checkpoints" / "chatsft_checkpoints"), tag
REAL_SOURCES = {
    "d6": ("base_checkpoints", "d6"),
    "smoketest_kvshare_win_d2": ("base_checkpoints", "smoketest_kvshare_win_d2"),
    "contest_h100run_gpt_d12": ("base_checkpoints", "contest_h100run_gpt_d12"),
    "contest_h100run_llama_d12": ("base_checkpoints", "contest_h100run_llama_d12"),
    "contest_h100run_llama_kvshare_win_d12": ("base_checkpoints", "contest_h100run_llama_kvshare_win_d12"),
    "contest_d12test_llama_kvshare_d12": ("base_checkpoints", "contest_d12test_llama_kvshare_d12"),
    "chatsft_contest_h100run_gpt_d12": ("chatsft_checkpoints", "contest_h100run_gpt_d12"),
}

TINY_SOURCES = [
    "tiny_gpt", "tiny_llama", "tiny_llama_kvshare", "tiny_llama_kvshare_win",
    "tiny_composed_gpt", "tiny_composed_llama", "tiny_composed_llama_kvshare",
    "tiny_composed_llama_kvshare_win",
]


def _load_golden(name):
    with open(os.path.join(GOLDENS_DIR, f"{name}.json"), encoding="utf-8") as f:
        return json.load(f)


def _assert_matches_golden(model, tokenizer, golden, prompt_tokens=None):
    assert state_dict_fingerprint(model.state_dict())["multiset"] == golden["state_dict_fingerprint"]["multiset"]
    assert accounting(model) == golden["accounting"]
    config = model.config
    assert fixed_logits_hash(model, config.vocab_size, config.sequence_len) == golden["logits_hash"]
    assert optimizer_layout(model) == golden["optimizer_layout"]

    if prompt_tokens is not None:
        prompt = list(prompt_tokens)
    else:
        prompt = tokenizer.encode(PROMPT_TEXT, prepend=tokenizer.get_bos_token_id())
    naive_tokens = list(generate_naive(model, prompt, max_tokens=GEN_TOKENS, temperature=0.0))
    assert naive_tokens == golden["generate_naive_tokens"]

    engine = Engine(model, tokenizer)
    batch_tokens, _ = engine.generate_batch(prompt, num_samples=1, max_tokens=GEN_TOKENS, temperature=0.0)
    assert batch_tokens[0][len(prompt):] == golden["generate_batch_tokens"]


@pytest.mark.parametrize("out_name", sorted(REAL_SOURCES))
def test_real_checkpoint_golden(out_name):
    subdir, tag = REAL_SOURCES[out_name]
    checkpoints_dir = os.path.join(get_base_dir(), subdir)
    if not os.path.isdir(os.path.join(checkpoints_dir, tag)):
        pytest.skip(f"{checkpoints_dir}/{tag} not present on this machine")

    golden = _load_golden(out_name)
    model, tokenizer, meta = load_model_from_dir(checkpoints_dir, DEVICE, phase="eval", model_tag=tag)
    model.eval()
    assert meta["model_config"] == golden["meta_model_config"]

    _assert_matches_golden(model, tokenizer, golden)

    checkpoint_dir = os.path.join(checkpoints_dir, tag)
    shard = optimizer_digest_from_dir(model, checkpoint_dir)
    assert shard == golden["optimizer_shard"]


@pytest.mark.parametrize("out_name", TINY_SOURCES)
def test_tiny_synthetic_golden(out_name, monkeypatch):
    golden = _load_golden(out_name)
    vocab_size = golden["meta_model_config"]["vocab_size"]
    monkeypatch.setattr(checkpoint_manager, "get_tokenizer", lambda: _FakeTokenizer(vocab_size))

    model, tokenizer, meta = load_model_from_dir(TINY_DIR, DEVICE, phase="eval", model_tag=out_name)
    model.eval()
    assert meta["model_config"] == golden["meta_model_config"]

    prompt_tokens = [i % vocab_size for i in range(1, 5)]
    _assert_matches_golden(model, tokenizer, golden, prompt_tokens=prompt_tokens)

    checkpoint_dir = os.path.join(TINY_DIR, out_name)
    shard = optimizer_digest_from_dir(model, checkpoint_dir)
    assert shard == golden["optimizer_shard"]
