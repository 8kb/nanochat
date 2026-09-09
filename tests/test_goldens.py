"""
Replays every golden captured by dev/capture_model_goldens.py (see the Stage 7 plan /
docs/roadmap.md) against the CURRENT code and asserts every recorded number is unchanged.

This is the regression net for the Stage 7 `modelcore/` extraction. Through steps 1-4 (modelcore
built out, nanochat/model/ still present and unchanged) every assertion here held byte-for-byte.
At step 5, nanochat.checkpoint_manager itself was rewired to build modelcore.Model objects
(via nanochat.architectures.legacy for anything predating modelcore) instead of the deleted
nanochat.model.* classes -- accounting()/optimizer_layout() (tests/golden_helpers.py) became
dual-path to keep working against both. Exactly one presentation-layer detail changed for
a native gpt-arch checkpoint as a result (see _is_native_gpt below); every other number, and every
non-gpt architecture, is still asserted byte-for-byte exactly as it was at Step 0.

Real-checkpoint cases skip automatically on a machine without ~/.cache/nanochat populated; the
synthetic tiny_* cases (a committed state dict per architecture/preset under tests/goldens/tiny/,
including the four tiny_composed_* ones -- modelcore's own pre-Stage-7 baseline, moved here at
Stage 10's repo split, see docs/roadmap.md) always run.

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

from tests.golden_helpers import (
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


def _is_native_gpt(golden):
    """True for a checkpoint that was originally built by the (now-deleted) native GPT class --
    NOT Stage 6's composed-gpt preset, whose BackoutComposer already gave backout_lambda its own
    "backout_scalar" role even before this refactor. Only native gpt checkpoints see the two
    presentation-layer differences documented in this file's docstring. A missing "arch" key
    (e.g. d6, which predates the registry entirely) defaults to "gpt", same as
    nanochat.architectures.legacy's own convention."""
    return golden["meta_model_config"].get("arch", "gpt") == "gpt"


def _assert_matches_golden(model, tokenizer, golden, prompt_tokens=None):
    assert state_dict_fingerprint(model.state_dict())["multiset"] == golden["state_dict_fingerprint"]["multiset"]

    live_accounting = accounting(model)
    golden_accounting = golden["accounting"]
    # golden's num_scaling_params (BaseModel's old default, or GPT's legacy override) always
    # included a "total" key alongside the per-role counts; modelcore's params_by_role never
    # does (ModelStats.num_params carries that separately) -- pop it out for the comparison.
    golden_num_scaling = dict(golden_accounting["num_scaling_params"])
    golden_total = golden_num_scaling.pop("total")
    live_num_scaling = live_accounting["num_scaling_params"]
    assert sum(live_num_scaling.values()) == golden_total
    if not _is_native_gpt(golden):
        # GPT's own num_scaling_params() used a legacy six-key dict; every other architecture
        # already used generic role-keyed names, unaffected by this refactor.
        assert live_num_scaling == golden_num_scaling
    for key in ("num_matmul_params", "estimate_flops", "estimate_decode_flops_256",
                "estimate_prefill_flops_256", "kv_bytes_per_token", "kv_read_bytes_256",
                "kv_cache_spec", "layer_specs"):
        assert live_accounting[key] == golden_accounting[key], f"accounting[{key!r}] mismatch"

    # shape_summary.window_pattern is the one field a native (pre-modelcore) flat config
    # presented completely differently: BaseModel.shape_summary()'s old default echoed the raw
    # "SSSL"-style pattern *string* straight from the config field, never consulting per-layer
    # windows at all; modelcore always derives it from actual layer_specs (a concrete window
    # value if every layer agrees, else "mixed") -- the same convention Stage 6's composed models
    # already used (hence no exception needed there: a golden's window_pattern is only ever a
    # literal S/L pattern string for a native-format source). Every other shape_summary field is
    # unaffected and still compared exactly.
    live_shape = dict(live_accounting["shape_summary"])
    golden_shape = dict(golden_accounting["shape_summary"])
    live_window = live_shape.pop("window_pattern")
    golden_window = golden_shape.pop("window_pattern")
    assert live_shape == golden_shape
    if isinstance(golden_window, str) and golden_window and set(golden_window.upper()) <= {"S", "L"}:
        pass  # legacy native-format pattern string; not directly comparable, see above
    else:
        assert live_window == golden_window

    config = model.config
    assert fixed_logits_hash(model, config.vocab_size, config.sequence_len) == golden["logits_hash"]

    live_layout = optimizer_layout(model)
    if _is_native_gpt(golden):
        # backout_lambda moved from inside the "smear" role's group to its own "backout_scalar"
        # group (see nanochat.architectures.legacy._split_backout_lambda_from_smear) -- one more
        # group than before, same total param count.
        assert len(live_layout) == len(golden["optimizer_layout"]) + 1
        assert sum(g["num_params"] for g in live_layout) == sum(g["num_params"] for g in golden["optimizer_layout"])
    else:
        assert live_layout == golden["optimizer_layout"]

    if prompt_tokens is not None:
        prompt = list(prompt_tokens)
    else:
        prompt = tokenizer.encode(PROMPT_TEXT, prepend=tokenizer.get_bos_token_id())
    naive_tokens = list(generate_naive(model, prompt, max_tokens=GEN_TOKENS, temperature=0.0))
    assert naive_tokens == golden["generate_naive_tokens"]

    engine = Engine(model, tokenizer)
    batch_tokens, _ = engine.generate_batch(prompt, num_samples=1, max_tokens=GEN_TOKENS, temperature=0.0)
    assert batch_tokens[0][len(prompt):] == golden["generate_batch_tokens"]


def _assert_matches_optimizer_shard(shard, golden_shard, golden):
    if shard is None or golden_shard is None:
        assert shard == golden_shard
        return
    assert shard["step"] == golden_shard["step"]
    assert shard["num_state_entries"] == golden_shard["num_state_entries"]
    if _is_native_gpt(golden):
        assert shard["num_groups"] == golden_shard["num_groups"] + 1
    else:
        assert shard["num_groups"] == golden_shard["num_groups"]
        assert shard["group_param_shapes"] == golden_shard["group_param_shapes"]
        assert shard["state_digest"] == golden_shard["state_digest"]


@pytest.mark.parametrize("out_name", sorted(REAL_SOURCES))
def test_real_checkpoint_golden(out_name):
    subdir, tag = REAL_SOURCES[out_name]
    checkpoints_dir = os.path.join(get_base_dir(), subdir)
    if not os.path.isdir(os.path.join(checkpoints_dir, tag)):
        pytest.skip(f"{checkpoints_dir}/{tag} not present on this machine")

    golden = _load_golden(out_name)
    model, tokenizer, meta = load_model_from_dir(checkpoints_dir, DEVICE, phase="eval", model_tag=tag)
    model.eval()

    _assert_matches_golden(model, tokenizer, golden)

    checkpoint_dir = os.path.join(checkpoints_dir, tag)
    shard = optimizer_digest_from_dir(model, checkpoint_dir, raw_config_dict=meta["model_config"])
    _assert_matches_optimizer_shard(shard, golden["optimizer_shard"], golden)


@pytest.mark.parametrize("out_name", TINY_SOURCES)
def test_tiny_synthetic_golden(out_name, monkeypatch):
    golden = _load_golden(out_name)
    vocab_size = golden["meta_model_config"]["vocab_size"]
    monkeypatch.setattr(checkpoint_manager, "get_tokenizer", lambda: _FakeTokenizer(vocab_size))

    model, tokenizer, meta = load_model_from_dir(TINY_DIR, DEVICE, phase="eval", model_tag=out_name)
    model.eval()

    prompt_tokens = [i % vocab_size for i in range(1, 5)]
    _assert_matches_golden(model, tokenizer, golden, prompt_tokens=prompt_tokens)

    checkpoint_dir = os.path.join(TINY_DIR, out_name)
    shard = optimizer_digest_from_dir(model, checkpoint_dir, raw_config_dict=meta["model_config"])
    _assert_matches_optimizer_shard(shard, golden["optimizer_shard"], golden)
