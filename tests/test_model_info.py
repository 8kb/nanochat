"""
Test scripts/model_info.py: config-mode (a hypothetical --arch/--depth, no checkpoint needed) and
checkpoint-mode (--checkpoints, reading an already-trained checkpoint's meta.json only -- no
weights loaded). The checkpoint-mode tests are the ones that matter for the contest workflow: they
exercise the tokenizer-fingerprint match/mismatch/unknown reporting this stage added.

python -m pytest tests/test_model_info.py -v
"""

import json
import argparse

import pytest

from nanochat.checkpoint_manager import save_checkpoint
from tests.conftest import build_tiny_model, TINY_KWARGS_BY_ARCH

from scripts import model_info


def _args(**overrides):
    defaults = dict(
        aspect_ratio=64, head_dim=16, max_seq_len=32, window_pattern=None, arch_opt=None,
        target_param_data_ratio=12, target_flops=-1.0, num_iterations=-1, total_batch_size=-1,
        weight_decay=0.28, gpu=None, num_gpus=1, mfu=0.4, kv_batch_size=1,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


@pytest.mark.parametrize("arch", list(TINY_KWARGS_BY_ARCH.keys()))
def test_inspect_one_config_mode_smoke(arch):
    """Every registered architecture can be inspected purely from --arch/--depth (no checkpoint,
    no data, no training) -- this is the tool's primary use case."""
    row = model_info.inspect_one(arch, 4, _args(), vocab_size=128)
    assert row["arch"] == arch
    assert row["params"]["total"] > 0
    assert row["flops"]["per_token"] > 0
    assert row["training_plan"]["num_iterations"] > 0


def _save_tiny_checkpoint(base_dir, tag, arch, tokenizer_fingerprint=None, core_metric=None):
    model = build_tiny_model(arch)
    checkpoint_dir = base_dir / "base_checkpoints" / tag
    meta = {
        "step": 3,
        "val_bpb": 1.23,
        "core_metric": core_metric,
        "tokenizer_fingerprint": tokenizer_fingerprint,
        "model_config": model.config.to_dict(),
        "total_batch_size": 1024,
        "user_config": {"depth": 4},
        "loop_state": {"total_training_time": 60.0},
    }
    save_checkpoint(str(checkpoint_dir), step=3, model_data=model.state_dict(), optimizer_data=None, meta_data=meta)
    return model


def test_inspect_checkpoint_reports_trained_stats(tmp_path, monkeypatch):
    monkeypatch.setattr(model_info, "get_base_dir", lambda: str(tmp_path))
    _save_tiny_checkpoint(tmp_path, "gpt_tiny", "gpt", tokenizer_fingerprint="abc123", core_metric=0.42)

    row = model_info.inspect_checkpoint("gpt_tiny", _args(), local_fingerprint="abc123")

    assert row["arch"] == "gpt"
    assert row["depth"] == 4
    assert row["trained"]["step"] == 3
    assert row["trained"]["tokens_trained"] == 3 * 1024
    assert row["trained"]["val_bpb"] == 1.23
    assert row["trained"]["core_metric"] == 0.42
    assert row["trained"]["tokenizer_fingerprint_status"] == "match"


def test_inspect_checkpoint_flags_tokenizer_mismatch(tmp_path, monkeypatch):
    monkeypatch.setattr(model_info, "get_base_dir", lambda: str(tmp_path))
    _save_tiny_checkpoint(tmp_path, "gpt_tiny", "gpt", tokenizer_fingerprint="abc123")

    row = model_info.inspect_checkpoint("gpt_tiny", _args(), local_fingerprint="different")
    assert row["trained"]["tokenizer_fingerprint_status"] == "MISMATCH"


def test_inspect_checkpoint_unknown_when_fingerprint_missing(tmp_path, monkeypatch):
    """A checkpoint saved before this stage has no tokenizer_fingerprint key at all -- nothing to
    compare, so 'unknown' rather than a false mismatch."""
    monkeypatch.setattr(model_info, "get_base_dir", lambda: str(tmp_path))
    _save_tiny_checkpoint(tmp_path, "gpt_tiny", "gpt", tokenizer_fingerprint=None)

    row = model_info.inspect_checkpoint("gpt_tiny", _args(), local_fingerprint="abc123")
    assert row["trained"]["tokenizer_fingerprint_status"] == "unknown"


def test_list_checkpoint_tags_explicit_list(tmp_path, monkeypatch):
    monkeypatch.setattr(model_info, "get_base_dir", lambda: str(tmp_path))
    assert model_info.list_checkpoint_tags("a, b") == ["a", "b"]


def test_list_checkpoint_tags_enumerates_all_when_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(model_info, "get_base_dir", lambda: str(tmp_path))
    _save_tiny_checkpoint(tmp_path, "gpt_tiny", "gpt")
    _save_tiny_checkpoint(tmp_path, "llama_tiny", "llama")
    assert model_info.list_checkpoint_tags("") == ["gpt_tiny", "llama_tiny"]


def test_list_checkpoint_tags_empty_when_no_checkpoints_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(model_info, "get_base_dir", lambda: str(tmp_path))
    assert model_info.list_checkpoint_tags("") == []
