"""
Test scripts/model_info.py: config-mode (a hypothetical --arch/--depth, no checkpoint needed) and
checkpoint-mode (--checkpoints, reading an already-trained checkpoint's meta.json only -- no
weights loaded). The checkpoint-mode tests are the ones that matter for the contest workflow: they
exercise the tokenizer-fingerprint match/mismatch/unknown reporting this stage added.

python -m pytest tests/test_model_info.py -v
"""

import argparse

import pytest

from modelcore import ModelManager
from nanochat.architectures import presets
from nanochat.checkpoint_manager import save_checkpoint

from scripts import model_info

PRESET_NAMES = ["gpt", "llama", "llama_kvshare", "llama_kvshare_win"]


def _args(**overrides):
    defaults = dict(
        arch="gpt", aspect_ratio=64, head_dim=16, max_seq_len=32, window_pattern=None, arch_opt=None,
        model_config=None, d_ref_scaling_params=None,
        target_param_data_ratio=12, target_flops=-1.0, num_iterations=-1, total_batch_size=-1,
        weight_decay=0.28, gpu=None, num_gpus=1, mfu=0.4, kv_batch_size=1,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


@pytest.mark.parametrize("preset", PRESET_NAMES)
def test_inspect_one_config_mode_smoke(preset):
    """Every preset can be inspected purely from --arch/--depth (no checkpoint, no data, no
    training) -- this is the tool's primary use case."""
    row = model_info.inspect_one(preset, 4, _args(arch=preset), vocab_size=128)
    assert row["arch"] == preset
    assert row["params"]["total"] > 0
    assert row["flops"]["per_token"] > 0
    assert row["training_plan"]["num_iterations"] > 0


@pytest.mark.parametrize("preset", PRESET_NAMES)
def test_inspect_one_via_model_config_matches_via_arch(preset):
    """--model-config <preset> (overriding --arch) resolves to exactly the same tree as bare
    --arch <preset> -- the two are the same code path (see model_info.build_config)."""
    via_model_config = model_info.inspect_one("gpt", 4, _args(model_config=preset), vocab_size=128)
    via_arch = model_info.inspect_one(preset, 4, _args(arch=preset), vocab_size=128)
    assert via_model_config["params"]["total"] == via_arch["params"]["total"]
    assert via_model_config["flops"]["per_token"] == via_arch["flops"]["per_token"]


def test_dump_config_round_trips_through_model_config():
    """The dump -> edit -> train bridge: --dump-config's output must be exactly what
    presets.resolve_model_config accepts back as a --model-config file."""
    import json
    import tempfile

    config = presets.expand("gpt", depth=4, aspect_ratio=16, head_dim=16, max_seq_len=32, vocab_size=128)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(ModelManager().config_to_dict(config), f)
        path = f.name
    reloaded = presets.resolve_model_config(path, depth=4, aspect_ratio=16, head_dim=16, max_seq_len=32, vocab_size=128)
    assert reloaded == config


def _save_tiny_checkpoint(base_dir, tag, preset, tokenizer_fingerprint=None, core_metric=None, **preset_kwargs):
    manager = ModelManager()
    kwargs = dict(depth=4, aspect_ratio=16, head_dim=16, max_seq_len=32, vocab_size=128)
    kwargs.update(preset_kwargs)
    config = presets.expand(preset, **kwargs)
    model = manager.create_model(config, device="cpu", seed=0)
    checkpoint_dir = base_dir / "base_checkpoints" / tag
    meta = {
        "step": 3,
        "val_bpb": 1.23,
        "core_metric": core_metric,
        "tokenizer_fingerprint": tokenizer_fingerprint,
        "model_config": manager.config_to_dict(config),
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


def test_inspect_checkpoint_migrates_a_legacy_flat_config(tmp_path, monkeypatch):
    """A checkpoint's model_config predating modelcore (no "format" key) must still be
    inspectable -- routed through nanochat.architectures.legacy just like checkpoint_manager."""
    monkeypatch.setattr(model_info, "get_base_dir", lambda: str(tmp_path))
    manager = ModelManager()
    config = presets.expand("llama", depth=4, aspect_ratio=16, head_dim=16, max_seq_len=32, vocab_size=128)
    model = manager.create_model(config, device="cpu", seed=0)
    flat_config = {
        "sequence_len": config.sequence_len, "vocab_size": config.vocab_size, "n_layer": 4,
        "n_head": config.n_embd // 16, "n_kv_head": config.n_embd // 16, "n_embd": config.n_embd,
        "window_pattern": "L", "arch": "llama",
    }
    checkpoint_dir = tmp_path / "base_checkpoints" / "llama_legacy"
    save_checkpoint(str(checkpoint_dir), step=0, model_data=model.state_dict(), optimizer_data=None,
                     meta_data={"step": 0, "model_config": flat_config})

    row = model_info.inspect_checkpoint("llama_legacy", _args(), local_fingerprint=None)
    assert row["arch"] == "llama"
    assert row["shape"]["n_layer"] == 4
    assert row["params"]["total"] > 0


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
