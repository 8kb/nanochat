"""
Test that checkpoint_manager.build_model can save and reload a current-format modelcore
checkpoint faithfully -- exact state dict, exact forward output -- for every preset architecture.
See tests/test_architectures.py for the separate (and much more thorough, real-checkpoint-backed)
coverage of migrating an *old* (pre-modelcore) checkpoint; this file only covers the
already-current-format path plus checkpoint_manager's own tokenizer-compatibility checks.

python -m pytest tests/test_checkpoint_roundtrip.py -v
"""

import logging

import torch
import pytest

from modelcore import ModelManager

from nanochat import checkpoint_manager
from nanochat.checkpoint_manager import save_checkpoint, build_model
from nanochat.architectures import presets

TINY_KWARGS = dict(depth=4, aspect_ratio=16, head_dim=32, max_seq_len=32, vocab_size=128)

manager = ModelManager()


class _FakeTokenizer:
    """Stub matching the methods build_model needs: the vocab-size compatibility check and the
    tokenizer-fingerprint mismatch warning."""
    def get_vocab_size(self):
        return TINY_KWARGS["vocab_size"]

    def fingerprint(self):
        return "local0000000000"


def _build_tiny(preset, **overrides):
    kwargs = {**TINY_KWARGS, **overrides}
    config = presets.expand(preset, **kwargs)
    return manager.create_model(config, device=torch.device("cpu"), seed=0)


@pytest.mark.parametrize("preset,kwargs", [
    ("gpt", {}),
    ("llama", {}),
    ("llama_kvshare", {"kv_share_frac": 0.5}),
    ("llama_kvshare_win", {"kv_share_frac": 0.5}),
])
def test_checkpoint_roundtrip_preserves_weights_and_forward_output(tmp_path, monkeypatch, preset, kwargs):
    monkeypatch.setattr(checkpoint_manager, "get_tokenizer", lambda: _FakeTokenizer())

    model = _build_tiny(preset, **kwargs)
    checkpoint_dir = str(tmp_path / f"{preset}_d_tiny")
    save_checkpoint(
        checkpoint_dir, step=0,
        model_data=model.state_dict(), optimizer_data=None,
        meta_data={"step": 0, "model_config": model.config.to_dict()},
    )

    reloaded, tokenizer, meta = build_model(checkpoint_dir, step=0, device=torch.device("cpu"), phase="eval")

    assert meta["model_config"]["format"] == "modelcore.v1"
    assert reloaded.config == model.config

    original_state = model.state_dict()
    reloaded_state = reloaded.state_dict()
    assert original_state.keys() == reloaded_state.keys()
    for key in original_state:
        assert torch.equal(original_state[key], reloaded_state[key]), f"mismatch in {key}"

    idx = torch.randint(0, model.config.vocab_size, (1, 5))
    with torch.no_grad():
        original_logits = model.forward(idx)
        reloaded_logits = reloaded.forward(idx)
    assert torch.equal(original_logits, reloaded_logits)


def test_checkpoint_roundtrip_warns_on_tokenizer_fingerprint_mismatch(tmp_path, monkeypatch, caplog):
    """A checkpoint whose tokenizer_fingerprint disagrees with the local tokenizer's must still
    load (vocab_size still matches) but should warn loudly -- this is the trap a contest run would
    otherwise hit silently: same vocab size, different token ids, garbage output."""
    monkeypatch.setattr(checkpoint_manager, "get_tokenizer", lambda: _FakeTokenizer())

    model = _build_tiny("gpt")
    checkpoint_dir = str(tmp_path / "d_mismatch")
    save_checkpoint(
        checkpoint_dir, step=0,
        model_data=model.state_dict(), optimizer_data=None,
        meta_data={"step": 0, "model_config": model.config.to_dict(), "tokenizer_fingerprint": "cloud0000000000"},
    )

    with caplog.at_level(logging.INFO, logger="nanochat.checkpoint_manager"):
        build_model(checkpoint_dir, step=0, device=torch.device("cpu"), phase="eval")
    assert any("tokenizer fingerprint mismatch" in record.message for record in caplog.records)


def test_checkpoint_roundtrip_silent_on_tokenizer_fingerprint_match(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(checkpoint_manager, "get_tokenizer", lambda: _FakeTokenizer())

    model = _build_tiny("gpt")
    checkpoint_dir = str(tmp_path / "d_match")
    save_checkpoint(
        checkpoint_dir, step=0,
        model_data=model.state_dict(), optimizer_data=None,
        meta_data={"step": 0, "model_config": model.config.to_dict(), "tokenizer_fingerprint": "local0000000000"},
    )

    with caplog.at_level(logging.INFO, logger="nanochat.checkpoint_manager"):
        build_model(checkpoint_dir, step=0, device=torch.device("cpu"), phase="eval")
    assert not any("tokenizer fingerprint mismatch" in record.message for record in caplog.records)


def test_checkpoint_roundtrip_silent_when_fingerprint_key_absent(tmp_path, monkeypatch, caplog):
    """Checkpoints saved before the fingerprint existed have no tokenizer_fingerprint key at all
    -- nothing to compare, so no warning (not every old checkpoint should look suspicious)."""
    monkeypatch.setattr(checkpoint_manager, "get_tokenizer", lambda: _FakeTokenizer())

    model = _build_tiny("gpt")
    checkpoint_dir = str(tmp_path / "d_no_fingerprint")
    save_checkpoint(
        checkpoint_dir, step=0,
        model_data=model.state_dict(), optimizer_data=None,
        meta_data={"step": 0, "model_config": model.config.to_dict()},
    )

    with caplog.at_level(logging.INFO, logger="nanochat.checkpoint_manager"):
        build_model(checkpoint_dir, step=0, device=torch.device("cpu"), phase="eval")
    assert not any("tokenizer fingerprint mismatch" in record.message for record in caplog.records)
