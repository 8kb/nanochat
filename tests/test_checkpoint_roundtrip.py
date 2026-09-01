"""
Test that checkpoint_manager.build_model can save and reload a model purely through the
architecture registry (nanochat.model.get_model_class / config_from_dict), with no direct
GPT/GPTConfig import in checkpoint_manager itself.

python -m pytest tests/test_checkpoint_roundtrip.py -v
"""

import torch

from nanochat import checkpoint_manager
from nanochat.checkpoint_manager import save_checkpoint, build_model
from tests.conftest import build_tiny_gpt, TINY_GPT_KWARGS


class _FakeTokenizer:
    """Stub matching the one method build_model needs: the vocab-size compatibility check."""
    def get_vocab_size(self):
        return TINY_GPT_KWARGS["vocab_size"]


def test_checkpoint_roundtrip_preserves_weights_and_forward_output(tmp_path, monkeypatch):
    monkeypatch.setattr(checkpoint_manager, "get_tokenizer", lambda: _FakeTokenizer())

    model = build_tiny_gpt()
    checkpoint_dir = str(tmp_path / "d_tiny")
    save_checkpoint(
        checkpoint_dir, step=0,
        model_data=model.state_dict(), optimizer_data=None,
        meta_data={"step": 0, "model_config": model.config.to_dict()},
    )

    reloaded, tokenizer, meta = build_model(checkpoint_dir, step=0, device=torch.device("cpu"), phase="eval")

    assert meta["model_config"]["arch"] == "gpt"
    assert type(reloaded) is type(model)
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


def test_checkpoint_roundtrip_legacy_config_without_arch_key(tmp_path, monkeypatch):
    """A checkpoint saved before the registry existed has no "arch" key in model_config; it
    must still load, defaulting to the gpt architecture."""
    monkeypatch.setattr(checkpoint_manager, "get_tokenizer", lambda: _FakeTokenizer())

    model = build_tiny_gpt()
    legacy_config = model.config.to_dict()
    del legacy_config["arch"]
    checkpoint_dir = str(tmp_path / "d_legacy")
    save_checkpoint(
        checkpoint_dir, step=0,
        model_data=model.state_dict(), optimizer_data=None,
        meta_data={"step": 0, "model_config": legacy_config},
    )

    reloaded, tokenizer, meta = build_model(checkpoint_dir, step=0, device=torch.device("cpu"), phase="eval")
    assert reloaded.config == model.config
