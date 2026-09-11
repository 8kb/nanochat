"""
Tests for nanochat.architectures.adapters (expand_adapters/expand_adapters_for_config -- the
low-code -> materialized adapter expansion) and checkpoint_manager.build_model's config_override
threading -- the nanochat-side half of PEFT support. See modelcore/tests/test_peft.py for the
modelcore-side coverage (AdapterLinear, the reconciling load, merge/disable, optimizer roles, ...);
this file only covers the pieces that live in nanochat.

python -m pytest tests/test_adapters.py -v
"""
import dataclasses

import pytest
import torch

from modelcore import AdapterSpec, ModelManager
from modelcore.peft import find_adapters

from nanochat import checkpoint_manager
from nanochat.architectures import presets
from nanochat.architectures.adapters import expand_adapters, expand_adapters_for_config, list_linear_targets
from nanochat.checkpoint_manager import build_model, save_checkpoint

TINY_KWARGS = dict(depth=4, aspect_ratio=16, head_dim=32, max_seq_len=32, vocab_size=128)

manager = ModelManager()


class _FakeTokenizer:
    """Stub matching what build_model needs -- see tests/test_checkpoint_roundtrip.py's identical fixture."""
    def get_vocab_size(self):
        return TINY_KWARGS["vocab_size"]

    def fingerprint(self):
        return "local0000000000"


def _build_tiny(preset="gpt", **overrides):
    kwargs = {**TINY_KWARGS, **overrides}
    config = presets.expand(preset, **kwargs)
    return manager.create_model(config, device=torch.device("cpu"), seed=0)


# -----------------------------------------------------------------------------
# expand_adapters / expand_adapters_for_config / list_linear_targets

def test_list_linear_targets_covers_every_block_projection():
    model = _build_tiny("gpt")
    targets = list_linear_targets(model)
    assert "body.blocks.0.attn.c_q" in targets
    assert "body.blocks.0.attn.c_proj" in targets
    assert "body.blocks.0.mlp.c_fc" in targets
    assert len(targets) == len(set(targets)), "no target should be listed twice"


def test_expand_adapters_materializes_every_layer_and_target():
    model = _build_tiny("gpt")
    adapters, frozen = expand_adapters(model, {
        "name": "sft0", "method": "lora", "r": 8, "alpha": 16,
        "targets": ["attn.c_q", "attn.c_v"], "layers": "all", "freeze_base": True,
    })
    n_layer = model.config.n_layer
    assert len(adapters) == n_layer * 2
    assert frozen == ["body"]
    expected = {f"body.blocks.{i}.attn.{t}" for i in range(n_layer) for t in ("c_q", "c_v")}
    assert {a.target for a in adapters} == expected
    assert all(a.name == "sft0" and a.type == "lora" and a.params == {"r": 8, "alpha": 16} for a in adapters)


def test_expand_adapters_respects_explicit_layers_and_freeze_base_false():
    model = _build_tiny("gpt")
    adapters, frozen = expand_adapters(model, {
        "name": "t", "targets": ["attn.c_proj"], "layers": [0, 2], "freeze_base": False,
    })
    assert {a.target for a in adapters} == {"body.blocks.0.attn.c_proj", "body.blocks.2.attn.c_proj"}
    assert frozen == []


def test_expand_adapters_raises_on_a_target_that_does_not_resolve():
    model = _build_tiny("gpt")
    with pytest.raises(ValueError, match="does not resolve to a Linear"):
        expand_adapters(model, {"name": "t", "targets": ["attn.nonexistent"]})


def test_expand_adapters_for_config_matches_expand_adapters_on_a_built_model():
    """expand_adapters_for_config's own meta-device probe must derive the exact same
    (adapters, frozen) as calling expand_adapters directly on an already-built model."""
    model = _build_tiny("llama")
    request = {"name": "t0", "targets": ["attn.c_proj"], "layers": "all"}
    via_model_adapters, via_model_frozen = expand_adapters(model, request)
    via_config_adapters, via_config_frozen = expand_adapters_for_config(model.config, request)
    assert [a.target for a in via_model_adapters] == [a.target for a in via_config_adapters]
    assert via_model_frozen == via_config_frozen


def test_hand_written_materialized_adapters_pass_through_expand_untouched():
    """A caller that already has concrete AdapterSpecs (e.g. loaded from JSON, or hand-written)
    doesn't need expand_adapters at all -- it can set ModelConfig.adapters/.frozen directly, the
    same "preset OR materialized tree" duality --model-config already has."""
    model = _build_tiny("gpt")
    hand_written = [AdapterSpec(target="body.blocks.0.attn.c_proj", name="manual", type="lora", params={"r": 2, "alpha": 4})]
    config = dataclasses.replace(model.config, adapters=hand_written, frozen=["body"])
    rebuilt = manager.create_model(config, device=torch.device("cpu"), seed=0)
    fqn, module = find_adapters(rebuilt)[0]
    assert fqn == "body.blocks.0.attn.c_proj"
    assert "manual" in module.deltas


# -----------------------------------------------------------------------------
# checkpoint_manager.build_model's config_override -- the actual nanochat-side feature

def test_build_model_config_override_attaches_adapters_to_an_existing_checkpoint(tmp_path, monkeypatch):
    """A plain checkpoint, no adapters at save time; passing a config_override with adapters/
    frozen at load time attaches them via ModelManager.load_model's reconciling load -- no
    state-dict surgery, matching modelcore's own test_peft.py acceptance test."""
    monkeypatch.setattr(checkpoint_manager, "get_tokenizer", lambda: _FakeTokenizer())

    model = _build_tiny("gpt")
    checkpoint_dir = str(tmp_path / "d_tiny")
    save_checkpoint(checkpoint_dir, step=0, model_data=model.state_dict(), optimizer_data=None,
                     meta_data={"step": 0, "model_config": model.config.to_dict()})

    adapters, frozen = expand_adapters(model, {"name": "sft0", "r": 4, "alpha": 8, "targets": ["attn.c_proj"], "layers": [0]})
    adapted_config = dataclasses.replace(model.config, adapters=adapters, frozen=frozen)

    reloaded, tokenizer, meta = build_model(checkpoint_dir, step=0, device=torch.device("cpu"), phase="eval",
                                             config_override=adapted_config)
    targets = find_adapters(reloaded)
    assert len(targets) == 1
    fqn, module = targets[0]
    assert fqn == "body.blocks.0.attn.c_proj"
    assert "sft0" in module.deltas

    idx = torch.randint(0, model.config.vocab_size, (1, 5))
    with torch.no_grad():
        original_out = model(idx)
        reloaded_out = reloaded(idx)
    assert torch.equal(original_out, reloaded_out), "a fresh (B=0) adapter must not change output"


def test_build_model_config_override_disable_via_hand_edit_round_trips(tmp_path, monkeypatch):
    """Hand-edit-config round trip at the nanochat entry point, mirroring modelcore's own
    test_peft.py::test_hand_edit_disable_then_add_adapter_round_trips_through_reload."""
    monkeypatch.setattr(checkpoint_manager, "get_tokenizer", lambda: _FakeTokenizer())

    base_model = _build_tiny("gpt")
    adapters, frozen = expand_adapters(base_model, {"name": "sft0", "r": 4, "alpha": 8, "targets": ["attn.c_proj"], "layers": [0]})
    trained_config = dataclasses.replace(base_model.config, adapters=adapters, frozen=frozen)
    trained_model = manager.create_model(trained_config, device=torch.device("cpu"), seed=0)
    _, module = find_adapters(trained_model)[0]
    with torch.no_grad():
        module.deltas["sft0"].lora_B.weight.normal_(std=0.1)  # simulate "trained"

    checkpoint_dir = str(tmp_path / "d_tiny")
    save_checkpoint(checkpoint_dir, step=0, model_data=trained_model.state_dict(), optimizer_data=None,
                     meta_data={"step": 0, "model_config": trained_model.config.to_dict()})

    idx = torch.randint(0, base_model.config.vocab_size, (1, 5))
    with torch.no_grad():
        trained_out = build_model(checkpoint_dir, step=0, device=torch.device("cpu"), phase="eval")[0](idx)

    disabled_config = dataclasses.replace(trained_config)
    disabled_config.adapters = [dataclasses.replace(a, enabled=False) for a in disabled_config.adapters]
    disabled_model, _, _ = build_model(checkpoint_dir, step=0, device=torch.device("cpu"), phase="eval",
                                        config_override=disabled_config)
    _, disabled_module = find_adapters(disabled_model)[0]
    assert disabled_module.enabled["sft0"] is False
    with torch.no_grad():
        disabled_out = disabled_model(idx)
    assert not torch.equal(disabled_out, trained_out)
