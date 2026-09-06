"""
Tests for modelcore/ -- the standalone model subsystem (Stage 7, see docs/roadmap.md). Fully
self-contained: builds config trees directly with ComponentSpec/ModelConfig, with no dependency
on nanochat.model or nanochat.architectures (those exist to *produce* such a tree from a depth
dial or an old checkpoint; modelcore only ever consumes the materialized result).

python -m pytest tests/test_modelcore.py -v
"""
import json
import os

import pytest
import torch

from modelcore import ComponentSpec, ModelConfig, ModelManager, OptimizerHparams
from modelcore.components.linear import Linear
from modelcore.roles import collect_param_roles

GOLDENS_DIR = os.path.join(os.path.dirname(__file__), "goldens")
TINY_DIR = os.path.join(GOLDENS_DIR, "tiny")


def _gpt_like(n_layer=4, n_head=2, n_kv_head=2, n_embd=64, head_dim=32, vocab_size=128, sequence_len=32, window=-1):
    blocks = [
        ComponentSpec("gpt_block", {
            "layer_idx": i, "n_head": n_head, "n_kv_head": n_kv_head, "window": window,
            "has_value_embed": (i % 2 == (n_layer - 1) % 2),
            "resid_lambda_init": 1.15 - 0.10 * i / max(n_layer - 1, 1),
            "x0_lambda_init": 0.20 - 0.15 * i / max(n_layer - 1, 1),
        })
        for i in range(n_layer)
    ]
    return ModelConfig(
        sequence_len=sequence_len, vocab_size=vocab_size, n_embd=n_embd,
        shared={"rope": ComponentSpec("rotary", {"head_dim": head_dim})},
        input=ComponentSpec("token_embedding", {"smear": True}),
        body=ComponentSpec("backout", {"backout_layer": n_layer // 2, "backout_lambda_init": 0.2, "blocks": blocks}),
        output=ComponentSpec("lm_head", {"softcap": 15}),
    )


def _plain_like(n_layer=4, n_head=2, n_kv_head=2, n_embd=64, head_dim=32, vocab_size=128, sequence_len=32,
                 window=-1, kv_slots=None):
    blocks = []
    for i in range(n_layer):
        params = {"layer_idx": i, "n_head": n_head, "n_kv_head": n_kv_head, "window": window}
        if kv_slots is not None:
            params["kv_slot"] = kv_slots[i]
            params["produces_kv"] = kv_slots[i] == i
        blocks.append(ComponentSpec("plain_block", params))
    return ModelConfig(
        sequence_len=sequence_len, vocab_size=vocab_size, n_embd=n_embd,
        shared={"rope": ComponentSpec("rotary", {"head_dim": head_dim})},
        input=ComponentSpec("token_embedding", {"smear": False}),
        body=ComponentSpec("stack", {"blocks": blocks}),
        output=ComponentSpec("lm_head", {"softcap": 15}),
    )


FLAVORS = {
    "gpt": lambda: _gpt_like(),
    "llama": lambda: _plain_like(),
    "llama_kvshare": lambda: _plain_like(kv_slots=[0, 1, 1, 1]),
    "llama_kvshare_win": lambda: _plain_like(window=8, kv_slots=[0, 1, 1, 1]),
}


@pytest.fixture(params=list(FLAVORS))
def config(request):
    return FLAVORS[request.param]()


@pytest.fixture
def manager():
    return ModelManager()


def build(manager, config, seed=0):
    return manager.create_model(config, device=torch.device("cpu"), seed=seed)


# -----------------------------------------------------------------------------
# Generic model behavior, parametrized over every flavor (mirrors tests/test_model_common.py)

def test_validate_config_accepts_every_flavor(manager, config):
    report = manager.validate_config(config)
    assert report.ok, report.errors


def test_forward_shape_and_finite(manager, config):
    model = build(manager, config)
    idx = torch.randint(0, config.vocab_size, (2, 8))
    logits = model(idx)
    assert logits.shape == (2, 8, config.vocab_size)
    assert torch.isfinite(logits).all()


def test_loss_is_finite_scalar(manager, config):
    model = build(manager, config)
    idx = torch.randint(0, config.vocab_size, (2, 8))
    targets = torch.randint(0, config.vocab_size, (2, 8))
    loss = model(idx, targets=targets)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_backward_populates_every_gradient(manager, config):
    model = build(manager, config)
    idx = torch.randint(0, config.vocab_size, (2, 8))
    targets = torch.randint(0, config.vocab_size, (2, 8))
    loss = model(idx, targets=targets)
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"{name} got no gradient"
        assert torch.isfinite(p.grad).all(), f"{name} has a non-finite gradient"


def test_params_by_role_partition_matches_total(manager, config):
    stats = manager.stats(config)
    assert sum(stats.params_by_role.values()) == stats.num_params
    model = build(manager, config)
    assert stats.num_params == sum(p.numel() for p in model.parameters())


def test_num_matmul_params_matches_manual_scan(manager, config):
    model = build(manager, config)
    stats = manager.stats(config)
    manual = sum(m.weight.numel() for m in model.modules() if isinstance(m, Linear))
    assert stats.num_matmul_params == manual


def test_optimizer_groups_partition_parameters_exactly(manager, config):
    model = build(manager, config)
    optimizer = manager.create_optimizer(model, OptimizerHparams())
    seen = []
    for group in optimizer.param_groups:
        seen.extend(group["params"])
    assert len(seen) == len(set(id(p) for p in seen)), "a parameter appeared in more than one group"
    assert {id(p) for p in seen} == {id(p) for p in model.parameters()}


def test_layer_specs_and_kv_cache_spec_are_consistent(manager, config):
    stats = manager.stats(config)
    assert len(stats.layer_specs) == config.n_layer
    n_kv_heads = {s.n_kv_head for s in stats.layer_specs}
    head_dims = {s.head_dim for s in stats.layer_specs}
    assert len(n_kv_heads) == 1 and len(head_dims) == 1
    slots = {i if s.kv_slot is None else s.kv_slot for i, s in enumerate(stats.layer_specs)}
    assert slots == set(range(len(slots)))
    assert stats.kv_cache_spec["num_kv_slots"] == len(slots)
    assert stats.kv_cache_spec["num_kv_slots"] <= len(stats.layer_specs)


# -----------------------------------------------------------------------------
# ModelManager: create/load/save, seeding, validation

def test_create_model_seed_is_reproducible(manager, config):
    a = build(manager, config, seed=123)
    b = build(manager, config, seed=123)
    for (na, pa), (nb, pb) in zip(a.named_parameters(), b.named_parameters()):
        assert na == nb
        assert torch.equal(pa, pb), f"{na} differs across identically-seeded builds"


def test_save_load_round_trip_preserves_weights_and_forward(manager, config, tmp_path):
    from modelcore.store import FileSystemStore

    model = build(manager, config, seed=0)
    store = FileSystemStore(str(tmp_path / "ckpt"), step=0)
    manager.save_model(model, store)

    reloaded = manager.load_model(store, device=torch.device("cpu"))
    assert reloaded.config == model.config
    original_state = model.state_dict()
    reloaded_state = reloaded.state_dict()
    assert original_state.keys() == reloaded_state.keys()
    for key in original_state:
        assert torch.equal(original_state[key], reloaded_state[key]), f"mismatch in {key}"

    idx = torch.randint(0, config.vocab_size, (1, 5))
    with torch.no_grad():
        assert torch.equal(model(idx), reloaded(idx))


def test_optimizer_save_load_round_trip(manager, config, tmp_path):
    from modelcore.store import FileSystemStore

    model = build(manager, config, seed=0)
    optimizer = manager.create_optimizer(model, OptimizerHparams())
    store = FileSystemStore(str(tmp_path / "ckpt"), step=0)
    manager.save_optimizer(optimizer, store, rank=0)

    reloaded_optimizer = manager.load_optimizer(model, store, rank=0)
    assert reloaded_optimizer is not None
    assert len(reloaded_optimizer.param_groups) == len(optimizer.param_groups)


def test_load_optimizer_returns_none_without_a_saved_shard(manager, config, tmp_path):
    from modelcore.store import FileSystemStore

    model = build(manager, config, seed=0)
    store = FileSystemStore(str(tmp_path / "ckpt"), step=0)
    manager.save_model(model, store)  # no save_optimizer call
    assert manager.load_optimizer(model, store, rank=0) is None


def test_validate_config_reports_every_error_not_just_the_first(manager):
    bad_blocks = [
        ComponentSpec("gpt_block", {
            "layer_idx": 0, "n_head": 3, "n_kv_head": 2, "window": -5, "has_value_embed": False,
            "resid_lambda_init": 1.0, "x0_lambda_init": 0.0,
        }),
    ]
    config = ModelConfig(
        sequence_len=32, vocab_size=128, n_embd=64,
        shared={"rope": ComponentSpec("rotary", {"head_dim": 32})},
        input=ComponentSpec("token_embedding", {"smear": True}),
        body=ComponentSpec("backout", {"backout_layer": 0, "blocks": bad_blocks}),
        output=ComponentSpec("lm_head", {"softcap": 15}),
    )
    report = manager.validate_config(config)
    assert not report.ok
    messages = " ".join(str(e) for e in report.errors)
    assert "divisible by n_head" in messages
    assert "window" in messages
    assert len(report.errors) >= 2, "expected multiple independent errors, not just the first"


def test_validate_config_reports_unknown_component_type(manager):
    config = ModelConfig(
        sequence_len=32, vocab_size=128, n_embd=64,
        shared={}, input=ComponentSpec("nonexistent_embedding", {}),
        body=ComponentSpec("stack", {"blocks": []}),
        output=ComponentSpec("lm_head", {"softcap": 15}),
    )
    report = manager.validate_config(config)
    assert not report.ok
    assert any("unknown component type" in e.message for e in report.errors)


def test_validate_config_reports_missing_input_body_output(manager):
    config = ModelConfig(sequence_len=32, vocab_size=128, n_embd=64)
    report = manager.validate_config(config)
    assert not report.ok
    paths = {e.path for e in report.errors}
    assert {"input", "body", "output"} <= paths


def test_create_model_raises_on_invalid_config(manager):
    config = ModelConfig(sequence_len=32, vocab_size=128, n_embd=64)  # no input/body/output
    with pytest.raises(ValueError):
        manager.create_model(config, device=torch.device("cpu"))


def test_kv_sharing_consumer_without_kv_slot_is_reported(manager):
    blocks = [
        ComponentSpec("plain_block", {"layer_idx": 0, "n_head": 2, "n_kv_head": 2, "window": -1}),
        ComponentSpec("plain_block", {"layer_idx": 1, "n_head": 2, "n_kv_head": 2, "window": -1, "produces_kv": False}),
    ]
    config = ModelConfig(
        sequence_len=32, vocab_size=128, n_embd=64,
        shared={"rope": ComponentSpec("rotary", {"head_dim": 32})},
        input=ComponentSpec("token_embedding", {"smear": False}),
        body=ComponentSpec("stack", {"blocks": blocks}),
        output=ComponentSpec("lm_head", {"softcap": 15}),
    )
    report = manager.validate_config(config)
    assert not report.ok
    assert any("explicit kv_slot" in e.message for e in report.errors)


# -----------------------------------------------------------------------------
# Cross-check against the pre-refactor goldens (tests/goldens/tiny_composed_*): proves modelcore's
# rewritten components (has_ve removed, n_layer dropped from Block, runtime injected instead of a
# COMPUTE_DTYPE global, ...) reproduce the old nanochat.model.composed path bit-for-bit.

GOLDEN_PRESETS = ["gpt", "llama", "llama_kvshare", "llama_kvshare_win"]


@pytest.mark.parametrize("preset", GOLDEN_PRESETS)
def test_matches_pre_refactor_composed_golden(manager, preset):
    golden_name = f"tiny_composed_{preset}"
    with open(os.path.join(GOLDENS_DIR, f"{golden_name}.json"), encoding="utf-8") as f:
        golden = json.load(f)
    config = manager.config_from_dict(golden["meta_model_config"])
    state = torch.load(os.path.join(TINY_DIR, golden_name, "model_000000.pt"), map_location="cpu")

    with torch.device("meta"):
        from modelcore.model import Model
        model = Model(config)
    model.to_empty(device="cpu")
    model.init_weights()
    model.load_state_dict(state, strict=True, assign=True)
    model.eval()

    stats = manager.stats(config)
    assert stats.num_params == golden["accounting"]["num_scaling_params"]["total"]
    assert stats.num_matmul_params == golden["accounting"]["num_matmul_params"]
    assert stats.flops_per_token == golden["accounting"]["estimate_flops"]
    assert stats.kv_bytes_per_token() == golden["accounting"]["kv_bytes_per_token"]
    assert stats.kv_cache_spec == golden["accounting"]["kv_cache_spec"]

    T = min(8, config.sequence_len)
    idx = (torch.arange(T) % config.vocab_size).long().unsqueeze(0)
    import hashlib
    with torch.no_grad():
        logits = model(idx)
    h = hashlib.sha256(logits.float().contiguous().numpy().tobytes()).hexdigest()
    assert h == golden["logits_hash"]
