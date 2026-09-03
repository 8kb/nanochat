"""
Architecture-agnostic model tests, parametrized over every registered architecture via the
tiny_model fixture (tests/conftest.py's TINY_KWARGS_BY_ARCH). Passing for every architecture is
the actual proof that nanochat.model.base's contracts (BaseModel, BaseEmbedding, BaseBlock,
BaseUnembedding, the parameter-role protocol) are real interfaces, not just GPT with extra
indirection. Architecture-specific behavior lives in tests/test_model_<arch>.py.

python -m pytest tests/test_model_common.py -v
"""

import torch

from nanochat.model.components.linear import Linear


def test_forward_logits_shape_and_finite(tiny_model):
    B, T = 2, 5
    idx = torch.randint(0, tiny_model.config.vocab_size, (B, T))
    logits = tiny_model.forward(idx)
    assert logits.shape == (B, T, tiny_model.config.vocab_size)
    assert torch.isfinite(logits).all()


def test_forward_loss_is_finite(tiny_model):
    B, T = 2, 5
    idx = torch.randint(0, tiny_model.config.vocab_size, (B, T))
    targets = torch.randint(0, tiny_model.config.vocab_size, (B, T))
    loss = tiny_model.forward(idx, targets=targets)
    assert loss.dim() == 0
    assert torch.isfinite(loss)


def test_backward_populates_every_parameter_gradient(tiny_model):
    tiny_model.train()
    B, T = 2, 5
    idx = torch.randint(0, tiny_model.config.vocab_size, (B, T))
    targets = torch.randint(0, tiny_model.config.vocab_size, (B, T))
    loss = tiny_model.forward(idx, targets=targets)
    loss.backward()
    missing = [name for name, p in tiny_model.named_parameters() if p.grad is None]
    assert not missing, f"parameters with no gradient: {missing}"


def test_num_scaling_params_totals_match_parameter_count(tiny_model):
    """Architecture-agnostic version of the total/group-sum check: doesn't assume any particular
    set of dict keys (GPT's legacy 6-key dict and BaseModel's generic role-keyed default use
    different key sets -- see nanochat/model/base.py)."""
    counts = tiny_model.num_scaling_params()
    assert counts["total"] == sum(p.numel() for p in tiny_model.parameters())
    group_sum = sum(v for k, v in counts.items() if k != "total")
    assert group_sum == counts["total"]


def test_num_matmul_params_matches_manual_sum(tiny_model):
    manual = sum(m.weight.numel() for m in tiny_model.modules() if isinstance(m, Linear))
    assert tiny_model.num_matmul_params() == manual
    assert tiny_model.num_matmul_params() > 0


def test_setup_optimizer_param_groups_partition_all_parameters_exactly(tiny_model):
    """setup_optimizer (via collect_param_roles) asserts internally that every parameter appears
    in exactly one role; this test also checks the reverse (no group parameter is missing from
    the model / duplicated across groups), independent of that internal assert."""
    optimizer = tiny_model.setup_optimizer()
    seen_ids = []
    for group in optimizer.param_groups:
        seen_ids.extend(id(p) for p in group["params"])
    model_ids = {id(p) for p in tiny_model.parameters()}
    assert len(seen_ids) == len(set(seen_ids)), "a parameter appears in more than one optimizer group"
    assert set(seen_ids) == model_ids, "optimizer groups do not exactly cover the model's parameters"


def test_layer_specs_and_kv_cache_spec_are_consistent(tiny_model):
    """num_kv_slots can be <= len(specs): an architecture with cross-layer KV sharing (see
    nanochat.model.llama_kvshare) allocates fewer KVCache slots than it has layers, with each
    layer's kv_slot pointing into a contiguous 0..M-1 range."""
    specs = tiny_model.layer_specs()
    assert len(specs) == tiny_model.config.n_layer
    n_kv_heads = {s.n_kv_head for s in specs}
    head_dims = {s.head_dim for s in specs}
    assert len(n_kv_heads) == 1 and len(head_dims) == 1
    slots = {i if s.kv_slot is None else s.kv_slot for i, s in enumerate(specs)}
    assert slots == set(range(len(slots))), "kv slots must be a contiguous 0..M-1 range"
    cache_spec = tiny_model.kv_cache_spec()
    assert cache_spec == {
        "num_heads": specs[0].n_kv_head,
        "head_dim": specs[0].head_dim,
        "num_kv_slots": len(slots),
    }
    assert cache_spec["num_kv_slots"] <= len(specs)
