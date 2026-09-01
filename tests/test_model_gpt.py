"""
Test the GPT architecture (nanochat/model/gpt/). Uses the tiny_gpt fixture from conftest.py --
CPU-only, no checkpoint or tokenizer needed.

python -m pytest tests/test_model_gpt.py -v
"""

import torch

from nanochat.model.components.linear import Linear


def test_forward_logits_shape_and_finite(tiny_gpt):
    B, T = 2, 5
    idx = torch.randint(0, tiny_gpt.config.vocab_size, (B, T))
    logits = tiny_gpt.forward(idx)
    assert logits.shape == (B, T, tiny_gpt.config.vocab_size)
    assert torch.isfinite(logits).all()


def test_forward_loss_is_finite(tiny_gpt):
    B, T = 2, 5
    idx = torch.randint(0, tiny_gpt.config.vocab_size, (B, T))
    targets = torch.randint(0, tiny_gpt.config.vocab_size, (B, T))
    loss = tiny_gpt.forward(idx, targets=targets)
    assert loss.dim() == 0
    assert torch.isfinite(loss)


def test_backward_populates_every_parameter_gradient(tiny_gpt):
    tiny_gpt.train()
    B, T = 2, 5
    idx = torch.randint(0, tiny_gpt.config.vocab_size, (B, T))
    targets = torch.randint(0, tiny_gpt.config.vocab_size, (B, T))
    loss = tiny_gpt.forward(idx, targets=targets)
    loss.backward()
    missing = [name for name, p in tiny_gpt.named_parameters() if p.grad is None]
    assert not missing, f"parameters with no gradient: {missing}"


def test_num_scaling_params_totals_match_parameter_count(tiny_gpt):
    counts = tiny_gpt.num_scaling_params()
    assert counts["total"] == sum(p.numel() for p in tiny_gpt.parameters())
    group_sum = counts["wte"] + counts["value_embeds"] + counts["lm_head"] + counts["transformer_matrices"] + counts["scalars"]
    assert group_sum == counts["total"]


def test_num_matmul_params_matches_manual_sum(tiny_gpt):
    manual = sum(m.weight.numel() for m in tiny_gpt.modules() if isinstance(m, Linear))
    assert tiny_gpt.num_matmul_params() == manual
    assert tiny_gpt.num_matmul_params() > 0


def test_setup_optimizer_param_groups_partition_all_parameters_exactly(tiny_gpt):
    """setup_optimizer asserts internally that every parameter appears in exactly one group;
    this test also checks the reverse (no group parameter is missing from the model / duplicated
    across groups), independent of that internal assert."""
    optimizer = tiny_gpt.setup_optimizer()
    seen_ids = []
    for group in optimizer.param_groups:
        seen_ids.extend(id(p) for p in group["params"])
    model_ids = {id(p) for p in tiny_gpt.parameters()}
    assert len(seen_ids) == len(set(seen_ids)), "a parameter appears in more than one optimizer group"
    assert set(seen_ids) == model_ids, "optimizer groups do not exactly cover the model's parameters"


def test_layer_specs_length_and_kv_cache_spec(tiny_gpt):
    specs = tiny_gpt.layer_specs()
    assert len(specs) == tiny_gpt.config.n_layer
    for spec in specs:
        assert spec.n_head == tiny_gpt.config.n_head
        assert spec.n_kv_head == tiny_gpt.config.n_kv_head
        assert spec.head_dim == tiny_gpt.config.n_embd // tiny_gpt.config.n_head
    cache_spec = tiny_gpt.kv_cache_spec()
    assert cache_spec == {
        "num_heads": tiny_gpt.config.n_kv_head,
        "head_dim": tiny_gpt.config.n_embd // tiny_gpt.config.n_head,
        "num_layers": tiny_gpt.config.n_layer,
    }
