"""
Llama-specific model tests (nanochat/model/llama/). Architecture-agnostic structural tests live
in tests/test_model_common.py, parametrized over every registered architecture. This file covers
what's specific to Llama: its use of BaseModel's generic (non-GPT) num_scaling_params() default,
the structural absence of GPT's residual-topology extras, and SwiGLUMLP itself.

python -m pytest tests/test_model_llama.py -v
"""

import torch

from nanochat.model.param_roles import collect_param_roles
from nanochat.model.components.mlp import SwiGLUMLP


def test_num_scaling_params_uses_generic_role_based_shape(tiny_llama):
    """Llama has no num_scaling_params() override, so it inherits BaseModel's generic
    {role: numel, ..., total} default -- distinct from GPT's hand-rolled six-key legacy dict."""
    counts = tiny_llama.num_scaling_params()
    assert counts["total"] == sum(p.numel() for p in tiny_llama.parameters())
    assert "embedding" in counts and "unembedding" in counts and "matrix" in counts
    assert "wte" not in counts and "value_embeds" not in counts and "scalars" not in counts


def test_no_gpt_residual_topology_extras(tiny_llama):
    """Structural proof of the roadmap's "boring baseline" claim: no value embeddings, smear, or
    per-layer resid/x0 scalars -- unlike GPT, which has all four roles."""
    roles = collect_param_roles(tiny_llama)
    assert "value_embedding" not in roles
    assert "smear" not in roles
    assert "resid_scalar" not in roles
    assert "x0_scalar" not in roles


def test_swiglu_mlp_shape_and_finite():
    mlp = SwiGLUMLP(n_embd=32)
    mlp.init_weights()
    x = torch.randn(2, 5, 32)
    y = mlp(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
