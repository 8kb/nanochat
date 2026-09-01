"""
Test the reusable building blocks under nanochat/model/components/. CPU-only, no real model
needed -- these operate directly on small tensors.

python -m pytest tests/test_model_components.py -v
"""

import torch
import pytest

from nanochat.model.components.rope import apply_rotary_emb, precompute_rotary_embeddings
from nanochat.model.components.norm import norm
from nanochat.model.components.linear import Linear
from nanochat.model.components.windows import compute_window_sizes


# -----------------------------------------------------------------------------
# rope

def test_apply_rotary_emb_preserves_norm():
    """Rotation is norm-preserving per (x1, x2) pair, so the full head vector's norm is unchanged."""
    torch.manual_seed(0)
    head_dim, seq_len = 8, 16
    cos, sin = precompute_rotary_embeddings(seq_len, head_dim, device="cpu", dtype=torch.float32)
    x = torch.randn(1, seq_len, 2, head_dim)  # (B, T, H, D)
    y = apply_rotary_emb(x, cos, sin)
    assert torch.allclose(x.norm(dim=-1), y.norm(dim=-1), atol=1e-5)


def test_apply_rotary_emb_relative_position_invariance():
    """q . k after RoPE depends only on the relative offset (j - i), not on absolute positions."""
    torch.manual_seed(0)
    head_dim, seq_len = 8, 32
    cos, sin = precompute_rotary_embeddings(seq_len, head_dim, device="cpu", dtype=torch.float32)
    q_vec = torch.randn(1, 1, 1, head_dim)
    k_vec = torch.randn(1, 1, 1, head_dim)

    def dot_at(i, j):
        q_rot = apply_rotary_emb(q_vec, cos[:, i:i + 1], sin[:, i:i + 1])
        k_rot = apply_rotary_emb(k_vec, cos[:, j:j + 1], sin[:, j:j + 1])
        return (q_rot * k_rot).sum().item()

    same_offset_a = dot_at(5, 2)    # offset 3
    same_offset_b = dot_at(20, 17)  # offset 3
    different_offset = dot_at(20, 10)  # offset 10
    assert abs(same_offset_a - same_offset_b) < 1e-4
    assert abs(same_offset_a - different_offset) > 1e-4


# -----------------------------------------------------------------------------
# norm

def test_norm_gives_unit_rms():
    x = torch.randn(4, 8, 16) * 5.0 + 3.0
    y = norm(x)
    rms = y.pow(2).mean(dim=-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-4)


# -----------------------------------------------------------------------------
# Linear

def test_linear_casts_activations_but_keeps_fp32_master_weight():
    lin = Linear(4, 8, bias=False)
    assert lin.weight.dtype == torch.float32
    x64 = torch.randn(2, 4, dtype=torch.float64)
    y = lin(x64)
    assert y.dtype == torch.float64  # matmul ran in the activation dtype
    assert lin.weight.dtype == torch.float32  # master weight untouched


# -----------------------------------------------------------------------------
# windows

def test_compute_window_sizes_full_context():
    ws = compute_window_sizes("L", n_layer=4, sequence_len=512)
    assert ws == [(512, 0)] * 4


def test_compute_window_sizes_last_layer_always_full_context():
    # Every layer requests a short window, but the final layer is always forced to full context.
    ws = compute_window_sizes("SS", n_layer=3, sequence_len=1024)
    assert ws[0][0] < 1024 and ws[1][0] < 1024
    assert ws[-1] == (1024, 0)


def test_compute_window_sizes_tiles_pattern_across_layers():
    ws = compute_window_sizes("SL", n_layer=4, sequence_len=2048)
    assert ws[0][0] < 2048  # S
    assert ws[1] == (2048, 0)  # L
    assert ws[2][0] < 2048  # S (pattern repeats)
    assert ws[3] == (2048, 0)  # L, also forced as the last layer


def test_compute_window_sizes_invalid_chars_assert():
    with pytest.raises(AssertionError):
        compute_window_sizes("X", n_layer=2, sequence_len=128)
