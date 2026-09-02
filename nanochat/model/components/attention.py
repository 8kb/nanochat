import torch
import torch.nn as nn

from nanochat.common import COMPUTE_DTYPE
from nanochat.model.base import AttentionLayerSpec
from nanochat.model.components.linear import Linear
from nanochat.model.components.norm import norm
# Our custom Flash Attention module that automatically uses FA3 when compatible and SDPA fallback otherwise
from nanochat.flash_attention import flash_attn


def has_ve(layer_idx, n_layer):
    """Returns True if a layer should have a Value Embedding (alternating, last layer always included)."""
    return layer_idx % 2 == (n_layer - 1) % 2


class CausalSelfAttention(nn.Module):
    """GQA + RoPE + QK-norm + optional value-residual (ResFormer-style) + FA3/SDPA sliding-window
    attention. Owns its window, its (optional) value-embedding table, and its layer_spec(); takes
    explicit dims rather than a config object so it can be reused by an architecture whose config
    has different field names."""
    PARAM_ROLES = {"value_embed": "value_embedding"}

    def __init__(self, n_embd, n_head, n_kv_head, layer_idx, window, rope, padded_vocab_size, has_value_embed):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.n_embd = n_embd
        self.head_dim = n_embd // n_head
        assert n_embd % n_head == 0
        assert n_kv_head <= n_head and n_head % n_kv_head == 0
        self.window = window
        self.rope = rope
        self.c_q = Linear(n_embd, n_head * self.head_dim, bias=False)
        self.c_k = Linear(n_embd, n_kv_head * self.head_dim, bias=False)
        self.c_v = Linear(n_embd, n_kv_head * self.head_dim, bias=False)
        self.c_proj = Linear(n_embd, n_embd, bias=False)
        kv_dim = n_kv_head * self.head_dim
        self.value_embed = nn.Embedding(padded_vocab_size, kv_dim) if has_value_embed else None
        self.ve_gate_channels = 12
        self.ve_gate = Linear(self.ve_gate_channels, n_kv_head, bias=False) if has_value_embed else None

    @torch.no_grad()
    def init_weights(self):
        s = 3**0.5 * self.n_embd**-0.5  # sqrt(3) multiplier makes sure Uniform achieves the same std as Normal
        torch.nn.init.uniform_(self.c_q.weight, -s, s)  # weights use Uniform to avoid outliers
        torch.nn.init.uniform_(self.c_k.weight, -s, s)
        torch.nn.init.uniform_(self.c_v.weight, -s, s)
        torch.nn.init.zeros_(self.c_proj.weight)  # projections are zero
        if self.value_embed is not None:
            torch.nn.init.uniform_(self.value_embed.weight, -s, s)  # init like c_v: uniform with same std
            if COMPUTE_DTYPE != torch.float16:
                self.value_embed.to(dtype=COMPUTE_DTYPE)
        if self.ve_gate is not None:
            # Gate weights init with small positive values so gates start slightly above neutral
            torch.nn.init.uniform_(self.ve_gate.weight, 0.0, 0.02)

    def layer_spec(self):
        return AttentionLayerSpec(n_head=self.n_head, n_kv_head=self.n_kv_head, head_dim=self.head_dim, window=self.window)

    def forward(self, x, idx, kv_cache):
        B, T, C = x.size()

        # Project the input to get queries, keys, and values
        # Shape: (B, T, H, D) - FA3's native layout, no transpose needed!
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # Value residual (ResFormer): mix in value embedding with input-dependent gate per head
        if self.value_embed is not None:
            ve = self.value_embed(idx).to(x.dtype).view(B, T, self.n_kv_head, self.head_dim)
            gate = 3 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))  # (B, T, n_kv_head), range (0, 3)
            v = v + gate.unsqueeze(-1) * ve

        # Apply Rotary Embeddings to queries and keys to get relative positional encoding
        q, k = self.rope(q, k, kv_cache)
        q, k = norm(q), norm(k)  # QK norm
        q = q * 1.2  # sharper attention (split scale between Q and K), TODO think through better
        k = k * 1.2

        # Flash Attention (FA3 or SDPA fallback)
        # window_size is (left, right) tuple: (N, 0) for causal, (-1, 0) for full context
        window_size = (self.window, 0)
        if kv_cache is None:
            # Training: causal attention with optional sliding window
            y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)
        else:
            # Inference: use flash_attn_with_kvcache which handles cache management
            k_cache, v_cache = kv_cache.get_layer_cache(self.layer_idx)
            y = flash_attn.flash_attn_with_kvcache(
                q, k_cache, v_cache,
                k=k, v=v,
                cache_seqlens=kv_cache.cache_seqlens,
                causal=True,
                window_size=window_size,
            )
            # Advance position after last layer processes
            if self.layer_idx == kv_cache.n_layers - 1:
                kv_cache.advance(T)

        # Re-assemble the heads and project back to residual stream
        y = y.contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y
