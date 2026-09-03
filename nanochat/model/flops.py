"""
Generic FLOPs / parameter / KV-cache-bytes accounting, expressed purely in terms of
nanochat.model.base.AttentionLayerSpec so any architecture that implements layer_specs() gets
these estimates for free (via nanochat.model.base.BaseModel's wrapper methods).

Ref: https://medium.com/@dzmitrybahdanau/the-flops-calculus-of-language-model-training-3b19c1f025e4
Ref: https://arxiv.org/abs/2204.02311 (PaLM paper)
Ref: https://arxiv.org/abs/2203.15556 (Chinchilla paper)
"""

from nanochat.model.components.linear import Linear


def num_matmul_params(model):
    """
    The number of parameters that participate in matmuls with the token stream,
    i.e. contribute 2 FLOPs/param to the forward pass. Counted structurally: every
    matmul in a nanochat.model architecture goes through the Linear class, while
    non-matmul params (embeddings = lookups, per-layer scalars) are nn.Embedding or raw
    Parameters.
    """
    return sum(m.weight.numel() for m in model.modules() if isinstance(m, Linear))


def _effective_window(window, cap):
    """window=-1 means unlimited/full context, capped at `cap` (sequence_len or context_len)."""
    return cap if window < 0 else min(window, cap)


def estimate_flops(layer_specs, num_matmul_params, sequence_len):
    """
    Return the estimated FLOPs per token for the model (forward + backward).
    Each matmul weight parameter contributes 2 FLOPs (multiply *, accumulate +) in forward, and 2X that in backward => 2+4=6.
    On top of that, 12 * h * q * effective_seq_len accounts for key @ query matmul flops inside attention.
    With sliding windows, effective_seq_len varies per layer (capped by window size).
    This is ~1% off from the exact formulas of Chinchilla paper, the difference is:
    - Chinchilla counts the embedding layer as flops (? weird, it's just a lookup => we ignore)
    - Chinchilla counts exp/sum/divide in attention softmax as flops (a little sus and very tiny => we ignore)
    """
    attn_flops = 0
    for spec in layer_specs:
        effective_seq = _effective_window(spec.window, sequence_len)
        attn_flops += 12 * spec.n_head * spec.head_dim * effective_seq
    return 6 * num_matmul_params + attn_flops


def estimate_decode_flops(layer_specs, num_matmul_params, context_len):
    """
    Forward FLOPs to decode one token at a given context length during inference:
    2 FLOPs per matmul param, plus attention over min(context, window) per layer.
    """
    attn_flops = 0
    for spec in layer_specs:
        w = _effective_window(spec.window, context_len)
        attn_flops += 4 * spec.n_head * spec.head_dim * w
    return 2 * num_matmul_params + attn_flops


def estimate_prefill_flops(layer_specs, num_matmul_params, num_tokens):
    """Forward FLOPs to prefill a prompt: causal, so token t attends to min(t, window)."""
    attn_flops = 0
    for spec in layer_specs:
        w = _effective_window(spec.window, num_tokens)
        attended_tokens = w * (w + 1) // 2 + (num_tokens - w) * w # ramp up to w, then flat
        attn_flops += 4 * spec.n_head * spec.head_dim * attended_tokens
    return 2 * num_matmul_params * num_tokens + attn_flops


def distinct_kv_specs(layer_specs):
    """One spec per distinct KV cache slot (see AttentionLayerSpec.kv_slot), rather than one per
    layer -- layers that share a slot (cross-layer KV sharing) must not be double-counted when
    accounting for what's actually *stored*. Layer order is preserved; a layer with kv_slot=None
    is its own slot at its own position, matching BaseModel.kv_cache_spec()'s convention."""
    seen = set()
    distinct = []
    for i, spec in enumerate(layer_specs):
        slot = i if spec.kv_slot is None else spec.kv_slot
        if slot not in seen:
            seen.add(slot)
            distinct.append(spec)
    return distinct


def kv_bytes_per_token(layer_specs, dtype_itemsize):
    """Bytes to *store* one token of KV cache during inference, per row -- one contribution per
    distinct KV slot, not per layer, so cross-layer KV sharing correctly shows a smaller footprint
    than one-slot-per-layer at the same layer count."""
    return sum(2 * spec.n_kv_head * spec.head_dim * dtype_itemsize for spec in distinct_kv_specs(layer_specs))


def kv_read_bytes(layer_specs, dtype_itemsize, context_len):
    """Bytes of KV cache *read* by one decode step at a given context length, per row. Summed per
    layer (not per slot): a layer that reuses another layer's stored K/V still issues its own read
    of that cache during its own attention call. Sliding window layers only attend to (and read)
    the last `window` tokens."""
    total = 0
    for spec in layer_specs:
        w = _effective_window(spec.window, context_len)
        total += 2 * spec.n_kv_head * spec.head_dim * dtype_itemsize * w
    return total
