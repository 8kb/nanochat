def compute_kv_slots(n_layer, kv_share_frac):
    """
    Cross-layer KV sharing (Gemma-3n style): the last kv_share_frac fraction of layers reuse the
    last KV-owning layer's slot instead of computing their own K/V.

    Returns a list of length n_layer mapping layer index -> KV cache slot. The distinct values
    are a contiguous 0..M-1 range, where M = n_layer - round(n_layer * kv_share_frac) is the
    number of layers that actually own (produce) K/V.
    """
    assert 0.0 <= kv_share_frac < 1.0, f"kv_share_frac must be in [0, 1): {kv_share_frac}"
    n_shared = round(n_layer * kv_share_frac)
    n_own = n_layer - n_shared
    assert n_own >= 1, f"kv_share_frac={kv_share_frac} leaves no KV-owning layers for n_layer={n_layer}"
    return [min(i, n_own - 1) for i in range(n_layer)]
