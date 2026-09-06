"""
Architecture derivation rules: how a depth dial (--depth/--aspect-ratio/--head-dim) or a stored
flat legacy config becomes concrete per-layer values. Every rule here used to live inside
modelcore (nanochat.model.components.attention.has_ve, .windows.compute_window_sizes,
.kv_sharing.compute_kv_slots) or be duplicated across GPTConfig.from_depth /
LlamaConfig.from_depth / nanochat.model.composed.presets._mup_dims; modelcore itself now only
ever consumes already-materialized per-layer values (see docs/roadmap.md's Stage 7). This module
is the one place that actually knows these policies -- nanochat.architectures.presets and
nanochat.architectures.legacy both call into it, rather than each re-deriving its own copy.
"""


def mup_dims(depth: int, aspect_ratio: int, head_dim: int) -> tuple[int, int]:
    """The muP-style depth dial every architecture preset shares: n_embd grows with depth *
    aspect_ratio, rounded up to a multiple of head_dim; n_head = n_embd // head_dim. GQA (n_kv_head
    < n_head) is reachable only via an explicit per-block override, not through this dial."""
    base_dim = depth * aspect_ratio
    n_embd = ((base_dim + head_dim - 1) // head_dim) * head_dim
    n_head = n_embd // head_dim
    return n_embd, n_head


def compute_window_sizes(pattern: str, n_layer: int, sequence_len: int) -> list[int]:
    """Per-layer sliding-window size (FA3's "left window" convention: -1 = unlimited/full
    context, else the number of preceding tokens attended to). `pattern` is tiled across layers;
    characters: L=long (full context), S=short (quarter context, rounded up to FA3's 128-token
    tile size). The final layer always gets full context, regardless of pattern."""
    pattern = pattern.upper()
    assert all(c in "SL" for c in pattern), f"Invalid window_pattern: {pattern}. Use only S and L."
    long_window = sequence_len
    short_window = -(-long_window // 4 // 128) * 128  # ceil to FA3 tile size (2048 -> 768)
    char_to_window = {"L": long_window, "S": short_window}
    windows = [char_to_window[pattern[i % len(pattern)]] for i in range(n_layer)]
    windows[-1] = long_window  # final layer always gets full context
    return windows


def compute_kv_slots(n_layer: int, kv_share_frac: float) -> list[int]:
    """Cross-layer KV sharing (Gemma-3n style): the last kv_share_frac fraction of layers reuse
    the last KV-owning layer's slot instead of computing their own K/V. Returns a list of length
    n_layer mapping layer index -> KV cache slot. The distinct values are a contiguous 0..M-1
    range, where M = n_layer - round(n_layer * kv_share_frac) is the number of layers that
    actually own (produce) K/V."""
    assert 0.0 <= kv_share_frac < 1.0, f"kv_share_frac must be in [0, 1): {kv_share_frac}"
    n_shared = round(n_layer * kv_share_frac)
    n_own = n_layer - n_shared
    assert n_own >= 1, f"kv_share_frac={kv_share_frac} leaves no KV-owning layers for n_layer={n_layer}"
    return [min(i, n_own - 1) for i in range(n_layer)]


def has_value_embed(layer_idx: int, n_layer: int) -> bool:
    """GPT's value-embedding parity rule: alternating, with the last layer always included."""
    return layer_idx % 2 == (n_layer - 1) % 2


def gpt_lambda_schedule(layer_idx: int, n_layer: int) -> tuple[float, float]:
    """GPT's per-layer resid/x0-lambda init schedule: stronger residual & more input-embedding
    blending at early layers, decaying with depth (a muP-style choice, not something a single
    layer or component could derive on its own). Returns (resid_lambda_init, x0_lambda_init)."""
    resid_lambda_init = 1.15 - (0.10 * layer_idx / max(n_layer - 1, 1))
    x0_lambda_init = 0.20 - (0.15 * layer_idx / max(n_layer - 1, 1))
    return resid_lambda_init, x0_lambda_init
