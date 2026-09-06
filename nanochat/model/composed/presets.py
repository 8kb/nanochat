"""
Compatibility/preset layer for composed architectures: expands today's --arch/--depth CLI dial
into a materialized ComposedConfig tree, reproducing gpt/llama/llama_kvshare/llama_kvshare_win's
own from_depth + __init__ derivation exactly (see docs/architecture.md's "Composed architectures").
The derivation rules (compute_window_sizes, compute_kv_slots, has_ve, GPT's per-layer resid/
x0-lambda schedule) run once, here, at expansion time -- the resulting tree has no rules left in
it, only concrete per-layer values a user can edit directly.
"""
import json
import os

from nanochat.model.components.windows import compute_window_sizes
from nanochat.model.components.kv_sharing import compute_kv_slots
from nanochat.model.components.attention import has_ve
from nanochat.model.composed.spec import ComponentSpec, ComposedConfig


def _mup_dims(depth, aspect_ratio, head_dim):
    """Same muP-style depth dial as GPTConfig.from_depth / LlamaConfig.from_depth. Returns
    (n_embd, n_head, n_kv_head) -- n_kv_head == n_head, same as the native from_depth methods
    (GQA is reachable only via an explicit per-block n_kv_head override, same as --arch-opt today)."""
    base_dim = depth * aspect_ratio
    n_embd = ((base_dim + head_dim - 1) // head_dim) * head_dim
    n_head = n_embd // head_dim
    return n_embd, n_head, n_head


def expand_gpt(depth, aspect_ratio=64, head_dim=128, max_seq_len=2048, vocab_size=32768, window_pattern="SSSL"):
    """Materializes exactly what nanochat.model.gpt.model.GPT.__init__ builds from GPTConfig."""
    n_embd, n_head, n_kv_head = _mup_dims(depth, aspect_ratio, head_dim)
    n_layer = depth
    window_sizes = compute_window_sizes(window_pattern, n_layer, max_seq_len)
    blocks = []
    for i in range(n_layer):
        window, _ = window_sizes[i]
        # Per-layer resid/x0 init schedule: stronger residual & more input blending at early
        # layers -- see nanochat.model.gpt.model.GPT.__init__ for the same formula un-materialized.
        resid_lambda_init = 1.15 - (0.10 * i / max(n_layer - 1, 1))
        x0_lambda_init = 0.20 - (0.15 * i / max(n_layer - 1, 1))
        blocks.append(ComponentSpec("gpt_block", {
            "layer_idx": i, "n_head": n_head, "n_kv_head": n_kv_head, "window": window,
            "has_value_embed": has_ve(i, n_layer),
            "resid_lambda_init": resid_lambda_init, "x0_lambda_init": x0_lambda_init,
        }))
    return ComposedConfig(
        sequence_len=max_seq_len, vocab_size=vocab_size, n_embd=n_embd,
        reference={"preset": "gpt", "kwargs": dict(aspect_ratio=aspect_ratio, head_dim=head_dim,
                                                    max_seq_len=max_seq_len, vocab_size=vocab_size,
                                                    window_pattern=window_pattern)},
        shared={"rope": ComponentSpec("rotary", {"head_dim": head_dim})},
        input=ComponentSpec("token_embedding", {"smear": True}),
        body=ComponentSpec("backout", {"backout_layer": n_layer // 2, "backout_lambda_init": 0.2, "blocks": blocks}),
        output=ComponentSpec("lm_head", {"softcap": 15}),
    )


def _expand_llama_like(preset_name, depth, aspect_ratio, head_dim, max_seq_len, vocab_size, window_pattern,
                        kv_share_frac=None):
    """Shared derivation for llama / llama_kvshare / llama_kvshare_win: all three build the same
    plain_block stack, differing only in whether cross-layer KV sharing is materialized."""
    n_embd, n_head, n_kv_head = _mup_dims(depth, aspect_ratio, head_dim)
    n_layer = depth
    window_sizes = compute_window_sizes(window_pattern, n_layer, max_seq_len)
    kv_slots = compute_kv_slots(n_layer, kv_share_frac) if kv_share_frac is not None else None
    blocks = []
    for i in range(n_layer):
        window, _ = window_sizes[i]
        params = {"layer_idx": i, "n_head": n_head, "n_kv_head": n_kv_head, "window": window}
        if kv_slots is not None:
            params["kv_slot"] = kv_slots[i]
            params["produces_kv"] = (kv_slots[i] == i)
        blocks.append(ComponentSpec("plain_block", params))
    reference_kwargs = dict(aspect_ratio=aspect_ratio, head_dim=head_dim, max_seq_len=max_seq_len,
                             vocab_size=vocab_size, window_pattern=window_pattern)
    if kv_share_frac is not None:
        reference_kwargs["kv_share_frac"] = kv_share_frac
    return ComposedConfig(
        sequence_len=max_seq_len, vocab_size=vocab_size, n_embd=n_embd,
        reference={"preset": preset_name, "kwargs": reference_kwargs},
        shared={"rope": ComponentSpec("rotary", {"head_dim": head_dim})},
        input=ComponentSpec("token_embedding", {"smear": False}),
        body=ComponentSpec("stack", {"blocks": blocks}),
        output=ComponentSpec("lm_head", {"softcap": 15}),
    )


def expand_llama(depth, aspect_ratio=64, head_dim=128, max_seq_len=2048, vocab_size=32768, window_pattern="L"):
    return _expand_llama_like("llama", depth, aspect_ratio, head_dim, max_seq_len, vocab_size, window_pattern)


def expand_llama_kvshare(depth, aspect_ratio=64, head_dim=128, max_seq_len=2048, vocab_size=32768,
                          window_pattern="L", kv_share_frac=0.5):
    return _expand_llama_like("llama_kvshare", depth, aspect_ratio, head_dim, max_seq_len, vocab_size,
                               window_pattern, kv_share_frac=kv_share_frac)


def expand_llama_kvshare_win(depth, aspect_ratio=64, head_dim=128, max_seq_len=2048, vocab_size=32768,
                              window_pattern="SSSL", kv_share_frac=0.5):
    return _expand_llama_like("llama_kvshare_win", depth, aspect_ratio, head_dim, max_seq_len, vocab_size,
                               window_pattern, kv_share_frac=kv_share_frac)


PRESETS = {
    "gpt": expand_gpt, "llama": expand_llama,
    "llama_kvshare": expand_llama_kvshare, "llama_kvshare_win": expand_llama_kvshare_win,
}


def expand_preset(name, depth, **kwargs) -> ComposedConfig:
    if name not in PRESETS:
        raise ValueError(f"Unknown composed preset {name!r}. Registered: {sorted(PRESETS)}")
    return PRESETS[name](depth, **kwargs)


def resolve_composed_config(model_config, depth, *, aspect_ratio, head_dim, max_seq_len, vocab_size,
                             window_pattern=None, arch_opts=None) -> ComposedConfig:
    """Resolve --model-config (a preset name registered in PRESETS, or a path to a materialized
    JSON tree) into the ComposedConfig to actually build at `depth`. See
    resolve_composed_reference_config for the muP scaling-law reference model (a different depth,
    same underlying preset)."""
    if os.path.isfile(model_config):
        assert not arch_opts, "--arch-opt is not supported with a --model-config JSON file; edit the tree directly"
        with open(model_config, "r", encoding="utf-8") as f:
            return ComposedConfig.from_dict(json.load(f))
    kwargs = dict(aspect_ratio=aspect_ratio, head_dim=head_dim, max_seq_len=max_seq_len, vocab_size=vocab_size)
    if window_pattern is not None:
        kwargs["window_pattern"] = window_pattern
    return expand_preset(model_config, depth, **kwargs, **(arch_opts or {}))


def resolve_composed_reference_config(resolved_config, ref_depth) -> ComposedConfig:
    """The muP scaling-law reference model (nanochat.scaling.derive_training_plan's d_ref) at
    ref_depth (12), for a config that resolve_composed_config already resolved (from a preset or a
    JSON file) to `resolved_config`. expand_preset always stamps a `reference` block on its output,
    so this works uniformly regardless of which source resolve_composed_config used -- a JSON file
    with no `reference` block (e.g. one written by hand rather than dumped from a preset) is the
    only case this can't handle; the caller should fall back to --d-ref-scaling-params instead."""
    assert resolved_config.reference is not None, (
        f"config has no 'reference' block, so its muP scaling-law reference model can't be "
        f"re-derived automatically at depth {ref_depth}; pass --d-ref-scaling-params instead"
    )
    return expand_preset(resolved_config.reference["preset"], ref_depth, **resolved_config.reference["kwargs"])
