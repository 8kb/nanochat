"""
Compatibility layer between the --arch/--depth CLI dial and modelcore's materialized ModelConfig
tree: expand(name, depth, ...) reproduces what each of nanochat's four original hand-written
architecture classes (gpt, llama, llama_kvshare, llama_kvshare_win -- since deleted, see
docs/upstream-sync.md) used to build directly, as a concrete tree instead of code. The derivation
rules themselves (mup_dims, compute_window_sizes, compute_kv_slots, has_value_embed,
gpt_lambda_schedule) live in nanochat.architectures.derive and run once, here, at expansion time
-- the resulting tree has no rules left in it, only concrete per-layer values.
"""
import json
import os

from modelcore import ComponentSpec, ModelConfig

from nanochat.architectures.derive import (
    compute_kv_slots, compute_window_sizes, gpt_lambda_schedule, has_value_embed, mup_dims,
)


def assemble_gpt(n_layer, n_head, n_kv_head, n_embd, head_dim, sequence_len, vocab_size, window_pattern,
                  reference=None) -> ModelConfig:
    """Builds the gpt-shaped tree from already-concrete dimensions -- shared by expand_gpt (which
    derives them from a depth dial) and nanochat.architectures.legacy (which reads them straight
    off an old checkpoint's stored fields, so a run that used --arch-opt still migrates exactly)."""
    windows = compute_window_sizes(window_pattern, n_layer, sequence_len)
    blocks = []
    for i in range(n_layer):
        resid_lambda_init, x0_lambda_init = gpt_lambda_schedule(i, n_layer)
        blocks.append(ComponentSpec("gpt_block", {
            "layer_idx": i, "n_head": n_head, "n_kv_head": n_kv_head, "window": windows[i],
            "has_value_embed": has_value_embed(i, n_layer),
            "resid_lambda_init": resid_lambda_init, "x0_lambda_init": x0_lambda_init,
        }))
    return ModelConfig(
        sequence_len=sequence_len, vocab_size=vocab_size, n_embd=n_embd, reference=reference,
        shared={"rope": ComponentSpec("rotary", {"head_dim": head_dim})},
        input=ComponentSpec("token_embedding", {"smear": True}),
        body=ComponentSpec("backout", {"backout_layer": n_layer // 2, "backout_lambda_init": 0.2, "blocks": blocks}),
        output=ComponentSpec("lm_head", {"softcap": 15}),
    )


def assemble_plain(n_layer, n_head, n_kv_head, n_embd, head_dim, sequence_len, vocab_size, window_pattern,
                    kv_share_frac=None, reference=None) -> ModelConfig:
    """Builds the llama-shaped (plain pre-norm stack) tree from already-concrete dimensions --
    shared by expand_llama(_kvshare(_win)) and nanochat.architectures.legacy. Covers llama,
    llama_kvshare, and llama_kvshare_win alike: all three build the same plain_block stack,
    differing only in whether cross-layer KV sharing is materialized (kv_share_frac)."""
    windows = compute_window_sizes(window_pattern, n_layer, sequence_len)
    kv_slots = compute_kv_slots(n_layer, kv_share_frac) if kv_share_frac is not None else None
    blocks = []
    for i in range(n_layer):
        params = {"layer_idx": i, "n_head": n_head, "n_kv_head": n_kv_head, "window": windows[i]}
        if kv_slots is not None:
            params["kv_slot"] = kv_slots[i]
            params["produces_kv"] = (kv_slots[i] == i)
        blocks.append(ComponentSpec("plain_block", params))
    return ModelConfig(
        sequence_len=sequence_len, vocab_size=vocab_size, n_embd=n_embd, reference=reference,
        shared={"rope": ComponentSpec("rotary", {"head_dim": head_dim})},
        input=ComponentSpec("token_embedding", {"smear": False}),
        body=ComponentSpec("stack", {"blocks": blocks}),
        output=ComponentSpec("lm_head", {"softcap": 15}),
    )


def expand_gpt(depth, aspect_ratio=64, head_dim=128, max_seq_len=2048, vocab_size=32768, window_pattern="SSSL") -> ModelConfig:
    n_embd, n_head = mup_dims(depth, aspect_ratio, head_dim)
    reference = {"preset": "gpt", "kwargs": dict(aspect_ratio=aspect_ratio, head_dim=head_dim,
                                                  max_seq_len=max_seq_len, vocab_size=vocab_size,
                                                  window_pattern=window_pattern)}
    return assemble_gpt(depth, n_head, n_head, n_embd, head_dim, max_seq_len, vocab_size, window_pattern,
                         reference=reference)


def _expand_llama_like(preset_name, depth, aspect_ratio, head_dim, max_seq_len, vocab_size, window_pattern,
                        kv_share_frac=None) -> ModelConfig:
    n_embd, n_head = mup_dims(depth, aspect_ratio, head_dim)
    reference_kwargs = dict(aspect_ratio=aspect_ratio, head_dim=head_dim, max_seq_len=max_seq_len,
                             vocab_size=vocab_size, window_pattern=window_pattern)
    if kv_share_frac is not None:
        reference_kwargs["kv_share_frac"] = kv_share_frac
    reference = {"preset": preset_name, "kwargs": reference_kwargs}
    return assemble_plain(depth, n_head, n_head, n_embd, head_dim, max_seq_len, vocab_size, window_pattern,
                           kv_share_frac=kv_share_frac, reference=reference)


def expand_llama(depth, aspect_ratio=64, head_dim=128, max_seq_len=2048, vocab_size=32768, window_pattern="L") -> ModelConfig:
    return _expand_llama_like("llama", depth, aspect_ratio, head_dim, max_seq_len, vocab_size, window_pattern)


def expand_llama_kvshare(depth, aspect_ratio=64, head_dim=128, max_seq_len=2048, vocab_size=32768,
                          window_pattern="L", kv_share_frac=0.5) -> ModelConfig:
    return _expand_llama_like("llama_kvshare", depth, aspect_ratio, head_dim, max_seq_len, vocab_size,
                               window_pattern, kv_share_frac=kv_share_frac)


def expand_llama_kvshare_win(depth, aspect_ratio=64, head_dim=128, max_seq_len=2048, vocab_size=32768,
                              window_pattern="SSSL", kv_share_frac=0.5) -> ModelConfig:
    return _expand_llama_like("llama_kvshare_win", depth, aspect_ratio, head_dim, max_seq_len, vocab_size,
                               window_pattern, kv_share_frac=kv_share_frac)


PRESETS = {
    "gpt": expand_gpt, "llama": expand_llama,
    "llama_kvshare": expand_llama_kvshare, "llama_kvshare_win": expand_llama_kvshare_win,
}


def expand(name, depth, **kwargs) -> ModelConfig:
    if name not in PRESETS:
        raise ValueError(f"Unknown preset {name!r}. Registered: {sorted(PRESETS)}")
    return PRESETS[name](depth, **kwargs)


def resolve_model_config(model_config, depth, *, aspect_ratio, head_dim, max_seq_len, vocab_size,
                          window_pattern=None, arch_opts=None) -> ModelConfig:
    """Resolve --model-config (a preset name registered in PRESETS, or a path to a materialized
    JSON tree) into the ModelConfig to actually build at `depth`. See resolve_reference_config for
    the muP scaling-law reference model (a different depth, same underlying preset)."""
    if os.path.isfile(model_config):
        assert not arch_opts, "--arch-opt is not supported with a --model-config JSON file; edit the tree directly"
        with open(model_config, "r", encoding="utf-8") as f:
            return ModelConfig.from_dict(json.load(f))
    kwargs = dict(aspect_ratio=aspect_ratio, head_dim=head_dim, max_seq_len=max_seq_len, vocab_size=vocab_size)
    if window_pattern is not None:
        kwargs["window_pattern"] = window_pattern
    return expand(model_config, depth, **kwargs, **(arch_opts or {}))


def resolve_reference_config(resolved_config: ModelConfig, ref_depth: int) -> ModelConfig:
    """The muP scaling-law reference model (nanochat.scaling.derive_training_plan's d_ref) at
    ref_depth (12), for a config that resolve_model_config already resolved (from a preset or a
    JSON file) to `resolved_config`. expand() always stamps a `reference` block on its output, so
    this works uniformly regardless of which source resolve_model_config used -- a JSON file with
    no `reference` block (e.g. one written by hand rather than dumped from a preset) is the only
    case this can't handle; the caller should fall back to --d-ref-scaling-params instead."""
    assert resolved_config.reference is not None, (
        f"config has no 'reference' block, so its muP scaling-law reference model can't be "
        f"re-derived automatically at depth {ref_depth}; pass --d-ref-scaling-params instead"
    )
    return expand(resolved_config.reference["preset"], ref_depth, **resolved_config.reference["kwargs"])
