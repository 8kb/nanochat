from dataclasses import dataclass

from nanochat.model.llama.config import LlamaConfig


@dataclass
class LlamaKVShareConfig(LlamaConfig):
    """LlamaConfig plus the cross-layer KV-sharing dial: the last kv_share_frac of layers reuse
    an earlier layer's K/V instead of computing their own (see
    nanochat.model.components.kv_sharing.compute_kv_slots). 0.5 => the last half of layers share
    one slot; 2/3 => the last two-thirds. Google's Gemma-3n on-device models use this trick to
    shrink both KV-cache memory and parameter count without changing depth.

    from_depth is inherited unchanged from LlamaConfig -- it builds via cls(...), so
    LlamaKVShareConfig.from_depth(...) returns a LlamaKVShareConfig with kv_share_frac at its
    default; override it via --arch-opt kv_share_frac=... (see scripts/base_train.py)."""
    kv_share_frac: float = 0.5
