"""
Base interface every architecture under nanochat/model/ must implement.

An architecture is a (BaseModelConfig subclass, BaseModel subclass) pair registered with
nanochat.model.registry.register_model. The rest of the codebase (checkpoint_manager, Engine,
training scripts) talks to models only through this interface, plus the config's arch tag, so
that adding a new architecture never requires touching those call sites.
"""

from dataclasses import dataclass, asdict
from typing import ClassVar

import torch.nn as nn


@dataclass
class BaseModelConfig:
    """Common config fields every architecture needs. Subclasses add their own architecture-
    specific fields (n_layer, n_embd, ...) and get to_dict()/registry round-tripping for free."""
    sequence_len: int
    vocab_size: int

    # Stamped onto the subclass by @register_model; identifies the architecture in checkpoints.
    arch: ClassVar[str] = "base"

    def to_dict(self) -> dict:
        """Flat, JSON-serializable dict for checkpoint metadata. Includes "arch" so
        nanochat.model.registry.config_from_dict can reconstruct the right config class."""
        d = asdict(self)
        d["arch"] = type(self).arch
        return d


@dataclass
class AttentionLayerSpec:
    """Per-layer attention geometry: what the KV cache needs to allocate for this layer, and
    what the FLOPs/KV-bytes accounting in nanochat.model.flops needs to charge for it.
    window=-1 means unlimited/full context; a non-negative window is the number of preceding
    tokens attended to (matches the FA3 "left window" convention used by CausalSelfAttention)."""
    n_head: int
    n_kv_head: int
    head_dim: int
    window: int = -1


class BaseModel(nn.Module):
    """Interface every architecture must implement, plus generic accounting helpers that work
    off of layer_specs() so a new architecture gets FLOPs/KV-cache-bytes estimation for free."""

    # -- architecture must implement --

    def init_weights(self):
        """Materialize all parameters/buffers. Called after to_empty(device), since __init__
        typically runs under torch.device("meta") (shapes/dtypes only, no real data)."""
        raise NotImplementedError

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean'):
        raise NotImplementedError

    def setup_optimizer(self, **kwargs):
        raise NotImplementedError

    def layer_specs(self) -> list[AttentionLayerSpec]:
        """One AttentionLayerSpec per transformer layer, in forward-pass order. Backs
        kv_cache_spec() (used by nanochat.engine.Engine to size the KV cache) and every
        FLOPs/KV-cache-bytes method below."""
        raise NotImplementedError

    def num_scaling_params(self) -> dict:
        """Detailed parameter counts for scaling-law analysis (see the GPT implementation for
        the expected shape of the returned dict: named groups summing to a 'total' key)."""
        raise NotImplementedError

    # -- backward-compat hooks for old checkpoints; no-ops by default --

    @classmethod
    def patch_config_dict(cls, model_config_kwargs, log=lambda msg: None):
        """Mutate a raw config kwargs dict in place (and return it) before config construction,
        to backfill fields that didn't exist when older checkpoints were saved."""
        return model_config_kwargs

    @classmethod
    def patch_state_dict(cls, model_data, model_config, log=lambda msg: None):
        """Mutate a raw state dict in place (and return it) before load_state_dict, to backfill
        parameters that didn't exist when older checkpoints were saved."""
        return model_data

    # -- generic implementations built on layer_specs(); rarely need overriding --

    def get_device(self):
        return next(self.parameters()).device

    def kv_cache_spec(self) -> dict:
        """What nanochat.engine.KVCache needs to allocate: num_layers, num_heads, head_dim.
        Requires uniform n_kv_head/head_dim across layers; architectures with heterogeneous
        per-layer KV geometry (e.g. mixed local/global head dims) should override this."""
        specs = self.layer_specs()
        assert specs, "layer_specs() returned no layers"
        n_kv_heads = {s.n_kv_head for s in specs}
        head_dims = {s.head_dim for s in specs}
        assert len(n_kv_heads) == 1 and len(head_dims) == 1, (
            "kv_cache_spec() requires uniform n_kv_head/head_dim across layers; "
            "override kv_cache_spec() for heterogeneous architectures"
        )
        return {"num_heads": specs[0].n_kv_head, "head_dim": specs[0].head_dim, "num_layers": len(specs)}

    def num_matmul_params(self):
        from nanochat.model import flops
        return flops.num_matmul_params(self)

    def estimate_flops(self):
        from nanochat.model import flops
        return flops.estimate_flops(self.layer_specs(), self.num_matmul_params(), self.config.sequence_len)

    def estimate_decode_flops(self, context_len):
        from nanochat.model import flops
        return flops.estimate_decode_flops(self.layer_specs(), self.num_matmul_params(), context_len)

    def estimate_prefill_flops(self, num_tokens):
        from nanochat.model import flops
        return flops.estimate_prefill_flops(self.layer_specs(), self.num_matmul_params(), num_tokens)

    def kv_bytes_per_token(self):
        from nanochat.common import COMPUTE_DTYPE
        from nanochat.model import flops
        return flops.kv_bytes_per_token(self.layer_specs(), COMPUTE_DTYPE.itemsize)

    def kv_read_bytes(self, context_len):
        from nanochat.common import COMPUTE_DTYPE
        from nanochat.model import flops
        return flops.kv_read_bytes(self.layer_specs(), COMPUTE_DTYPE.itemsize, context_len)
