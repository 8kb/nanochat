"""
modelcore -- a standalone model subsystem: configs, architectures-as-data, and the machinery to
create/load/save models and optimizers and compute their stats. Knows nothing about nanochat,
checkpoints, tokenizers, or CLI flags; see nanochat/architectures/ and
nanochat/checkpoint_manager.py for the layer that does.

This module is being built up in stages (see the Stage 7 plan in docs/roadmap.md); today it
holds the pieces with no nanochat dependencies at all (runtime, the MuonAdamW optimizer, the
flash-attention kernel shim, the inference-time KV cache). ModelManager and the component/config
system land in later stages of the same extraction.
"""
from modelcore.cache import KVCache
from modelcore.runtime import Runtime, DEFAULT_RUNTIME

__all__ = ["KVCache", "Runtime", "DEFAULT_RUNTIME"]
