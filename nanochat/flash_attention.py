"""
Backward-compat shim: the FA3/SDPA switching interface now lives in modelcore/kernels/ (Stage 7
-- see docs/roadmap.md), which has no nanochat dependency at all. Kept here so existing
`from nanochat.flash_attention import ...` call sites keep working unchanged.
"""
from modelcore.kernels.flash_attn import flash_attn, HAS_FA3, FA3_LOAD_ERROR, USE_FA3

__all__ = ["flash_attn", "HAS_FA3", "FA3_LOAD_ERROR", "USE_FA3"]
