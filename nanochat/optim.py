"""
Backward-compat shim: MuonAdamW now lives in modelcore/optim/ (Stage 7 -- see docs/roadmap.md),
which has no nanochat dependency at all. Kept here so existing `from nanochat.optim import
MuonAdamW` call sites keep working unchanged.
"""
from modelcore.optim import MuonAdamW

__all__ = ["MuonAdamW"]
