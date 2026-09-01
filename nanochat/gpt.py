"""
Backward-compat shim. The GPT model now lives in nanochat/model/ (see nanochat/model/gpt/ for
the model itself and nanochat/model/components/ for its reusable building blocks) as part of
making nanochat support multiple architectures side by side. See docs/architecture.md.

This module re-exports the same names upstream nanochat's nanochat/gpt.py exports, so upstream
diffs that `from nanochat.gpt import ...` keep applying without modification. Prefer importing
from nanochat.model directly in new code.
"""

from nanochat.model.gpt.config import GPTConfig
from nanochat.model.gpt.model import GPT
from nanochat.model.components.linear import Linear
from nanochat.model.components.norm import norm
from nanochat.model.components.rope import apply_rotary_emb
from nanochat.model.components.attention import has_ve, CausalSelfAttention
from nanochat.model.components.mlp import MLP
from nanochat.model.components.block import Block

__all__ = [
    "GPTConfig", "GPT", "Linear", "norm", "apply_rotary_emb",
    "has_ve", "CausalSelfAttention", "MLP", "Block",
]
