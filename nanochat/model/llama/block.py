import torch

from nanochat.model.base import BaseBlock
from nanochat.model.components.norm import norm
from nanochat.model.components.attention import CausalSelfAttention
from nanochat.model.llama.mlp import SwiGLUMLP


class PlainBlock(BaseBlock):
    """Plain pre-norm residual block: x = x + attn(norm(x)); x = x + mlp(norm(x)). No per-layer
    resid/x0-lambda mixing, no value embeddings, no smear/backout -- unlike GPT's Block
    (nanochat/model/components/block.py), this is a deliberately boring baseline. Reuses
    CausalSelfAttention unmodified (GQA/RoPE/QK-norm are not GPT-specific tricks); only the MLP
    (SwiGLU) and the absence of lambda mixing differ.

    No PARAM_ROLES declaration needed: attn's matrices default to role "matrix" via Linear, and
    mlp (SwiGLUMLP) is the same."""

    def __init__(self, n_embd, n_head, n_kv_head, layer_idx, window, rope, padded_vocab_size):
        super().__init__()
        self.attn = CausalSelfAttention(n_embd, n_head, n_kv_head, layer_idx, window, rope, padded_vocab_size, has_value_embed=False)
        self.mlp = SwiGLUMLP(n_embd)

    @torch.no_grad()
    def init_weights(self):
        self.attn.init_weights()
        self.mlp.init_weights()

    def layer_spec(self):
        return self.attn.layer_spec()

    def forward(self, x, x0, idx, kv_cache):
        # x0 is part of the BaseBlock contract (see nanochat/model/base.py) but unused here --
        # this topology has no x0 residual.
        x = x + self.attn(norm(x), idx, kv_cache)
        x = x + self.mlp(norm(x))
        return x
