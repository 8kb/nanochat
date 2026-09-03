import torch
import torch.nn as nn

from nanochat.model.base import BaseBlock
from nanochat.model.components.norm import norm
from nanochat.model.components.attention import CausalSelfAttention, has_ve
from nanochat.model.components.mlp import MLP, SwiGLUMLP


class Block(BaseBlock):
    """CausalSelfAttention + MLP, plus the per-layer resid/x0-lambda residual mixing (inspired by
    modded-nanogpt; see docs/upstream/LOG.md's 2026-01 entries on why these help):
    resid_lambda scales the residual stream at this layer (init ~1.0 = neutral), x0_lambda blends
    the initial embedding back in (init ~0.0 = disabled). The per-layer schedule for their real
    init values is a model-level (muP-ish) decision computed by the caller and passed in; this
    module only applies it."""
    PARAM_ROLES = {"resid_lambda": "resid_scalar", "x0_lambda": "x0_scalar"}

    def __init__(self, n_embd, n_head, n_kv_head, layer_idx, n_layer, window, rope, padded_vocab_size,
                 resid_lambda_init, x0_lambda_init):
        super().__init__()
        self.attn = CausalSelfAttention(n_embd, n_head, n_kv_head, layer_idx, window, rope, padded_vocab_size, has_ve(layer_idx, n_layer))
        self.mlp = MLP(n_embd)
        self.resid_lambda = nn.Parameter(torch.empty(()))  # fake init, real init in init_weights()
        self.x0_lambda = nn.Parameter(torch.empty(()))     # fake init, real init in init_weights()
        self._resid_lambda_init = resid_lambda_init
        self._x0_lambda_init = x0_lambda_init

    @torch.no_grad()
    def init_weights(self):
        self.attn.init_weights()
        self.mlp.init_weights()
        self.resid_lambda.fill_(self._resid_lambda_init)
        self.x0_lambda.fill_(self._x0_lambda_init)

    def layer_spec(self):
        return self.attn.layer_spec()

    def forward(self, x, x0, idx, kv_cache):
        x = self.resid_lambda * x + self.x0_lambda * x0
        x = x + self.attn(norm(x), idx, kv_cache)
        x = x + self.mlp(norm(x))
        return x


class PlainBlock(BaseBlock):
    """Plain pre-norm residual block: x = x + attn(norm(x)); x = x + mlp(norm(x)). No per-layer
    resid/x0-lambda mixing, no value embeddings, no smear/backout -- unlike GPT's Block above,
    this is a deliberately boring baseline. Reuses CausalSelfAttention unmodified (GQA/RoPE/
    QK-norm are not GPT-specific tricks); only the MLP (SwiGLU) and the absence of lambda mixing
    differ. kv_slot/produces_kv pass straight through to CausalSelfAttention (both default to
    today's one-slot-per-layer behavior) so this block also serves cross-layer KV sharing
    (nanochat.model.llama_kvshare) without a fork.

    No PARAM_ROLES declaration needed: attn's matrices default to role "matrix" via Linear, and
    mlp (SwiGLUMLP) is the same. A KV-sharing consumer block (produces_kv=False) simply has fewer
    Linear submodules -- nothing to declare either way."""

    def __init__(self, n_embd, n_head, n_kv_head, layer_idx, window, rope, padded_vocab_size,
                 kv_slot=None, produces_kv=True):
        super().__init__()
        self.attn = CausalSelfAttention(n_embd, n_head, n_kv_head, layer_idx, window, rope, padded_vocab_size,
                                         has_value_embed=False, kv_slot=kv_slot, produces_kv=produces_kv)
        self.mlp = SwiGLUMLP(n_embd)

    @torch.no_grad()
    def init_weights(self):
        self.attn.init_weights()
        self.mlp.init_weights()

    def layer_spec(self):
        return self.attn.layer_spec()

    def forward(self, x, x0, idx, kv_cache, kv_bus=None):
        # x0 is part of the BaseBlock contract (see nanochat/model/base.py) but unused here --
        # this topology has no x0 residual.
        x = x + self.attn(norm(x), idx, kv_cache, kv_bus)
        x = x + self.mlp(norm(x))
        return x
