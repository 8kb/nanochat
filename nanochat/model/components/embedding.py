import torch
import torch.nn as nn

from nanochat.common import COMPUTE_DTYPE
from nanochat.model.base import BaseEmbedding
from nanochat.model.components.linear import Linear
from nanochat.model.components.norm import norm


class Smear(nn.Module):
    """Mix the previous token's (normed) embedding into the current position -- cheap bigram-like
    info, gated by a small input-dependent projection. During KV-cache decode there is no "full
    sequence" to slice from, so it reads/writes kv_cache.state["prev_embedding"] to carry the
    needed context across single-token forward() calls."""
    PARAM_ROLES = {"gate": "smear", "lambda_": "smear"}

    def __init__(self, gate_channels=24):
        super().__init__()
        self.gate_channels = gate_channels
        self.gate = Linear(gate_channels, 1, bias=False)
        self.lambda_ = nn.Parameter(torch.empty(()))  # fake init, real init in init_weights()

    @torch.no_grad()
    def init_weights(self):
        torch.nn.init.zeros_(self.lambda_)
        torch.nn.init.uniform_(self.gate.weight, 0.0, 0.02)

    def _gate(self, x_slice):
        return self.lambda_.to(x_slice.dtype) * torch.sigmoid(self.gate(x_slice))

    def forward(self, x, kv_cache):
        T = x.size(1)
        if kv_cache is None:
            # Training / naive generate: full sequence available, use fast slice
            assert T > 1, "Training forward pass should have T > 1"
            gate = self._gate(x[:, 1:, :self.gate_channels])
            return torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)
        # KV cache inference: read prev embedding from cache, store current for next step
        x_pre_smear = kv_cache.state.get("prev_embedding")
        kv_cache.state["prev_embedding"] = x[:, -1:, :]
        if T > 1:
            # Prefill: apply smear to positions 1+, same as training
            gate = self._gate(x[:, 1:, :self.gate_channels])
            return torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)
        elif x_pre_smear is not None:
            # Decode: single token, use cached prev embedding
            gate = self._gate(x[:, :, :self.gate_channels])
            return x + gate * x_pre_smear
        return x


class TokenEmbedding(BaseEmbedding):
    """wte lookup + COMPUTE_DTYPE cast + norm() + optional Smear. Token ids -> residual-stream
    activations, ready for the trunk."""
    PARAM_ROLES = {"wte": "embedding"}

    def __init__(self, padded_vocab_size, n_embd, smear=True):
        super().__init__()
        self.wte = nn.Embedding(padded_vocab_size, n_embd)
        self.smear = Smear() if smear else None

    @torch.no_grad()
    def init_weights(self):
        torch.nn.init.normal_(self.wte.weight, mean=0.0, std=0.8)
        if self.smear is not None:
            self.smear.init_weights()
        # Cast embeddings to COMPUTE_DTYPE: optimizer can tolerate reduced-precision embeddings
        # and it saves memory. Exception: fp16 requires fp32 embeddings because GradScaler cannot
        # unscale fp16 gradients.
        if COMPUTE_DTYPE != torch.float16:
            self.wte.to(dtype=COMPUTE_DTYPE)

    def forward(self, idx, kv_cache=None):
        x = self.wte(idx)
        x = x.to(COMPUTE_DTYPE)  # ensure activations are in compute dtype (no-op usually, but active for fp16)
        x = norm(x)
        if self.smear is not None:
            x = self.smear(x, kv_cache)
        return x
