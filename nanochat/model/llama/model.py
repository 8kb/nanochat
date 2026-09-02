"""
Llama-style architecture: SwiGLU MLP, plain pre-norm residual blocks, no value embeddings /
smear / backout / per-layer resid-x0 lambdas -- a deliberately boring baseline whose only job is
to prove nanochat.model.base's contracts (BaseModel, BaseEmbedding, BaseBlock, BaseUnembedding,
the parameter-role protocol) are real interfaces and not just GPT with extra indirection. Reuses
GPT's TokenEmbedding (smear disabled), CausalSelfAttention, RotaryEmbedding, and LMHead verbatim;
only the MLP (SwiGLUMLP) and the block-level residual mixing (PlainBlock, no lambdas) differ. See
docs/architecture.md.
"""

import torch
import torch.nn as nn

from nanochat.common import print0
from nanochat.optim import MuonAdamW
from nanochat.model.base import BaseModel
from nanochat.model.param_roles import collect_param_roles, build_param_groups
from nanochat.model.registry import register_model
from nanochat.model.components.rotary import RotaryEmbedding
from nanochat.model.components.embedding import TokenEmbedding
from nanochat.model.components.unembedding import LMHead
from nanochat.model.components.windows import compute_window_sizes
from nanochat.model.llama.block import PlainBlock
from nanochat.model.llama.config import LlamaConfig


@register_model("llama", LlamaConfig)
class Llama(BaseModel):
    def __init__(self, config, pad_vocab_size_to=64):
        """NOTE: __init__ may run under torch.device("meta") -- see docs/architecture.md."""
        super().__init__()
        self.config = config
        padded_vocab_size = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        if padded_vocab_size != config.vocab_size:
            print0(f"Padding vocab_size from {config.vocab_size} to {padded_vocab_size} for efficiency")
        head_dim = config.n_embd // config.n_head

        self.embedding = TokenEmbedding(padded_vocab_size, config.n_embd, smear=False)
        self.rope = RotaryEmbedding(head_dim, config.sequence_len)

        window_sizes = compute_window_sizes(config.window_pattern, config.n_layer, config.sequence_len)
        blocks = []
        for i in range(config.n_layer):
            window, _ = window_sizes[i]
            blocks.append(PlainBlock(config.n_embd, config.n_head, config.n_kv_head, i, window, self.rope, padded_vocab_size))
        self.blocks = nn.ModuleList(blocks)

        self.unembedding = LMHead(config.n_embd, config.vocab_size, padded_vocab_size)

    @torch.no_grad()
    def init_weights(self):
        self.embedding.init_weights()
        self.rope.init_weights()
        for block in self.blocks:
            block.init_weights()
        self.unembedding.init_weights()

    def layer_specs(self):
        return [block.layer_spec() for block in self.blocks]

    def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0, scalar_lr=0.5):
        # scalar_lr is accepted (unused) so scripts/base_train.py's single, arch-agnostic
        # setup_optimizer(...) call keeps working regardless of --arch.
        model_dim = self.config.n_embd
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print0(f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")
        policy = {
            "unembedding": dict(kind='adamw', lr=unembedding_lr * dmodel_lr_scale, betas=(0.8, 0.96), eps=1e-10, weight_decay=0.01),
            "embedding": dict(kind='adamw', lr=embedding_lr * dmodel_lr_scale, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.001),
            "matrix": dict(kind='muon', lr=matrix_lr, momentum=0.95, ns_steps=5, beta2=0.9, weight_decay=weight_decay),
        }
        param_groups = build_param_groups(collect_param_roles(self), policy)
        optimizer = MuonAdamW(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean'):
        x = self.embedding(idx, kv_cache)
        for block in self.blocks:
            x = block(x, None, idx, kv_cache)
        return self.unembedding(x, targets, loss_reduction)
