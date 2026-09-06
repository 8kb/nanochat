"""
Composer contract + two composers. A composer owns a set of blocks (or other composers) and how
they connect into one residual-stream transform -- what nanochat.model.gpt.model.GPT._forward_trunk
/ nanochat.model.llama.model.Llama.forward's block loops used to hardcode per architecture class,
now a nameable, swappable thing a config's `body` names via its own "#type". See
docs/architecture.md's "Composed architectures" section.
"""

import torch
import torch.nn as nn


class BaseComposer(nn.Module):
    """Contract: forward(x, idx, kv_cache) -> x; layer_specs() -> list[AttentionLayerSpec], in
    forward-pass order (a composer wrapping several block lists concatenates them so the generic
    FLOPs/KV accounting in nanochat.model.flops keeps working unchanged)."""

    def init_weights(self):
        raise NotImplementedError

    def forward(self, x, idx, kv_cache):
        raise NotImplementedError

    def layer_specs(self):
        raise NotImplementedError


class StackComposer(BaseComposer):
    """Plain sequential residual stack, no x0 residual -- covers llama / llama_kvshare /
    llama_kvshare_win. Threads a fresh kv_bus dict through every block each forward pass (a
    producer layer writes its K/V into it, a consumer layer reads an earlier layer's out of it --
    see nanochat.model.components.attention.CausalSelfAttention); a block that ignores kv_bus
    (BaseBlock.forward's kv_bus=None default) is unaffected."""

    def __init__(self, blocks):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)

    def init_weights(self):
        for block in self.blocks:
            block.init_weights()

    def layer_specs(self):
        return [b.layer_spec() for b in self.blocks]

    def forward(self, x, idx, kv_cache):
        kv_bus = {}
        for block in self.blocks:
            x = block(x, None, idx, kv_cache, kv_bus)
        return x


class BackoutComposer(BaseComposer):
    """GPT's residual topology: x0 (post-embedding activations) blended into every block via each
    block's own resid/x0 lambdas, plus a mid-depth "backout" subtraction before the final norm.
    Reproduces nanochat.model.gpt.model.GPT._forward_trunk exactly; owns backout_lambda itself
    (SOLID module ownership, per docs/architecture.md, applied to residual topology -- GPT's own
    backout_lambda historically dangled on the top-level model class instead)."""
    PARAM_ROLES = {"backout_lambda": "backout_scalar"}

    def __init__(self, blocks, backout_layer, backout_lambda_init=0.2):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)
        self.backout_layer = backout_layer
        self.backout_lambda = nn.Parameter(torch.empty(()))  # fake init, real init in init_weights()
        self._backout_lambda_init = backout_lambda_init

    @torch.no_grad()
    def init_weights(self):
        for block in self.blocks:
            block.init_weights()
        self.backout_lambda.fill_(self._backout_lambda_init)

    def layer_specs(self):
        return [b.layer_spec() for b in self.blocks]

    def forward(self, x, idx, kv_cache):
        x0 = x  # save initial (post-embedding) activations for the x0 residual
        x_backout = None
        for i, block in enumerate(self.blocks):
            x = block(x, x0, idx, kv_cache)
            if i == self.backout_layer:
                x_backout = x
        if x_backout is not None:
            x = x - self.backout_lambda.to(x.dtype) * x_backout
        return x
