"""
GPT model (rewrite, a lot simpler)
Notable features:
- rotary embeddings (and no positional embeddings)
- QK norm
- untied weights for token embedding and lm_head
- relu^2 activation in MLP
- norm after token embedding
- no learnable params in rmsnorm
- no bias in linear layers
- Group-Query Attention (GQA) support for more efficient inference
- Flash Attention 3 integration

Structurally: GPT wires together three contracts -- BaseEmbedding, BaseBlock, BaseUnembedding
(see nanochat/model/base.py) -- and knows what its own choice of them means (padded vocab size,
the per-layer resid/x0-lambda schedule, backout, sliding-window pattern), but nothing below them.
See docs/architecture.md.
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
from nanochat.model.components.block import Block
from nanochat.model.gpt.config import GPTConfig
from nanochat.model.gpt import migrations


@register_model("gpt", GPTConfig)
class GPT(BaseModel):
    PARAM_ROLES = {"backout_lambda": "smear"}

    def __init__(self, config, pad_vocab_size_to=64):
        """
        NOTE a major footgun: this __init__ function runs in meta device context (!!)
        Therefore, any calculations inside here are shapes and dtypes only, no actual data.
        => We actually initialize all data (parameters, buffers, etc.) in init_weights() instead.
        """
        super().__init__()
        self.config = config
        # Pad vocab for efficiency (DDP, tensor cores). This is just an optimization - outputs are cropped in forward().
        # https://huggingface.co/docs/transformers/main_classes/model#transformers.PreTrainedModel.resize_token_embeddings
        padded_vocab_size = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        if padded_vocab_size != config.vocab_size:
            print0(f"Padding vocab_size from {config.vocab_size} to {padded_vocab_size} for efficiency")
        head_dim = config.n_embd // config.n_head

        self.embedding = TokenEmbedding(padded_vocab_size, config.n_embd)
        self.rope = RotaryEmbedding(head_dim, config.sequence_len)

        # Compute per-layer window sizes for sliding window attention
        # window_size is (left, right) tuple: (-1, 0) for full context, (N, 0) for sliding window
        window_sizes = compute_window_sizes(config.window_pattern, config.n_layer, config.sequence_len)
        n_layer = config.n_layer
        blocks = []
        for i in range(n_layer):
            window, _ = window_sizes[i]
            # Per-layer resid init: stronger residual at early layers, weaker at deep layers.
            # Decaying x0 init: earlier layers get more input embedding blending. Real values are
            # only set in Block.init_weights(); the model-level schedule lives here since it's a
            # muP-style choice about depth, not something a single layer can derive on its own.
            resid_lambda_init = 1.15 - (0.10 * i / max(n_layer - 1, 1))
            x0_lambda_init = 0.20 - (0.15 * i / max(n_layer - 1, 1))
            blocks.append(Block(
                config.n_embd, config.n_head, config.n_kv_head, i, n_layer, window, self.rope,
                padded_vocab_size, resid_lambda_init, x0_lambda_init,
            ))
        self.blocks = nn.ModuleList(blocks)

        self.unembedding = LMHead(config.n_embd, config.vocab_size, padded_vocab_size)
        # Backout: subtract cached mid-layer residual before final norm to remove low-level features
        self.backout_lambda = nn.Parameter(torch.empty(()))  # fake init, real init in init_weights()

    @torch.no_grad()
    def init_weights(self):
        """
        Initialize the full model: each submodule initializes its own parameters (see their
        respective init_weights()), in this order:

        embedding (wte, smear):  see TokenEmbedding.init_weights
        rope:                    see RotaryEmbedding.init_weights
        for each block:
            attn.c_q/c_k/c_v:    uniform, std=1/sqrt(n_embd); attn.c_proj: zeros
            attn.value_embed:    uniform, std=1/sqrt(n_embd) (if present)
            mlp.c_fc:            uniform, 0.4x std=1/sqrt(n_embd); mlp.c_proj: zeros
            resid_lambda/x0_lambda: per-layer schedule computed in __init__
        unembedding (lm_head):   normal, std=0.001
        backout_lambda:          constant 0.2
        """
        self.embedding.init_weights()
        self.rope.init_weights()
        for block in self.blocks:
            block.init_weights()
        self.unembedding.init_weights()
        torch.nn.init.constant_(self.backout_lambda, 0.2)

    def layer_specs(self):
        """One AttentionLayerSpec per transformer layer. Backs kv_cache_spec() (used by
        nanochat.engine.Engine) and all FLOPs/KV-bytes accounting in nanochat.model.flops (via
        BaseModel's wrapper methods)."""
        return [block.layer_spec() for block in self.blocks]

    def num_scaling_params(self):
        """
        Return detailed parameter counts for scaling law analysis.
        Different papers use different conventions:
        - Kaplan et al. excluded embedding parameters
        - Chinchilla included all parameters
        Ref: https://arxiv.org/abs/2203.15556 (Chinchilla paper)
        Ref: https://arxiv.org/abs/2001.08361 (Kaplan et al. original scaling laws paper)

        Returns a dict with counts for each parameter group, so downstream analysis
        can experiment with which combination gives the cleanest scaling laws. Keys are
        load-bearing: runs/scaling_laws.sh greps them out of base_train.py's stdout.
        """
        n = {role: sum(p.numel() for p in params) for role, params in collect_param_roles(self).items()}
        return {
            'wte': n.get('embedding', 0),
            'value_embeds': n.get('value_embedding', 0),
            'lm_head': n.get('unembedding', 0),
            'transformer_matrices': n.get('matrix', 0),
            'scalars': n.get('resid_scalar', 0) + n.get('x0_scalar', 0) + n.get('smear', 0),
            'total': sum(n.values()),
        }

    def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0, scalar_lr=0.5):
        model_dim = self.config.n_embd

        # Scale the LR for the AdamW parameters by ∝1/√dmodel (tuned for 768 dim model)
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print0(f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")

        # Role -> optimizer hyperparameters. Order is load-bearing: it is the on-disk
        # optimizer param_group layout (see nanochat.model.param_roles.build_param_groups).
        policy = {
            "unembedding": dict(kind='adamw', lr=unembedding_lr * dmodel_lr_scale, betas=(0.8, 0.96), eps=1e-10, weight_decay=0.01),
            "embedding": dict(kind='adamw', lr=embedding_lr * dmodel_lr_scale, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.001),
            "value_embedding": dict(kind='adamw', lr=embedding_lr * dmodel_lr_scale * 0.5, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.01),
            "resid_scalar": dict(kind='adamw', lr=scalar_lr * 0.01, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.05),
            "x0_scalar": dict(kind='adamw', lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),  # higher beta1 for x0
            "smear": dict(kind='adamw', lr=0.2, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0),
            "matrix": dict(kind='muon', lr=matrix_lr, momentum=0.95, ns_steps=5, beta2=0.9, weight_decay=weight_decay),
        }
        param_groups = build_param_groups(collect_param_roles(self), policy)

        optimizer = MuonAdamW(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def _forward_trunk(self, x, idx, kv_cache):
        """Run the stack of blocks with backout subtraction. Overridable by architectures with a
        different depth/residual topology (weight tying, layer looping, skip connections, MTP
        heads, ...) without touching the embedding/unembedding code in forward()."""
        x0 = x  # save initial (post-embedding) activations for the x0 residual
        n_layer = len(self.blocks)
        backout_layer = n_layer // 2  # cache at halfway point
        x_backout = None
        for i, block in enumerate(self.blocks):
            x = block(x, x0, idx, kv_cache)
            if i == backout_layer:
                x_backout = x
        # Subtract mid-layer residual to remove low-level features before logit projection
        if x_backout is not None:
            x = x - self.backout_lambda.to(x.dtype) * x_backout
        return x

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean'):
        x = self.embedding(idx, kv_cache)
        x = self._forward_trunk(x, idx, kv_cache)
        if kv_cache is not None:
            kv_cache.advance(idx.size(1))
        return self.unembedding(x, targets, loss_reduction)

    @classmethod
    def patch_config_dict(cls, model_config_kwargs, log=lambda msg: None):
        return migrations.patch_missing_config_keys(model_config_kwargs, log)

    @classmethod
    def patch_state_dict(cls, model_data, model_config, log=lambda msg: None):
        model_data = migrations.patch_missing_state_keys(model_data, model_config, log)
        model_data = migrations.patch_state_dict_layout(model_data, model_config, log)
        return model_data

    @classmethod
    def patch_optimizer_state_dict(cls, optimizer_data, model_config, log=lambda msg: None):
        return migrations.patch_optimizer_state_dict(optimizer_data, model_config, log)
