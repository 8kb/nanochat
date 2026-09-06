"""
ComposedModel: one generic BaseModel implementation, driven entirely by a ComposedConfig tree
instead of hardcoded per-architecture assembly. See docs/architecture.md's "Composed
architectures" section.
"""

import torch
import torch.nn as nn

from nanochat.common import print0
from nanochat.optim import MuonAdamW
from nanochat.model.base import BaseModel
from nanochat.model.param_roles import collect_param_roles, build_param_groups
from nanochat.model.registry import register_model
from nanochat.model.composed.spec import ComposedConfig
from nanochat.model.composed.registry import build_component
from nanochat.model.composed import catalog  # noqa: F401 -- import for its registration side effects


@register_model("composed", ComposedConfig)
class ComposedModel(BaseModel):
    def __init__(self, config):
        """NOTE: may run under torch.device("meta") -- see docs/architecture.md."""
        super().__init__()
        self.config = config
        padded_vocab_size = config.padded_vocab_size
        if padded_vocab_size != config.vocab_size:
            print0(f"Padding vocab_size from {config.vocab_size} to {padded_vocab_size} for efficiency")

        # Build context: derived globals every component may ask to have injected (see each
        # catalog entry's `needs`), plus `shared` components (e.g. rope) built once here and
        # injected into whatever asks for them by name -- see nanochat.model.composed.registry.
        ctx = {
            "n_embd": config.n_embd, "vocab_size": config.vocab_size,
            "padded_vocab_size": padded_vocab_size, "sequence_len": config.sequence_len,
            "n_layer": config.n_layer,
        }
        shared = {}
        for name, spec in config.shared.items():
            shared[name] = build_component(spec, ctx)
            ctx[name] = shared[name]
        self.shared = nn.ModuleDict(shared)

        self.embedding = build_component(config.input, ctx)
        self.body = build_component(config.body, ctx)
        self.unembedding = build_component(config.output, ctx)

    @torch.no_grad()
    def init_weights(self):
        self.embedding.init_weights()
        for m in self.shared.values():
            m.init_weights()
        self.body.init_weights()
        self.unembedding.init_weights()

    def layer_specs(self):
        return self.body.layer_specs()

    def shape_summary(self):
        """A composed model's per-layer attention geometry can vary (that's the point), so this
        reports "mixed" rather than a single scalar where layers disagree -- see
        nanochat.model.base.BaseModel.shape_summary for the flat-config default this overrides."""
        specs = self.layer_specs()
        n_heads = {s.n_head for s in specs}
        n_kv_heads = {s.n_kv_head for s in specs}
        windows = {s.window for s in specs}
        return {
            "n_layer": self.config.n_layer, "n_embd": self.config.n_embd,
            "n_head": next(iter(n_heads)) if len(n_heads) == 1 else "mixed",
            "n_kv_head": next(iter(n_kv_heads)) if len(n_kv_heads) == 1 else "mixed",
            "sequence_len": self.config.sequence_len,
            "window_pattern": next(iter(windows)) if len(windows) == 1 else "mixed",
        }

    def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0, scalar_lr=0.5):
        # Same hyperparameters as GPT.setup_optimizer (the closest reference), plus a
        # "backout_scalar" role for BackoutComposer's own scalar -- covers every role any
        # currently-cataloged component can produce, regardless of which the tree actually uses
        # (build_param_groups skips a policy role with no params present).
        model_dim = self.config.n_embd
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print0(f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")
        policy = {
            "unembedding": dict(kind='adamw', lr=unembedding_lr * dmodel_lr_scale, betas=(0.8, 0.96), eps=1e-10, weight_decay=0.01),
            "embedding": dict(kind='adamw', lr=embedding_lr * dmodel_lr_scale, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.001),
            "value_embedding": dict(kind='adamw', lr=embedding_lr * dmodel_lr_scale * 0.5, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.01),
            "resid_scalar": dict(kind='adamw', lr=scalar_lr * 0.01, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.05),
            "x0_scalar": dict(kind='adamw', lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),
            "smear": dict(kind='adamw', lr=0.2, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0),
            "backout_scalar": dict(kind='adamw', lr=0.2, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0),
            "matrix": dict(kind='muon', lr=matrix_lr, momentum=0.95, ns_steps=5, beta2=0.9, weight_decay=weight_decay),
        }
        param_groups = build_param_groups(collect_param_roles(self), policy)
        optimizer = MuonAdamW(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean'):
        x = self.embedding(idx, kv_cache)
        x = self.body(x, idx, kv_cache)
        if kv_cache is not None:
            kv_cache.advance(idx.size(1))
        return self.unembedding(x, targets, loss_reduction)
