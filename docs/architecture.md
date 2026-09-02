# Architecture contract

`nanochat/model/` holds every transformer architecture this fork supports, side by side. The
rest of the codebase — `checkpoint_manager`, `Engine`, the training scripts — talks to a model
only through the interface described here, so adding an architecture never requires touching
those call sites.

```
nanochat/model/
├── __init__.py            public API: BaseModel, BaseModelConfig, AttentionLayerSpec,
│                          BaseEmbedding, BaseBlock, BaseUnembedding, register_model,
│                          get_model_class, get_config_class, config_from_dict, GPT, GPTConfig
├── base.py                the contract: BaseModelConfig, BaseModel, BaseEmbedding, BaseBlock,
│                          BaseUnembedding, AttentionLayerSpec
├── registry.py             arch name -> (config class, model class)
├── param_roles.py           parameter-role protocol backing setup_optimizer()/num_scaling_params()
├── flops.py                  FLOPs / KV-cache-bytes accounting, generic over layer_specs()
├── components/            reusable building blocks any architecture can import
│   ├── linear.py            Linear         -- weight-casting nn.Linear (see "Precision" below)
│   ├── norm.py               norm()         -- parameter-free RMSNorm
│   ├── rope.py                 apply_rotary_emb(), precompute_rotary_embeddings() (free functions)
│   ├── rotary.py                 RotaryEmbedding  -- owns cos/sin buffers, shared across layers
│   ├── attention.py               has_ve(), CausalSelfAttention  (FA3/SDPA, GQA, value residual)
│   ├── mlp.py                      MLP           -- relu² MLP
│   ├── block.py                     Block         -- BaseBlock: attn + MLP + resid/x0 lambdas
│   ├── embedding.py                   Smear, TokenEmbedding  -- BaseEmbedding
│   ├── unembedding.py                   LMHead      -- BaseUnembedding
│   └── windows.py                        compute_window_sizes()  -- sliding-window pattern tiling
└── gpt/                    the default architecture
    ├── config.py             GPTConfig(BaseModelConfig)
    ├── model.py                GPT(BaseModel) -- wires Embedding/Block/Unembedding together
    └── migrations.py            old-checkpoint backward-compat patches (config/state/optimizer)
```

## The three module contracts

Everything below `BaseModel` is one of three contracts (`nanochat/model/base.py`). A model may
know about these contracts and what its own choice of them means; a contract's implementation
must not know anything about the model wiring it in.

```python
class BaseEmbedding(nn.Module):    # token ids -> residual-stream activations
    def init_weights(self): ...
    def forward(self, idx, kv_cache=None): ...

class BaseBlock(nn.Module):        # one residual-stream transform step
    def init_weights(self): ...
    def forward(self, x, x0, idx, kv_cache): ...
    def layer_spec(self) -> AttentionLayerSpec | None: ...   # None if not attention-shaped

class BaseUnembedding(nn.Module):  # residual-stream activations -> logits, or loss
    def init_weights(self): ...
    def forward(self, x, targets=None, loss_reduction="mean"): ...
```

GPT's implementations: `nanochat/model/components/embedding.py`'s `TokenEmbedding` (wte lookup +
`COMPUTE_DTYPE` cast + `norm()` + an optional `Smear` submodule that mixes in the previous token's
embedding, reading/writing `kv_cache.state["prev_embedding"]` during KV-cached decode);
`nanochat/model/components/block.py`'s `Block` (`CausalSelfAttention` + `MLP`, plus the per-layer
`resid_lambda`/`x0_lambda` residual mixing — see "Parameter roles" below for where their *values*
come from); `nanochat/model/components/unembedding.py`'s `LMHead` (final `norm()` + projection +
vocab crop + tanh softcap, and the loss when `targets` is given).

Each owns everything about its own concern: `TokenEmbedding` owns the smear logic and its KV-cache
state key; `Block` owns its attention geometry (`layer_spec()` delegates to its
`CausalSelfAttention`), its value-embedding table (if any), and its two per-layer scalars;
`LMHead` owns the loss computation. `GPT` itself only decides *how many* blocks, *what* per-layer
window/schedule each gets, and wires the three together — it does not reach into a block's
`attn.c_q.weight` or similar (contrast with pre-Stage-2 `GPT.init_weights()`, which did).

Position encoding is deliberately **not** part of `BaseBlock.forward`'s signature.
`nanochat/model/components/rotary.py`'s `RotaryEmbedding` owns the `cos`/`sin` buffers and the
offset-into-KV-cache logic; `GPT` constructs one instance and passes it into every attention layer
(a shared submodule, not a per-layer copy). That keeps position encoding an attention-internal
concern a future architecture can swap (RoPE / NoPE / ALiBi) without touching the block or trunk
contract.

`x0` (the initial post-embedding activation, blended back in via `x0_lambda`) is computed inside
`GPT._forward_trunk`, not inside `TokenEmbedding` — it belongs to the trunk's residual topology,
not to the embedding step, and a future architecture with a different topology (or none at all —
see "Non-transformer architectures" below) may not want an `x0` residual at all.

## `BaseModel` (`nanochat/model/base.py`)

```python
class BaseModelConfig:            # a dataclass
    sequence_len: int
    vocab_size: int
    arch: ClassVar[str]           # stamped by @register_model
    def to_dict(self) -> dict     # -> {**dataclasses.asdict(self), "arch": ...}

class AttentionLayerSpec:         # a dataclass, one per transformer layer
    n_head: int
    n_kv_head: int
    head_dim: int
    window: int = -1              # -1 = full/unlimited context

class BaseModel(nn.Module):
    # you implement these:
    def init_weights(self): ...
    def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean'): ...
    def setup_optimizer(self, **kwargs) -> torch.optim.Optimizer: ...
    def layer_specs(self) -> list[AttentionLayerSpec]: ...
    def num_scaling_params(self) -> dict: ...

    # optional, no-op by default (see "Old-checkpoint migrations" below):
    @classmethod
    def patch_config_dict(cls, model_config_kwargs, log=...): return model_config_kwargs
    @classmethod
    def patch_state_dict(cls, model_data, model_config, log=...): return model_data
    @classmethod
    def patch_optimizer_state_dict(cls, optimizer_data, model_config, log=...): return optimizer_data

    # you get these for free, built on layer_specs():
    def get_device(self)
    def kv_cache_spec(self) -> dict          # {num_heads, head_dim, num_layers}
    def num_matmul_params(self)
    def estimate_flops(self)
    def estimate_decode_flops(self, context_len)
    def estimate_prefill_flops(self, num_tokens)
    def kv_bytes_per_token(self)
    def kv_read_bytes(self, context_len)
```

`layer_specs()` is the pivot the free methods are built on: it is one `AttentionLayerSpec` per
layer, in forward-pass order (typically `[block.layer_spec() for block in self.blocks]`).
`kv_cache_spec()` (what `nanochat.engine.Engine` uses to size its `KVCache`) and every
FLOPs/KV-bytes estimate in `nanochat/model/flops.py` are expressed purely in terms of it — a new
architecture gets all of that accounting for free by implementing `layer_specs()` correctly, no
matter how its per-layer attention geometry varies.

`kv_cache_spec()`'s default implementation requires uniform `n_kv_head`/`head_dim` across layers
(true for GPT). An architecture with heterogeneous per-layer KV geometry (e.g. mixed local/global
attention with different head dims) should override `kv_cache_spec()` directly.

### Implicit requirements not spelled out above

- The model must set `self.config = config` in `__init__` (used by `estimate_flops()` via
  `self.config.sequence_len`).
- `__init__` should assume it may run under `torch.device("meta")` (shapes/dtypes only, no real
  data) — see "The meta-device footgun" below. Real initialization goes in `init_weights()`.
- Every matmul-participating parameter should go through `nanochat.model.components.linear.Linear`
  rather than a raw `nn.Linear` or `nn.Parameter`, or `num_matmul_params()` (and everything built
  on it: `estimate_flops`, `estimate_decode_flops`, `estimate_prefill_flops`) will silently
  undercount it.
- Every parameter needs a declared role (see "Parameter roles" below), or `setup_optimizer()` /
  `num_scaling_params()` — if built the standard way, on `collect_param_roles` — raise.

### Non-transformer architectures

Not a goal today, but not precluded: `layer_specs()` may return `[]` for a model with no
attention-shaped layers at all, in which case it overrides `kv_cache_spec()` and the
`estimate_*`/`kv_*_bytes` methods directly instead of relying on the generic implementations.
Similarly, `BaseBlock.layer_spec()` defaults to returning `None`, for a block that isn't
attention-shaped inside an otherwise-attention model.

## Parameter roles

`nanochat/model/param_roles.py`. Every parameter in the model needs a declared **role** — a string like `"matrix"`, `"embedding"`,
`"smear"` — so `setup_optimizer()` knows which optimizer (Muon vs AdamW) and hyperparameters to
give it, and `num_scaling_params()` knows which bucket to count it in. A module declares roles for
the parameters it *directly* owns via a `PARAM_ROLES` class attribute, mapping an attribute name
to a role string. The attribute can be a `Parameter` (that parameter takes the role) or a child
submodule (every parameter in that submodule's whole subtree takes the role):

```python
class Smear(nn.Module):
    PARAM_ROLES = {"gate": "smear", "lambda_": "smear"}
```

`collect_param_roles(module)` walks the tree resolving these declarations. A
`nanochat.model.components.linear.Linear`'s `.weight` defaults to role `"matrix"` when not
otherwise declared (matches `nanochat.model.flops.num_matmul_params`'s "Linear marks a matmul"
convention); **anything else undeclared raises** rather than falling through into a default bucket
— this is what stands in for the old hand-maintained "covers every parameter exactly once" assert,
and it catches the specific failure mode this protocol exists to prevent: a bare `nn.Embedding` or
`nn.Parameter` scalar silently defaulting into Muon's shape-based matrix grouping just because it
happened to live inside the same submodule as attention matrices.

A module that wants to own its whole subtree's role assignment in one place (rather than one role
per attribute) can define `param_roles(self) -> dict[str, list[Parameter]]` instead — the escape
hatch. `LMHead` uses it for optional weight tying: when constructed with `weight=<a Parameter
another module already owns>`, it declares no role for that parameter at all, since claiming
`"unembedding"` for a parameter the embedding already claimed `"embedding"` for would raise
(a parameter claimed for the same role via two different paths is fine — that's what tying without
a role conflict looks like — but two *different* roles for the same parameter is a real error, and
`collect_param_roles` raises on it either way, using the role assignment to decide which case it is).

`build_param_groups(role_params, policy)` turns `{role: [params]}` plus an *ordered*
`{role: hyperparameters}` policy dict into `nanochat.optim.MuonAdamW` param groups, splitting any
`kind="muon"` role into one group per parameter shape (Muon stacks same-shape params for its fused
step). `GPT.setup_optimizer` is that policy table:

```python
policy = {   # order is load-bearing -- see "Optimizer state is checkpointed positionally" below
    "unembedding":     dict(kind="adamw", lr=..., betas=..., eps=..., weight_decay=...),
    "embedding":       dict(kind="adamw", ...),
    "value_embedding": dict(kind="adamw", ...),
    "resid_scalar":    dict(kind="adamw", ...),
    "x0_scalar":       dict(kind="adamw", ...),
    "smear":           dict(kind="adamw", ...),
    "matrix":          dict(kind="muon", ...),
}
param_groups = build_param_groups(collect_param_roles(self), policy)
```

`num_scaling_params()` sums `p.numel()` per role from the same `collect_param_roles(self)` call.
Its six returned dict keys (`wte`, `value_embeds`, `lm_head`, `transformer_matrices`, `scalars`,
`total`) are a fixed, greppable output mapping — see "Two more load-bearing contracts" below — kept
stable even though the underlying role names are more granular (`scalars` sums three roles:
`resid_scalar` + `x0_scalar` + `smear`).

### Optimizer state is checkpointed positionally

`torch.optim.Optimizer.state_dict()` flattens every parameter across every group into one global
flat index, in group order. Reordering `setup_optimizer()`'s policy dict, or changing how many
parameters a role holds (a `[n_layer]` tensor becoming `n_layer` separate scalars, say), shifts
that indexing — and a same-*size* reorder corrupts state silently, since `load_state_dict` only
validates group sizes, not identity. This is why the policy dict's key order above is treated as
part of the on-disk format: changing it needs a `patch_optimizer_state_dict` migration (see below),
not just a code change.

## Registering an architecture

```python
# nanochat/model/<arch>/model.py
from nanochat.model.base import BaseModel
from nanochat.model.registry import register_model
from nanochat.model.<arch>.config import MyConfig

@register_model("my_arch", MyConfig)
class MyModel(BaseModel):
    ...
```

The decorator stamps `MyConfig.arch = "my_arch"` and registers the (config, model) pair.
Register it by importing the module somewhere that always runs — either add
`from nanochat.model.<arch> import MyModel, MyConfig` to `nanochat/model/__init__.py` (this is
what `gpt/` does), or import it explicitly wherever you invoke it (e.g. a script's own imports).
Only architectures that get imported are registered — if `get_model_class("my_arch")` raises
`ValueError: Unknown architecture`, the import never happened.

`BaseModelConfig.to_dict()` is what goes into `meta["model_config"]` in every checkpoint, and
`nanochat.model.registry.config_from_dict()` is the inverse — pop the `"arch"` key, look up the
config class, and construct it from the rest. This round-trip is how
`nanochat/checkpoint_manager.py` reconstructs the right model class without importing any
specific architecture. A checkpoint's `model_config` with no `"arch"` key (saved before this
registry existed) defaults to `"gpt"`.

### Adding an architecture, step by step

1. `mkdir nanochat/model/<arch>` with `config.py`, `model.py`, `__init__.py` (mirror `gpt/`).
2. `config.py`: a `@dataclass class MyConfig(BaseModelConfig)` with your architecture's fields.
   If you want it to work with `scripts/base_train.py`'s `--depth` dial, add a
   `from_depth(depth, aspect_ratio, head_dim, max_seq_len, vocab_size, window_pattern)`
   classmethod (see `GPTConfig.from_depth` for the muP-style derivation GPT uses) — otherwise
   `base_train.py` raises a clear assertion telling you it's missing.
3. `model.py`: implement `BaseModel`, reusing whatever fits from `nanochat/model/components/`.
   `CausalSelfAttention`/`MLP` take explicit dims (not a config object), so they're reusable by a
   config with different field names. Reach for `Block` directly (or `TokenEmbedding`/`LMHead`) if
   you're only changing depth/residual topology; write your own `BaseBlock` if you're changing
   attention itself or want a non-attention layer.
4. Add the import to `nanochat/model/__init__.py` so `@register_model` actually runs.
5. Declare a role for every parameter you introduce (`PARAM_ROLES` or `param_roles()`) — see
   "Parameter roles" above.
6. Add a fixture + a couple of tests mirroring `tests/conftest.py`'s `tiny_gpt` /
   `tests/test_model_gpt.py` (forward shape, loss finite, backward populates every parameter's
   grad, `setup_optimizer` partitions all parameters exactly once).
7. Try it: `python -m scripts.base_train --arch=my_arch --depth=2 --num-iterations=3 ...` (see
   the CPU smoke-test invocation in "Verifying a change is behavior-preserving" below).

## Old-checkpoint migrations

Checkpoints predate fields, parameters, and (as of Stage 2) whole module layouts. Three hooks on
`BaseModel`, all no-ops by default, overridden per-architecture; GPT's overrides live in
`nanochat/model/gpt/migrations.py`:

- **`patch_config_dict(model_config_kwargs, log)`** — backfills missing config fields (e.g.
  `window_pattern` defaulting to `"L"`, full context, for checkpoints saved before sliding-window
  attention existed).
- **`patch_state_dict(model_data, model_config, log)`** — backfills missing parameters (e.g.
  `resid_lambdas`/`x0_lambdas` defaulting to their identity values) and, for GPT specifically,
  renames the Stage 2 module restructure's keys (`transformer.wte.weight` ->
  `embedding.wte.weight`, `resid_lambdas[i]` -> `blocks.{i}.resid_lambda`, and so on — see the
  table in [upstream-sync.md](upstream-sync.md)). The rename runs *after* the backfill, since a
  checkpoint missing `resid_lambdas` entirely necessarily also predates the module restructure.
- **`patch_optimizer_state_dict(optimizer_data, model_config, log)`** — migrates a raw
  `torch.optim.Optimizer.state_dict()` (`{"state": {flat_index: {...}}, "param_groups": [...]}`)
  for a parameter that split/merged/moved since the checkpoint was saved. GPT's override splits
  the resid/x0 scalar groups' per-index state (renumbering every later flat index to make room)
  when a `[n_layer]` tensor became `n_layer` per-block scalars. Unlike the other two hooks, this
  one isn't called automatically by `checkpoint_manager` — optimizer state is loaded separately
  from model state (see `nanochat/checkpoint_manager.py`'s `load_optimizer_state`), so the two call
  sites that resume optimizer state (`scripts/base_train.py`'s `--resume-from-step` and
  `scripts/chat_sft.py`'s `--load-optimizer`) call it explicitly, right before
  `optimizer.load_state_dict(...)`.

`nanochat/checkpoint_manager.py:build_model` calls the first two automatically:

```python
model_cls = get_model_class(arch)
model_config_kwargs = model_cls.patch_config_dict(model_config_kwargs, log=log0)
model_config = config_from_dict(model_config_kwargs)
model_data = model_cls.patch_state_dict(model_data, model_config, log=log0)
```

## The meta-device footgun

`GPT.__init__` (and any architecture's `__init__`, and any submodule's `__init__`) may run under
`torch.device("meta")` — shapes and dtypes only, no real storage. `scripts/base_train.py`'s
`build_model_meta` and `checkpoint_manager.build_model` both do this so they can inspect a model's
config/param counts before committing to allocate it. Any real data (weight init, RNG-dependent
buffers) must live in `init_weights()`, called after `model.to_empty(device=...)` materializes real
storage. This is why `checkpoint_manager.build_model` calls `init_weights()` even when *loading* a
checkpoint (right before `load_state_dict(..., assign=True)` overwrites everything) — GPT's rotary
`cos`/`sin` buffers (owned by `nanochat.model.components.rotary.RotaryEmbedding`) are
`persistent=False` (not saved in the checkpoint) and only get populated with real values inside
`init_weights()`.

Each submodule owns its own `init_weights()` (called by its parent's, recursively, down to `GPT`'s)
rather than one function reaching into every submodule's internals by attribute path — this is
what makes a submodule usable by a different architecture's `__init__` without that architecture
needing to know the submodule's field names. One consequence: **the exact sequence of RNG calls
during a from-scratch `init_weights()` changed** in Stage 2 relative to pre-Stage-2 nanochat (same
individual `torch.nn.init.*` calls, different order), so a freshly-initialized model with a given
seed gets different actual weight values than before. This does not affect *loading* an existing
checkpoint (its saved values fully override whatever `init_weights()` produced), only bit-for-bit
reproducibility of new from-scratch runs — the same category of intentional, documented deviation
as Stage 1's top-k sampling change (see [upstream-sync.md](upstream-sync.md)).

A module built directly (not via `with torch.device("meta")`) should still work correctly on
whatever device is ambient — see `RotaryEmbedding.__init__`, which reads the current default
device explicitly (`torch.empty(0).device`) rather than assuming a meta context, so it's directly
usable standalone (e.g. in a unit test) without an external `to_empty(device)` call first.

## Precision

There's no `torch.amp.autocast` anywhere in this codebase. Precision is controlled by one global,
`COMPUTE_DTYPE` (`nanochat/common.py`, overridable via the `NANOCHAT_DTYPE` env var): model
weights stay fp32 (for optimizer precision), and `nanochat.model.components.linear.Linear` casts
them to `COMPUTE_DTYPE` in `forward()`. On CPU/MPS (this repo's day-to-day dev machine) that's
`float32` by default. Any new architecture should route its matmul weights through `Linear`
rather than a raw `nn.Linear`, both for this precision policy and so FLOPs accounting sees it
(see "Implicit requirements" above).

## Two more load-bearing contracts

- **`num_scaling_params()`'s six dict keys** (`wte`, `value_embeds`, `lm_head`,
  `transformer_matrices`, `scalars`, `total`) are greppable output: `runs/scaling_laws.sh` greps
  `^key ` lines out of `scripts/base_train.py`'s stdout dump of this dict, and
  `dev/scaling_analysis.ipynb` reads the resulting CSV columns. Keep these exact key names even if
  the underlying role names (see "Parameter roles" above) are more granular.
- **The `setup_optimizer()` policy dict's key order** is the on-disk optimizer `param_group`
  layout — see "Optimizer state is checkpointed positionally" above.

## Verifying a change is behavior-preserving

There's a trained checkpoint at `~/.cache/nanochat/base_checkpoints/d6` on this machine (from
`runs/runcpu.sh`), including its optimizer shard. Before and after a change to shared code
(`nanochat/model/base.py`, `nanochat/model/param_roles.py`, `nanochat/model/flops.py`,
`nanochat/engine.py`, `nanochat/checkpoint_manager.py`), compare:

```python
from nanochat.checkpoint_manager import load_model
from nanochat.common import compute_init, autodetect_device_type
from nanochat.engine import generate_naive

_, _, _, _, device = compute_init(autodetect_device_type())
model, tokenizer, meta = load_model("base", device, phase="eval", model_tag="d6")
prompt = tokenizer.encode("The chemical formula of water is", prepend=tokenizer.get_bos_token_id())
tokens = list(generate_naive(model, prompt, max_tokens=64, temperature=0.0))
# also worth checking: model.estimate_flops(), .num_matmul_params(), .num_scaling_params(),
# .kv_bytes_per_token(), .estimate_decode_flops(256), .estimate_prefill_flops(256)
```

Every value should match exactly. This is how Stage 1's `nanochat/gpt.py` -> `nanochat/model/`
extraction, and Stage 2's parameter-role/module-ownership restructure, were both checked end to
end. If the change also touches optimizer grouping (e.g. a `setup_optimizer` policy change), also
check that `load_optimizer_state("base", device, rank=0, model_tag="d6")`'s real shard loads into
a freshly built optimizer without error (a same-size group reorder passes size validation but
corrupts silently — see "Optimizer state is checkpointed positionally" above, and
`nanochat/model/gpt/migrations.py:patch_optimizer_state_dict` for the pattern if it doesn't).

For an end-to-end smoke test of the training path on CPU/MPS:

```bash
python -m scripts.base_train --depth=2 --head-dim=32 --window-pattern=L --max-seq-len=128 \
  --device-batch-size=1 --total-batch-size=256 --num-iterations=3 \
  --eval-every=-1 --core-metric-every=-1 --sample-every=-1 --model-tag=smoke --run=dummy
```

Delete `~/.cache/nanochat/base_checkpoints/smoke` afterward — it's a throwaway.
