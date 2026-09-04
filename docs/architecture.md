# Architecture contract

`nanochat/model/` holds every transformer architecture this fork supports, side by side. The
rest of the codebase — `checkpoint_manager`, `Engine`, the training scripts — talks to a model
only through the interface described here, so adding an architecture never requires touching
those call sites.

```
nanochat/model/
├── __init__.py            public API: BaseModel, BaseModelConfig, AttentionLayerSpec,
│                          BaseEmbedding, BaseBlock, BaseUnembedding, register_model,
│                          get_model_class, get_config_class, config_from_dict, apply_arch_opts,
│                          GPT, GPTConfig, Llama, LlamaConfig, LlamaKVShare, LlamaKVShareConfig
├── base.py                the contract: BaseModelConfig, BaseModel, BaseEmbedding, BaseBlock,
│                          BaseUnembedding, AttentionLayerSpec (incl. kv_slot)
├── registry.py             arch name -> (config class, model class); apply_arch_opts()
├── param_roles.py           parameter-role protocol backing setup_optimizer()/num_scaling_params()
├── flops.py                  FLOPs / KV-cache-bytes accounting, generic over layer_specs()
├── components/            reusable building blocks any architecture can import
│   ├── linear.py            Linear         -- weight-casting nn.Linear (see "Precision" below)
│   ├── norm.py               norm()         -- parameter-free RMSNorm
│   ├── rope.py                 apply_rotary_emb(), precompute_rotary_embeddings() (free functions)
│   ├── rotary.py                 RotaryEmbedding  -- owns cos/sin buffers, shared across layers
│   ├── attention.py               has_ve(), CausalSelfAttention  (FA3/SDPA, GQA, value residual,
│   │                                cross-layer KV sharing -- see "Cross-layer KV sharing" below)
│   ├── mlp.py                      MLP, SwiGLUMLP  -- relu² MLP, Llama-style gated MLP
│   ├── block.py                     Block, PlainBlock  -- BaseBlock implementations
│   ├── embedding.py                   Smear, TokenEmbedding  -- BaseEmbedding
│   ├── unembedding.py                   LMHead      -- BaseUnembedding
│   ├── windows.py                        compute_window_sizes()  -- sliding-window pattern tiling
│   └── kv_sharing.py                       compute_kv_slots()  -- cross-layer KV-slot assignment
├── gpt/                    the original architecture
│   ├── config.py             GPTConfig(BaseModelConfig)
│   ├── model.py                GPT(BaseModel) -- wires Embedding/Block/Unembedding together
│   └── migrations.py            old-checkpoint backward-compat patches (config/state/optimizer)
├── llama/                  the second architecture -- see "Worked example: llama" below
│   ├── config.py             LlamaConfig(BaseModelConfig)
│   └── model.py                Llama(BaseModel) -- reuses components/ verbatim, incl. PlainBlock
├── llama_kvshare/          the third architecture -- see "Worked example: llama_kvshare" below
│   ├── config.py             LlamaKVShareConfig(LlamaConfig) -- adds kv_share_frac
│   └── model.py                LlamaKVShare(BaseModel) -- Llama + cross-layer KV sharing
└── llama_kvshare_win/      the fourth architecture -- see "Worked example: llama_kvshare_win" below
    ├── config.py             LlamaKVShareWinConfig(LlamaKVShareConfig) -- window_pattern default only
    └── model.py                LlamaKVShareWin(LlamaKVShare) -- no new logic at all
```

`nanochat/scaling.py` (`derive_training_plan`) and `scripts/model_info.py` are outside `nanochat/model/`
but round out the same contract: the former is the training-horizon math `scripts/base_train.py`
and `scripts/model_info.py` both call, the latter reports every number in this doc (params, FLOPs,
KV bytes, training horizon) for any registered architecture purely from a meta-device build --
no GPU, no data, no training. See "Verifying a change is behavior-preserving" below.

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
    kv_slot: int | None = None    # None = "owns a slot at its own position"; see "Cross-layer
                                   # KV sharing" below for a layer that reuses another's slot

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
    def kv_cache_spec(self) -> dict          # {num_heads, head_dim, num_kv_slots}
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
attention with different head dims) should override `kv_cache_spec()` directly. `num_kv_slots` can
be *fewer* than `len(layer_specs())`: a layer whose `AttentionLayerSpec.kv_slot` points at an
earlier layer's slot doesn't get its own `KVCache` allocation (see "Cross-layer KV sharing" below).

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

`BaseModel.num_scaling_params()` has a generic default, also built on `collect_param_roles`:
`{role: numel, ..., "total": ...}`, one key per role name actually present. `GPT` overrides it to
sum `p.numel()` per role into a **fixed, six-key legacy dict** (`wte`, `value_embeds`, `lm_head`,
`transformer_matrices`, `scalars`, `total` — `scalars` sums three roles: `resid_scalar` +
`x0_scalar` + `smear`) — see "Two more load-bearing contracts" below for why that dict's exact key
names matter and can't just follow the role names. `Llama` has no override and just returns the
generic `{"embedding": ..., "matrix": ..., "unembedding": ..., "total": ...}` shape. Code that
wants the underlying counts without caring which dict shape applies (e.g.
`scripts/base_train.py`'s `get_scaling_params`, which needs "matrix + unembedding params") should
call `collect_param_roles(self)` directly and sum by role name — role names are stable across
architectures by construction; `num_scaling_params()`'s dict keys are a presentation layer GPT
happens to override.

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
   `base_train.py` raises a clear assertion telling you it's missing. Any field `from_depth`
   doesn't take a fixed kwarg for (e.g. `LlamaKVShareConfig.kv_share_frac`) is still reachable
   from the CLI via `--arch-opt field_name=value` (`scripts/base_train.py` and
   `scripts/model_info.py` both accept it; `nanochat.model.registry.apply_arch_opts` validates the
   field exists and parses the value with `ast.literal_eval`, so it never silently no-ops a typo).
3. `model.py`: implement `BaseModel`, reusing whatever fits from `nanochat/model/components/`.
   `CausalSelfAttention`/`MLP` take explicit dims (not a config object), so they're reusable by a
   config with different field names. Reach for `Block` directly (or `TokenEmbedding`/`LMHead`) if
   you're only changing depth/residual topology; write your own `BaseBlock` if you're changing
   attention itself or want a non-attention layer.
4. Add the import to `nanochat/model/__init__.py` so `@register_model` actually runs.
5. Declare a role for every parameter you introduce (`PARAM_ROLES` or `param_roles()`) — see
   "Parameter roles" above.
6. Add your architecture's kwargs to `tests/conftest.py`'s `TINY_KWARGS_BY_ARCH` — it then
   automatically gets a `tiny_<arch>` fixture and is included in the `tiny_model` fixture's
   `["gpt", "llama", ...]` parametrization, so `tests/test_model_common.py`'s architecture-generic
   suite (forward shape, loss finite, backward grad coverage, `setup_optimizer` partition,
   `layer_specs`/`kv_cache_spec` consistency) runs against it for free. Add a
   `tests/test_model_<arch>.py` for anything specific to your architecture.
7. Try it: `python -m scripts.base_train --arch=my_arch --depth=2 --num-iterations=3 ...` (see
   the CPU smoke-test invocation in "Verifying a change is behavior-preserving" below). If your
   architecture isn't `"gpt"`, checkpoints save under `<arch>_d<depth>` by default (see
   "Checkpoint tags and architecture-aware discovery" below), so a same-depth `gpt` run won't
   collide with it.

### Worked example: `llama`

`nanochat/model/llama/` (SwiGLU MLP, plain pre-norm blocks, none of GPT's value embeddings /
smear / backout / per-layer resid-x0 lambdas) exists specifically to test whether the contracts
above are real interfaces or just GPT with extra indirection. What it reused verbatim from
`nanochat/model/components/`: `CausalSelfAttention` (GQA/RoPE/QK-norm are not GPT-specific —
constructed with `has_value_embed=False`), `RotaryEmbedding`, `TokenEmbedding` (constructed with
`smear=False`), `LMHead`, and (since Stage 4) `SwiGLUMLP`/`PlainBlock` themselves — both now live
in `nanochat/model/components/mlp.py` and `block.py` alongside GPT's `MLP`/`Block`, since
`llama_kvshare` (below) needed `PlainBlock` too and a second consumer is what earns a component its
place there (`PlainBlock`'s `x0` argument is accepted per the `BaseBlock` contract but unused —
this topology has no `x0` residual). `Llama` itself (`nanochat/model/llama/model.py`) needed **no
`PARAM_ROLES` declarations anywhere** in its whole tree: every parameter it owns is either a
`Linear` weight (defaults to role `"matrix"`) or reused directly from a GPT component that already
declares its own roles. It also needed no `patch_config_dict`/`patch_state_dict`/
`patch_optimizer_state_dict` overrides — a brand-new architecture has no legacy checkpoints, so
`BaseModel`'s no-op defaults are exactly correct. This is what the contracts are supposed to make
possible: a second architecture is mostly reuse, with new code only where it's genuinely different.

## Cross-layer KV sharing

`nanochat/model/llama_kvshare/` (Gemma-3n-style) makes the last `kv_share_frac` fraction of layers
reuse an earlier layer's K/V instead of computing their own — fewer parameters (no `c_k`/`c_v` on
those layers), less prefill compute, and a smaller KV cache, all at the same depth. It's also the
first architecture to break an assumption that used to be baked into three places: "one KV-cache
slot per layer." Fixing that generically (so GPT and Llama, which still are one-slot-per-layer,
stay byte-identical) is what this section documents.

**The slot/layer decoupling.** `AttentionLayerSpec.kv_slot` (see above) identifies which `KVCache`
slot a layer's K/V lives in, separate from its position in `layer_specs()`. `BaseModel.kv_cache_spec()`
counts *distinct* slots (`num_kv_slots`), which `nanochat.engine.KVCache` allocates (`KVCache.n_slots`,
`KVCache.get_slot_cache(slot)` — both renamed from `n_layers`/`get_layer_cache` in this stage, the
forcing function that found every reader). `nanochat.model.components.kv_sharing.compute_kv_slots(n_layer,
kv_share_frac)` is the assignment: the first `n_layer - round(n_layer * kv_share_frac)` layers each
own a slot at their own index; every later layer's slot is the last owning layer's.

**`CausalSelfAttention` gained `kv_slot`/`produces_kv` constructor kwargs** (both default to
today's one-slot-per-layer behavior) and a `kv_bus=None` forward kwarg. A producer layer
(`produces_kv=True`, the default) computes `k`/`v` as before, then — if given a `kv_bus` dict —
writes `kv_bus[self.kv_slot] = (k, v)` after RoPE/QK-norm/the ×1.2 scale, i.e. exactly the tensors
it's about to attend with. A consumer layer (`produces_kv=False`, so no `c_k`/`c_v` at all) reads
`k, v = kv_bus[self.kv_slot]` and only projects/rotates its own queries
(`RotaryEmbedding.apply_to_q`). Both then call the *same* `flash_attn_with_kvcache(q, k_cache,
v_cache, k=k, v=v, ...)` — a model wires this by threading one `kv_bus = {}` dict through its block
loop each forward pass (see `LlamaKVShare.forward`); GPT and Llama simply never pass one, so nothing
about them changes.

**Why the consumer passes the producer's own `k`/`v` back in, instead of `k=None`** (a real trap,
worth knowing if you touch this code): `flash_attn_with_kvcache(k=None)` means different things on
the two backends. `nanochat/flash_attention.py`'s SDPA fallback reads `k_cache[:, :pos+T_new]`
regardless of whether `k`/`v` were passed — it doesn't care. FA3's real kernel, given `k=None`,
uses `seqlen_k = cache_seqlens`, which *excludes* the tokens the producer just wrote earlier in the
same forward pass, silently misaligning attention by `T_new` positions. Re-passing the producer's
own tensors sidesteps this: the cache write becomes a value-idempotent no-op (those exact tensors
are already there), and both backends then behave identically to the producer's own call. This is
also why `kv_cache.advance()` moved out of `CausalSelfAttention.forward` entirely (it used to fire
on `layer_idx == kv_cache.n_layers - 1`, which breaks the moment `n_layers` != slot count) and into
each model's own `forward`, called once after the whole block loop.

**`nanochat/model/flops.py`'s `kv_bytes_per_token`** now sums one contribution per *distinct* slot
(`distinct_kv_specs`), not per layer — otherwise it would over-report stored KV bytes by exactly
the sharing factor, the one number this architecture exists to shrink. `kv_read_bytes` stays
per-layer: a consumer layer still issues its own read of the shared cache during its own attention
call, so its DRAM traffic isn't smaller just because it doesn't own the slot.

### Worked example: `llama_kvshare`

`nanochat/model/llama_kvshare/` is ~90 lines total across `config.py`/`model.py`. `LlamaKVShareConfig`
adds one field (`kv_share_frac: float = 0.5`) to `LlamaConfig`; `from_depth` is inherited
unchanged (it builds via `cls(...)`, so it just carries the extra field's default through — reach
it with `--arch-opt kv_share_frac=0.667`). `LlamaKVShare.__init__` calls `compute_kv_slots(n_layer,
kv_share_frac)` once, then builds each `PlainBlock` with that layer's `kv_slot` and
`produces_kv=(slot == layer_index)` — otherwise identical to `Llama.__init__`. `forward` threads a
fresh `kv_bus = {}` through the block loop and calls `kv_cache.advance(...)` once at the end. Like
`llama`, it needs **no `PARAM_ROLES` declarations** (a consumer block simply has fewer `Linear`
submodules than a producer one — nothing new to declare either way) and no migration hooks.

### Worked example: `llama_kvshare_win`

`nanochat/model/llama_kvshare_win/` is the payoff case for the component contracts: a whole fourth
contest entrant costs one config subclass and a one-line `model.py`. `LlamaKVShareWinConfig`
(`nanochat/model/llama_kvshare_win/config.py`) changes exactly one thing from
`LlamaKVShareConfig` — `window_pattern` defaults to `"SSSL"` instead of `"L"` — plus a `from_depth`
override that exists *only* to change that same default in the inherited classmethod's own
signature (`LlamaConfig.from_depth` hardcodes `window_pattern="L"`, so the dataclass field default
alone wouldn't reach a `--depth`-driven run). `nanochat/model/llama_kvshare_win/model.py` is a
`class LlamaKVShareWin(LlamaKVShare): pass` under `@register_model("llama_kvshare_win", ...)` — no
new model logic at all, because `LlamaKVShare.__init__` already reads `config.window_pattern` and
passes it to `compute_window_sizes` (see "Worked example: `llama_kvshare`" above); the sliding-window
attention itself is `CausalSelfAttention`'s existing `window_size` handling, shared by every
architecture. It subclasses `LlamaKVShare` (rather than re-registering it under a second name) so
`get_model_class("llama_kvshare_win")`, tracebacks, and `repr`s all name the real class.

Windowing and cross-layer KV sharing compose without any extra wiring because they act at different
points: KV sharing decides *which layer's* K/V a given layer attends to (`kv_slot`, wired once in
`LlamaKVShare.__init__`), while the window is a *mask applied at attention time* by the consuming
layer itself (`CausalSelfAttention.forward`'s `window_size=(self.window, 0)`), not a truncation of
what the producer stores. A short-window consumer layer sharing a long-window producer's K/V is
therefore fine in both the training path (`kv_bus`, a full uncropped K/V handed to every consumer,
each applying its own window at attention time) and the cached-inference path
(`kv_cache.get_slot_cache`, same reasoning).

## Checkpoint tags and architecture-aware discovery

A checkpoint's directory name (its "tag") defaults to `d<depth>` for `gpt`, and
`<arch>_d<depth>` for anything else (`scripts/base_train.py`; unaffected by an explicit
`--model-tag`). This keeps `gpt`'s existing checkpoints' names unchanged, while giving every other
architecture a distinct default — without it, `--arch=llama --depth=2` after a `--arch=gpt
--depth=2` run would land in the *same* directory and overwrite the first run's files (not just
confuse auto-discovery; genuine data corruption, since both would write `model_005000.pt` etc. to
the same path).

Callers that don't pass an explicit tag (`model_tag=None`) fall back to
`checkpoint_manager.find_largest_model`, which by default just picks the largest `d<number>`
directory it finds — regardless of architecture. Pass `arch=` (to `find_largest_model` itself, or
through `load_model`/`load_model_from_dir`/`load_optimizer_state`, which all accept and forward
it) to filter candidates to that architecture first: it peeks at each candidate tag's latest
`meta_*.json` (`model_config.arch`, defaulting to `"gpt"` for checkpoints predating that key) via
`checkpoint_manager._checkpoint_arch`, no directory-naming assumption required. `scripts/
base_train.py` and `scripts/base_eval.py` both have a `--arch` flag wired to this.

`scripts/chat_sft.py` now has the same `--arch` flag, threaded into its `load_model`/
`load_optimizer_state` calls, and arch-qualifies its own output tag the same way `base_train.py`
does (`chatsft_checkpoints/<arch>_d<depth>/`, avoiding the cross-architecture collision that would
otherwise land two different architectures' SFT checkpoints in the same directory). It also stamps
the resolved base checkpoint's tag/step into the SFT checkpoint's own meta, so a chat checkpoint is
traceable back to the base run it came from without needing the contest's CSV. Not yet done: the
rest of the pipeline (`chat_rl.py`, `chat_cli.py`, `infer_bench.py`, `chat_eval.py`) still doesn't
pass `arch=` anywhere, so their auto-discovery stays architecture-blind — fine as long as only one
architecture's checkpoints exist under a given `*_checkpoints/` directory at a time (or the caller
passes an explicit `--model-tag`/`-g`, which every contest script does), but worth revisiting if
that stops holding.

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

- **`GPT.num_scaling_params()`'s six dict keys** (`wte`, `value_embeds`, `lm_head`,
  `transformer_matrices`, `scalars`, `total`) are greppable output: `runs/scaling_laws.sh` greps
  `^key ` lines out of `scripts/base_train.py`'s stdout dump of this dict, and
  `dev/scaling_analysis.ipynb` reads the resulting CSV columns. Keep these exact key names even if
  the underlying role names (see "Parameter roles" above) are more granular. Other architectures
  don't need to match this shape — it's specifically what GPT's override preserves; see "Parameter
  roles" above for the generic default every other architecture gets instead.
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

`scripts/model_info.py --arch gpt --depth 6` reports the same `num_scaling_params()`/`estimate_flops()`/
`kv_bytes_per_token()` numbers without touching a checkpoint at all (meta-device only) — a faster
first check when the change is purely about accounting, not weights.

Every value should match exactly. This is how Stage 1's `nanochat/gpt.py` -> `nanochat/model/`
extraction, and Stage 2's parameter-role/module-ownership restructure, were both checked end to
end. If the change also touches optimizer grouping (e.g. a `setup_optimizer` policy change), also
check that `load_optimizer_state("base", device, rank=0, model_tag="d6")`'s real shard loads into
a freshly built optimizer without error (a same-size group reorder passes size validation but
corrupts silently — see "Optimizer state is checkpointed positionally" above, and
`nanochat/model/gpt/migrations.py:patch_optimizer_state_dict` for the pattern if it doesn't).

For an end-to-end smoke test of the training path on CPU/MPS (add `--arch=llama`,
`--arch=llama_kvshare`, or any other registered architecture to exercise it instead — the same
command works unmodified; add `--arch-opt kv_share_frac=...` to vary the KV-sharing fraction):

```bash
python -m scripts.base_train --depth=2 --head-dim=32 --window-pattern=L --max-seq-len=128 \
  --device-batch-size=1 --total-batch-size=256 --num-iterations=3 \
  --eval-every=-1 --core-metric-every=-1 --sample-every=-1 --model-tag=smoke --run=dummy
```

Delete `~/.cache/nanochat/base_checkpoints/smoke` afterward — it's a throwaway.

A brand-new architecture has no golden checkpoint to diff against; verify it directly instead —
`model.setup_optimizer()`'s groups partition `model.parameters()` exactly (see
`tests/test_model_common.py`), a forward/backward pass produces finite output and populates every
gradient, and (per "Checkpoint tags and architecture-aware discovery" above) two architectures
smoke-trained at the same `--depth` land in different checkpoint directories and
`find_largest_model(dir, arch=...)` resolves each independently.
