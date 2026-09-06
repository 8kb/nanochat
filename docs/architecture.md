# Architecture contract

Stage 7 (see [roadmap.md](roadmap.md)) split the model subsystem in two: `modelcore/`, a
standalone package that knows only a materialized config tree and nothing about architecture
*names*, CLI flags, checkpoints, or tokenizers; and `nanochat/architectures/`, which turns a
`--depth` dial or an old checkpoint into a tree for `modelcore` to build. `modelcore` has zero
`nanochat` imports — it is a directory move away from being a separate package entirely.

```
modelcore/               standalone model subsystem
├── manager.py              ModelManager -- the one entrypoint
├── model.py                Model -- the one model class, built from a config tree
├── config/
│   ├── spec.py                ComponentSpec, ModelConfig, AttentionLayerSpec
│   └── validate.py            validate_config() -- structural + component-owned semantic checks
├── catalog.py               component registry: "#type" name -> (cls, needs, validate)
├── components/               linear, norm, rope, rotary, attention, mlp, block, embedding, unembedding
├── composers/                 base, stack, backout
├── roles.py                  parameter-role protocol (optimizer grouping)
├── stats.py                  FLOPs/param/KV-bytes accounting, ModelStats
├── store.py                  ArtifactStore protocol + FileSystemStore
├── runtime.py                Runtime (compute dtype, log sink) -- injected, not a global
├── optim/                     MuonAdamW
├── kernels/                   FA3/SDPA flash-attention interface
└── cache.py                   KVCache

nanochat/architectures/   everything that knows an architecture *by name*
├── derive.py                mup_dims, compute_window_sizes, compute_kv_slots, has_value_embed,
│                             gpt_lambda_schedule -- the depth-dial derivation rules
├── presets.py                expand(name, depth, ...) -> ModelConfig; assemble_gpt/assemble_plain
└── legacy.py                 migrate_checkpoint/migrate_optimizer_state -- old checkpoint -> current

nanochat/                 everything else
├── checkpoint_manager.py     naming policy (tags, steps) + meta.json extras; hands ModelManager
│                             a migrated config and lets it build the model
├── engine.py                  inference: Engine (KV-cached generate), generate_naive; KVCache
│                             re-exported from modelcore.cache
├── optim.py, flash_attention.py   one-line re-export shims onto modelcore.optim/modelcore.kernels
└── scaling.py                 muP training-plan math (architecture-agnostic, untouched by Stage 7)
```

## `ModelManager`: the one entrypoint

Everything a caller needs — create, load, save, or validate a model or its optimizer, or measure
a config's cost — goes through one object:

```python
class ModelManager:
    # config
    def config_from_dict(self, d: dict) -> ModelConfig
    def config_to_dict(self, config: ModelConfig) -> dict
    def validate_config(self, config: ModelConfig) -> ValidationReport   # every error, not just the first

    # create
    def create_model(self, config, *, device, seed: int | None = None) -> Model
    def create_optimizer(self, model, hparams: OptimizerHparams | None = None) -> MuonAdamW

    # load / save
    def load_model(self, store, *, device, config=None, train: bool = False) -> Model
    def load_optimizer(self, model, store, *, rank=0, hparams=None) -> MuonAdamW | None
    def save_model(self, model, store) -> None
    def save_optimizer(self, optimizer, store, *, rank=0) -> None

    # stats & runtime helpers
    def stats(self, config: ModelConfig) -> ModelStats
    def new_kv_cache(self, model, *, batch_size, seq_len, device=None) -> KVCache
```

`create_model`/`load_model` own the meta-device dance (`torch.device("meta")` → `to_empty(device)`
→ `init_weights()`, and — for `load_model` — `load_state_dict(..., assign=True)` right after) so
no caller writes it themselves. Both call `validate_config` first and raise on any error — an
invalid tree never reaches `Model.__init__`.

`Model` itself (`modelcore/model.py`) is deliberately thin: `__call__(idx, targets=None,
kv_cache=None, loss_reduction=...)`, `.config`, `.get_device()`. It carries **no** accounting or
optimizer methods — no `layer_specs()`, `kv_cache_spec()`, `estimate_flops()`,
`num_scaling_params()`, `setup_optimizer()`. Those need a model only to read shapes/roles, which
`ModelManager.stats()`/`create_optimizer()` do from the outside. This is deliberate: a model
object answers "what do I compute", not "how much does that cost" or "how do I optimize myself".

## The materialized config tree

There is exactly one config shape `modelcore` understands (`modelcore/config/spec.py`):

```python
@dataclass
class ComponentSpec:
    type: str            # the catalog's "#type" name
    params: dict          # already-concrete constructor kwargs; may nest more ComponentSpecs

@dataclass
class ModelConfig:
    sequence_len: int
    vocab_size: int
    n_embd: int                    # the only truly global, uniform-across-layers value
    pad_vocab_size_to: int = 64
    reference: dict | None = None  # optional provenance: {"preset": name, "kwargs": {...}}
    shared: dict = {}              # name -> ComponentSpec, e.g. {"rope": ...}
    input: ComponentSpec | None = None    # embedding
    body: ComponentSpec | None = None     # the composer (a stack of blocks, or nested composers)
    output: ComponentSpec | None = None   # unembedding
```

Every component — embedding, block, unembedding, composer, a shared thing like RoPE — is one
`ComponentSpec`: its type under a `"#type"` key, already-concrete constructor kwargs as siblings.
A composer is a component like any other; its per-layer block list is just one of its own params
(conventionally `"blocks"`), so nothing in this schema privileges "a stack of blocks" — a composer
with several block lists, or one nesting another composer, needs no schema change.
`ModelConfig.n_layer` is a derived property (`_count_blocks` sums every `"blocks"`-named list,
recursing into nested composers), not a stored field, since it can vary with tree content.

`to_dict()`/`from_dict()` stamp/read a `"format": "modelcore.v1"` key. A dict with no `"format"`
key predates modelcore entirely and needs `nanochat.architectures.legacy` first — see "Old
checkpoints" below. There is no `"arch"` field anywhere in this schema; `reference` (when present)
is the closest equivalent, and it's provenance, not something `modelcore` ever reads back to
decide how to build the tree.

### Only concrete, already-decided values — no rules

A config tree carries no *derivation rules*, only their already-computed output. Every value that
used to be a rule lives in `nanochat/architectures/derive.py` instead, run once at tree-expansion
time:

- `has_value_embed`: a plain `bool` a `gpt_block`'s params carry directly — not `None` meaning
  "derive the alternating-by-parity pattern from `n_layer`". `Block` (the class backing
  `"gpt_block"`) doesn't take `n_layer` at all, because it never needs to re-derive anything.
- `window`: a concrete int (or `-1` for full context) per block — not a pattern string like
  `"SSSL"` tiled at construction time.
- `kv_slot`/`produces_kv`: concrete per-block values — not a `kv_share_frac` float a component
  would need to interpret.

This is the fix for the abstraction leak the whole redesign started from: a component like
`CausalSelfAttention` cannot have a method like the old `has_ve(layer_idx, n_layer)` (a policy
about *which* layers get a value embedding), because by the time modelcore ever sees a config,
that decision is already made. Materializing it is `nanochat/architectures/presets.py`'s job.

## Component contracts

Three module contracts (`modelcore/components/contracts.py`), each owning everything about its
own concern and nothing about how it's assembled into a model:

```python
class BaseEmbedding(nn.Module):
    def init_weights(self): ...
    def forward(self, idx, kv_cache=None): ...

class BaseBlock(nn.Module):
    def init_weights(self): ...
    def forward(self, x, x0, idx, kv_cache, kv_bus=None): ...
    def layer_spec(self): return None   # AttentionLayerSpec, or None if not attention-shaped

class BaseUnembedding(nn.Module):
    def init_weights(self): ...
    def forward(self, x, targets=None, loss_reduction="mean"): ...
```

Plus one more, for whatever owns `body` (`modelcore/composers/base.py`):

```python
class BaseComposer(nn.Module):
    def init_weights(self): ...
    def forward(self, x, idx, kv_cache): ...
    def layer_specs(self): ...   # list[AttentionLayerSpec], in forward-pass order
```

A component may know it must implement one of these contracts; it must not know anything about
*who* wires it in. Position encoding is deliberately not baked into `BaseBlock.forward`'s
signature (an architecture can swap RoPE for something else without touching the block/trunk
contract); `x0` (the post-embedding residual) is computed by whichever composer needs it
(`BackoutComposer`), not by the embedding.

### Every parameter needs a declared role

Same protocol as before Stage 7, moved to `modelcore/roles.py`: a module declares
`PARAM_ROLES = {attr_name: role_string}` for parameters/submodules it directly owns; a `Linear`'s
`.weight` defaults to role `"matrix"` when undeclared; anything else undeclared raises rather than
silently defaulting into the wrong optimizer bucket. `collect_param_roles(module)` walks the tree;
the escape hatch `param_roles(self) -> dict[str, list[Parameter]]` lets a module (e.g. a tied
`LMHead`) own its whole subtree's assignment in one place.

`build_param_groups(role_params, policy)` turns `{role: [params]}` plus an **ordered**
`{role: hyperparameters}` policy into `MuonAdamW` param groups — the order is the on-disk
optimizer format (state is checkpointed and reloaded positionally by flat index across every
group). `ModelManager.create_optimizer` owns the one policy table every model in `modelcore` uses,
covering every role any cataloged component can produce (`build_param_groups` skips a policy role
with no params present, so a plain `llama`-shaped tree's optimizer groups look exactly like they
always did — nothing extra).

## The catalog

`modelcore/catalog.py` maps a `"#type"` name to `(cls, needs, validate)`. A component
self-registers at its own class definition:

```python
@register_component("gpt_block", needs=("n_embd", "padded_vocab_size", "rope", "runtime"),
                     validate=_validate_gpt_block)
class Block(BaseBlock):
    ...
```

`needs` names build-context values injected as constructor kwargs — derived globals
(`n_embd`, `vocab_size`, `padded_vocab_size`, `sequence_len`, `runtime`) that `Model.__init__`
computes once, plus `shared` components (built once and injected by name, e.g. `rope`) — so a
spec's `params` only ever needs to carry what's *not* derivable from context. `build_component`
resolves any nested `ComponentSpec` (or list of them) first, then calls `cls(**resolved_params,
**needed)`.

`modelcore/components/__init__.py` and `modelcore/composers/__init__.py` import every
component/composer module so its decorator runs; importing `modelcore` (or `modelcore.manager`)
imports both, so the catalog is always fully populated by the time a caller reaches
`ModelManager`.

### Validation

`validate_config(config)` never raises and never stops at the first problem — it returns a
`ValidationReport` with every error found, each anchored to a path (`body.blocks[3].n_kv_head`,
`shared.rope.head_dim`, `input.#type`):

- **Structural checks (core-owned, generic across any component)**: `input`/`body`/`output`
  present; every `#type` registered; every `needs` name available in the build context; no unknown
  or missing constructor params (checked via `inspect.signature`).
- **Semantic checks (component-owned, via the catalog's `validate` hook)**: `n_embd` divisible by
  `n_head`, a KV-sharing consumer (`produces_kv=False`) can't have `has_value_embed=True`,
  `produces_kv=False` requires an explicit `kv_slot` — things only the component knows the rule
  for. `modelcore/components/block.py`'s `_validate_gpt_block`/`_validate_plain_block` are the
  reference implementation.
- **Cross-layer checks (core-owned, but generic over any composer's block list, not hardcoded to
  one composer type)**: KV slots form a contiguous `0..M-1` range, a consumer's `kv_slot` points
  at an earlier producer, `n_kv_head`/`head_dim` are uniform across every attention-shaped block —
  computed structurally from block params, without building a real model.

`create_model`/`load_model`/`stats` all call `validate_config` first and raise `ValueError` on any
error, so an invalid tree is caught before a single tensor is allocated.

## `ModelStats` and accounting

`ModelManager.stats(config)` builds a model on `torch.device("meta")` (shapes/dtypes only, no real
weight values ever allocated — cheap regardless of model size) and returns a frozen snapshot:

```python
@dataclass(frozen=True)
class ModelStats:
    n_layer: int
    params_by_role: dict          # generic role -> numel, for every architecture uniformly
    num_params: int
    num_matmul_params: int
    layer_specs: list[AttentionLayerSpec]
    kv_cache_spec: dict            # what modelcore.cache.KVCache needs to allocate
    shape_summary: dict            # n_layer/n_embd/n_head/n_kv_head/sequence_len/window_pattern
    flops_per_token: int
    has_sliding_window: bool

    @property
    def num_scaling_params(self) -> int: ...   # matrix + unembedding roles (cleanest scaling laws)
    def decode_flops(self, context_len): ...
    def prefill_flops(self, num_tokens): ...
    def kv_bytes_per_token(self): ...
    def kv_read_bytes(self, context_len): ...
```

`shape_summary` reports a concrete value for `n_head`/`n_kv_head`/`window_pattern` when every
layer agrees, else the string `"mixed"` — one implementation for every tree, uniform or not,
rather than a separate "flat config" code path. `params_by_role` is the generic role-keyed dict
for *every* architecture — there's no more architecture-specific override (GPT used to present a
frozen legacy six-key dict; that presentation now lives one layer up, in
`scripts/base_train.py`'s `_legacy_scaling_keys`, which reads `params_by_role` and maps it to the
exact key names `runs/scaling_laws.sh` still greps).

`AttentionLayerSpec.window = -1` means unlimited/full context; a non-negative int is the number of
preceding tokens attended to. `kv_slot = None` means "this layer owns a slot at its own position";
a layer that reuses an earlier layer's K/V sets it to that layer's slot instead — see "Cross-layer
KV sharing" below.

## `ArtifactStore`: how a model/optimizer gets its bytes

`modelcore/store.py` defines a narrow protocol — read/write a model state dict, an optimizer state
dict per rank, a config dict — and `FileSystemStore`, the default implementation: one checkpoint
directory + step, matching `nanochat`'s existing on-disk layout exactly (`model_{step:06d}.pt`,
`meta_{step:06d}.json`'s `"model_config"` key, `optim_{step:06d}_rank{N}.pt}`) so wiring it in
changed no file format. `write_config` merges into the meta.json's `"model_config"` key rather
than overwriting the file, since `nanochat.checkpoint_manager` writes its own sibling keys
(`val_bpb`, `user_config`, `tokenizer_fingerprint`, ...) into the same file — each side only ever
touches the key(s) it owns.

`nanochat.checkpoint_manager` doesn't actually route through `FileSystemStore` for its own
richer format (it needs finer control over meta.json's extra fields than the generic store
contract offers) — it calls `ModelManager.create_model`/`config_to_dict`/
`nanochat.architectures.legacy.migrate_checkpoint` directly, using raw `torch.save`/`torch.load`
for the tensor I/O exactly as before. `FileSystemStore` is there for a caller (or a future
standalone user of `modelcore`) with a plain directory and no need for that richness — see
`tests/test_modelcore.py`'s save/load round-trip tests for the intended usage.

## `Runtime`: no more ambient globals

`modelcore/runtime.py`'s `Runtime` carries the values a component needs that aren't part of the
config: `compute_dtype` (replacing the old bare `nanochat.common.COMPUTE_DTYPE` global) and a log
sink. A component that needs it declares `needs=("runtime",)`, same mechanism as `rope` or
`n_embd`. `nanochat.common.COMPUTE_DTYPE`/`COMPUTE_DTYPE_REASON` now source from
`modelcore.runtime.DEFAULT_RUNTIME` (the direction of the dependency reversed: `modelcore` still
has zero `nanochat` imports, so everything else adapts to it) — `NANOCHAT_DTYPE` and every
existing reader of those two names keep working unchanged.

## Adding a component, step by step

1. Write the `nn.Module`, obeying whichever contract it fits (`BaseEmbedding`/`BaseBlock`/
   `BaseUnembedding`/`BaseComposer`) — take explicit constructor kwargs, not a config object, so
   it doesn't assume any particular tree shape around it.
2. Declare `PARAM_ROLES` for every parameter it directly owns (or a `param_roles()` override); a
   `Linear`'s weight needs no declaration.
3. Register it: `@register_component("my_thing", needs=(...))`, naming exactly the build-context
   values its constructor needs beyond what's in the spec's own `params`. Add a `validate=`
   function if it has real constructor-level constraints worth catching before a model is built.
4. Import the module from `modelcore/components/__init__.py` (or `composers/__init__.py`) so the
   decorator runs.
5. If it needs testing at the tree level rather than in isolation, add a case to
   `tests/test_modelcore.py`'s `FLAVORS` dict, or a preset in `nanochat/architectures/presets.py`
   if it should be reachable via `--arch`/`--model-config`.

## Cross-layer KV sharing

Unchanged in mechanism from before Stage 7, just materialized instead of derived at model-build
time. `AttentionLayerSpec.kv_slot` decouples layer index from KV-cache slot:
`kv_cache_spec()["num_kv_slots"]` (via `modelcore.stats.kv_cache_spec`) can be less than the layer
count when a layer's `kv_slot` points at an earlier layer's slot. `KVCache`'s constructor kwarg
and attribute are `num_kv_slots`/`n_slots`, and `get_slot_cache(slot)` returns that slot's
`(k_cache, v_cache)` view.

A layer built with `produces_kv=False` has no `c_k`/`c_v` at all; at forward time it reads an
earlier layer's already-RoPE'd/normed/scaled K/V out of a `kv_bus` dict (threaded through one
composer's forward pass) instead of computing its own — it only projects and rotates its own
queries (`RotaryEmbedding.apply_to_q`). Passing the producer's own K/V tensors back into
`flash_attn_with_kvcache` for the consumer (rather than `k=None`) sidesteps a real FA3-vs-SDPA
divergence in what `k=None` means: with a real cache, `k=None` tells FA3 "nothing new to insert
this call", so it reads exactly `cache_seqlens` cached tokens — correct for the producer, which
already wrote the shared slot earlier in the *same* forward pass, but wrong by exactly the new
token count for a consumer if it relied on `k=None` too.

`kv_cache.advance()` belongs to `Model.forward`, called once after the whole block/composer loop
runs — not to any one attention layer (a same-layer-count assumption breaks the moment a model has
fewer KV slots than layers).

`nanochat.architectures.derive.compute_kv_slots(n_layer, kv_share_frac)` is the one place the
"last `kv_share_frac` fraction of layers reuse the last KV-owning layer's slot" policy lives —
called once by `presets.py` (or `legacy.py`, reconstructing from a checkpoint's stored
`kv_share_frac`) to materialize concrete `kv_slot`/`produces_kv` values into the tree.

## `nanochat/architectures/`: presets and legacy migration

Everything that knows an architecture *by name* lives outside `modelcore`, since `modelcore` only
ever consumes an already-materialized tree.

**`derive.py`** — the depth-dial derivation rules, each with exactly one implementation now
(previously duplicated up to three times, or embedded inside a component): `mup_dims` (the muP
depth/aspect-ratio/head-dim dial), `compute_window_sizes`, `compute_kv_slots`, `has_value_embed`
(the old `has_ve`'s alternating-parity rule), `gpt_lambda_schedule` (the per-layer resid/x0-lambda
init decay).

**`presets.py`** — `expand(name, depth, **kwargs) -> ModelConfig` reproduces exactly what each of
the four deleted native architecture classes' own `from_depth` + `__init__` used to build, as a
concrete tree instead of code. `assemble_gpt`/`assemble_plain` factor out the actual tree assembly
(looping over layers, building `ComponentSpec`s) so `presets.py` (deriving dims from a depth dial)
and `legacy.py` (reading them straight off a stored checkpoint's fields) share one implementation
of "how do I build the tree", differing only in "where do the numbers come from".
`resolve_model_config(model_config, depth, ...)` dispatches a `--model-config` value between a
registered preset name and a path to a materialized JSON tree; `resolve_reference_config` re-
derives the muP d12 scaling-law reference model from a resolved config's own `reference` block.

**`legacy.py`** — `migrate_checkpoint(config_dict, model_data, log=...) -> (ModelConfig, dict)` and
the separate `migrate_optimizer_state(...)` (optimizer state loads independently of model state,
so it's migrated independently too) handle every checkpoint generation on disk. See "Old
checkpoints" below for the details.

## Old checkpoints

A `model_config` dict with no `"format"` key predates modelcore entirely — everything below
applies; one that already has `"format": "modelcore.v1"` passes straight through
`ModelConfig.from_dict` with no migration at all. `nanochat.checkpoint_manager.build_model` always
routes through `legacy.migrate_checkpoint` first; it's a fast no-op for the current-format case.

**Config**: missing `"arch"` defaults to `"gpt"` (the only architecture old enough to predate that
key too); missing `"window_pattern"` defaults to `"L"`. Stage 6's `"composed"` architecture is
already a materialized tree in exactly modelcore's shape (it predates modelcore only in name,
stamping `"arch": "composed"` where modelcore stamps `"format"`) — `ModelConfig.from_dict` reads
it directly, ignoring the leftover `"arch"` key it never looks at. Every other architecture is a
flat, per-layer-derivable config: `legacy.py` reconstructs the exact materialized tree from the
checkpoint's own *stored* fields (`n_layer`, `n_head`, `n_kv_head`, `n_embd`, `window_pattern`,
`kv_share_frac`) via the same `derive.py` helpers `presets.py` uses — not from `--depth`, so a run
that used `--arch-opt` to diverge from the muP dial's defaults still migrates losslessly.

**State dict**: a pre-Stage-2 (2026) flat GPT layout (`transformer.wte.weight`,
`transformer.h.{i}.*`, `resid_lambdas`/`x0_lambdas` as single `[n_layer]` tensors,
`smear_gate.weight`/`smear_lambda`, `value_embeds.{i}.weight`) is renamed to the post-Stage-2
per-module layout first (`patch_gpt_state_dict_layout`; a no-op if already in that layout,
detected by `"transformer.wte.weight"` presence). Every checkpoint then gets the `body.` prefix
Stage 6/7 introduced (`blocks.{i}.*` → `body.blocks.{i}.*`, `backout_lambda` →
`body.backout_lambda`) via `patch_body_prefix` — a no-op if already prefixed.

**Optimizer state** (two independent fixes, both flat-index positional since
`torch.optim.Optimizer.state_dict()` numbers every parameter sequentially across every group):

1. `resid_lambdas`/`x0_lambdas` were each a single `[n_layer]` parameter (one flat index); they
   became `n_layer` independent per-block scalars. `_patch_resid_x0_split` splits that group's
   per-index state into `n_layer` entries and renumbers every later flat index to make room. A
   no-op if the resid group already has more than one param (already in the new layout).
2. The old native `GPT.PARAM_ROLES` mapped `backout_lambda` to role `"smear"` (grouped alongside
   `embedding.smear`'s gate/lambda_); `modelcore`'s `BackoutComposer` gives it its own role,
   `"backout_scalar"`, positioned immediately after `"smear"` in `ModelManager.create_optimizer`'s
   policy order. Because flat indices are assigned purely by walking every group's params in
   sequence, `backout_lambda`'s flat index doesn't move at all when this happens — only the
   *group boundary* does, so `_split_backout_lambda_from_smear` only needs to split one group's
   param list into two, with **no** state renumbering. A no-op unless the smear group has the old
   3-member shape (gate, lambda_, backout_lambda) — i.e. this only ever applies to a native
   gpt-arch checkpoint; every other architecture's roles are unaffected, so its group count and
   flat indices are identical before and after this refactor.

Verified against every real checkpoint on the development machine (`d6` — genuinely pre-Stage-2,
no `"arch"` key at all — and four d12 contest runs) in `tests/test_architectures.py`: migrated
config, state dict, accounting numbers, forward logits, and (where a shard exists) the live
post-migration optimizer state all match recordings taken before this refactor
(`tests/goldens/*.json`, captured by `dev/capture_model_goldens.py`).

## Checkpoint tags and architecture-aware discovery

Unchanged in outward behavior: the default save tag is `d<depth>` for `gpt`, `<arch>_d<depth>`
otherwise, so two architectures at the same `--depth` don't collide.
`nanochat.checkpoint_manager.arch_of(model_config_dict)` is the new naming-policy helper backing
this: a current-format config's `reference.preset` (defaulting to `"custom"` for a hand-written
tree with no reference block), or a legacy config's `"arch"` key (defaulting to `"gpt"`).
`find_largest_model(..., arch=...)` and `chat_sft.py`'s output-tag naming both use it — previously
they read `model.config.arch`, an attribute that no longer exists on `modelcore.ModelConfig` at
all (there's no registry key to read; `arch_of` reads the *raw* dict a checkpoint's meta.json
carries, which is where that information still genuinely lives).

## The meta-device footgun

Unchanged: `Model.__init__` (and any component's `__init__`) may run under `torch.device("meta")`
— shapes and dtypes only, no real storage. `ModelManager.create_model`/`load_model` do this
internally so a caller never repeats the `to_empty(device)` → `init_weights()` dance. `RotaryEmbedding`'s
`cos`/`sin` buffers are `persistent=False` (never saved to a checkpoint) and only get real values
inside `init_weights()` — this is why `load_model` calls `init_weights()` even when *loading* a
checkpoint, right before `load_state_dict(..., assign=True)` overwrites everything else.

Each submodule owns its own `init_weights()` (called by its parent's, recursively) rather than one
function reaching into every submodule's internals by attribute path — this is what makes a
component usable inside a different tree without that tree needing to know the component's field
names. One consequence, carried over from before Stage 7: the exact sequence of RNG calls during a
from-scratch `init_weights()` is not guaranteed stable across a refactor of shared code (same
individual `torch.nn.init.*` calls, potentially different order) — this does not affect *loading*
an existing checkpoint (its saved values fully override whatever `init_weights()` produced), only
bit-for-bit reproducibility of a brand-new from-scratch run at a given seed. This is why
`tests/goldens/tiny/*`'s regression proof loads real saved weights into a freshly-built model
rather than comparing two independently-seeded `init_weights()` calls.

## Precision

Still one global-*feeling* value, `COMPUTE_DTYPE` — but it's `modelcore.runtime.Runtime.compute_dtype`
now, injected into any component that declares `needs=("runtime",)`, not read off a bare module
attribute. `nanochat.common.COMPUTE_DTYPE`/`COMPUTE_DTYPE_REASON` (overridable via `NANOCHAT_DTYPE`)
re-export the default runtime's value for every existing reader. Model weights stay fp32 (for
optimizer precision); `modelcore.components.linear.Linear` casts to `COMPUTE_DTYPE` in `forward()`.
Any new component should route its matmul weights through `Linear` rather than a raw `nn.Linear`,
both for this precision policy and so `modelcore.stats.num_matmul_params` sees it — it's the
structural marker that accounting scans for.

## Verifying a change is behavior-preserving

`tests/goldens/*.json` (captured once, before Stage 7's refactor, by `dev/capture_model_goldens.py`)
records — for `d6` (genuinely pre-Stage-2), the four real d12 contest checkpoints, one chatsft
checkpoint, and a seeded synthetic model of every architecture and preset — the state-dict
fingerprint, every accounting number, greedy-generation token ids (both the naive and KV-cached
paths), a forward-logits hash, and the optimizer layout plus a full state-tensor digest.
`tests/test_goldens.py` replays every one of them against the current code; real-checkpoint cases
skip automatically on a machine without `~/.cache/nanochat` populated.

For a change to shared code (anything under `modelcore/`, or `nanochat/architectures/`,
`nanochat/checkpoint_manager.py`, `nanochat/engine.py`):

```bash
python -m pytest tests/test_goldens.py tests/test_modelcore.py tests/test_architectures.py -v
```

`tests/test_modelcore.py` additionally cross-checks every preset-equivalent tree's accounting and
forward output against the same goldens directly through `modelcore.ModelManager`, independent of
`nanochat.checkpoint_manager`; `tests/test_architectures.py` does the same through
`nanochat.architectures.legacy.migrate_checkpoint` against the real on-disk checkpoints — the
authoritative proof that migration reproduces pre-refactor behavior exactly, including the
backout-lambda optimizer-state case above.

`scripts/model_info.py --arch gpt --depth 6` reports the same accounting numbers without touching
a checkpoint at all (meta-device only) — a faster first check when a change is purely about
accounting, not weights.

For an end-to-end smoke test of the training path on CPU/MPS (any preset; `--arch-opt
kv_share_frac=...` to vary the KV-sharing fraction):

```bash
python -m scripts.base_train --depth=2 --head-dim=32 --window-pattern=L --max-seq-len=128 \
  --device-batch-size=1 --total-batch-size=256 --num-iterations=3 \
  --eval-every=-1 --core-metric-every=-1 --sample-every=-1 --model-tag=smoke --run=dummy
```

Delete `~/.cache/nanochat/base_checkpoints/smoke` afterward — it's a throwaway. `--resume-from-step`
(pointing at that run's final step) exercises optimizer save+load through `ModelManager` end to
end.

A brand-new component or composer has no golden to diff against; verify it directly instead —
`manager.validate_config(config).ok`, `manager.create_optimizer(model)`'s groups partition
`model.parameters()` exactly (see `tests/test_modelcore.py`'s generic suite, parametrized over
every flavor), a forward/backward pass produces finite output and populates every gradient, and
two presets/configs at the same `--depth` land in different checkpoint directories.
