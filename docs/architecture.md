# Architecture contract

Stage 7 (see [roadmap.md](roadmap.md)) split the model subsystem in two: `modelcore`, a
standalone package that knows only a materialized config tree and nothing about architecture
*names*, CLI flags, checkpoints, or tokenizers; and `nanochat/architectures/`, which turns a
`--depth` dial or an old checkpoint into a tree for `modelcore` to build. Stage 8 made `modelcore`
self-contained (its own tests, docs, and packaging metadata), and Stage 10 moved it into its own
repository, [8kb/modelcore](https://github.com/8kb/modelcore) — see
[modelcore's architecture.md](https://github.com/8kb/modelcore/blob/main/docs/architecture.md) for
the core contract itself, and this repo's `pyproject.toml`'s `[tool.uv.sources]` for how it's
pinned as a dependency here. This document covers the nanochat side: how the app consumes
`ModelManager` and (Stage 9) `datacore.DataManager`, how an old checkpoint gets there, and how to
verify a change is behavior-preserving.

```
modelcore                 separate repo (github.com/8kb/modelcore), pinned git dependency --
                          ModelManager (the one entrypoint), components, composers, catalog,
                          roles, stats, store, runtime, precision, optim, kernels; its own
                          tests/docs/README/pyproject. See its own docs/architecture.md.

nanochat/architectures/   everything that knows an architecture *by name*
├── derive.py                mup_dims, compute_window_sizes, compute_kv_slots, has_value_embed,
│                             gpt_lambda_schedule -- the depth-dial derivation rules
├── presets.py                expand(name, depth, ...) -> ModelConfig; assemble_gpt/assemble_plain
└── legacy.py                 migrate_checkpoint/migrate_optimizer_state -- old checkpoint -> current

nanochat/                 everything else
├── checkpoint_manager.py     naming policy (tags, steps) + meta.json extras; LegacyCheckpointStore
│                             adapts an old checkpoint onto ModelManager.load_model
├── engine.py                  inference: Engine (calculator/tool-use, KV-cached generate) built on
│                             modelcore.generate.Decoder; re-exports generate_naive/KVCache
├── optim.py, flash_attention.py   one-line re-export shims onto modelcore.optim/modelcore.kernels
└── scaling.py                 muP training-plan math (architecture-agnostic, untouched by Stage 7)
```

## Consuming `ModelManager`

Every script that needs a model goes through one `ModelManager` instance and never touches a
component, the catalog, or the meta-device dance directly:

```python
from modelcore import ModelManager, OptimizerHparams
manager = ModelManager()

config = presets.expand(args.arch, depth=args.depth, ...)   # nanochat.architectures.presets
model = manager.create_model(config, device=device, seed=0)
optimizer = manager.create_optimizer(model, OptimizerHparams(...))
stats = manager.stats(config)   # params/FLOPs/KV-cache, no weights needed
```

`scripts/base_train.py` no longer imports `modelcore.components.linear.Linear` directly — the fp8
eval swap-back it used to need that for goes through `manager.enable_fp8`/`manager.fp8_disabled`
instead (see
[modelcore's architecture.md](https://github.com/8kb/modelcore/blob/main/docs/architecture.md#fp8-precision)).
A comment near its `--fp8` handling still names `Linear`/`modelcore.precision.fp8` for context, but
the module itself imports only `Model`, `ModelManager`, `OptimizerHparams`, and
`modelcore.kernels.flash_attn.build_doc_args`.

## Consuming `DataManager`

Training and eval scripts read data through one `datacore.DataManager` instance, exactly the same
pattern as `ModelManager` — see [datacore/docs/architecture.md](https://github.com/8kb/datacore/blob/main/docs/architecture.md)
for the full contract. `nanochat`'s job is producing the dataset (once, offline, via
`scripts/data_prep.py`) and naming it; `DataManager` knows nothing about ClimbMix, SmolTalk, or
this fork's checkpoint conventions:

```python
from datacore import DataManager, FileSystemDatasetStore
manager = DataManager()
dataset = manager.open(FileSystemDatasetStore(dataset_dir))   # raises FileNotFoundError if unprepared

assert dataset.info.sequence_len == args.max_seq_len            # hard error otherwise, not a warning
assert dataset.info.tokenizer_fingerprint == tokenizer.fingerprint()

for inputs, targets, state in manager.batches(dataset, "train", args.device_batch_size,
                                               device=device, rank=ddp_rank, world_size=ddp_world_size,
                                               resume=dataloader_resume_state_dict, infinite=True):
    ...
```

`scripts/data_prep.py` is the preparation entrypoint (`--kind=base` for the pretraining corpus via
`nanochat.dataset`'s ClimbMix identity, `--kind=sft` for the `tasks/` mixture rendered through
`RustBPETokenizer.render_conversation`) — see its own docstring and
[datacore/docs/architecture.md](https://github.com/8kb/datacore/blob/main/docs/architecture.md) for the on-disk format, the
cursor-based resumable read order, and why `sequence_len`/tokenizer fingerprint mismatches raise
rather than warn. `nanochat/dataset.py` keeps only the corpus *identity* (`BASE_URL`, `MAX_SHARD`,
the local directory, the legacy `base_data` fallback) — the download mechanism is
`datacore.download`, and `parquets_iter_batched` (used by `scripts/tok_train.py`/`tok_eval.py`,
which run before any tokenizer — and therefore any `DataManager` — exists) stays put rather than
folding into a `datacore` source, since its row-group-level DDP striding is a different granularity
than `datacore.sources.ParquetDirectorySource`'s one-batch-per-file boundary.

## `nanochat.checkpoint_manager`: naming policy + the `ArtifactStore` adapter

`nanochat/checkpoint_manager.py` owns *naming* — which directory, which step, which tag, plus
`meta.json`'s extra fields (`val_bpb`, `user_config`, `tokenizer_fingerprint`, `dataloader_state`,
...) — and hands the actual model/optimizer bytes to `modelcore` through an `ArtifactStore`:

- `save_checkpoint`/`load_checkpoint` go through `modelcore.store.FileSystemStore` instead of raw
  `torch.save`/`torch.load`.
- `build_model` is `manager.load_model(LegacyCheckpointStore(...), device=..., train=...)`, where
  `LegacyCheckpointStore(FileSystemStore)` overrides `read_config`/`read_model_state` to run an old
  checkpoint through `nanochat.architectures.legacy.migrate_checkpoint` on first read (memoized) —
  `ModelManager` itself never learns legacy formats exist. The checkpoint's *raw* `model_config` in
  `meta.json` is left untouched on disk and read separately for `arch_of()`/tag naming, since a
  legacy checkpoint's original `"arch"` key is exactly the naming information `reference` (always
  unset by `legacy.migrate_config`) can't reproduce.

On-disk layout and file names are unchanged from before Stage 7 — this is naming/metadata policy
layered on top of `modelcore`'s artifact format, not a different format.

## `nanochat.engine`: `Engine` on top of `Decoder`

`nanochat/engine.py` keeps everything tokenizer/tool-use-shaped: `RowState`, the calculator
sandbox (`use_calculator`), and the hardcoded chat special tokens
(`<|python_start|>`/`<|assistant_end|>`/...). `Engine.generate` drives a
`modelcore.generate.Decoder` (via `self.manager.new_decoder(...)`) for the actual prefill/KV-cache/
model-stepping — `Engine` never allocates or clones a `KVCache` itself anymore.
`sample_next_token`/`generate_naive`/`KVCache` are re-exported from `modelcore` for existing
`from nanochat.engine import ...` call sites.

## `nanochat/architectures/`: presets and legacy migration

Everything that knows an architecture *by name* lives outside `modelcore`, since `modelcore` only
ever consumes an already-materialized tree.

**`derive.py`** — the depth-dial derivation rules, each with exactly one implementation:
`mup_dims` (the muP depth/aspect-ratio/head-dim dial), `compute_window_sizes`, `compute_kv_slots`,
`has_value_embed` (the old `has_ve`'s alternating-parity rule), `gpt_lambda_schedule` (the
per-layer resid/x0-lambda init decay).

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
`ModelConfig.from_dict` with no migration at all. `LegacyCheckpointStore` always routes through
`legacy.migrate_checkpoint` first; it's a fast no-op for the current-format case.

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
post-migration optimizer state all match recordings taken before the Stage 7 refactor
(`tests/goldens/*.json`, captured by `dev/capture_model_goldens.py`).

## Checkpoint tags and architecture-aware discovery

The default save tag is `d<depth>` for `gpt`, `<arch>_d<depth>` otherwise, so two architectures at
the same `--depth` don't collide. `nanochat.checkpoint_manager.arch_of(model_config_dict)` is the
naming-policy helper backing this: a current-format config's `reference.preset` (defaulting to
`"custom"` for a hand-written tree with no reference block), or a legacy config's `"arch"` key
(defaulting to `"gpt"`). `find_largest_model(..., arch=...)` and `chat_sft.py`'s output-tag naming
both use it — there's no registry key to read on `modelcore.ModelConfig` itself; `arch_of` reads
the *raw* dict a checkpoint's meta.json carries, which is where that information still genuinely
lives.

## Verifying a change is behavior-preserving

`tests/goldens/*.json` (captured once, before Stage 7's refactor, by `dev/capture_model_goldens.py`;
live replay helpers moved to `tests/golden_helpers.py` at Stage 8) records — for `d6` (genuinely
pre-Stage-2), the four real d12 contest checkpoints, one chatsft checkpoint, and a seeded synthetic
model of every native architecture — the state-dict fingerprint, every accounting number,
greedy-generation token ids (both the naive and KV-cached paths), a forward-logits hash, and the
optimizer layout plus a full state-tensor digest. `tests/test_goldens.py` replays every one of them
against the current code; real-checkpoint cases skip automatically on a machine without
`~/.cache/nanochat` populated. (The four `tiny_composed_*` presets' own goldens moved to
`modelcore/tests/goldens/` at Stage 8, then back into `tests/goldens/` at Stage 10 once modelcore
became a separate repo — reproducing a pre-Stage-7 checkpoint is a nanochat regression concern, not
modelcore's, so modelcore's own extracted repo carries none of this data.)

For a change to shared code (anything `modelcore`/`datacore` provide, or `nanochat/architectures/`,
`nanochat/checkpoint_manager.py`, `nanochat/engine.py`):

```bash
python -m pytest tests/test_goldens.py tests/test_architectures.py -v
```

`tests/test_architectures.py` cross-checks every preset-equivalent tree's accounting and forward
output against the `tiny_composed_*` goldens two ways: through `presets.expand` +
`modelcore.ModelManager` directly (independent of `nanochat.checkpoint_manager`), and through
`nanochat.architectures.legacy.migrate_checkpoint` against the real on-disk checkpoints — the
authoritative proof that migration reproduces pre-refactor behavior exactly, including the
backout-lambda optimizer-state case above.

`scripts/model_info.py --arch gpt --depth 6` reports the same accounting numbers without touching
a checkpoint at all (meta-device only) — a faster first check when a change is purely about
accounting, not weights.

For an end-to-end smoke test of the training path on CPU/MPS (any preset; `--arch-opt
kv_share_frac=...` to vary the KV-sharing fraction), prepare a tiny dataset first (Stage 9 —
training reads a prepared dataset, not raw parquet, so this step is required now):

```bash
python -m scripts.data_prep --kind=base --dataset=smoke --sequence-len=128 --max-shards=2
python -m scripts.base_train --depth=2 --head-dim=32 --window-pattern=L --max-seq-len=128 \
  --device-batch-size=1 --total-batch-size=256 --num-iterations=3 --dataset=smoke \
  --eval-every=-1 --core-metric-every=-1 --sample-every=-1 --model-tag=smoke --run=dummy
```

Delete `~/.cache/nanochat/base_checkpoints/smoke` and `~/.cache/nanochat/prepared/smoke` afterward
— both are throwaway. `--resume-from-step` (pointing at that run's final step) exercises optimizer
save+load through `ModelManager` (and `FileSystemStore`) *and* the exact-cursor dataloader resume
through `DataManager` end to end.

For a change purely inside `modelcore` itself (a new component, a new precision scheme, ...), work
in its own repo, [8kb/modelcore](https://github.com/8kb/modelcore) — see
[its architecture.md](https://github.com/8kb/modelcore/blob/main/docs/architecture.md#verifying-a-change-is-behavior-preserving).
Its suite runs standalone there (`python -m pytest modelcore/tests -v`) and includes a mechanical
check (`test_standalone.py`) that it never grows a dependency back on nanochat. Once a change is
ready, bump this repo's `pyproject.toml`'s `[tool.uv.sources]` tag (or point it at a local clone
via `uv pip install -e ../modelcore` for the inner dev loop) and re-run this repo's own suite
against it. For a change purely inside `datacore` (a new packer, a format change, ...), same shape
in [8kb/datacore](https://github.com/8kb/datacore) — see
[its architecture.md](https://github.com/8kb/datacore/blob/main/docs/architecture.md#verifying-a-change-is-behavior-preserving)
(`python -m pytest datacore/tests -v`, its own `test_standalone.py`) — plus
`tests/test_data_packing_parity.py` in this repo to cross-check the packing algorithms against
`dev/capture_data_goldens.py`'s frozen pre-datacore reference.
