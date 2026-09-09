# AGENTS.md

Repo map and non-obvious invariants for anyone (human or agent) working in this fork. Read
[../llmllab/AGENTS.md](../llmllab/AGENTS.md) first for family-wide conventions (style, RunPod ops,
the standalone-subsystem pattern) that apply here too but aren't repeated below.
Read [modelcore's AGENTS.md](https://github.com/8kb/modelcore/blob/main/AGENTS.md) and
[architecture.md](https://github.com/8kb/modelcore/blob/main/docs/architecture.md) before touching
anything that depends on `modelcore`, [datacore's AGENTS.md](https://github.com/8kb/datacore/blob/main/AGENTS.md)
and [architecture.md](https://github.com/8kb/datacore/blob/main/docs/architecture.md) before
touching anything that depends on `datacore`, [docs/architecture.md](docs/architecture.md) before
touching `nanochat/architectures/` or how the app consumes `ModelManager`, and
[docs/upstream-sync.md](docs/upstream-sync.md) before touching anything that used to live in
`nanochat/gpt.py` (now deleted — see that doc's "Stage 7" section for where its code lives today).

`modelcore` and `datacore` are separate repositories
([8kb/modelcore](https://github.com/8kb/modelcore), [8kb/datacore](https://github.com/8kb/datacore)
— Stage 10, see docs/roadmap.md), consumed here as pinned git dependencies (`pyproject.toml`'s
`[tool.uv.sources]`), not directories in this tree. `uv sync` installs both into `.venv/`;
`import modelcore`/`import datacore` resolve from there. Working on either subsystem itself means
cloning its own repo (both are siblings of this one under `../`, see
[../llmllab/AGENTS.md](../llmllab/AGENTS.md)), not editing a copy inside this one — see each
repo's own README/AGENTS.md for its local dev loop, or use `uv pip install -e ../modelcore` (after
`uv sync`) to point this checkout at a local sibling clone for cross-repo development.

## What this fork is

A fork of [karpathy/nanochat](https://github.com/karpathy/nanochat) repurposed as a playground
for trying different model *architectures*, not just different hyperparameters. Original upstream
docs are preserved at `docs/upstream/`. See [docs/roadmap.md](docs/roadmap.md) for the staged
plan and current progress.

## Repo map

```
modelcore              -- separate repo (github.com/8kb/modelcore), pinned git dependency, not a
                          directory here. Model subsystem: ModelManager (the one entrypoint --
                          create/load/save model+optimizer, stats, validate, precision, decoding),
                          Model, config/ (ComponentSpec, ModelConfig), catalog.py's component
                          registry, components/, composers/, roles.py, stats.py, store.py,
                          runtime.py, precision/fp8.py, optim/ (MuonAdamW), kernels/ (FA3/SDPA),
                          cache.py (KVCache). See its own AGENTS.md/docs/architecture.md.
datacore               -- separate repo (github.com/8kb/datacore), pinned git dependency, not a
                          directory here. Data subsystem: DataManager (the one entrypoint --
                          prepare a dataset, open one, read batches), store.py, packing.py
                          (BestFitCropPacker/BestFitPadPacker), writer.py, reader.py (the only
                          module that imports torch, lazily), sources.py, download.py, tokenizer.py.
                          See its own AGENTS.md/docs/architecture.md.
nanochat/             everything that knows nanochat's own conventions
├── architectures/       expand a --depth dial (presets.py) or migrate an old checkpoint (legacy.py)
│                        into a modelcore.ModelConfig; derive.py holds the depth-dial derivation rules
├── engine.py             inference: Engine (calculator/tool-use, KV-cached generate) built on
│                        modelcore.generate.Decoder; generate_naive/KVCache re-exported
├── checkpoint_manager.py  naming policy (tags, steps) + meta.json extras; LegacyCheckpointStore
│                        adapts an old checkpoint onto ModelManager.load_model
├── optim.py, flash_attention.py   one-line re-export shims onto modelcore.optim/modelcore.kernels
├── tokenizer.py            BPE tokenizer wrapper (satisfies datacore.Tokenizer unmodified)
├── dataset.py               ClimbMix identity (URL, shard count, local dir) -- download/parquet
│                        mechanism lives in datacore.download/datacore.sources
├── core_eval.py / loss_eval.py   base-model evaluation (CORE benchmark, bits-per-byte)
├── execution.py            sandboxed Python execution (tool use)
└── scaling.py               muP training-plan math (architecture-agnostic)
scripts/              entry points, run as `python -m scripts.<name>`
├── data_prep.py            prepares a datacore dataset (--kind=base|sft); CPU-only, run before
│                        base_train.py/chat_sft.py, never on a billed GPU pod
└── ...
tasks/                task/dataset definitions for eval (arc, mmlu, gsm8k, humaneval, smoltalk)
tests/                nanochat's pytest suite — see "What runs on this Mac" below
                        (modelcore/datacore each own their own standalone suite, in their own repo)
runs/                 shell scripts wiring scripts/ together (speedrun.sh, runcpu.sh, ...)
docs/                 this fork's documentation; docs/upstream/ holds the original nanochat docs
dev/                  images, notebooks, dev/repackage_data_reference.py, dev/capture_model_goldens.py,
                        dev/capture_data_goldens.py (frozen pre-datacore packing-algorithm reference)
```

## Invariants owned elsewhere

These are `modelcore`'s or `datacore`'s own invariants, not this repo's — full explanation and
"why" live in their `AGENTS.md`s. One line each here for the local consequence:

- **Meta-device `__init__`, no `torch.amp.autocast`, `Linear` as the matmul marker, every
  parameter needs a declared role, a config tree carries only concrete values, optimizer state is
  positional, `kv_cache.advance()` belongs to `Model.forward`, `doc_args` built outside
  `torch.compile`, `build_doc_args`'s `max_docs` is dataset-tuned not worst-case,
  `AttentionLayerSpec.kv_slot`, `ArtifactStore` is a real code path** — all
  [modelcore's](https://github.com/8kb/modelcore/blob/main/AGENTS.md). Consequence here: a new
  architecture added under `nanochat/architectures/` inherits every one of these automatically by
  going through `ModelManager`/`presets.py`'s existing helpers — bypassing them (a raw `nn.Linear`,
  a hand-rolled save path) is how each of these invariants gets violated in practice.
- **A prepared dataset's `sequence_len`/tokenizer fingerprint must match at read time (raises, not
  warns); the dataloader cursor is an exact, world-size-independent sequence count** — both
  [datacore's](https://github.com/8kb/datacore/blob/main/AGENTS.md). Consequence here:
  `scripts/base_train.py`/`scripts/chat_sft.py` open their dataset via `--dataset` (default:
  `scripts/data_prep.py:default_dataset_name`, derived from `--max-seq-len`/tokenizer fingerprint)
  and hard-error if either doesn't match; a pre-datacore checkpoint's old `{pq_idx, rg_idx, epoch}`
  dataloader state (no `"format"` key) is refused unless `--ignore-dataloader-state` is passed —
  model/optimizer weights still load fine either way, only the data-stream position is affected.

## Invariants that will bite you (nanochat's own)

- **`runs/scaling_laws.sh` and `runs/miniseries.sh` grep exact stdout text** out of
  `scripts/base_train.py`: `runs/scaling_laws.sh` greps six `^key ` lines
  (`wte`, `value_embeds`, `lm_head`, `transformer_matrices`, `scalars`, `total`) from
  `_legacy_scaling_keys(model_stats.params_by_role)`'s dump — a presentation-layer helper mapping
  `modelcore`'s generic role names to these load-bearing legacy key names, used for every
  architecture uniformly now (previously GPT's own frozen dict); `runs/miniseries.sh` greps a
  single `"Number of parameters: N (scaling: M)"` line. Both also grep `"Calculated number of
  iterations"`, `"Total batch size"`, `"Validation bpb:"`, `"CORE metric:"`. Changing those print
  statements' format breaks those scripts silently.
- **Rotary `cos`/`sin` buffers are `persistent=False`** (not saved in checkpoints) — this is why
  `ModelManager.load_model` calls `model.init_weights()` even when *loading* a checkpoint, right
  before `load_state_dict(..., assign=True)` overwrites everything else.
- **A checkpoint's `model_config` has no `"arch"` key any more** — `ModelConfig.to_dict()` stamps
  `"format": "modelcore.v1"` instead, plus an optional `reference: {"preset": name, "kwargs": {...}}`
  block for provenance. `nanochat.checkpoint_manager.arch_of(model_config_dict)` is the
  naming-policy helper that reads either shape: `reference.preset` (defaulting to `"custom"`) for
  a current-format config, or the legacy `"arch"` key (defaulting to `"gpt"`, for checkpoints
  saved before that key existed) for anything predating modelcore. Nothing should read
  `model.config.arch` directly — that attribute doesn't exist on `modelcore.ModelConfig` at all.
- **A checkpoint predating modelcore always routes through `nanochat.architectures.legacy` first.**
  `checkpoint_manager.build_model` calls `legacy.migrate_checkpoint` unconditionally — a no-op
  (straight to `ModelConfig.from_dict`) for an already-current `"format"` dict, full
  reconstruction-from-stored-fields otherwise. See
  [docs/architecture.md](docs/architecture.md#old-checkpoints) for exactly what each generation
  needs (pre-Stage-2 flat state-dict layout, the `body.` key prefix, the resid/x0 optimizer split,
  and — gpt-arch only — `backout_lambda`'s role-rename optimizer fix).
- **Checkpoint tag naming is architecture-aware; auto-discovery isn't, by default.**
  `scripts/base_train.py`'s default save tag is `d<depth>` for `gpt` (unchanged) and
  `<arch>_d<depth>` otherwise — two architectures at the same `--depth` would otherwise write into
  the same directory. `checkpoint_manager.find_largest_model` (and `load_model`/
  `load_model_from_dir`/`load_optimizer_state`, which forward to it) accepts an optional `arch=`
  filter for callers that don't have an explicit `--model-tag`; passing nothing (the default
  everywhere except `scripts/base_train.py`/`scripts/base_eval.py`) picks the largest checkpoint
  regardless of architecture. See [docs/architecture.md](docs/architecture.md) "Checkpoint tags
  and architecture-aware discovery".
- **Checkpoint meta carries `tokenizer_fingerprint` and `core_metric`.**
  `RustBPETokenizer.fingerprint()` (`nanochat/tokenizer.py`) is a content hash of the vocab, not
  the pickle file -- it identifies *what a token id means*. `scripts/base_train.py` writes it (plus
  whatever `core_metric` the final-step CORE eval produced, `None` if that eval didn't run)
  alongside every checkpoint. `checkpoint_manager.build_model` only *warns* (doesn't raise) on a
  mismatch against the local tokenizer, and stays silent when the key is absent (every checkpoint
  saved before this existed) -- the vocab_size-only compatibility check it had before this would
  happily load a checkpoint trained against a *different* tokenizer of the same size and produce
  silent garbage, which is exactly the failure mode a multi-machine architecture comparison
  (`runs/contest.sh`, see [docs/contest.md](docs/contest.md)) would otherwise hit undetected.
- **`nanochat/default_tokenizer/` is a committed, portable default tokenizer** (532KB:
  `tokenizer.pkl` + `token_bytes.pt`) -- content-derived, so a checked-in copy is exactly as valid
  as a freshly-trained one. `runs/contest.sh`/`runs/contest_d12.sh` copy it into
  `$NANOCHAT_BASE_DIR/tokenizer/` if that directory doesn't already have *both* files (checking
  only `tokenizer.pkl` used to be enough to silently pass a partial copy that then crashed at
  training start, since `token_bytes.pt` is also required), before ever falling back to
  `scripts/tok_train.py`. Don't regenerate it casually -- it's the tokenizer every architecture
  contest checkpoint is trained against.
- **`chat_sft.py` checkpoints stamp `base_model_tag`/`base_model_step`** into their own meta.json,
  recording exactly which base checkpoint they were fine-tuned from. `chat_sft.py`'s
  auto-generated output tag is also arch-qualified (`f"{arch}_d{depth}"` for non-gpt, via
  `arch_of(meta["model_config"])`) to avoid two architectures' SFT runs at the same depth silently
  overwriting one directory.
- **A new architecture is a preset, not a class.** Adding one means: a `nanochat/architectures/derive.py`
  helper for any new derivation rule it needs, an `expand_<name>` function in
  `nanochat/architectures/presets.py` (reusing `assemble_gpt`/`assemble_plain` if its tree shape
  already fits one of them), and an entry in `PRESETS`. There is no model *class* to register and
  no registry to add it to — `modelcore` builds every tree through the same `Model` class,
  regardless of which preset produced it. See
  [docs/architecture.md](docs/architecture.md#nanochatarchitectures-presets-and-legacy-migration).
- **`tests/goldens/*.json` (including the four `tiny_composed_*` ones -- modelcore's own
  pre-Stage-7 baseline, moved here from `modelcore/tests/goldens/` at Stage 10's repo split) is the
  regression net for anything touching `modelcore`, `nanochat/architectures/`,
  `nanochat/checkpoint_manager.py`, or `nanochat/engine.py`.** Captured once
  (`dev/capture_model_goldens.py`, now frozen — its `main()`/`capture_synthetic()` depend on code
  this refactor deleted; the live digest helpers it used moved to `tests/golden_helpers.py`)
  before Stage 7's redesign, from every real checkpoint on this machine plus a seeded synthetic
  model of every architecture/preset. `tests/test_goldens.py` replays it;
  `tests/test_architectures.py` cross-checks the same numbers through the new API directly (both
  via `presets.expand` and via `ModelManager.config_from_dict` straight off the golden). Run both
  after any change to those areas — see
  [docs/architecture.md](docs/architecture.md#verifying-a-change-is-behavior-preserving).

## Before you spend money on a pod

Read [../llmllab/docs/runpod-ops.md](../llmllab/docs/runpod-ops.md) first — the six-question
pre-spend gate, CPU pod sizing, SSH quirks, GPU-stock reality checks, and network-volume tradeoffs
all live there now (Stage 8, see `docs/contest.md`, is the pre-spend gate's own worked failure
case: a paid CPU pod and 2x H100 run that could not have tested what it was named for, because
`scripts/chat_sft.py` had no `--doc-masking` wiring at the time).

Two nanochat-specific facts worth not re-deriving:

- **CPU pod sizing for `--kind=sft` data prep needs ≥16GB.** `SmolTalk`/`MMLU`/`GSM8K` load their
  full source datasets into memory before any `--max-conversations` cap applies — see the repo
  map's own "CPU-only, run before base_train.py/chat_sft.py, never on a billed GPU pod"
  (`data_prep.py`'s entry) and [../llmllab/docs/runpod-ops.md](../llmllab/docs/runpod-ops.md)'s CPU
  pod flavor section for how to actually get a big-enough CPU pod (`cpu3m`/4vcpu/32GB, not the
  4GB-capped `cpu3c` both `runpodctl` and the MCP tools default to).
- **`scripts/data_prep.py --kind=sft`'s `--sft-padding-id` should stay at its `None` default** until
  a tokenizer exists with a genuinely free pad token id. Every id in the current tokenizer is a real
  special token (Stage 8 tried `<|output_end|>`); passing one in is strictly worse than falling back
  to `bos_token_id`, which `build_doc_args`'s fold-in heuristic already handles correctly.

## What runs on this Mac

See [../llmllab/AGENTS.md](../llmllab/AGENTS.md) for the machine-level facts (no CUDA, `COMPUTE_DTYPE`
default, the MPS/`torch.compile` cold-start behavior, `print()` buffering). Set up with:

```bash
uv sync --extra cpu --group dev && source .venv/bin/activate
```

`uv sync` also builds and installs `modelcore`/`datacore` from their pinned git tags (Stage 10) --
needs network access to github.com the first time or after bumping either pin.

Runs fine locally: everything in `tests/`. modelcore's and datacore's own suites (each in its own
repo, cloned separately -- `uv run pytest` there) run clean here too except
`modelcore/tests/test_optim.py` (module-level `skipif(not cuda_available)`) and the
`TestFA3VsSDPA` class in `modelcore/tests/test_kernels.py` (needs an sm80/sm89/sm90 GPU for the
real FA3 kernel — the SDPA fallback classes in that file run fine on CPU). `scripts/base_train.py` /
`scripts/chat_sft.py` run at small `--depth`/`--max-seq-len`/`--device-batch-size` (see
`runs/runcpu.sh`) against a `scripts/data_prep.py`-prepared dataset at the same `--max-seq-len` (or
`--sequence-len` for SFT). `scripts/infer_bench.py` hard-asserts CUDA and does not run here.

Untested on this machine as a result: the `bfloat16` compute path, the real FA3 kernel path
(vs. the SDPA fallback it's checked against), the real fp8 `_scaled_mm` numerics
(`modelcore/precision/fp8.py` — the role/accounting bookkeeping around it is CPU-tested, see
`modelcore/tests/test_precision.py`), and multi-GPU/DDP gradient reduction in `modelcore/optim/`.
Keep changes to those paths conservative and prefer reasoning from the code plus the existing
(CUDA-gated) tests over "I ran it and it worked." (FP8 itself *has* now been verified end to end on
real 2x H100 hardware — see `docs/contest.md`'s "Stage 4 results" — so the disclaimer here is about
this machine specifically, not the feature.)

**Do not attempt large multi-hour training runs in this environment** — see
[../llmllab/AGENTS.md](../llmllab/AGENTS.md); use tiny smoke configs (see
[docs/architecture.md](docs/architecture.md#verifying-a-change-is-behavior-preserving)) to check
plumbing, not to produce a usable model.

## Style

See [../llmllab/AGENTS.md](../llmllab/AGENTS.md#style).
