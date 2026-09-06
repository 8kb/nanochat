# AGENTS.md

Repo map and non-obvious invariants for anyone (human or agent) working in this fork. Read
[docs/architecture.md](docs/architecture.md) before touching `modelcore/` or
`nanochat/architectures/`, and [docs/upstream-sync.md](docs/upstream-sync.md) before touching
anything that used to live in `nanochat/gpt.py` (now deleted — see that doc's "Stage 7" section
for where its code lives today).

## What this fork is

A fork of [karpathy/nanochat](https://github.com/karpathy/nanochat) repurposed as a playground
for trying different model *architectures*, not just different hyperparameters. Original upstream
docs are preserved at `docs/upstream/`. See [docs/roadmap.md](docs/roadmap.md) for the staged
plan and current progress.

## Repo map

```
modelcore/            standalone model subsystem (zero nanochat imports) — see docs/architecture.md
├── manager.py           ModelManager: the one entrypoint (create/load/save model+optimizer, stats, validate)
├── model.py              Model: the one model class, built from a materialized config tree
├── config/                ComponentSpec, ModelConfig, AttentionLayerSpec; validate_config()
├── catalog.py             component registry: "#type" name -> (cls, needs, validate)
├── components/             linear, norm, rope, rotary, attention, mlp, block, embedding, unembedding
├── composers/              base, stack, backout
├── roles.py               parameter-role protocol (optimizer grouping)
├── stats.py                FLOPs/param/KV-bytes accounting, ModelStats
├── store.py                ArtifactStore protocol + FileSystemStore
├── runtime.py              Runtime: compute dtype + log sink (injected, not a global)
├── optim/                  MuonAdamW (single combined optimizer, ZeRO-2 sharded)
├── kernels/                 unified FA3/SDPA attention interface
└── cache.py                 KVCache
nanochat/             everything that knows nanochat's own conventions
├── architectures/       expand a --depth dial (presets.py) or migrate an old checkpoint (legacy.py)
│                        into a modelcore.ModelConfig; derive.py holds the depth-dial derivation rules
├── engine.py             inference: Engine (KV-cached generate), generate_naive; KVCache re-exported
├── checkpoint_manager.py  naming policy (tags, steps) + meta.json extras; hands ModelManager a config
├── optim.py, flash_attention.py   one-line re-export shims onto modelcore.optim/modelcore.kernels
├── tokenizer.py            BPE tokenizer wrapper
├── dataloader.py / dataset.py   pretraining data
├── core_eval.py / loss_eval.py   base-model evaluation (CORE benchmark, bits-per-byte)
├── execution.py            sandboxed Python execution (tool use)
├── scaling.py               muP training-plan math (architecture-agnostic)
└── fp8.py                    FP8 training (CUDA/Hopper only)
scripts/              entry points, run as `python -m scripts.<name>`
tasks/                task/dataset definitions for eval (arc, mmlu, gsm8k, humaneval, smoltalk)
tests/                pytest suite — see "What runs on this Mac" below
runs/                 shell scripts wiring scripts/ together (speedrun.sh, runcpu.sh, ...)
docs/                 this fork's documentation; docs/upstream/ holds the original nanochat docs
dev/                  images, notebooks, dev/repackage_data_reference.py, dev/capture_model_goldens.py
```

## Invariants that will bite you

- **`__init__` may run under `torch.device("meta")`.** `Model.__init__` (and any component's) must
  not compute anything that depends on real tensor *values* — only shapes/dtypes. Real
  initialization goes in `init_weights()`, called after `model.to_empty(device=...)`.
  `ModelManager.create_model`/`load_model` own this dance; nothing else should repeat it. See
  "The meta-device footgun" in [docs/architecture.md](docs/architecture.md).
- **No `torch.amp.autocast`.** Precision is `modelcore.runtime.Runtime.compute_dtype`, injected
  into any component declaring `needs=("runtime",)` — not a bare global read off an attribute.
  `nanochat.common.COMPUTE_DTYPE` (override via `NANOCHAT_DTYPE` env var) re-exports the default
  runtime's value for existing readers. Model weights stay fp32; `modelcore.components.linear.Linear`
  casts to `COMPUTE_DTYPE` in `forward()`. Route every matmul-participating parameter through it.
- **`Linear` is the structural marker for "matmul params".**
  `modelcore.stats.num_matmul_params` finds every FLOPs-relevant parameter by scanning for
  `isinstance(m, Linear)`. A new matmul that uses a raw `nn.Linear` or bare `nn.Parameter`
  silently disappears from `ModelStats.flops_per_token`/`decode_flops`/`prefill_flops` and every
  FLOPs/s or MFU number derived from them.
- **Every parameter needs a declared role.** `modelcore.roles.collect_param_roles` walks the
  module tree and raises on any parameter it can't assign a role to (a `Linear.weight` defaults to
  `"matrix"`; anything else needs a `PARAM_ROLES` class attribute or a `param_roles()` override).
  `ModelManager.create_optimizer`/`ModelStats.params_by_role` are built on this, so a new
  `nn.Parameter` or submodule that forgets to declare a role raises at construction — far better
  than it silently defaulting into the wrong optimizer (e.g. Muon's shape-based matrix grouping).
  See [docs/architecture.md](docs/architecture.md#component-contracts).
- **A config tree carries only concrete, already-decided values, never a derivation rule.**
  `has_value_embed` is a plain bool per block, `window` a concrete int, `kv_slot`/`produces_kv`
  concrete per-block values — never a pattern string or a fraction a component would need to
  interpret. Every rule that produces these values (`has_value_embed`'s alternating-parity policy,
  `compute_window_sizes`, `compute_kv_slots`, the muP depth dial) lives in
  `nanochat/architectures/derive.py`, run once at tree-expansion time, outside `modelcore` entirely.
  A component asking "which layer am I" or "how many layers are there" to re-derive a policy is
  exactly the abstraction leak this fork's Stage 7 redesign eliminated — don't reintroduce it.
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
- **Optimizer state is checkpointed and reloaded positionally.** `torch.optim.Optimizer.state_dict()`
  flattens every parameter across every group into one global index order; a parameter that
  splits, merges, or moves group changes that indexing, and a same-size reorder corrupts state
  silently (no shape-mismatch error) rather than loudly. `ModelManager.create_optimizer`'s policy
  dict order is therefore part of the on-disk format, not just a style choice — see
  [docs/architecture.md](docs/architecture.md#component-contracts). A change that does reorder or
  resplit needs a migration in `nanochat/architectures/legacy.py` (see `_patch_resid_x0_split`/
  `_split_backout_lambda_from_smear` for the pattern) or old optimizer shards fail to load —
  `scripts/base_train.py`'s `--resume-from-step` and `scripts/chat_sft.py`'s `--load-optimizer`
  are the two call sites that route through `migrate_optimizer_state`.
- **`kv_cache.advance()` belongs to `Model.forward`, not the last attention layer.** It fires once,
  after the whole block/composer loop runs — broken the moment a model has fewer KV slots than
  layers (cross-layer KV sharing), since no layer's index then equals the slot count.
- **`AttentionLayerSpec.kv_slot` decouples layer index from KV-cache slot.**
  `ModelStats.kv_cache_spec["num_kv_slots"]` can be `<= n_layer`: a layer whose `kv_slot` points
  at an earlier layer's slot (cross-layer KV sharing) shares that `modelcore.cache.KVCache`
  allocation instead of getting its own. `KVCache`'s constructor kwarg and attribute are
  `num_kv_slots`/`n_slots`, and `get_slot_cache(slot)` returns that slot's view — see
  "Cross-layer KV sharing" in [docs/architecture.md](docs/architecture.md) for the full mechanism,
  including a real FA3-vs-SDPA divergence in what `k=None` means to `flash_attn_with_kvcache` that
  a naive sharing implementation would hit.
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
- **`tests/goldens/*.json` is the regression net for anything touching `modelcore`,
  `nanochat/architectures/`, `nanochat/checkpoint_manager.py`, or `nanochat/engine.py`.** Captured
  once (`dev/capture_model_goldens.py`, now effectively frozen — it depends on code this refactor
  deleted) before Stage 7's redesign, from every real checkpoint on this machine plus a seeded
  synthetic model of every architecture/preset. `tests/test_goldens.py` replays it;
  `tests/test_modelcore.py`/`tests/test_architectures.py` cross-check the same numbers through the
  new API directly. Run all three after any change to those areas — see
  [docs/architecture.md](docs/architecture.md#verifying-a-change-is-behavior-preserving).

## What runs on this Mac

Dev machine: Apple Silicon (M4), macOS, **no CUDA**. `COMPUTE_DTYPE` defaults to `float32` here
(see `modelcore/runtime.py`'s `detect_compute_dtype`). Set up with:

```bash
uv sync --extra cpu --group dev && source .venv/bin/activate
```

Runs fine locally: everything in `tests/` except `tests/test_optim.py` (module-level
`skipif(not cuda_available)`) and the `TestFA3VsSDPA` class in
`tests/test_attention_fallback.py` (needs an sm80/sm89/sm90 GPU for the real FA3 kernel — the
SDPA fallback classes in that file run fine on CPU). `scripts/base_train.py` /
`scripts/chat_sft.py` run at small `--depth`/`--max-seq-len`/`--device-batch-size` (see
`runs/runcpu.sh`). `scripts/infer_bench.py` hard-asserts CUDA and does not run here.

Untested on this machine as a result: the `bfloat16` compute path, the real FA3 kernel path
(vs. the SDPA fallback it's checked against), FP8 training (`nanochat/fp8.py`), and multi-GPU/DDP
gradient reduction in `modelcore/optim/`. Keep changes to those paths conservative and prefer
reasoning from the code plus the existing (CUDA-gated) tests over "I ran it and it worked."

**Do not attempt large multi-hour training runs in this environment** (no GPU, thermal/power
constraints of a laptop) — use tiny smoke configs (see
[docs/architecture.md](docs/architecture.md#verifying-a-change-is-behavior-preserving)) to check
plumbing, not to produce a usable model.

## Style

Match the surrounding code: minimal comments explaining *why*, not *what*; no giant config
objects or factory indirection beyond what `modelcore/catalog.py`'s registry already adds; prefer
extending an existing component/composer over adding a new abstraction layer.
