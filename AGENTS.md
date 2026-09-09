# AGENTS.md

Repo map and non-obvious invariants for anyone (human or agent) working in this fork. Read
[modelcore/docs/architecture.md](modelcore/docs/architecture.md) before touching anything under
`modelcore/`, [datacore/docs/architecture.md](datacore/docs/architecture.md) before touching
anything under `datacore/`, [docs/architecture.md](docs/architecture.md) before touching
`nanochat/architectures/` or how the app consumes `ModelManager`, and
[docs/upstream-sync.md](docs/upstream-sync.md) before touching anything that used to live in
`nanochat/gpt.py` (now deleted — see that doc's "Stage 7" section for where its code lives today).

## What this fork is

A fork of [karpathy/nanochat](https://github.com/karpathy/nanochat) repurposed as a playground
for trying different model *architectures*, not just different hyperparameters. Original upstream
docs are preserved at `docs/upstream/`. See [docs/roadmap.md](docs/roadmap.md) for the staged
plan and current progress.

## Repo map

```
modelcore/            standalone model subsystem (zero nanochat imports) — see modelcore/docs/architecture.md
├── manager.py           ModelManager: the one entrypoint (create/load/save model+optimizer, stats,
│                         validate, precision, decoding)
├── model.py              Model: the one model class, built from a materialized config tree
├── generate.py            sample_next_token, generate_naive, Decoder (cached prefill+decode)
├── config/                ComponentSpec, ModelConfig, AttentionLayerSpec; validate_config()
├── catalog.py             component registry: "#type" name -> (cls, needs, validate)
├── components/             linear, norm, rope, rotary, attention, mlp, block, embedding, unembedding
├── composers/              base, stack, backout
├── roles.py               parameter-role protocol (optimizer grouping)
├── stats.py                FLOPs/param/KV-bytes accounting, ModelStats
├── store.py                ArtifactStore protocol + FileSystemStore
├── runtime.py              Runtime: compute dtype + log sink (injected, not a global)
├── precision/fp8.py         Float8Linear + convert_to_float8_training (ModelManager.enable_fp8)
├── optim/                  MuonAdamW (single combined optimizer, ZeRO-2 sharded)
├── kernels/                 unified FA3/SDPA attention interface
├── cache.py                 KVCache
└── tests/, docs/, README.md, pyproject.toml   modelcore's own suite, contract, and packaging
datacore/             standalone data subsystem (zero nanochat imports) — see datacore/docs/architecture.md
├── manager.py           DataManager: the one entrypoint (prepare a dataset, open one, read batches)
├── store.py              DatasetStore protocol + FileSystemDatasetStore + the manifest schema
├── packing.py             Packer protocol; BestFitCropPacker, BestFitPadPacker
├── writer.py              rolls PackedRow into volumes, flushed at a cap or a source boundary
├── reader.py              memmap volumes + the cursor-based, DDP-sharded, resumable batch iterator
│                        (the only module that imports torch, lazily)
├── sources.py             TextSource/TokenSource protocols; ParquetDirectorySource
├── download.py            generic resumable HTTP shard downloader
├── tokenizer.py            Tokenizer protocol + CharTokenizer (dependency-free test double)
└── tests/, docs/, README.md, pyproject.toml   datacore's own suite, contract, and packaging
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
                        (modelcore/tests/, datacore/tests/ are each component's own standalone suite)
runs/                 shell scripts wiring scripts/ together (speedrun.sh, runcpu.sh, ...)
docs/                 this fork's documentation; docs/upstream/ holds the original nanochat docs
dev/                  images, notebooks, dev/repackage_data_reference.py, dev/capture_model_goldens.py,
                        dev/capture_data_goldens.py (frozen pre-datacore packing-algorithm reference)
```

## Invariants that will bite you

- **`__init__` may run under `torch.device("meta")`.** `Model.__init__` (and any component's) must
  not compute anything that depends on real tensor *values* — only shapes/dtypes. Real
  initialization goes in `init_weights()`, called after `model.to_empty(device=...)`.
  `ModelManager.create_model`/`load_model` own this dance; nothing else should repeat it. See
  "The meta-device footgun" in [modelcore/docs/architecture.md](modelcore/docs/architecture.md).
- **No `torch.amp.autocast`.** Precision is `modelcore.runtime.Runtime.compute_dtype`, injected
  into any component declaring `needs=("runtime",)` — not a bare global read off an attribute.
  `nanochat.common.COMPUTE_DTYPE` (override via `MODELCORE_DTYPE`, or the back-compat
  `NANOCHAT_DTYPE`, env var) re-exports the default runtime's value for existing readers. Model
  weights stay fp32; `modelcore.components.linear.Linear` casts to `COMPUTE_DTYPE` in `forward()`.
  Route every matmul-participating parameter through it.
- **`Linear` is the structural marker for "matmul params".**
  `modelcore.stats.num_matmul_params` finds every FLOPs-relevant parameter by scanning for
  `isinstance(m, Linear)`. A new matmul that uses a raw `nn.Linear` or bare `nn.Parameter`
  silently disappears from `ModelStats.flops_per_token`/`decode_flops`/`prefill_flops` and every
  FLOPs/s or MFU number derived from them. This is also why
  `modelcore.precision.fp8.Float8Linear` subclasses `Linear` rather than a bare `nn.Linear` — it
  used to subclass `nn.Linear` directly (`nanochat/fp8.py`, pre-Stage-8), which meant an
  fp8-converted model's `collect_param_roles` raised outright (`Float8Linear.weight has no
  declared role`) the moment `ModelManager.create_optimizer` tried to build its param groups.
- **Every parameter needs a declared role.** `modelcore.roles.collect_param_roles` walks the
  module tree and raises on any parameter it can't assign a role to (a `Linear.weight` defaults to
  `"matrix"`; anything else needs a `PARAM_ROLES` class attribute or a `param_roles()` override).
  `ModelManager.create_optimizer`/`ModelStats.params_by_role` are built on this, so a new
  `nn.Parameter` or submodule that forgets to declare a role raises at construction — far better
  than it silently defaulting into the wrong optimizer (e.g. Muon's shape-based matrix grouping).
  See [modelcore/docs/architecture.md](modelcore/docs/architecture.md#component-contracts).
- **A config tree carries only concrete, already-decided values, never a derivation rule.**
  `has_value_embed` is a plain bool per block, `window` a concrete int, `kv_slot`/`produces_kv`
  concrete per-block values — never a pattern string or a fraction a component would need to
  interpret. Every rule that produces these values (`has_value_embed`'s alternating-parity policy,
  `compute_window_sizes`, `compute_kv_slots`, the muP depth dial) lives in
  `nanochat/architectures/derive.py`, run once at tree-expansion time, outside `modelcore` entirely.
  A component asking "which layer am I" or "how many layers are there" to re-derive a policy is
  exactly the abstraction leak this fork's Stage 7 redesign eliminated — don't reintroduce it.
- **`ArtifactStore` is a real code path, not aspirational.** `nanochat.checkpoint_manager.save_checkpoint`/
  `load_checkpoint`/`build_model` route through `modelcore.store.FileSystemStore` (and, for an old
  checkpoint, `LegacyCheckpointStore(FileSystemStore)`, which migrates on first read and memoizes) —
  they don't call `torch.save`/`torch.load` directly. A new checkpoint-touching code path should go
  through the store too, not add a third way to read/write the same files.
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
  [modelcore/docs/architecture.md](modelcore/docs/architecture.md#component-contracts). A change that does reorder or
  resplit needs a migration in `nanochat/architectures/legacy.py` (see `_patch_resid_x0_split`/
  `_split_backout_lambda_from_smear` for the pattern) or old optimizer shards fail to load —
  `scripts/base_train.py`'s `--resume-from-step` and `scripts/chat_sft.py`'s `--load-optimizer`
  are the two call sites that route through `migrate_optimizer_state`.
- **`kv_cache.advance()` belongs to `Model.forward`, not the last attention layer.** It fires once,
  after the whole block/composer loop runs — broken the moment a model has fewer KV slots than
  layers (cross-layer KV sharing), since no layer's index then equals the slot count.
- **Intra-document masking's `doc_args` must be built outside `torch.compile`.**
  `modelcore.kernels.flash_attn.build_doc_args(idx, bos_token_id)` derives per-row document
  boundaries via `nonzero()`, and `scripts/base_train.py --doc-masking` calls it in the training
  loop, before `model(x, y, doc_args=...)` — never inside the compiled model itself. See
  [modelcore/docs/architecture.md](modelcore/docs/architecture.md#intra-document-masking) for why
  (a real, measured recompile cost otherwise) and why positions are deliberately not reset per
  document (RoPE + QK-norm make it a no-op).
- **`build_doc_args`'s `max_docs` default is a dataset-tuned guess, not a safe worst case.** It
  sizes the FA3 varlen kernel's backward-pass scratch allocation directly — defaulting it to
  `batch_size * sequence_len` (every token its own document) OOM'd a real 2x H100 run trying to
  allocate 28GB of scratch for a declared batch of 131,072 sequences when the real batch had ~270
  documents. `DEFAULT_MAX_DOCS_PER_ROW=64` is tuned against ClimbMix's measured ~4.2 documents/row
  at `sequence_len=2048`; `--doc-masking-max-docs-per-row` overrides it for a different
  dataset/sequence-length combination.
- **`AttentionLayerSpec.kv_slot` decouples layer index from KV-cache slot.**
  `ModelStats.kv_cache_spec["num_kv_slots"]` can be `<= n_layer`: a layer whose `kv_slot` points
  at an earlier layer's slot (cross-layer KV sharing) shares that `modelcore.cache.KVCache`
  allocation instead of getting its own. `KVCache`'s constructor kwarg and attribute are
  `num_kv_slots`/`n_slots`, and `get_slot_cache(slot)` returns that slot's view — see
  "Cross-layer KV sharing" in [modelcore/docs/architecture.md](modelcore/docs/architecture.md) for the full mechanism,
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
- **A prepared dataset's `sequence_len` and tokenizer fingerprint are fixed, and must match, or
  training raises.** `scripts/base_train.py`/`scripts/chat_sft.py` open their `datacore.Dataset`
  via `--dataset` (default: derived from `--max-seq-len`/tokenizer fingerprint,
  `scripts/data_prep.py:default_dataset_name`) and hard-error -- not warn -- if `--max-seq-len`
  doesn't equal `dataset.info.sequence_len` or the local tokenizer's fingerprint doesn't match
  `dataset.info.tokenizer_fingerprint`. Unlike the checkpoint fingerprint check above, this one
  raises: there is no scenario where training on a mismatched tokenization was intended, and it
  produces silent garbage. Batch size, world size, rank, and split are the only things free at
  read time -- see [datacore/docs/architecture.md](datacore/docs/architecture.md).
- **The dataloader state in checkpoint meta is an exact global sequence cursor, not an
  approximation.** `meta["dataloader_state_dict"]` is now `{"format": "datacore.v1", "cursor",
  "epoch", "num_sequences", "batch_size", "world_size"}` -- `cursor` is the count of sequences
  consumed by all ranks so far, world-size-independent by construction (resuming at a different
  `--nproc_per_node` than the run that saved it still produces a gap-free, duplicate-free
  continuation). A pre-datacore checkpoint's `{pq_idx, rg_idx, epoch}` state (detected by the
  absent `"format"` key) is **refused**, not translated -- `scripts/base_train.py` raises with an
  actionable message unless `--ignore-dataloader-state` is passed, since there is no faithful
  mapping into a sequence cursor. Model and optimizer weights still load fine either way; only the
  data-stream position is affected.
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
- **`tests/goldens/*.json` (plus `modelcore/tests/goldens/tiny_composed_*.json`) is the regression
  net for anything touching `modelcore`, `nanochat/architectures/`,
  `nanochat/checkpoint_manager.py`, or `nanochat/engine.py`.** Captured once
  (`dev/capture_model_goldens.py`, now frozen — its `main()`/`capture_synthetic()` depend on code
  this refactor deleted; the live digest helpers it used moved to `tests/golden_helpers.py`)
  before Stage 7's redesign, from every real checkpoint on this machine plus a seeded synthetic
  model of every architecture/preset. `tests/test_goldens.py` replays it;
  `modelcore/tests/test_manager.py`/`tests/test_architectures.py` cross-check the same numbers
  through the new API directly. Run all three after any change to those areas — see
  [docs/architecture.md](docs/architecture.md#verifying-a-change-is-behavior-preserving).

## Before you spend money on a pod

Stage 8 (see `docs/contest.md`) paid for a CPU pod and a 2x H100 pod on a run that could not have
tested what it was named for: `scripts/chat_sft.py` had no `--doc-masking` wiring, so a new
`padding_id`-filled SFT dataset changed nothing reachable by that script. Answer these six
questions **in writing, before creating any billed pod** — not as a formality, each has a
falsifiable answer:

1. **What number does this produce, and against what number is it compared?** Quote the baseline's
   actual value and where it's recorded (a `docs/contest.md` stage, or a log under
   `runs/results/`). No existing baseline means this is two runs, not one.
2. **What is the single variable that differs?** More than one differing ⇒ the result isn't
   attributable to anything.
3. **Trace the consumer.** Name the `file:line` where the flag/parameter under test is *read* in
   the exact script being launched. **No consumer ⇒ stop — the run cannot test it.** This is the
   question Stage 8 failed, and it's answerable by `grep` alone.
4. **What can be checked for free or nearly free first?** A local CPU smoke run, a manifest read, a
   dataset diff, a `pytest` — all free. A CPU pod is ~$0.05/run; a wrong GPU hour is not. Do the
   cheap check and report its result before creating the GPU pod.
5. **Expected effect size vs. known noise.** Every result in the doc-masking line so far sits at or
   near noise (upstream's own d16 attempt, `docs/upstream/LOG.md:715-741`: 0.85427→0.85407; this
   fork's Stage 7 wall-time-adjusted; Stage 8's 0.24%). State what would make *this* run
   distinguishable from noise, or don't run it.
6. **Written cost estimate before launch**: pod flavor × expected minutes × $/hr.

Two operational facts worth not re-deriving:

- **CPU pod sizing for `--kind=sft` data prep needs ≥16GB.** `SmolTalk`/`MMLU`/`GSM8K` load their
  full source datasets into memory before any `--max-conversations` cap applies. Neither the MCP
  `create-pod` tool nor `runpodctl` (`create pod` or `pod create`) can select a CPU flavor/vCPU
  count — both land on `cpu3c` (2 vcpu, 4GB, enforced as a hard cgroup limit regardless of what the
  host reports), which OOMs this job at exit 137 with no output. Work around it via
  `POST https://api.runpod.io/v2/pods` directly with `cpu: {id, vcpuCount}` (`cpu3m`/4vcpu/32GB is
  known-good), using the API key already configured for `runpodctl` in `~/.runpod/config.toml`. See
  the repo map's own "CPU-only, run before base_train.py/chat_sft.py, never on a billed GPU pod"
  (above, `data_prep.py`'s entry) — this is the CPU-side counterpart: size the CPU pod correctly
  instead of discovering the OOM after paying for the attempt.
- **`scripts/data_prep.py --kind=sft`'s `--sft-padding-id` should stay at its `None` default** until
  a tokenizer exists with a genuinely free pad token id. Every id in the current tokenizer is a real
  special token (Stage 8 tried `<|output_end|>`); passing one in is strictly worse than falling back
  to `bos_token_id`, which `build_doc_args`'s fold-in heuristic already handles correctly.
- **RunPod's SSH proxy (`ssh.runpod.io`, the route used when a pod has no public IP —
  `ssh.direct` is `null` in the create/get-pod response) needs an account-registered key, not the
  container's `PUBLIC_KEY` env var, and only supports an interactive PTY channel.** `startSsh: true`
  on pod creation injects whatever's registered via `GET/PUT /v2/account/ssh-keys` into
  `PUBLIC_KEY` — check that endpoint first (`~/.runpod/ssh/runpodctl-ssh-key` is typically already
  the registered key from a prior `runpodctl` use) rather than generating and threading through a
  new one. Plain `ssh host cmd` fails outright ("doesn't support PTY"); use `ssh -tt host < script`
  (commands piped via stdin) instead. `scp`/`sftp` don't work over this proxy at all (no subsystem
  support) — to get a locally-edited file onto the pod, base64-encode it and pipe
  `base64 -d > path <<'EOF' ... EOF` through the same stdin channel, and verify with `md5sum` on
  both ends (the interactive shell's echoed terminal output looks garbled but the actual bytes
  received are unaffected).
- **Adding *any* exposed port to a pod (even one you don't otherwise need, e.g. `8000/http`) makes
  RunPod allocate it a real public IP**, populating `ssh.direct` in the create/get-pod response —
  after which normal `ssh -p <port> root@<ip>` and, critically, **real `scp`/`rsync`** work, unlike
  the PTY-only proxy above. Worth doing any time you need to move more than a few KB (a checkpoint,
  a dataset) rather than reaching for base64-over-stdin, which is fine for small text files but not
  gigabytes. Changing a running pod's `ports` (via `update-pod`) restarts the container and
  reallocates the port mapping — reread the pod's current `ssh.direct` port after the update rather
  than reusing the one from creation.
- **A GPU type's "LOW" stock label in `GET /v2/catalog/datacenters?include=GPU_AVAILABILITY` is not
  "zero."** A specific datacenter can have literally no stock for a GPU type/count (pod creation
  fails with "no longer any instances available") while the *global* aggregate
  (`get-capacity`/`list-gpu-types`) still reads "High," and a datacenter separately labeled "LOW"
  can still provision successfully. Confirm by actually attempting creation (terminating
  immediately if it succeeds and isn't needed yet) rather than trusting the label either way — a
  failed attempt costs nothing, a wrongly-abandoned option costs the alternative's overhead (e.g.
  migrating data to a different datacenter that doesn't actually need it).
- **A network volume is pinned to its datacenter; a pod can mount at most one.** If the datacenter
  with your data has no GPU stock but another one does, moving *only* what's strictly needed (e.g.
  a trained checkpoint) via direct `scp` is usually cheaper than migrating an entire dataset — a
  dataset built by `scripts/data_prep.py` from public sources is often faster to just re-prepare
  fresh on the new pod than to transfer, and doing so is a real (if content-deterministic, per
  Stage 10) way to verify the pipeline reproduces byte-for-byte-equivalent document/sequence counts.

## What runs on this Mac

Dev machine: Apple Silicon (M4), macOS, **no CUDA**. `COMPUTE_DTYPE` defaults to `float32` here
(see `modelcore/runtime.py`'s `detect_compute_dtype`). Set up with:

```bash
uv sync --extra cpu --group dev && source .venv/bin/activate
```

Runs fine locally: everything in `tests/`, `modelcore/tests/`, and `datacore/tests/` except
`modelcore/tests/test_optim.py` (module-level `skipif(not cuda_available)`) and the
`TestFA3VsSDPA` class in `modelcore/tests/test_kernels.py` (needs an sm80/sm89/sm90 GPU for the
real FA3 kernel — the SDPA fallback classes in that file run fine on CPU). `scripts/base_train.py` /
`scripts/chat_sft.py` run at small `--depth`/`--max-seq-len`/`--device-batch-size` (see
`runs/runcpu.sh`) against a `scripts/data_prep.py`-prepared dataset at the same `--max-seq-len` (or
`--sequence-len` for SFT). `scripts/infer_bench.py` hard-asserts CUDA and does not run here.

**MPS's first real op after `torch.compile` can genuinely take minutes**, not seconds — a cold
Metal shader cache means the very first training run in a fresh shell can look hung (CPU busy in
`waitUntilCompleted`/`MPSStream::synchronize`, no new stdout) for several minutes before proceeding
normally; a second run against the same shapes is fast. Real behavior, not a bug — don't mistake it
for a hang while testing a change on this machine. Redirecting stdout to a file also fully
block-buffers Python's `print()` (line-buffering is TTY-only), which compounds the appearance of a
hang — use `python -u`/`PYTHONUNBUFFERED=1` when diagnosing one for real.

Untested on this machine as a result: the `bfloat16` compute path, the real FA3 kernel path
(vs. the SDPA fallback it's checked against), the real fp8 `_scaled_mm` numerics
(`modelcore/precision/fp8.py` — the role/accounting bookkeeping around it is CPU-tested, see
`modelcore/tests/test_precision.py`), and multi-GPU/DDP gradient reduction in `modelcore/optim/`.
Keep changes to those paths conservative and prefer reasoning from the code plus the existing
(CUDA-gated) tests over "I ran it and it worked." (FP8 itself *has* now been verified end to end on
real 2x H100 hardware — see `docs/contest.md`'s "Stage 4 results" — so the disclaimer here is about
this machine specifically, not the feature.)

**Do not attempt large multi-hour training runs in this environment** (no GPU, thermal/power
constraints of a laptop) — use tiny smoke configs (see
[docs/architecture.md](docs/architecture.md#verifying-a-change-is-behavior-preserving)) to check
plumbing, not to produce a usable model.

## Style

Match the surrounding code: minimal comments explaining *why*, not *what*; no giant config
objects or factory indirection beyond what `modelcore/catalog.py`'s registry already adds; prefer
extending an existing component/composer over adding a new abstraction layer.
