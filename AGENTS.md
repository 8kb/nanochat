# AGENTS.md

Repo map and non-obvious invariants for anyone (human or agent) working in this fork. Read
[docs/architecture.md](docs/architecture.md) before touching `nanochat/model/`, and
[docs/upstream-sync.md](docs/upstream-sync.md) before touching anything that used to live in
`nanochat/gpt.py`.

## What this fork is

A fork of [karpathy/nanochat](https://github.com/karpathy/nanochat) repurposed as a playground
for trying different model *architectures*, not just different hyperparameters. Original upstream
docs are preserved at `docs/upstream/`. See [docs/roadmap.md](docs/roadmap.md) for the staged
plan and current progress.

## Repo map

```
nanochat/            the library
├── model/              pluggable architectures — see docs/architecture.md
├── gpt.py               backward-compat shim re-exporting nanochat.model.gpt symbols
├── engine.py             inference: KVCache, Engine (KV-cached generate), generate_naive
├── checkpoint_manager.py  save/load; reconstructs models via the nanochat.model registry
├── optim.py               MuonAdamW (single combined optimizer, ZeRO-2 sharded)
├── tokenizer.py            BPE tokenizer wrapper
├── dataloader.py / dataset.py   pretraining data
├── core_eval.py / loss_eval.py   base-model evaluation (CORE benchmark, bits-per-byte)
├── execution.py            sandboxed Python execution (tool use)
├── flash_attention.py       unified FA3/SDPA attention interface
└── fp8.py                    FP8 training (CUDA/Hopper only)
scripts/              entry points, run as `python -m scripts.<name>`
tasks/                task/dataset definitions for eval (arc, mmlu, gsm8k, humaneval, smoltalk)
tests/                pytest suite — see "What runs on this Mac" below
runs/                 shell scripts wiring scripts/ together (speedrun.sh, runcpu.sh, ...)
docs/                 this fork's documentation; docs/upstream/ holds the original nanochat docs
dev/                  images, notebooks, dev/repackage_data_reference.py
```

## Invariants that will bite you

- **`__init__` may run under `torch.device("meta")`.** `GPT.__init__` (and any architecture's)
  must not compute anything that depends on real tensor *values* — only shapes/dtypes. Real
  initialization goes in `init_weights()`, called after `model.to_empty(device=...)`. See
  "The meta-device footgun" in [docs/architecture.md](docs/architecture.md).
- **No `torch.amp.autocast`.** Precision is one global, `COMPUTE_DTYPE`
  (`nanochat/common.py`, override via `NANOCHAT_DTYPE` env var). Model weights stay fp32; the
  custom `nanochat.model.components.linear.Linear` casts to `COMPUTE_DTYPE` in `forward()`. Route
  every matmul-participating parameter through it.
- **`Linear` is the structural marker for "matmul params".**
  `nanochat.model.flops.num_matmul_params` finds every FLOPs-relevant parameter by scanning for
  `isinstance(m, Linear)`. A new matmul that uses a raw `nn.Linear` or bare `nn.Parameter`
  silently disappears from `estimate_flops`, `estimate_decode_flops`, `estimate_prefill_flops`,
  and every FLOPs/s or MFU number derived from them.
- **Every parameter needs a declared role.** `nanochat.model.param_roles.collect_param_roles`
  walks the module tree and raises on any parameter it can't assign a role to (a `Linear.weight`
  defaults to `"matrix"`; anything else needs a `PARAM_ROLES` class attribute or a `param_roles()`
  override). `setup_optimizer()`/`num_scaling_params()` are built on this, so a new `nn.Parameter`
  or submodule that forgets to declare a role raises at construction — far better than it silently
  defaulting into the wrong optimizer (e.g. Muon's shape-based matrix grouping). See
  [docs/architecture.md](docs/architecture.md#parameter-roles).
- **`runs/scaling_laws.sh` and `runs/miniseries.sh` grep exact stdout text** out of
  `scripts/base_train.py`: `runs/scaling_laws.sh` greps six `^key ` lines from the
  `num_scaling_params()` dump (`wte`, `value_embeds`, `lm_head`, `transformer_matrices`,
  `scalars`, `total` — all load-bearing key names, even though the underlying role names are more
  granular); `runs/miniseries.sh` greps a single `"Number of parameters: N (scaling: M)"` line.
  Both also grep `"Calculated number of iterations"`, `"Total batch size"`, `"Validation bpb:"`,
  `"CORE metric:"`. Changing those print statements' format breaks those scripts silently.
- **Rotary `cos`/`sin` buffers are `persistent=False`** (not saved in checkpoints) — this is why
  `checkpoint_manager.build_model` calls `model.init_weights()` even when *loading* a checkpoint,
  right before `load_state_dict(..., assign=True)` overwrites everything else.
- **Checkpoint `model_config` carries an `"arch"` key** (from `BaseModelConfig.to_dict()`), read
  by `nanochat.model.registry.config_from_dict` to pick the right config/model class. Missing
  `"arch"` (checkpoints saved before Stage 1) defaults to `"gpt"`.
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
  silently (no shape-mismatch error) rather than loudly. `setup_optimizer()`'s policy dict order
  is therefore part of the on-disk format, not just a style choice — see
  [docs/architecture.md](docs/architecture.md#parameter-roles). A change that does reorder or
  resplit needs a `patch_optimizer_state_dict` migration (see
  `nanochat/model/gpt/migrations.py`'s Stage 2 resid/x0-lambda split for the pattern) or old
  optimizer shards fail to load — `scripts/base_train.py`'s `--resume-from-step` and
  `scripts/chat_sft.py`'s `--load-optimizer` are the two call sites that route through it.
- **`kv_cache.advance()` belongs to `Model.forward`, not the last attention layer.** It used to
  fire inside `CausalSelfAttention.forward` on `self.layer_idx == kv_cache.n_layers - 1` -- broken
  the moment a model has fewer KV slots than layers (cross-layer KV sharing), since no layer's
  index then equals the slot count. Every `BaseModel.forward` now calls
  `kv_cache.advance(idx.size(1))` itself, once, after its whole block loop runs (see
  `GPT.forward`/`Llama.forward`/`LlamaKVShare.forward`).
- **`AttentionLayerSpec.kv_slot` decouples layer index from KV-cache slot.**
  `BaseModel.kv_cache_spec()["num_kv_slots"]` can be `<= n_layer`: a layer whose `kv_slot` points
  at an earlier layer's slot (cross-layer KV sharing, `nanochat.model.llama_kvshare`) shares that
  `nanochat.engine.KVCache` allocation instead of getting its own. `KVCache`'s constructor kwarg
  and attribute are `num_kv_slots`/`n_slots` (not `num_layers`/`n_layers`), and
  `get_layer_cache(layer_idx)` is now `get_slot_cache(slot)` -- see "Cross-layer KV sharing" in
  [docs/architecture.md](docs/architecture.md) for the full mechanism, including a real FA3-vs-SDPA
  divergence in what `k=None` means to `flash_attn_with_kvcache` that a naive sharing
  implementation would hit.
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
  recording exactly which base checkpoint they were fine-tuned from (in addition to `arch`, already
  present via `model_config`). `chat_sft.py`'s auto-generated output tag is also arch-qualified
  (`f"{arch}_d{depth}"` for non-gpt, matching `base_train.py`'s existing pattern) to avoid two
  architectures' SFT runs at the same depth silently overwriting one directory.
- **An architecture that only changes config defaults should subclass the model it's based on, not
  re-register it under a new name.** `nanochat.model.llama_kvshare_win.LlamaKVShareWin(LlamaKVShare):
  pass` is the pattern — the config subclass carries the real change (`LlamaKVShareWinConfig`
  overrides `window_pattern`'s default), the model class exists only so `get_model_class(...)`,
  tracebacks, and checkpoint `repr`s name the right architecture. If the changed default is also a
  `from_depth(...)` kwarg (e.g. `window_pattern`), `from_depth` itself must be overridden too, not
  just the dataclass field — the parent classmethod's own signature default (`LlamaConfig.from_depth`
  hardcodes `window_pattern="L"`) otherwise silently wins over the subclass's field default on any
  `--depth`-driven run.
- **`nanochat/model/composed/` (`--arch composed`) is a materialized-config-tree alternative to a
  hardcoded architecture class, additive alongside gpt/llama/llama_kvshare(_win) — see
  [docs/architecture.md](docs/architecture.md#composed-architectures).** A composed model's
  state-dict paths live under `body.` (e.g. `body.blocks.0.attn.c_q.weight`, not
  `blocks.0.attn.c_q.weight` — the composer is a real submodule wrapping what a native
  architecture keeps at the top level); `docs/architecture.md`'s "Composed architectures" section
  has the exact remap `tests/test_model_composed.py` uses to prove a composed preset matches its
  native architecture bit-for-bit. `ComposedModel.setup_optimizer`'s policy dict order (like every
  `setup_optimizer`) is part of the on-disk optimizer format, but *which* roles a given tree
  actually produces varies with tree content — a composed checkpoint's optimizer shard is only
  guaranteed to reload against the same config it was saved with, not across an edited tree.
  `ComposedConfig.n_layer` is a derived property (total block count, however the composer arranges
  them), not a stored field — `--arch-opt n_layer=...` correctly rejects it.

## What runs on this Mac

Dev machine: Apple Silicon (M4), macOS, **no CUDA**. `COMPUTE_DTYPE` defaults to `float32` here
(see `nanochat/common.py`'s `_detect_compute_dtype`). Set up with:

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
gradient reduction in `nanochat/optim.py`. Keep changes to those paths conservative and prefer
reasoning from the code plus the existing (CUDA-gated) tests over "I ran it and it worked."

**Do not attempt large multi-hour training runs in this environment** (no GPU, thermal/power
constraints of a laptop) — use tiny smoke configs (see
[docs/architecture.md](docs/architecture.md#verifying-a-change-is-behavior-preserving)) to check
plumbing, not to produce a usable model.

## Style

Match the surrounding code: minimal comments explaining *why*, not *what*; no giant config
objects or factory indirection beyond what `nanochat/model/`'s registry already adds; prefer
extending an existing module over adding a new abstraction layer.
