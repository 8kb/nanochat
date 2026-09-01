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
- **`GPT.num_scaling_params()` and `GPT.setup_optimizer()` both assert they cover every
  parameter exactly once.** Add a new `nn.Parameter` or submodule to `GPT` and forget to add it
  to both, and you get an `AssertionError` at model construction or at optimizer setup — annoying,
  but far better than a parameter silently missing from the optimizer or the parameter count.
- **`runs/scaling_laws.sh` and `runs/miniseries.sh` grep exact stdout text** out of
  `scripts/base_train.py` (lines matching `^wte `, `^lm_head `, `CORE metric:`,
  `Validation bpb:`). Changing those print statements' format breaks those scripts silently.
- **Rotary `cos`/`sin` buffers are `persistent=False`** (not saved in checkpoints) — this is why
  `checkpoint_manager.build_model` calls `model.init_weights()` even when *loading* a checkpoint,
  right before `load_state_dict(..., assign=True)` overwrites everything else.
- **Checkpoint `model_config` carries an `"arch"` key** (from `BaseModelConfig.to_dict()`), read
  by `nanochat.model.registry.config_from_dict` to pick the right config/model class. Missing
  `"arch"` (checkpoints saved before Stage 1) defaults to `"gpt"`.

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
