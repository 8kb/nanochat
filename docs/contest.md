# Running the architecture contest on RunPod

`runs/contest.sh` trains all three registered architectures (`gpt`, `llama`, `llama_kvshare`) on
the same tokenizer and the same compute budget, so the comparison is actually apples-to-apples,
then leaves everything needed to compare them locally in the checkpoint directory. This page is
the runbook for the piece that costs real money: renting the GPUs. **Nothing in this repo does
that automatically** — provisioning a pod is a manual step you take deliberately.

Do not skip the dry run in step 3. It costs nothing and tells you exactly what you're about to
spend before a single GPU-second runs.

## 1. Pod spec

- **4x A100 40GB** (RunPod community or secure cloud, whichever's available). The defaults below
  (`TARGET_FLOPS=5e18`, `d16`, `--device-batch-size=16`) are sized for this; see "Sizing" below to
  change it.
- **~100GB persistent volume** at `/workspace`. Budget: ~9GB for 100 pretraining data shards,
  ~4.3GB for the three d16 model checkpoints (fp32 weights: `params * 4 bytes`, so llama_kvshare's
  smaller parameter count is smaller on disk too), ~26MB for the CORE eval bundle, plus optimizer
  shards while training is in flight (see "What to bring back" — these are *not* worth keeping,
  but they exist on disk during the run).
- Any recent PyTorch + CUDA container image; `runs/contest.sh`'s setup step runs `uv sync
  --extra gpu`, which pulls the rest.

## 2. One-time setup on the pod

```bash
export NANOCHAT_BASE_DIR=/workspace/.cache/nanochat
mkdir -p "$NANOCHAT_BASE_DIR"
git clone <this repo's URL> nanochat && cd nanochat
wandb login   # optional but recommended -- see "Logging" below; skip and use WANDB_RUN=dummy otherwise
```

`runs/contest.sh` handles the rest of setup (venv, dataset shards, tokenizer) itself unless you
pass `SKIP_SETUP=1`.

## 3. Always dry-run first

```bash
DRY_RUN=1 bash runs/contest.sh mycontest
```

This downloads nothing extra, trains nothing, and prints exactly what `scripts/model_info.py`
(Stage 4) computes for every row: shape, params, FLOPs/token, KV slots, and a GPU-hours / dollar
estimate (`GPU_NAME`/`MFU`/`PRICE_PER_GPU_HOUR` env vars feed that last part — the defaults are
`"NVIDIA A100"`, `0.4`, `$1.50/GPU-hour`; adjust `PRICE_PER_GPU_HOUR` to your actual RunPod rate).
At the defaults (`TARGET_FLOPS=5e18`, d16, 4x A100), expect:

| arch | scaling params | FLOPs/token | KV slots | GPU-hours |
|---|---|---|---|---|
| gpt | 234.9M | 1.585e9 | 16 | ~2.78 |
| llama | 239.1M | 1.837e9 | 16 | ~2.78 |
| llama_kvshare | 222.3M | 1.736e9 | 8 | ~2.78 |

Total ≈ 8.4 GPU-hours ≈ 2.1h wall clock on 4 GPUs ≈ $12–15 at community-cloud A100 rates. (All
three land at the same GPU-hours by construction — `TARGET_FLOPS` is the same for every row,
that's what "iso-FLOPs" means; a cheaper architecture spends the saved compute on more tokens
instead of finishing early.)

If the numbers look wrong (wrong depth, wrong GPU count, unexpected `--arch-opt`), fix the
`CONTEST_ROWS` array at the top of `runs/contest.sh` or the env vars and dry-run again. Only once
this table looks right, drop `DRY_RUN` and let it actually train.

## 4. Launch for real

```bash
screen -L -Logfile contest.log -S contest bash runs/contest.sh mycontest
# detach: Ctrl-A D. Reattach: screen -r contest
```

`mycontest` becomes the run label: results land in
`$NANOCHAT_BASE_DIR/contest_mycontest_results/results.csv`, checkpoints in
`$NANOCHAT_BASE_DIR/base_checkpoints/contest_mycontest_{arch}_d{depth}/`. Omit the label to default
to today's date, matching `runs/miniseries.sh`'s convention.

**Logging**: `WANDB_RUN` defaults to `contest_<label>`, which requires `wandb login` (step 2). Set
`WANDB_RUN=dummy` to skip wandb entirely (no plots, just the CSV and the terminal log).

### What to watch for in the log

- **`✓ Using Flash Attention 3`** near the top of each row's log. FA3 loads for A100 (sm80) via the
  `kernels` hub but isn't guaranteed — if you instead see `WARNING: Flash Attention 3 not
  available, using PyTorch SDPA fallback`, training still runs correctly, just slower (this is what
  every local Mac verification of this repo exercises, since there's no CUDA here).
- **GPT's row runs with `window_pattern=SSSL`** (its own architecture default; Llama and
  LlamaKVShare default to `L`, full attention) — this is intentional (each architecture competes as
  its author defined it, not with a pattern forced to match). If FA3 didn't load, the SDPA fallback
  has no sliding-window kernel and will print its own warning; GPT's row will simply be slower per
  step than the other two, not wrong.
- **`Total training FLOPs estimate`** for each row should be within rounding of `TARGET_FLOPS`
  regardless of architecture — that's the iso-FLOPs contract holding.

### Interruption / resume

`runs/contest.sh` is idempotent per row: re-running the same command with the same label skips any
row whose label/arch/depth is already a line in `results.csv`. If a spot instance dies mid-row,
that row's checkpoint directory has no *final*-step checkpoint (only `--save-every=-1` intermediate
saves, i.e. none, unless you changed that), so just re-launch the same command — the finished rows
skip, the interrupted one restarts from scratch (base_train.py's own `--resume-from-step` is not
wired into `runs/contest.sh`; a full row is short enough at these depths that resuming mid-row
isn't worth the complexity).

## 5. Bring the results home

```bash
rsync -avz --include='model_*.pt' --include='meta_*.json' --exclude='optim_*' \
    pod:/workspace/.cache/nanochat/base_checkpoints/contest_mycontest_*/ \
    ~/.cache/nanochat/base_checkpoints/contest_mycontest_TAG/
rsync -avz pod:/workspace/.cache/nanochat/contest_mycontest_results/ ~/.cache/nanochat/contest_mycontest_results/
rsync -avz pod:/workspace/.cache/nanochat/tokenizer/ ~/.cache/nanochat/tokenizer/   # only if this machine has no tokenizer yet
```

**Deliberately excluded: `optim_*_rank*.pt`.** They're sharded per rank (ZeRO-2), collectively
~2x the model weights, and useless for anything this doc does (inference, eval, sampling) — only
`--resume-from-step`/`chat_sft --load-optimizer` read them, and you're not resuming a finished pod.

**Do bring the tokenizer** if this machine doesn't already have the one the pod trained with (step
2's setup trains a fresh one if `$NANOCHAT_BASE_DIR/tokenizer/` is empty). If this machine already
has a tokenizer — e.g. from `runs/runcpu.sh` — do **not** overwrite it; the whole point of the
tokenizer fingerprint below is to catch exactly this mismatch, and the fix is to use the *pod's*
tokenizer, not silently keep a different local one.

## 6. Compare locally

No GPU needed for any of this.

```bash
# Static + trained-model comparison for all three, side by side, straight from meta.json:
python -m scripts.model_info --checkpoints "contest_mycontest_gpt_d16,contest_mycontest_llama_d16,contest_mycontest_llama_kvshare_d16"
# or, to sweep every contest checkpoint in one go:
python -m scripts.model_info --checkpoints "$(ls ~/.cache/nanochat/base_checkpoints | grep contest_mycontest | paste -sd, -)"
```

Each row reports params/FLOPs/KV-cache (from the checkpoint's own saved config, not a guess) plus
what training actually produced: step, tokens trained, val bpb, CORE (if it was evaluated —
`runs/contest.sh` only evaluates CORE on the final step), wall-clock, and a **tokenizer fingerprint
status** (`match` / `MISMATCH` / `unknown`). Take a `MISMATCH` seriously: it means this checkpoint
was trained against a different tokenizer than the one currently at
`~/.cache/nanochat/tokenizer/`, and outputs will be garbage even though the vocab *size* still
lines up (see `AGENTS.md`'s tokenizer-fingerprint invariant for why this check exists at all).

For qualitative samples, on CPU/MPS (works fine for inference at these sizes, unlike training):

```bash
python -m scripts.base_eval --model-tag contest_mycontest_gpt_d16 --eval sample --device-type mps
```

`--eval sample` alone skips the (slow-on-CPU) CORE/bpb passes. `base_eval.py`'s CSV output is now
named after the resolved model tag (`base_eval/<tag>_<step>.csv`), so evaluating all three in the
same `NANOCHAT_BASE_DIR` no longer overwrites one file three times.

## Sizing: changing the contest

Everything above assumes the defaults. To change what's being compared:

- **Depth**: edit the `16` in each `CONTEST_ROWS` entry (all three must move together to stay
  comparable), or pass `--depth=<N>` if you fork the script per-arch. Re-run the dry run — the
  budget table updates automatically.
- **Compute budget**: `TARGET_FLOPS` env var. `scripts/model_info.py --arch ... --depth ...
  --target-flops=<N>` (Stage 4) tells you the resulting token count and GPU-hours before you commit.
- **Iso-params instead of iso-FLOPs**: this repo's default matches compute, not parameter count
  (`gpt` has ~2x llama_kvshare's *total* params at d16 because of its tied value-embedding/lm_head
  structure, but nearly identical *scaling* params — see the table above). To match params instead,
  tune `llama_kvshare`'s `kv_share_frac` (via `--arch-opt kv_share_frac=<f>` in its `CONTEST_ROWS`
  entry) until `scripts/model_info.py`'s params column lines up, then switch `TARGET_FLOPS` to
  `--target-param-data-ratio` in both the preflight and the row's args if you want each model
  individually compute-optimal instead of budget-matched.
- **Data**: `NUM_SHARDS` (default 100). At `TARGET_FLOPS=5e18`, gpt's row (the most token-hungry
  of the three) needs ≈3.16B tokens; this tokenizer's vocab averages ≈4.7 characters/token on this
  dataset, and the BOS-aligned dataloader keeps ≈65% of tokens after cropping (see
  `nanochat/dataloader.py`), so each ~253M-character shard yields ≈35M usable training tokens —
  ≈91 shards needed, 100 leaves a ~10% margin. Raise `NUM_SHARDS` if you raise `TARGET_FLOPS` or
  `--depth` significantly. The validation shard (always the last one, `shard_06542.parquet`) is
  identical across every row regardless of `NUM_SHARDS`, which is what makes the three val-bpb
  numbers comparable to each other.
- **`--fp8`** is not wired into `runs/contest.sh` and is H100-only (`nanochat/fp8.py`) — irrelevant
  on A100s; if you move the contest to H100s, add `--fp8` to each row's args and expect a real
  speedup, but note it changes precision, so keep it on or off for all three rows equally.

## Local rehearsal (no cloud, plumbing check only)

To confirm the harness itself works before touching a cloud account (this is exactly how this
stage was verified — no A100 was rented to write this doc):

```bash
NPROC_PER_NODE=1 DEVICE_BATCH_SIZE=2 SKIP_SETUP=1 WANDB_RUN=dummy \
EXTRA_TRAIN_ARGS="--depth=2 --num-iterations=3 --max-seq-len=128 --total-batch-size=256 --core-metric-every=-1 --eval-tokens=2048" \
bash runs/contest.sh rehearsal
```

`EXTRA_TRAIN_ARGS` is appended last to every `base_train.py` call, so it overrides each row's
`--depth`/`--target-flops` and forces a 3-step toy run regardless of size. This proves the launcher
selection (`torchrun` vs. plain `python`), per-row skip-on-resume, tokenizer sharing, distinct
checkpoint tags, and log parsing all work — but because the *preflight* JSON (used for the CSV's
static params/FLOPs/KV columns) still reflects each row's nominal `--depth=16`, not the
`EXTRA_TRAIN_ARGS`-overridden depth actually trained, `results.csv`'s static columns are wrong for
a rehearsal run specifically (the dynamic columns — val bpb, iterations, tokens trained — are
correct, read from the real log). `scripts.model_info --checkpoints` on the resulting checkpoints
reports the true trained shape, since it reads the checkpoint's own saved config rather than the
preflight file. This mismatch cannot happen in a real contest run, which never overrides `--depth`
this way. `NPROC_PER_NODE=1` also switches the launcher to plain `python -m scripts.base_train`
(no `torchrun`), so this works on a CPU/MPS machine with no CUDA at all.
