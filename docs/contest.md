# Running the architecture contest on RunPod

`runs/contest.sh` trains all three registered architectures (`gpt`, `llama`, `llama_kvshare`) on
the same tokenizer and the same compute budget, so the comparison is actually apples-to-apples;
then it SFT (chat) fine-tunes and `chat_eval`s each resulting base checkpoint, so the contest
compares both base *and* chat models, not base alone. It leaves everything needed to compare them
locally in the checkpoint directory. This page is the runbook for the piece that costs real money:
renting the GPUs. **Nothing in this repo does that automatically** — provisioning a pod is a
manual step you take deliberately.

A persistent-volume pipeline (pre-staged data/tokenizer, checkpoints written straight to durable
storage, no live rsync-while-billing) is planned but not yet built — this doc still describes the
single-ephemeral-pod workflow. Until then, the tokenizer step below is already free (committed to
the repo, see "One-time setup"), which removes most of what a volume would have saved anyway.

Do not skip the dry run in step 3. It costs nothing and tells you exactly what you're about to
spend before a single GPU-second runs.

## 1. Pod spec

- **4x A100 80GB** (SXM or PCIe, secure or community cloud, whichever's in stock) —
  **4x A100 40GB is the nominal target but is frequently NOT orderable**: RunPod's real 40GB SKU
  (`NVIDIA A100-SXM4-40GB`) caps out at 2 GPUs in community cloud and 0 in secure, with LOW stock,
  as of this doc. 80GB works fine at the same settings (`--device-batch-size=16` was already
  conservative for 40GB, so it fits with headroom on 80GB) and was what the verified run in
  "Verified on real cloud GPUs" below actually used, at $1.39/hr (community) to $1.59/hr (secure)
  per GPU. Check live stock with `list-gpu-types`/`get-gpu-type` (Runpod MCP) or `runpodctl gpu
  list` before committing to a data center.
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
```

`runs/contest.sh` handles the rest of setup (venv, dataset shards, tokenizer) itself unless you
pass `SKIP_SETUP=1`. The tokenizer specifically is **free**: `nanochat/default_tokenizer/`
(committed to the repo, 532KB — content-derived, no machine identity, so a checked-in copy is
exactly as valid as a freshly-trained one) is copied into `$NANOCHAT_BASE_DIR/tokenizer/`
automatically if it's not already there, so `scripts.tok_train` never runs on a fresh pod at all.

**Wandb credentials are sourced automatically now — no manual login step.** `runs/contest.sh` and
`runs/contest_d12.sh` both do `[ -f /etc/rp_environment ] && source /etc/rp_environment` near the
top. This closes a real gap found running this for real: a RunPod account Secret referenced as
`{{ RUNPOD_SECRET_WANDB_API_KEY }}` in a pod's env (create-pod's `env` field) does **not** resolve
into the process environment the way you'd expect via the API used by the RunPod MCP/`create-pod`
tool as of this doc — it lands in `/etc/rp_environment` on the pod, which `~/.bashrc` only sources
under an **interactive**-shell guard, so a script-launched (non-interactive) run used to crash with
`api_key not configured (no-tty)` even though the secret really was there — which is also why
`wandb login` could report "already logged in" in a human's interactive SSH session while a script
launched right after still failed. Sourcing the file directly in the scripts themselves means this
now just works regardless of how the script is launched. `WANDB_RUN=dummy` still skips wandb
entirely if you don't want it at all.

## 3. Always dry-run first

```bash
DRY_RUN=1 bash runs/contest.sh mycontest
```

This downloads nothing extra, trains nothing, and prints exactly what `scripts/model_info.py`
(Stage 4) computes for every row: shape, params, FLOPs/token, KV slots, and a GPU-hours / dollar
estimate (`GPU_NAME`/`MFU`/`PRICE_PER_GPU_HOUR` env vars feed that last part — the defaults are
`"NVIDIA A100"`, `0.4`, `$1.50/GPU-hour`; adjust `PRICE_PER_GPU_HOUR` to your actual RunPod rate).
`GPU-hours` here is total GPU-*resource* consumption — what you're billed at a per-GPU-hour rate —
which does **not** shrink with more GPUs (the same total FLOPs cost the same resource-hours whether
spread across 1 GPU or 8); the preflight table also prints a separate wall-clock estimate, which
*does* shrink with `NPROC_PER_NODE`. (An earlier version of this script conflated the two — divided
by `num_gpus` once to get wall-clock time, then labeled that number "GPU-hours" and fed it straight
into a per-GPU-hour price, undercounting the real dollar cost by a factor of `num_gpus`. Fixed; if
you see a `"GPU-hours"` figure elsewhere that's roughly `1/num_gpus` of the numbers below, it's the
old bug's output.)

At the defaults (`TARGET_FLOPS=5e18`, d16, 4x A100, `--mfu 0.4`), expect:

| arch | scaling params | FLOPs/token | KV slots | GPU-hours |
|---|---|---|---|---|
| gpt | 234.9M | 1.585e9 | 16 | ~11.13 |
| llama | 239.1M | 1.837e9 | 16 | ~11.13 |
| llama_kvshare | 222.3M | 1.736e9 | 8 | ~11.13 |

Total ≈ **33.4 GPU-hours** ≈ **8.4h wall clock** on 4 GPUs ≈ **$46–53** at 80GB-A100 rates
($1.39/hr community to $1.59/hr secure, per GPU — see "Pod spec" above). (All three land at the
same GPU-hours by construction — `TARGET_FLOPS` is the same for every row, that's what "iso-FLOPs"
means; a cheaper architecture spends the saved compute on more tokens instead of finishing early.)
`--mfu 0.4` is optimistic for the SDPA fallback (see "What to watch for" below) — the verified d12
shakedown below measured 33–58% depending on architecture; at a more conservative `--mfu 0.33` the
same contest is ≈40.5 GPU-hours ≈ 10.1h wall clock ≈ $56–64. **This is a real, half-day, ~$50 run —
size accordingly, and consider the d12 shakedown (below) first if you haven't run this harness on
real cloud GPUs yet.**

If the numbers look wrong (wrong depth, wrong GPU count, unexpected `--arch-opt`), fix the
`CONTEST_ROWS` array at the top of `runs/contest.sh` or the env vars and dry-run again. Only once
this table looks right, drop `DRY_RUN` and let it actually train.

## 4. Launch for real

```bash
screen -L -Logfile contest.log -S contest bash runs/contest.sh mycontest
# detach: Ctrl-A D. Reattach: screen -r contest
```

`mycontest` becomes the run label: base results land in
`$NANOCHAT_BASE_DIR/contest_mycontest_results/results.csv`, checkpoints in
`$NANOCHAT_BASE_DIR/base_checkpoints/contest_mycontest_{arch}_d{depth}/`. Chat (SFT) results land
in `chat_results.csv` in the same results dir, checkpoints in
`$NANOCHAT_BASE_DIR/chatsft_checkpoints/contest_mycontest_{arch}_d{depth}/` — same tag, different
namespace, so base and chat checkpoints pair up 1:1 by tag. Omit the label to default to today's
date, matching `runs/miniseries.sh`'s convention.

**Logging**: `WANDB_RUN` defaults to `contest_<label>` and now authenticates automatically (see
"One-time setup" above — no `wandb login` step needed). Set `WANDB_RUN=dummy` to skip wandb
entirely (no plots, just the CSVs and the terminal logs).

### What to watch for in the log

- **`✓ Using Flash Attention 3`** near the top of each row's log. FA3 loads for A100 (sm80) via the
  `kernels` hub but isn't guaranteed — if you instead see `WARNING: Flash Attention 3 not available
  (<reason>), using PyTorch SDPA fallback`, training still runs correctly, just slower (this is
  what every local Mac verification of this repo exercises, since there's no CUDA here). The
  `<reason>` is real now (`nanochat.flash_attention.FA3_LOAD_ERROR`) — it used to be silently
  discarded (`except Exception: return None` with no logging), so a genuinely fixable failure (HF
  hub unreachable, a broken `kernels` import, ...) was indistinguishable from "this GPU just
  doesn't have a kernel." If you want to dig further on a live pod:
  `python -c "from kernels import get_kernel, has_kernel; import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_capability()); print(has_kernel('kernels-community/flash-attn3'))"`.
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
rsync -avz --include='model_*.pt' --include='meta_*.json' --exclude='optim_*' \
    pod:/workspace/.cache/nanochat/chatsft_checkpoints/contest_mycontest_*/ \
    ~/.cache/nanochat/chatsft_checkpoints/contest_mycontest_TAG/
rsync -avz pod:/workspace/.cache/nanochat/contest_mycontest_results/ ~/.cache/nanochat/contest_mycontest_results/
```

**Deliberately excluded: `optim_*_rank*.pt`.** They're sharded per rank (ZeRO-2), collectively
~2x the model weights, and useless for anything this doc does (inference, eval, sampling) — only
`--resume-from-step`/`chat_sft --load-optimizer` read them, and you're not resuming a finished pod.

**No need to bring the tokenizer back anymore** — the pod used the same repo-committed
`nanochat/default_tokenizer/` this machine already has (see "One-time setup"), so there's nothing
to sync. This also means the tokenizer-mismatch trap the fingerprint check below exists for
shouldn't come up for a normal contest run; it stays relevant if you deliberately mix in a
checkpoint trained some other way (e.g. `runs/runcpu.sh`, which still trains its own tokenizer).

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

For the **chat (SFT)** checkpoints, `chat_results.csv` already has per-task accuracy and the
ChatCORE metric from the contest run itself (see "Launch for real"); to re-run or dig into a
specific task locally:

```bash
python -m scripts.chat_eval -i sft -g contest_mycontest_gpt_d16 -a GSM8K --device-type mps
```

Each chat checkpoint's `meta.json` also records `base_model_tag`/`base_model_step` — which exact
base checkpoint it was fine-tuned from — so a downloaded SFT checkpoint is traceable on its own,
without needing the run's CSV.

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
- **The SFT step has no iso-FLOPs budget of its own.** It's `--num-iterations=-1` (a full epoch of
  the SmolTalk+GSM8K+MMLU mixture) by default for every row, matching upstream nanochat's own SFT
  convention (`runs/speedrun.sh`) — not compute-matched the way base training is. Size this into
  the total budget separately: roughly one SFT run's worth of compute per architecture, on top of
  the base contest's cost.

## Local rehearsal (no cloud, plumbing check only)

To confirm the harness itself works before touching a cloud account (this is exactly how this
stage was verified — no A100 was rented to write the original version of this doc; the base+SFT
extension below was likewise verified as a full CPU/MPS rehearsal before ever running on a pod):

```bash
NPROC_PER_NODE=1 DEVICE_BATCH_SIZE=2 SKIP_SETUP=1 WANDB_RUN=dummy \
EXTRA_TRAIN_ARGS="--depth=2 --num-iterations=3 --max-seq-len=128 --total-batch-size=256 --core-metric-every=-1 --eval-tokens=2048" \
EXTRA_SFT_ARGS="--num-iterations=3 --max-seq-len=128 --eval-every=-1 --chatcore-every=200 --mmlu-epochs=0 --gsm8k-epochs=0" \
EXTRA_CHATEVAL_ARGS="--max-problems=2 --task-name=ARC-Easy" \
bash runs/contest.sh rehearsal
```

`EXTRA_TRAIN_ARGS`/`EXTRA_SFT_ARGS`/`EXTRA_CHATEVAL_ARGS` are each appended last to every call of
their respective script, so they override each row's own settings and force a tiny toy run
regardless of size — `EXTRA_CHATEVAL_ARGS` above also narrows `chat_eval` to one fast task
(`ARC-Easy`, capped at 2 problems) instead of its default of all five tasks, since the full
generative sweep (GSM8K/HumanEval sampling included) is slow even at toy sizes. This proves the
launcher selection (`torchrun` vs. plain `python`, shared by `base_train` and `chat_sft` via the
same `launch_module` helper), per-row skip-on-resume (base *and* chat, independently), tokenizer
sharing, distinct arch-qualified checkpoint tags (both `base_checkpoints/` and
`chatsft_checkpoints/`), base→SFT provenance stamping, and log parsing all work — but because the
*preflight* JSON (used for the CSV's static params/FLOPs/KV columns) still reflects each row's
nominal `--depth=16`, not the `EXTRA_TRAIN_ARGS`-overridden depth actually trained, `results.csv`'s
static columns are wrong for a rehearsal run specifically (the dynamic columns — val bpb,
iterations, tokens trained, and everything in `chat_results.csv` — are correct, read from the real
logs). `scripts.model_info --checkpoints` on the resulting checkpoints reports the true trained
shape, since it reads the checkpoint's own saved config rather than the preflight file. This
mismatch cannot happen in a real contest run, which never overrides `--depth` this way.
`NPROC_PER_NODE=1` also switches the launcher to plain `python` (no `torchrun`), so this works on a
CPU/MPS machine with no CUDA at all.

## Cloud shakedown: `runs/contest_d12.sh`

Before spending ~$50 on the real d16 contest, prove the harness on real cloud GPUs cheaply: same
three rows, `--depth=12` and `TARGET_FLOPS=1e18` instead of `--depth=16`/`5e18` — about **1/33rd**
the GPU-hours (a d12 row is both shallower *and*, at a fixed FLOPs budget, needs proportionally
fewer tokens than d16 despite the "cheaper architectures get more tokens" effect within a single
depth). Kept as its own file (`runs/contest_d12.sh`) rather than a `CONTEST_ROWS` edit to
`runs/contest.sh`, so that script's committed defaults stay the real d16 contest:

```bash
DRY_RUN=1 bash runs/contest_d12.sh d12test   # always dry-run first, same as the real contest
bash runs/contest_d12.sh d12test             # ~1.7 GPU-hours, ~$10 at $1.50/GPU-hour reference rate
```

**Verified end-to-end on a real 4x A100-SXM4-80GB pod** (secure cloud, US-MD-1, $6.36/hr for the
pod):

| arch | val bpb | CORE | wall-clock | MFU achieved |
|---|---|---|---|---|
| gpt | 0.8388 | **0.1509** | 40.4 min | 33% |
| llama | 0.8746 | 0.1110 | 23.2 min | 57% |
| llama_kvshare | 0.8732 | 0.1191 | 23.1 min | 58% |

FA3 did not load on this pod (SDPA fallback, as this doc already warns is possible) — gpt's
`SSSL` sliding-window pattern paid for that noticeably more than llama's/llama_kvshare's plain full
attention (33% MFU vs 57–58%), which is exactly the "GPT will simply be slower per step" case
called out in "What to watch for" above. gpt still won on quality (best val bpb and CORE) despite
being slowest — a genuine result, not a harness artifact. Total wall clock for all three rows:
86.6 minutes; total pod time including setup (venv, 46 shards, tokenizer training) was closer to
100 minutes, at $6.36/hr ≈ **$10.60** — close to the corrected estimate above.

Real gotchas hit running this for real (all fixed in the code/docs you're reading now, documented
here so they don't need rediscovering): the GPU-hours/cost formula bug described in "Always
dry-run first" above; the wandb-secret-sourcing gotcha in "One-time setup" above; the missing FA3
failure reason noted in "What to watch for" above (FA3 fell back to SDPA on this run with no
diagnostic at the time); a `column: command not found` on this pod's minimal image, breaking the
script's final pretty-printed summary (fixed — falls back to plain `cat` when `column` isn't
installed); and **4x A100 40GB was not orderable** at the time of this run (see "Pod spec" above)
— used 4x A100-SXM4-80GB instead, which needed no other changes.

**Note:** the table above is from the base-only version of this run, before the SFT/chat extension
in "Launch for real" existed — it's been verified locally (CPU/MPS rehearsal, above) but not yet on
a real pod. Re-running `contest_d12.sh` on a real pod (optionally narrowed to one architecture via
`CONTEST_ROWS`, e.g. just `llama_kvshare`, for the cheapest possible real-cloud validation) is the
natural next real-money step before trusting the SFT extension for the full d16 contest.

## Lessons from the first real cloud run

Everything below was found running this harness for real (not in local rehearsal) and is now
fixed in the code/docs on this page — kept here as one place to check before the next real run,
rather than scattered across commit messages.

- **The GPU-hours/cost estimate was wrong by a factor of `num_gpus` (4x).** `scripts/model_info.py`
  divided by `num_gpus` to compute wall-clock time, then labeled that number "GPU-hours" and fed it
  straight into a per-GPU-hour price — undercounting real dollar cost by 4x. The estimate said
  ~$2.50/25min for the d12 shakedown; the real run cost ~$10.60/100min. Fixed: `gpu_hours` is now
  true GPU-resource-hours (independent of `num_gpus`), with a separate `wall_clock_hours` field for
  the ETA display. See "Always dry-run first" above for the corrected numbers.
- **wandb credentials didn't reach a non-interactively-launched training script**, even though the
  RunPod Secret genuinely resolved into `/etc/rp_environment` — the interactive-shell guard in
  `.bashrc` was the gap. Now sourced automatically by the scripts themselves; see "One-time setup"
  above.
- **FA3 silently discarded its own failure reason.** `except Exception: return None` with no
  logging meant a real run falling back to SDPA looked identical whether the cause was fixable
  (HF hub unreachable, a broken import) or not (genuinely unsupported hardware). Now captured into
  `nanochat.flash_attention.FA3_LOAD_ERROR` and printed in the fallback warning; see "What to watch
  for" above.
- **GPT's "total params" column overstates its size relative to compute.** GPT's `value_embeds`
  (a per-layer embedding lookup added into the attention value stream, alternating layers, ~150M
  params at d12) is *not* a matmul — it costs `O(kv_dim)` per token regardless of table size, so
  it's excluded from both `scripts/model_info.py`'s "scaling params" figure and FLOPs/token
  (deliberately, matching the repo's existing Chinchilla-style accounting: embedding lookups aren't
  "flops"). **"Scaling params," not "total params," is the fair axis for comparing architectures**
  — at d12, gpt/llama both report ~110M scaling params despite gpt's total params being over 2x
  llama's. It isn't free, though: dense gradient reduce + 2 fp32 AdamW moment buffers for that
  table, amortized per step, plus a meaningfully larger on-disk checkpoint (fp32 weights scale with
  *total* params, not scaling params).
- **An accidental key exposure.** While debugging the wandb gap above, a `grep -i wandb
  /etc/rp_environment` printed the real API key in plaintext into a conversation transcript instead
  of just checking whether it was set. The user was asked to rotate the key immediately. The fix
  going forward: check a secret's *presence/length* (`echo ${#VAR}`), never grep or print its
  *value* — even when the whole point is confirming it resolved.
