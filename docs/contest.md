# Running the architecture contest on RunPod

`runs/contest.sh` trains all four registered architectures (`gpt`, `llama`, `llama_kvshare`,
`llama_kvshare_win`) on the same tokenizer and the same compute budget, so the comparison is
actually apples-to-apples;
then it SFT (chat) fine-tunes and `chat_eval`s each resulting base checkpoint, so the contest
compares both base *and* chat models, not base alone. It leaves everything needed to compare them
locally in the checkpoint directory. This page is the runbook for the piece that costs real money:
renting the GPUs. **Nothing in this repo does that automatically** — provisioning a pod is a
manual step you take deliberately.

A persistent Network Volume now exists and is pre-staged (see "Attaching the persistent volume"
below): `3w7toelc6z` ("nanochat-contest-archive"), 50GB Standard tier, **US-KS-2**, holding all
101 data shards, the tokenizer, the CORE eval bundle, and a full `.venv` (`uv sync --extra gpu`
already run). A pod that mounts it at `/workspace` skips setup almost entirely. The rest of this
doc still also describes the plain single-ephemeral-pod workflow (no volume) for a one-off run
that doesn't want persistent infra, or for GPU types in a different data center than the volume
(see the DC-mismatch note below).

Do not skip the dry run in step 3. It costs nothing and tells you exactly what you're about to
spend before a single GPU-second runs.

## 0. Authenticate `runpodctl` (only needed for attaching an existing network volume)

The RunPod MCP tools (used for everything else in this doc — creating pods, checking GPU stock,
creating/deleting network volumes) work via OAuth with no key on disk. But attaching an
*existing* network volume to a new pod isn't exposed by the MCP `create-pod` tool (as of this doc
— it only supports creating a fresh pod-local volume disk, not referencing a volume ID; the
underlying RunPod v2 API does support this via `mounts.network[0]`, so it's a gap in the MCP tool
specifically). That one operation needs `runpodctl` instead, which needs a real
`RUNPOD_API_KEY` — get one at console.runpod.io/user/settings → API Keys, scoped to Pod
create/get/terminate only (no serverless/templates/registries/billing/secrets/network-volume
endpoints needed).

**`runpodctl doctor`** is the easy way to set this up interactively (prompts for the key, writes
`~/.runpod/config.toml`, and can also register an SSH key) — simpler than hand-editing the TOML
file. Run it yourself in your own terminal (not scripted, since it prompts). Verify with
`runpodctl user` (prints account info, not the key). `runpodctl doctor` also auto-registers an
SSH key at `~/.runpod/ssh/runpodctl-ssh-key` — use this key (not any other) for every pod created
via `runpodctl`, since RunPod injects account-registered keys at boot.

## 0.5. Attaching the persistent volume

A pod must be created **in the volume's data center** (US-KS-2 for `3w7toelc6z`) to mount it —
this is a real constraint, not a preference: a network volume is DC-pinned, and a pod created
elsewhere simply can't reference it. Check GPU stock in that specific DC before creating the pod
(`get-gpu-type` with the DC in its `dataCenters` list, or `runpodctl gpu list`).

```bash
runpodctl pod create \
  --compute-type gpu \
  --gpu-id "NVIDIA A100-SXM4-80GB" --gpu-count 4 \
  --data-center-ids US-KS-2 \
  --network-volume-id 3w7toelc6z --volume-mount-path /workspace \
  --template-id runpod-torch-v280 \
  --env '{"WANDB_API_KEY":"{{ RUNPOD_SECRET_WANDB_API_KEY }}"}' \
  --wait
```

**The `WANDB_API_KEY` secret must be requested explicitly in `--env` at pod creation** — it is
*not* injected account-wide automatically. Forgetting this (easy to do on a CPU pod that doesn't
need it, then reusing the same habit on a GPU pod that does) silently means no wandb logging.

Once attached, `NANOCHAT_BASE_DIR=/workspace/.cache/nanochat` is already populated (data,
tokenizer, eval bundle) and `/workspace/nanochat/.venv` already has `uv sync --extra gpu` done —
`SKIP_SETUP=1` is safe, or leave setup unset and every step just skips because the files already
exist.

**If a stage's GPU type isn't available in the volume's DC** (expected for H100/H200 — no single
RunPod DC currently offers standard-storage volumes plus A100 *and* H100 *and* H200 stock
together), that pod trains in whichever DC actually has the GPU, writes checkpoints to its own
local disk as normal, and results come home via the ordinary "Bring the results home" rsync step
below rather than a direct volume mount — the volume isn't useless there, it just isn't reachable
from that pod's filesystem directly.

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
  --extra gpu`, which pulls the rest -- including `modelcore`/`datacore` from their own public
  GitHub repos (Stage 10), so the pod needs outbound network access to github.com at sync time.

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
| llama_kvshare_win | 222.3M | 1.510e9 | 8 | ~11.13 |

Total ≈ **44.5 GPU-hours** ≈ **11.1h wall clock** on 4 GPUs ≈ **$62–71** at 80GB-A100 rates
($1.39/hr community to $1.59/hr secure, per GPU — see "Pod spec" above). (All four land at the
same GPU-hours by construction — `TARGET_FLOPS` is the same for every row, that's what "iso-FLOPs"
means; a cheaper architecture spends the saved compute on more tokens instead of finishing early —
`llama_kvshare_win`'s lower FLOPs/token than `llama_kvshare` at an identical param count means it
trains on proportionally more tokens for the same GPU-hours, not fewer.) `--mfu 0.4` is optimistic
for the SDPA fallback (see "What to watch for" below) — the verified d12 shakedown below measured
33–58% depending on architecture; at a more conservative `--mfu 0.33` the same contest is ≈53.9
GPU-hours ≈ 13.5h wall clock ≈ $75–86. **This is a real, half-to-full-day, ~$65-85 run — size
accordingly, and consider the d12 shakedown (below) first if you haven't run this harness on real
cloud GPUs yet.**

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
- **GPT's and `llama_kvshare_win`'s rows run with `window_pattern=SSSL`** (each architecture's own
  default; Llama and LlamaKVShare default to `L`, full attention) — this is intentional (each
  architecture competes as its author defined it, not with a pattern forced to match). If FA3
  didn't load, the SDPA fallback attention runs sliding windows via an explicit mask rather than a
  fused kernel and will print its own warning; GPT's and `llama_kvshare_win`'s rows will simply be
  slower per step than the other two, not wrong.
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

### Watching a long run without getting stale information

A run spans multiple architectures and (with the SFT extension) two training phases each, easily
tens of minutes to hours. Whether you're tailing the log by hand or scripting a poll loop, two real
failure modes showed up watching this project's own runs:

- **A fixed-size `tail -N` window silently stops reporting progress once the log outgrows it.**
  Diffing "new content" by comparing line/byte counts against a capped `tail -n 1000` (or similar)
  plateaus once the file exceeds that window — the count stops growing even though the file still
  is, so a naive polling loop goes quiet forever after that point while the job keeps running.
  Watch the **full file's** size (`wc -c`) or track a byte offset (`tail -c +$OFFSET`) instead of a
  windowed snapshot. This caused a real stuck-looking report mid-run: reported "still at step 900"
  for several updates while the row had actually already finished and training had moved three
  architectures ahead.
- **A single point-in-time check can go stale between when you run it and when you report it.**
  If you ask "is it done yet" and get "still running," that's true only at that instant — always
  re-verify (list the actual process, tail the real log) right before reporting status rather than
  trusting a check from a few minutes/messages earlier, especially right after a step that's known
  to take a while (like a large `uv sync`).
- **`timeout`/`gtimeout` isn't installed on macOS by default** (it's GNU coreutils, not BSD) — a
  script or command assuming it exists fails immediately at that one step rather than running the
  intended long job. Check the actual exit code / that the job you meant to start is actually
  running, don't assume "no visible error" means "did what I intended."

## 5. Bring the results home

**Terminate the GPU pod first — the moment its actual GPU work (training/eval) finishes — before
running anything below.** The checkpoints are already durable on the volume at that point; the
commands here just move files off it, no GPU involved, so running them while the training pod is
still up bills its full per-GPU rate for pure I/O. Once it's terminated, pull the copy either via
the RunPod S3 API against the volume directly (no pod at all, see "Attaching the persistent
volume" above) or, if a pod is genuinely needed, a fresh cheap CPU pod (~$0.06/hr) attached to the
same volume — never the GPU pod that trained. See "Lessons" below for what skipping this cost in
practice the one time it was done wrong.

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
# Static + trained-model comparison for all four, side by side, straight from meta.json:
python -m scripts.model_info --checkpoints "contest_mycontest_gpt_d16,contest_mycontest_llama_d16,contest_mycontest_llama_kvshare_d16,contest_mycontest_llama_kvshare_win_d16"
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

- **Depth**: edit the `16` in each `CONTEST_ROWS` entry (all rows must move together to stay
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
  of the rows) needs ≈3.16B tokens. Tokenization/packing now happens once, offline
  (`scripts/data_prep.py`, replacing the old realtime BOS-aligned dataloader), and the retention
  ratio it actually achieved is a *measured* number, not an estimate — `python -m
  scripts.data_prep --describe --dataset=<name>` reports it per split, and the prep run's own
  stdout does too (previously this was a hand-estimated ≈65% at T=2048; measure your own corpus's
  real number instead of assuming it). Raise `NUM_SHARDS` (and re-run `scripts.data_prep`) if you
  raise `TARGET_FLOPS` or `--depth` significantly. The validation shard (always the last one,
  `shard_06542.parquet`) is identical across every row regardless of `NUM_SHARDS`, which is what
  makes the val-bpb numbers comparable to each other.
- **`--fp8`** is not wired into `runs/contest.sh` and is H100-only (`modelcore/precision/fp8.py`) — irrelevant
  on A100s; if you move the contest to H100s, add `--fp8` to each row's args, but keep it on or off
  for every row equally (it changes precision). Confirmed working on real H100 hardware for the
  first time in "Stage 4 results" below (via the dedicated `runs/contest_fp8_d13.sh`, not
  `runs/contest.sh` itself) — mixed result, not a clean win: ~9.6% faster but a small, consistent
  val-bpb regression and *higher*, not lower, peak memory. Measure before assuming either direction.
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
# SKIP_SETUP=1 skips runs/contest.sh's own data_prep step too, so prepare small toy datasets
# first, at the same --max-seq-len the rehearsal below uses (--mmlu-epochs/--gsm8k-epochs live on
# data_prep now, not on chat_sft.py -- see scripts/data_prep.py --kind=sft):
python -m scripts.data_prep --kind=base --sequence-len=128 --max-shards=2
python -m scripts.data_prep --kind=sft --sequence-len=128 --mmlu-epochs=0 --gsm8k-epochs=0 --max-conversations=500

NPROC_PER_NODE=1 DEVICE_BATCH_SIZE=2 SKIP_SETUP=1 WANDB_RUN=dummy \
EXTRA_TRAIN_ARGS="--depth=2 --num-iterations=3 --max-seq-len=128 --total-batch-size=256 --core-metric-every=-1 --eval-tokens=2048" \
EXTRA_SFT_ARGS="--num-iterations=3 --max-seq-len=128 --eval-every=-1 --chatcore-every=200" \
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

Before spending ~$65-85 on the real d16 contest, prove the harness on real cloud GPUs cheaply:
`--depth=12` and `TARGET_FLOPS=1e18` instead of `--depth=16`/`5e18` — about **1/33rd** the
GPU-hours per row (a d12 row is both shallower *and*, at a fixed FLOPs budget, needs
proportionally fewer tokens than d16 despite the "cheaper architectures get more tokens" effect
within a single depth). Kept as its own file (`runs/contest_d12.sh`) rather than a `CONTEST_ROWS`
edit to `runs/contest.sh`, so that script's committed defaults stay the real d16 contest.
`runs/contest_d12.sh`'s default rows are `gpt`, `llama`, `llama_kvshare_win` — `llama_kvshare`
(without windowing) is intentionally left out here since it was already measured against this
exact pipeline once (see "Stage 1 results" below) and re-training it would just be spending money
to get the same number back; `runs/contest.sh` (the real d16 contest) still trains all four:

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

## Stage 1 results: `llama_kvshare` d12 on the persistent-volume pipeline

The first real run against the persistent-volume pipeline (this doc's "Attaching the persistent
volume" section), narrowed to one architecture (`CONTEST_ROWS` trimmed to just `llama_kvshare`) to
exercise base→SFT→chat_eval end to end at the smallest real cost before trusting it for a full
contest. 4x A100-SXM4-80GB, US-KS-2, volume `3w7toelc6z`:

| step | metric | value |
|---|---|---|
| base training | val bpb | 0.8716 |
| base training | CORE | 0.1307 |
| base training | wall-clock | 23.03 min |
| base training | FA3 | **active** (`✓ Using Flash Attention 3`) |
| SFT | val bpb | 0.3831 |
| SFT | wall-clock | 9.35 min |
| chat_eval (`--max-problems=100`) | ARC-Easy | 47.00% |
| chat_eval | ARC-Challenge | 28.00% |
| chat_eval | MMLU | 30.00% |
| chat_eval | GSM8K | 0.00% |
| chat_eval | HumanEval | 5.00% |
| chat_eval | **ChatCORE** | **0.0900** |

Real cost: ~$6-6.5 GPU time across two pods (the first was killed mid-run to apply the `hf_transfer`
fix below, cleanly; the retry that produced the numbers above includes recovery from the runaway
`chat_eval` incident, also below).

**This is the reference row for the H100 contest below** (`llama_kvshare` itself is intentionally
not re-trained there). `val_bpb`/`CORE`/`ChatCORE` are hardware-independent at identical
iterations/data/seed, so they're directly comparable across the A100→H100 move; `train_time_sec`
and MFU are not (different hardware, and the H100 run additionally has FA3 actually active — this
A100 run's FA3 success came from a fix applied *after* the original silent SDPA fallback, see
"Lessons" below).

## Stage 3 results: `gpt`/`llama`/`llama_kvshare_win` d12 on 4x→2x H100

The full d12 contest against `llama_kvshare_win` (the fourth architecture — `llama_kvshare` plus
sliding-window attention, see `docs/architecture.md`'s worked example), on a fresh persistent
volume in a Hopper-capable DC. `llama_kvshare` itself was intentionally excluded from this run's
rows (already measured above, on this exact pipeline) so the run only pays for new information.

**Infra**: 50GB Standard volume `w6ndh50xcl`, **US-GA-2** — checked live that US-KS-2 (the Stage 1
volume's DC) has zero H100/H200 stock at all, so this needed a second volume in a different DC
(see "Lessons" below for how to pick one). **4x H100 wasn't orderable** despite showing live
stock (`get-gpu-type` reported "LOW", not "NONE") — confirmed by directly probing GPU counts (4x
failed three times; 3x and 2x both succeeded) — ran on **2x H100 SXM 80GB** instead. FA3 confirmed
active directly (`HAS_FA3: True`) and in every row's log — the first real exercise of the Hopper
(`major==9`) code path, which the Stage 1 A100 run never touched.

| arch | val bpb | CORE | ChatCORE | ARC-Easy | ARC-Chal. | MMLU | GSM8K | HumanEval | scaling params | KV cache |
|---|---|---|---|---|---|---|---|---|---|---|
| gpt | 0.8434 | 0.1553 | 0.0833 | 43% | 29% | 29% | 2% | 5% | 110.1M | 75.5MB |
| llama | 0.8791 | 0.1109 | 0.0507 | 37% | 31% | 23% | 0% | 4% | 110.1M | 75.5MB |
| llama_kvshare_win | 0.8693 | 0.1359 | 0.0620 | 47% | 25% | 24% | 0% | 3% | 103.0M | 37.7MB |

`llama_kvshare_win` placed 2nd on both CORE and ChatCORE (behind `gpt`, ahead of `llama`) with the
**fewest scaling params and smallest KV cache** of the three — the KV-sharing + windowing
combination is paying off on quality-per-byte, not just raw quality. Its CORE (0.1359) is also
higher than Stage 1's plain `llama_kvshare` (0.1307) — suggestive that windowing helps at this
depth, though the comparison spans different hardware and FA3 status, so it isn't a controlled
ablation.

**A mid-run tuning fix worth carrying forward**: `DEVICE_BATCH_SIZE=16` (inherited from the A100
default) left only 16-20GB of the H100's 80GB in use — the training loop was overhead-bound
(data loading, Python, DDP sync), not compute-bound, so FA3 and H100's 3.2x higher peak FLOPS
weren't showing up in wall-clock. A clean same-architecture check via wandb's own
`total_training_time` metric (`llama`, no windowing, so FA3-vs-SDPA is a smaller factor): **A100
23.19 min (4 GPUs, no FA3) vs. H100 20.68 min (2 GPUs, FA3) — only ~11% faster**, confirming the
overhead-bound diagnosis. Killing the in-progress `llama_kvshare_win` row (minimal sunk cost — no
checkpoint had saved yet) and relaunching with `DEVICE_BATCH_SIZE=64` pushed memory to 57GB/80GB
(71%, safe) and **MFU from ~30-35% to ~42-44%**, finishing the row in 33.1 min instead of the
~40 min it was on pace for. **Use a batch size sized for the actual GPU's memory, not a value
inherited from a previous GPU class** — checking `nvidia-smi` memory usage early in a run on new
hardware is the concrete, checkable signal.

Checkpoints (base + SFT, no optimizer state) were rsynced home before terminating the pod — RunPod
S3 keys are still deferred, so this remains the only way to get local access without another
billed pod.

## Stage 4 results: FP8 sanity + `kvshare4_win` d13 on H100

Not a multi-architecture contest — one architecture, two base-training-only rows (`bf16` vs.
`--fp8`), run via a dedicated script rather than `runs/contest.sh`/`contest_d12.sh`:
`runs/contest_fp8_d13.sh`. Two questions, answered by one run: does `--fp8` actually work on real
Hopper hardware (it never had — see below), and what does a more aggressive KV-sharing point look
like (4 KV-owning layers of 13, vs. Stage 3's 6 of 12)?

**Why FP8 needed a sanity check at all**: `Float8Linear` used to subclass `torch.nn.Linear` instead
of `modelcore.components.linear.Linear`, so `collect_param_roles` raised `ValueError` and
`create_optimizer` died at startup — `--fp8` was a hard crash on any CUDA box, on every architecture,
always. Fixed on `stage7-modelcore-extraction` (commit `8e59911`) by subclassing core's own
`Linear`, but that fix had only ever run on CPU, where the `_scaled_mm` kernel doesn't execute — see
`AGENTS.md`'s "untested on this machine" list. This run is that fix's first execution on real
hardware.

**Architecture**: `llama_kvshare_win`, depth 13 (not 12 — `compute_window_sizes` forces the final
layer to full context unconditionally, which would have silently turned a 12-layer `...S S` tail
into `...S L`; at 13 layers the pattern already ends in `L`, so the override is a no-op),
`--window-pattern=LLLLSSLSSLSSL --arch-opt kv_share_frac=0.6923`:

```
layer:   1  2  3  4  5  6  7  8  9 10 11 12 13
window:  L  L  L  L  S  S  L  S  S  L  S  S  L
kv slot: 0  1  2  3  3  3  3  3  3  3  3  3  3
```

**Infra**: same volume as Stage 3 (`w6ndh50xcl`, US-GA-2). 4x H100 wasn't orderable (same
"reports stock, still fails to order" pattern as Stage 3) — fell back to 2x H100 SXM 80GB directly
rather than probing further downward, since each successful probe is a real billed pod.
`device-batch-size=64` (`grad_accum=2` at 2 GPUs) confirmed safe via a 20-step probe of each
condition before committing to the full 1,793-iteration run.

| row | val bpb | tok/sec | bf16_mfu¹ | peak mem | fp8 converted | train time |
|---|---|---|---|---|---|---|
| bf16 | 0.881810 | 825,689 | 44.41% | 60.0 GB | — | 21.6 min |
| fp8 | 0.883870 | 904,694 | 48.66%¹ | 69.9 GB | 74/74 | 20.2 min |

¹ `bf16_mfu` divides by *bf16* peak FLOPS for both rows (`base_train.py`), so it isn't a real
ceiling for the fp8 row — `tok/sec` is the honest cross-row number.

**FP8 sanity: confirmed.** All 74 linear layers converted (0 skipped — every shape in this model
is 16-aligned with `min(in,out) >= 128`), no NaN, no crash, `fp8_disabled` round-tripped cleanly
through every eval cycle during training (not just in a unit test). The role/accounting bug fix
holds under real training.

**Quality**: fp8's val bpb is 0.2% worse than bf16 — tracked step-by-step during the run (grepped
both logs at identical steps 560-569, not just the final checkpoint): fp8 ran consistently
~0.004–0.006 higher loss than bf16 across that window, never crossing or diverging. The shape of a
small, stable precision tax, not instability.

**Speed**: fp8 was **~9.6% faster** (904.7K vs 825.7K tok/sec) — a genuine surprise against
upstream's own benchmarking (`docs/upstream/LOG.md`: "d12 was still slower with FP8; d26+ shows
gains"). One run isn't a controlled ablation, but it's a real data point at a different shape
(13 layers, `n_embd=896`, heavy KV sharing) than upstream measured.

**Memory**: fp8 used *more* peak memory than bf16 (69.9GB vs 60.0GB), not less — the opposite of
the ~9GB activation savings other fp8 write-ups describe. Not investigated further; flagged rather
than explained away.

**Cost**: ~53 min of 2x H100 pod time ($6.98/hr) plus ~2 min of CPU pod time for the pull ≈ **$6.20**
total — under the ~$10-14 estimate, since both rows finished slightly faster than planned and no
probe needed a retry.

## Stage 5 results: `kvshare4_win` d13 at Chinchilla ratio, fp8, on H100

Same architecture as Stage 4 (`llama_kvshare_win`, depth 13, `--window-pattern=LLLLSSLSSLSSL
--arch-opt kv_share_frac=0.6923`), but trained to a proper compute-optimal horizon instead of an
arbitrary FLOPs cap: `--target-param-data-ratio=20` (Chinchilla) instead of `--target-flops=1e18`.
Single row, `--fp8` on throughout (no bf16 comparison this time — Stage 4 already established fp8
works; the point here is a real quality number at a real training budget). Base training + a full,
uncapped CORE eval at the end (`--core-metric-every=999999 --core-metric-max-per-task=-1`) — no
SFT, no chat_eval, no wandb (`--run=dummy`).

**Infra**: same volume (`w6ndh50xcl`, US-GA-2), 2x H100 SXM 80GB again (4x reported stock but
failed to order a second time — same pattern as Stages 3 and 4, no further probing). The volume's
58 shards from the Stage 4 run weren't enough for this budget's 2.92B tokens (roughly 84-90 needed
by this doc's own ≈35M-usable-tokens/shard estimate) — topped up to 91 via
`python -m nanochat.dataset -n 90` before training (idempotent: only the missing shards downloaded).
`device-batch-size=64` reused from Stage 4's probe rather than re-probed — same architecture, same
per-step memory footprint, so a fresh probe would only have spent money confirming what was already
known.

| tokens trained | iterations | tokens:param | val bpb (= minimum) | CORE | tok/sec | peak mem | fp8 converted |
|---|---|---|---|---|---|---|---|
| 2,921,857,024 | 5,573 | 20.00 | 0.833913 | **0.1597** | 924,694 (median) | 69,870.63 MB | 74/74 |

**Training was clean end to end**: val bpb decreased monotonically from step 0 (3.170390) to the
final step (0.833913) with no rebound — confirmed by grepping every `Validation bpb` line in the
log, not just trusting the final number. fp8 held up over the full ~53-minute training loop (vs.
Stage 4's ~20 minutes) with peak memory essentially identical to Stage 4's shorter run (69.87GB
both times) — no drift, no NaN, no crash.

**Vs. Stage 4's arbitrary-FLOPs run**: val bpb improved 0.883870 → 0.833913 (~6% lower) on 3.1x
more tokens (2.92B vs. 940M) at the proper ratio instead of a FLOPs budget picked for cheapness —
exactly the direction you'd expect, and a useful confirmation that `--target-param-data-ratio`
behaves sanely on this architecture.

**Vs. this repo's own Stage 3 d12 contest** (`gpt` CORE 0.1553, `llama` 0.1109, `llama_kvshare_win`
0.1359, all at a d12/~1e18-ish budget): 0.1597 is the best CORE recorded in this repo so far — but
not a controlled ablation against those three, since this run differs in both depth (13 vs. 12) and
training horizon (Chinchilla ratio vs. an arbitrary FLOPs cap).

**Vs. upstream nanochat's own leaderboard** (`docs/upstream/README.md`): GPT-2's own CORE is
0.2565, and nanochat's `d24`/`d26`-class leaderboard entries score 0.2578-0.2690 at ~730-918M
scaling params and ~9-11B tokens over 13-24 GPU-hours on an 8xH100 node. This run's 0.1597 sits
well below that threshold — expected, not a regression, given ~5x fewer scaling params (146.1M),
~3-4x fewer tokens, and ~7-12x less compute (1.94 GPU-hours). The val-bpb comparison is the more
meaningful one: this fork and nanochat's leaderboard runs #4-6 both train on NVIDIA ClimbMix (the
README itself flags runs #1-3 as using a different, non-comparable dataset before that switch), so
0.8339 vs. their 0.7180-0.7185 is a real same-data comparison — worse, exactly as scale predicts.
The CORE methodology itself lines up cleanly (this run's log shows the same 22-task DCLM battery
upstream describes), so 0.1597 is comparable in *kind*, just not in scale.

**Cost**: pod ran 81.3 min on 2x H100 ($6.98/hr) ≈ **$9.45**, plus a negligible CPU-pod pull ≈
**$9.50** total — the shard top-up, the longer training loop (52.61 min vs. Stage 4's ~20), and the
full uncapped CORE eval (~7 min across 22 tasks, vs. Stage 4's `-1`/disabled) all add up relative
to Stage 4's $6.20.

## Stage 6 results: `kvshare4_win` d13 at tokens:param ratio 10, on the `datacore` pipeline

Same architecture as Stages 4-5 (`llama_kvshare_win`, depth 13, `--window-pattern=LLLLSSLSSLSSL
--arch-opt kv_share_frac=0.6923`), `--fp8`, but at a cheaper horizon
(`--target-param-data-ratio=10`, half of Stage 5's Chinchilla ratio) and, more importantly, the
first real base-training run to go through the Stage 9 `datacore` pipeline end to end: a
`scripts.data_prep`-prepared dataset (`climbmix_t2048_e348819205de14ab`) read via `DataManager`,
not raw parquet. CORE disabled (`--core-metric-every=-1`) — val bpb is the axis this stage cares
about, not a leaderboard number. A full SFT pass followed (1 epoch on the matching
`sft_t2048_e348819205de14ab` dataset) as a second, independent exercise of the pipeline; its number
is recorded here but isn't part of the base-training ablation below.

```
torchrun --standalone --nproc_per_node=2 -m scripts.base_train -- \
  --arch=llama_kvshare_win --depth=13 --window-pattern=LLLLSSLSSLSSL \
  --arch-opt kv_share_frac=0.6923 --target-param-data-ratio=10 --fp8 \
  --device-batch-size=64 --core-metric-every=-1 \
  --model-tag=kvshare4win_d13_ratio10 --run=dummy
```

**Infra**: same volume (`w6ndh50xcl`, US-GA-2), 2x H100 SXM 80GB. Shard prep (`nanochat.dataset`
+ `scripts.data_prep --kind=base --sequence-len=2048 --max-shards=45`, then `--kind=sft`) ran on a
separate CPU pod against the same volume, terminated before the GPU pod started — the CPU-work
discipline this repo's AGENTS.md calls for.

| params | scaling params | iterations | total batch | tokens:param | **val bpb** | tok/sec | bf16 MFU | peak mem | netto time |
|---|---|---|---|---|---|---|---|---|---|
| 175,472,640 | 146,112,512 | 2,786 | 524,288 | 10.00 | **0.874398** | ~915,000 | ~49% | 69.87 GB | 26.52 min |

Final step was also the minimum (no rebound). **Vs. Stage 5** (same architecture, ratio 20, 2.92B
tokens): val bpb 0.874398 vs. 0.833913 — worse, exactly as expected for half the tokens:param
ratio (~1.46B tokens here).

SFT (1 epoch, `sft_t2048_e348819205de14ab`, off the `kvshare4win_d13_ratio10` base checkpoint):
val bpb **0.3805**, 9.44 min, peak mem 60.02 GB. Not compared against anything — no prior SFT run
at this depth/ratio exists — recorded for the checkpoint's own provenance.

This run is the unmasked baseline for the intra-document attention masking work
(`modelcore/kernels/flash_attn.py`'s `build_doc_args`/varlen path) — see that section's own
results, appended below once run.

## Stage 7 results: intra-document attention masking (`--doc-masking`), same shape as Stage 6

Byte-identical command to Stage 6 plus `--doc-masking`, on the same volume/dataset, so Stage 6 is
the direct baseline:

```
torchrun --standalone --nproc_per_node=2 -m scripts.base_train -- \
  --arch=llama_kvshare_win --depth=13 --window-pattern=LLLLSSLSSLSSL \
  --arch-opt kv_share_frac=0.6923 --target-param-data-ratio=10 --fp8 --doc-masking \
  --device-batch-size=64 --core-metric-every=-1 \
  --model-tag=kvshare4win_d13_ratio10_docmask --run=dummy
```

**First attempt OOM'd** — `flash_attn_varlen_func`'s backward pass tried to allocate 28.44GB of
scratch and crashed on the very first training step, against a model that otherwise fit in ~70GB.
Root cause: `build_doc_args`'s `max_docs` (which sizes the kernel's declared segment count, and
therefore its backward-scratch allocation, not just `cu_seqlens`'s own tensor size) defaulted to
the true worst case (`batch_size * sequence_len` = 131,072 possible documents) instead of a
realistic one. Fixed by defaulting to `DEFAULT_MAX_DOCS_PER_ROW=64 * batch_size` (a >15x margin
over ClimbMix's measured ~4.2 documents/row at this sequence length, from this same prepared
dataset's manifest: 3,727,360 documents / 893,729 sequences) — see the commit fixing
`modelcore/kernels/flash_attn.py` for the full account. Terminated the crashed pod immediately,
fixed and tested locally, then retried on a fresh pod; the retry ran clean end to end.

| params | scaling params | iterations | total batch | tokens:param | **val bpb** | tok/sec | bf16 MFU | peak mem | netto time | fp8 converted |
|---|---|---|---|---|---|---|---|---|---|---|
| 175,472,640 | 146,112,512 | 2,786 | 524,288 | 10.00 | **0.872063** | ~832,000-840,000 | ~44.8% | 69.87 GB | 29.17 min | 74/74 |

**Same tokens (1.46B, same 2,786 steps), masking vs. Stage 6's unmasked baseline**: val bpb
0.872063 vs. 0.874398 — masking **0.27% lower** (better). Final step was again the minimum for
both runs. Peak memory is essentially unchanged (69.87GB both), confirming the `max_docs` fix
eliminated the OOM without materially changing the model's own memory footprint — the extra cost
is all in the varlen kernel's own (now-bounded) scratch.

**Same wall-clock budget**: doc-masking is measurably slower — 29.17min vs. 26.52min for the same
2,786 steps, a **9.99% throughput cost** (830-840k tok/sec vs. baseline's 915-920k, 44.8% vs. 49%
bf16 MFU), consistent almost exactly with the wall-time ratio (29.17/26.52 = 1.0999). Interpolating
the baseline's own logged checkpoints (every 250 steps) against the wall-clock-equivalent baseline
step for each masked checkpoint (`masked_step * 1.0999`) shows masking trailing by roughly 0.5-0.9%
through most of training, narrowing back to roughly even by the last ~300 steps — so the small
final-step edge is not simply "masking is better," it's masking spending ~10% more wall-clock time
to land in the same place, plus a small extra edge that shows up late. Neither framing is dramatic:
this is the same "essentially identical, noise-level" territory `docs/upstream/LOG.md`'s own varlen
attempt found at d16, now reproduced on this fork's kv-sharing + sliding-window architecture, with
the added, measured cost of the varlen kernel itself.

One asymmetry worth flagging for anyone re-reading these two numbers later: this run's val bpb is
computed *with* the same intra-document masking as training (`modelcore.ModelManager.evaluate_bpb`
takes `bos_token_id`/`doc_masking_max_docs_per_row` and masks val batches
identically) — deliberate, so val bpb stays comparable to the training loss it's evaluating, but it
means the two runs' val bpb aren't measuring exactly the same quantity (unmasked vs. masked
attention over the same held-out tokens), only the same *procedure* each run actually trained
under.

**Not run**: applying `doc_args` only to the architecture's full-context (`window: 2048`) layers
and leaving the six `window: 512` layers on the cheaper fixed-window kernel — a real design
question this result raises (cross-document leakage is largest exactly in the full-context layers,
where a token can otherwise see all the way back to row start; a short-window layer's 512-token
reach already limits how much of it is even reachable), left for a follow-up rather than this run.

**Cost**: pod ran 29.17 min training (plus setup/sync) on 2x H100 ($6.98/hr) ≈ **$3.40** for this
run; the crashed first attempt added ~2 min (~$0.23) before being caught and terminated.

## Stage 8 results: `padding_id` end to end — a fresh SFT dataset, SFT off the doc-masked base

First real use of the `padding_id` plumbing (Stage 7's own follow-up commit): a new SFT dataset
prepared with a real, non-`bos_token_id` pad filler, then a full SFT pass off Stage 7's doc-masked
base checkpoint. `chat_sft.py` itself has no `--doc-masking` wiring yet — this exercises the new
dataset and the masked base checkpoint's SFT-time behavior, not SFT-time masking.

```
# CPU pod (cpu3m: 4 vcpu, 32GB -- cpu3c's default 4GB OOM'd mid-import; see "Lessons" below)
python -m scripts.data_prep --kind=sft --dataset=sft_t2048_padid_e348819205de14ab \
  --sequence-len=2048 --sft-padding-id=32767   # <|output_end|> -- see the padding_id commit's
                                                # note on why no id in this tokenizer is truly free

# GPU pod (2x H100)
torchrun --standalone --nproc_per_node=2 -m scripts.chat_sft -- \
  --model-tag=kvshare4win_d13_ratio10_docmask --dataset=sft_t2048_padid_e348819205de14ab \
  --chatcore-every=-1 --run=dummy
```

**Verified the new dataset changes nothing but the padding byte**: compared the new dataset
against Stage 6's original `sft_t2048_e348819205de14ab` directly (same document stream, same
`buffer_size` — packing *decisions* don't depend on the fill value). Same train sequence count
(237,453 both). Diffed a full volume byte-for-byte: 274/603 rows differ, every differing position
has `mask=0` in both datasets (pure padding, never touching the loss), and everywhere else is
byte-identical. So this run's training data is, for `chat_sft.py`'s purposes, identical to Stage
6's — the only real variable between the two SFT runs is the base checkpoint (Stage 7's doc-masked
one here vs. Stage 6's unmasked one).

| params | scaling params | steps (1 epoch) | tok/sec | mfu | peak mem | wall time | **val bpb** |
|---|---|---|---|---|---|---|---|
| 175,472,640 | 146,112,512 | 927 | ~840,000-865,000 | ~45-46.5% | 60,024.01 MiB | 9.41 min | **0.3796** |

**Vs. Stage 6's SFT** (identical training data, unmasked base checkpoint): steps, tok/sec, mfu,
and peak memory are all indistinguishable (60,024.01 MiB peak mem to the byte in both runs) — as
expected, since neither run applies doc-masking at SFT time, so both do the exact same computation
shape. Val bpb tracked within noise at every logged checkpoint and pulled slightly ahead by the
end:

| step | Stage 6 (unmasked base) | Stage 8 (doc-masked base) |
|---|---|---|
| 0 | 0.6325 | 0.6384 |
| 200 | 0.4548 | 0.4549 |
| 400 | 0.4414 | 0.4411 |
| 600 | 0.4170 | 0.4163 |
| 800 | 0.3902 | 0.3894 |
| 927 (final) | 0.3805 | **0.3796** |

0.24% lower (better) at the final step, the same "consistent but small" pattern Stage 7's own base
comparison showed on a wall-time-adjusted basis. Since the SFT data and procedure are proven
identical between the two runs, this is the cleanest signal yet that Stage 7's doc-masked base
checkpoint carries a small, real (if practically negligible) edge into SFT — not proof either way
on whether SFT-time doc-masking itself would help, which needs `chat_sft.py` wiring this stage
didn't build.

**Lessons**:
- **Neither the MCP `create-pod` tool nor `runpodctl`'s `pod create`/`create pod` expose CPU
  flavor or vCPU count.** Both always land on the smallest flavor (`cpu3c`, 2 vcpu, 4GB, enforced
  as a hard cgroup limit despite the host reporting far more RAM) with no override flag. `scripts.
  data_prep --kind=sft` OOM'd on it before printing anything (`SmolTalk`/`MMLU`/`GSM8K` load their
  full source datasets into memory before any `--max-conversations` cap applies) — exit 137, silent
  otherwise. Worked around by calling the REST v2 API directly (`POST /v2/pods` with `cpu: {id,
  vcpuCount}`, per `GET /v2/catalog/cpus`'s `cpu3m`/`cpu5g`/etc. — the flavor ids these tools
  themselves can't select), using the same API key `runpodctl` already had configured. `cpu5g`/`8
  vcpu` had no stock in US-GA-2 at request time; `cpu3g`/`4 vcpu` (16GB) and `cpu3m`/`4 vcpu` (32GB,
  matching the Stage 6 prep's own `cpu5g`/`8 vcpu` memory budget) both did — used `cpu3m`.
- **There is no free token id for `padding_id` in the current tokenizer without adding a real
  special token.** `vocab_size=32768` is already an exact multiple of `pad_vocab_size_to` (64), so
  there's no spare embedding-table slot either. `<|user_start|>` (`bos_token_id+1`) was tried first
  and rejected on inspection of the actual packed rows — it appears in every real conversation, so
  reusing it as filler just relocates the semantic overload rather than removing it. `<|output_end|>`
  only appears in GSM8K's calculator-tool conversations (`tasks/gsm8k.py`'s `python`/`python_output`
  message parts, rendered via `RustBPETokenizer.render_conversation`) — a small slice of the full
  mixture, SmolTalk-dominated by volume — so it's the pragmatic choice without a tokenizer change,
  not a truly free id. A genuinely clean pad token needs a new tokenizer, out of scope here and
  already on the roadmap for future real experiments.

## Stage 9: process fixes after Stage 8 — a pre-spend gate, dataset stats, and `chat_sft.py` doc-masking

Stage 8 was framed as testing `padding_id` "end to end," but couldn't have: `scripts/chat_sft.py`
had no `--doc-masking` wiring, so the new pad filler sat only at `mask=0`, causally-last positions
— unreachable and loss-free either way. This was caught after the pods ran, not before. No pod was
created for this stage; everything below is local, free, and closes the gap so the next real run
can actually test what it's named for.

**`AGENTS.md` gained a "Before you spend money on a pod" checklist** — six questions (comparison
baseline, single variable, the exact `file:line` where the flag under test is read, a free/cheap
check to run first, expected effect size vs. known noise, a written cost estimate) that must be
answered before any billed pod is created. Also recorded there: `--sft-padding-id` should stay at
its `None` default until a tokenizer has a genuinely free pad id — every id Stage 8 could have
tried is a real special token, so passing one in is strictly worse than the existing bos-fold-in
heuristic, not an improvement. **`sft_t2048_padid_e348819205de14ab` is retired**, not built on.

**`scripts/data_prep.py --describe` gained `--deep` and `--compare-to`** — a row-level scan
(document count/length percentiles, documents/row including the max that
`--doc-masking-max-docs-per-row` should be set from, padding token share, loss-mask share) computed
purely by reading a dataset's existing volumes, so it works on any dataset regardless of when it
was prepared — no re-prep. `--compare-to` runs the scan on a second dataset and diffs it against
the first.

**Run for real** on a CPU pod (`cpu3g`, 4 vcpu/16GB, US-GA-2 — cheaper than Stage 8's `cpu3m` since
this only reads already-prepared volumes, no raw HF dataset loading) against both real SFT datasets
on volume `w6ndh50xcl`. Two pods total (~15 min combined uptime, ~$0.04): the RunPod SSH proxy
turned out to need an account-registered key and PTY allocation, not the usual `PUBLIC_KEY`
env var / non-interactive `ssh host cmd` — worked around by using the already-registered
`runpodctl-ssh-key` and piping commands through the interactive shell's stdin; `scp`/`sftp` don't
work over this proxy at all (no subsystem support), so the locally-fixed file was pushed as base64
through the same stdin channel, verified byte-identical via `md5sum` both times:

```
python -m scripts.data_prep --describe --deep --compare-to=sft_t2048_padid_e348819205de14ab \
  --dataset=sft_t2048_e348819205de14ab
```

The *first* run of this exposed a real bug in the tool itself, caught by comparing its output
against the manifest's own recorded token-utilization ratio rather than trusting it blindly:
`sft_t2048_e348819205de14ab` predates the `padding_id` field entirely (it's the original Stage 6
dataset), so `DatasetInfo.padding_id` reads back `None` for it — but the deep-scan was treating
*that* `None` identically to a crop packer's "no padding concept at all" `None`, silently reporting
0% padding for a dataset the manifest's own ratio said was ~0.2%+ padded, and miscounting each
padded row's bos-valued tail as a spurious extra one-token document (898,289 vs. the true 789,759).
Fixed by having `deep_scan` fall back to `bos_token_id` — `BestFitPadPacker`'s own documented
default — whenever `padding_id` is `None` *and* the packer is `bestfit_pad`, matching what the
packer actually wrote to disk; only a genuine `bestfit_crop` dataset skips padding detection
entirely now. A second bug surfaced by the fix landing on real (not just synthetic) data: the
per-chunk document-length computation used `np.roll(rows_idx, -1)`, which wraps the last document
in a chunk around to compare against the first — spuriously "matching" (and computing a bogus
length) whenever a chunk holds documents from only one row, a real case (any dataset's final
partial chunk can shrink to exactly one row), not just a tiny-test artifact. Both are pinned by new
regression tests (`tests/test_data_prep.py`) against a reconstructed old-format manifest and a
one-row dataset. Re-run after both fixes, the two real datasets came back **identical in every
measured respect** — document counts (789,759 train), lengths, documents/row (max 35 train / 34
val — this is what `--doc-masking-max-docs-per-row` would be set from), padding (0.22% train /
0.79% val), and mask1/mask0 token counts, all exactly equal between the two datasets, confirming
(now reproducibly, at full scale, not just via Stage 8's one-off byte-diff) that they differ only
in which id fills the pad tail. Pod terminated immediately after the second run.

This is also the tool that answers the blocking pre-check for any future SFT-time doc-masking run:
SFT rows pack up to 35 documents/row (vs. `DEFAULT_MAX_DOCS_PER_ROW=64`'s tuning for pretraining's
~4.2/row) — comfortably under the default, so no `--doc-masking-max-docs-per-row` override would be
needed for *this* dataset, but the number is now measured, not assumed the way Stage 7's original
guess (which did OOM) was.

**`scripts/chat_sft.py` gained `--doc-masking` / `--doc-masking-max-docs-per-row`**, mirroring
`scripts/base_train.py`'s wiring exactly (`doc_args` built outside the `torch.compile`'d model, in
the micro-batch loop; `ModelManager.evaluate_bpb` given the same `bos_token_id`/`padding_id`/
`doc_masking_max_docs_per_row`). `datacore.reader.DatasetInfo` now also surfaces `padding_id` and
`bos_token_id` (previously readable only from the raw manifest dict), which both `base_train.py`
and `chat_sft.py` use to pass the dataset's actual resolved `padding_id` into `build_doc_args`
rather than always falling back to its bos-run heuristic. Verified locally (CPU/MPS, per
`AGENTS.md`'s "What runs on this Mac"): a tiny SFT dataset against a real local `d6` checkpoint,
`--doc-masking` on, ran to completion with finite loss and val bpb at every step — this is the
check whose *absence* let Stage 8 spend money on an untestable premise.

This closes the gap for the SFT-time masking experiment Stage 8 should have been (base pretraining
masked ~4.2 documents/row; an SFT row packs several short conversations, so cross-document
attention is a much larger share of total attention mass there — a real reason to expect a
different result than the noise-level one in Stage 7). That run is still gated on its own pre-spend
checklist (an SFT-scale `--describe --deep` run to size `--doc-masking-max-docs-per-row`, and
confirming the pad-fill choice doesn't change which dataset should serve as the control arm) and is
not launched by this stage.

## Stage 10 results: SFT-time intra-document masking (`chat_sft.py --doc-masking`), off the doc-masked base

The run Stage 8 should have been, using Stage 9's `chat_sft.py` wiring: a real, single-variable
`--doc-masking` comparison at SFT time, holding the base checkpoint fixed.

```
# GPU pod (2x H100 SXM, US-MO-1 -- see "Infra" below for why not US-GA-2)
torchrun --standalone --nproc_per_node=2 -m scripts.chat_sft -- \
  --model-tag=kvshare4win_d13_ratio10_docmask --dataset=sft_t2048_e348819205de14ab \
  --doc-masking --chatcore-every=-1 --run=dummy
```

**Pre-spend gate, answered before launch**: number/baseline = SFT val bpb at step 927 vs. Stage 8's
0.3796; single variable = `--doc-masking` on at SFT time, same base checkpoint and (content-
identical, per Stage 9's `--compare-to`) dataset; consumer = `scripts/chat_sft.py`'s new
`build_doc_args` call; cheap pre-check = Stage 9's CPU-pod scan already measured this dataset's
real documents/row max (35), safely under `DEFAULT_MAX_DOCS_PER_ROW=64`; effect-size argument = SFT
rows pack far more documents/row than pretraining's ~4.2, a real structural reason to expect a
larger, more distinguishable effect than Stage 7's noise-level pretraining result.

**Infra: US-GA-2 had no multi-GPU Hopper-class stock at all when this ran** — 2x H100 SXM, 2x H100
NVL, and 2x H200 all failed with "no longer any instances available," and COMMUNITY cloud caps
H100 at 1 GPU/pod regardless of stock. `GET /v2/catalog/datacenters?include=GPU_AVAILABILITY`
confirmed real stock existed elsewhere (**US-MO-1** had 2x H100 SXM, verified by an actual
create-then-immediately-terminate probe, not just trusting the "LOW" label) but no other
datacenter hosts volume `w6ndh50xcl`. Rather than migrate the ~1.5GB dataset too, the SFT dataset
was re-prepared fresh on the GPU pod's own CPU (`scripts.data_prep --kind=sft`, no `--dataset`
override) — a deliberate, scoped exception to the "never non-GPU work on a billed GPU pod" rule,
made for this one task specifically to avoid a second cross-datacenter transfer. It reproduced
**exactly** the same numbers as the original (789,759 train documents, 237,453 sequences) —
confirming the pipeline is genuinely content-deterministic, not just probably-deterministic. Only
the 1.4GB base checkpoint (model + both optimizer shards, for `--load-optimizer=1` parity with
Stage 8) was moved, via direct `scp` — RunPod's SSH proxy (`ssh.runpod.io`) only supports an
interactive PTY shell with no `scp`/`sftp` subsystem at all, but adding *any* exposed port
(`8000/http`, unrelated to the transfer itself) causes RunPod to allocate the pod a real public IP,
after which normal `ssh`/`scp` on the direct address works fine. ~11 MB/s over that link; verified
byte-identical both ends via `md5sum` before trusting it.

**Speed** (2x H100 SXM both runs, same GPU type; different pods since US-GA-2 had no capacity):

| | Stage 8 (SFT masking off) | Stage 10 (SFT masking on) |
|---|---|---|
| tok/sec | ~840,000–865,000 | ~778,000–805,000 (avg ~789K) |
| MFU | ~45–46.5% | ~41.7–43.2% (avg ~42.4%) |
| wall time (927 steps) | 9.41 min | 10.17 min |
| peak memory | 60,024.01 MiB | 60,022.27 MiB (identical) |

**~8% throughput cost** for SFT-time masking — the same ballpark as Stage 7's ~10% at pretraining
time.

**Quality** — val bpb, identical base checkpoint and dataset, only SFT-time masking differs:

| step | Stage 8 (masking off) | Stage 10 (masking on) |
|---|---|---|
| 0 | 0.6384 | 0.6301 |
| 200 | 0.4549 | 0.4533 |
| 400 | 0.4411 | 0.4405 |
| 600 | 0.4163 | 0.4159 |
| 800 | 0.3894 | 0.3891 |
| 927 (final) | **0.3796** | **0.3794** |

**0.053% better** — smaller than Stage 8's own base-checkpoint-masking effect (0.24%), and well
inside the noise band every result in this line has landed in since upstream's own original
attempt (`docs/upstream/LOG.md:715-741`). The hypothesis motivating this run — that SFT-time
masking should matter *more* than pretraining-time masking, since a packed SFT row holds far more
documents (up to 35/row measured in Stage 9) than a pretraining row (~4.2/row) — was not borne out:
holding the already-masked base checkpoint fixed, adding masking at SFT time too produced a
*smaller* delta than masking the base checkpoint alone did. Combined with the ~8% throughput cost,
this closes the doc-masking line of inquiry for this architecture: real, measured, consistently
small-positive, and not worth its cost at this scale.

**Cost**: ~26 min on 2x H100 SXM ($6.98/hr) ≈ $3.02 for the training run, plus ~9 min on a `cpu3g`
CPU pod for the checkpoint transfer (≈ $0.02) and a few seconds each for capacity-probe pods
(≈ $0.03) — **≈ $3.07 total**, all pods terminated immediately after their step finished.

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
- **FA3's real root cause was a missing `hf_transfer` package.** `HF_HUB_ENABLE_HF_TRANSFER=1` ships
  set in RunPod's own pytorch image, but the `hf_transfer` package itself wasn't installed, so
  `kernels`' hub download raised `ValueError` before it ever reached the network — invisible before
  `FA3_LOAD_ERROR` existed, and found by that diagnostic on its **first real use**. Fixed by adding
  `hf_transfer>=0.1.9` to `pyproject.toml`; verified on real infra both directly
  (`HAS_FA3: True, FA3_LOAD_ERROR: None`) and in the Stage 1 training log
  (`✓ Using Flash Attention 3`).
- **FA3's speedup is attention-pattern-dependent, and an early claim about it over-generalized.**
  After the fix, `llama_kvshare` (full causal attention, no windowing) measured ~57.7% MFU with FA3
  vs. ~56.6% with SDPA — ~2%, not the dramatic win predicted from the earlier gpt-on-A100 result.
  That earlier gap (33% SDPA vs. 57-58% FA3, see the d12 shakedown table above) is specific to
  **windowed** patterns like gpt's `SSSL`: SDPA falls off its fused kernel onto an explicit mask for
  those (see "SDPA has no support for sliding window attention" fixed to "SDPA's sliding window
  support falls back to an explicit mask" in `scripts/base_train.py`), while full causal attention
  already has an efficient fused SDPA path. This is exactly why `llama_kvshare_win` (Stage 3, adds
  windowing to `llama_kvshare`) is worth measuring on a GPU where FA3 actually loads — it's the
  first architecture in this contest where FA3-vs-SDPA and KV-sharing-vs-not are both live at once.
- **`chat_eval` cost more than training.** `--max-problems` defaults to `None` = the full test set;
  GSM8K (~1319 problems) and HumanEval (~164) are generative and unbatched per rank, and a single
  architecture's eval ran past 25 minutes still only partway through GSM8K — more than base training
  (23 min) plus SFT (9.35 min) combined. Now capped via `CHATEVAL_MAX_PROBLEMS` (default 100). The
  trap: **`-1` is not an "uncapped" sentinel** — `chat_eval.py` has no such value, and `-1` evaluates
  *zero* problems (`min(len(task), -1)`); the true uncapped form is `CHATEVAL_MAX_PROBLEMS=""`
  (empty string), which omits the flag entirely.
- **`chat_eval` was also wasting 3 of 4 billed GPUs.** It already shards problems across ranks and
  `all_reduce`s the aggregate (`scripts/chat_eval.py`), but `runs/contest.sh`/`contest_d12.sh` were
  launching it with plain `python` instead of `torchrun` — using 1 of however many GPUs the pod was
  billed for. Now runs through the same `launch_module` helper as `base_train`/`chat_sft`. General
  lesson: on a multi-GPU pod, *every* pipeline step is billed at the full pod rate, so a
  single-rank step isn't a neutral choice — it's a markup equal to GPU count on that step alone.
- **Resume granularity is coarser than it looks.** The per-row skip in `runs/contest*.sh` is keyed
  on a row already existing in `chat_results.csv`, so a failure *after* SFT finishes but *during*
  `chat_eval` re-runs the whole SFT step on the next attempt. Worked around once by invoking
  `chat_eval` directly against the already-finished checkpoint and hand-appending the CSV row
  (`python -m scripts.chat_eval -i sft -g <tag> --max-problems=100`, then constructing the row with
  the same regex the script uses) rather than re-running the row through the harness. Finer-grained
  checkpointing of the SFT step would close this gap but hasn't been made.
- **Minimal pod images lack tools you assume exist.** `column` is absent on RunPod's own pod image
  (`print_csv()`'s `cat` fallback exists because of this), and separately, `timeout`/`gtimeout`
  (GNU coreutils) is absent on **macOS** by default — a local rehearsal command wrapped in
  `timeout 1100 bash ...` died instantly with a single easy-to-miss `command not found: timeout`
  line, under a background task that still reported exit code 0. Neither failure is loud; check the
  actual exit code and that the intended long job is actually running, don't infer it from "no
  visible error."
- **Network-volume DC choice is stickier than it looks.** US-KS-2 (this doc's existing volume) was
  picked for A100 + storage overlap and, checked live, carries **no H100 and no H200 stock at all**
  — a Hopper contest needs an entirely separate volume and cold-start kit in a different DC, exactly
  as anticipated when the first volume was created. Cross-reference `get-gpu-type` (per GPU tier,
  gives per-DC availability) against `list-data-centers` (gives each DC's `networkVolumeTypes`) to
  find a DC with both live stock for the GPU tier you need *and* the storage tier you want — pick
  the volume's DC for the GPU tier you'll need **last**, not first, since the volume choice is what's
  sticky, not the pod's.
- **`get-gpu-type`'s "LOW" availability is not a promise of stock at the count you ask for.**
  US-GA-2 showed "LOW" (not "NONE") for H100 SXM, but 4x failed three times in a row before 3x and
  then 2x both succeeded. **Probing GPU counts by calling `pod create` at each count creates a real
  billed pod on success, not just a dry check** — two stray pods got created probing this and had
  to be terminated within seconds of creation (negligible cost here, but a real trap). If you need
  to find the actual available count, accept the first success rather than continuing to probe
  downward, or terminate immediately between attempts.
- **A batch size tuned for one GPU class silently underutilizes a different one.** `DEVICE_BATCH_SIZE=16`
  (the A100-sized default) left only 16-20GB of an H100's 80GB in use (~20-25%) — the training loop
  was overhead-bound (data loading, Python, DDP sync per step), not compute-bound, so FA3 and
  H100's 3.2x higher peak FLOPS barely showed up in wall-clock: a same-architecture check via
  wandb's `total_training_time` metric (`llama`, no windowing) measured only ~11% faster net
  training time on 2x H100+FA3 than 4x A100 without FA3. Bumping to `DEVICE_BATCH_SIZE=64` (memory
  headroom allowed it, confirmed via `nvidia-smi` before committing) pushed MFU from ~30-35% to
  ~42-44% with no change to `val_bpb`/`CORE` (batch size only changes wall-clock/cost, not what's
  actually trained, since `total_batch_size` stays fixed and grad-accum steps just shrink). Low
  `nvidia-smi` memory usage early in a run on new hardware is the concrete, checkable signal that
  the batch size needs raising — check it before, not after, a long run.
- **Pulled the H100 contest's results home over the GPU pod's own SSH session instead of
  terminating it first.** The checkpoints were already durable on the volume the instant training
  finished — the rsync was pure file I/O, no GPU involved — but it ran while the 2x H100 pod
  ($6.98/hr = **$0.116/min**) was still up, rather than terminating it immediately and either using
  S3 (still deferred, see below) or a fresh CPU pod (~$0.06/hr, the same class already used for
  volume prep in this exact session) for the pull. **One minute on that GPU pod cost more than 1.9
  hours would have on the CPU pod** — a real, quantifiable waste, and an inconsistency with the
  session's own established logic (Phase 0's whole point was doing non-GPU work on a cheap pod).
  Two fixes, not one: (1) **stop deferring RunPod S3 keys** — set them up once and this entire
  question disappears, no pod of any kind needed to pull from a volume ever again; (2) **whenever a
  pod is genuinely needed for pure data movement, it must be the cheapest pod type attached to that
  volume, never the GPU pod that was just training** — terminate the GPU pod the moment its actual
  GPU work (training/eval) is done, full stop, before running anything that isn't GPU work.

## Stage 11: first prod test of Stage 14/15 (modelcore/benchcore pin bumps), and a Blackwell reality check

Landed the `modelcore` v0.2.0 / `benchcore` v0.1.1 pin bumps (`TODO.md`'s pending item —
`evaluate_bpb`/`token_bytes` moving into `modelcore`/`datacore`, Stage 14, and the `benchcore`
extraction, Stage 15) and prod-tested them for real, on GPU hardware, for the first time — every
verification of that work up to this point had been local CPU/MPS only
(`nanochat/AGENTS.md`'s own disclaimer). Also the first real attempt at either Blackwell chip
(B200, B300) on this account, and the first real run with a deliberately shrunk `--eval-tokens`
budget (1,048,576 vs. the ~41.9M default) to see how much of a run's wall time validation eval
actually costs.

**Landing the pins surfaced a genuine gap that pure local testing had masked**: `modelcore`'s
Stage 14 commit added `modelcore/tests/test_evaluate.py`, which imports `numpy` — never declared
as a dev dependency, invisible because the only verification path so far was
`uv pip install -e ../modelcore` from `nanochat/`, whose own venv already had `numpy` transitively.
Fixed before tagging `v0.2.0`. Separately, `benchcore`'s own tagged `v0.1.0` still pinned
`datacore@v0.2.0` while `nanochat` pins `datacore@v0.2.1` — a commit (`9106308`) had already fixed
this in `benchcore`'s `main` but never tagged it, on the mistaken assumption that a host's own
`[tool.uv.sources]` override always wins; in practice `uv`'s universal resolver treats two
non-workspace packages' conflicting source pins for the same dependency as a hard error
(`uv sync` refused with "conflicting URLs for package datacore"), not a root-project override.
Tagged `v0.1.1` to fix it for real. Full account and the exact commands in `TODO.md`.

**A second, more consequential gap only showed up when `scripts/base_train.py` actually ran**:
Stage 15 folded the CORE-eval loop directly into `scripts/base_eval.py`'s `main()` but never left
behind the standalone `evaluate_core` function `base_train.py` imports at module load time —
`from scripts.base_eval import evaluate_core` has been raising `ImportError` unconditionally since
Stage 15 landed, even with `--core-metric-every=-1` (the import runs before argparse). Nothing in
`tests/` imports `scripts.base_train` at module scope, so `python -m pytest -q` stayed green the
entire time. Fixed by re-adding `evaluate_core` as a thin wrapper around `BenchManager.core`
(which already implements the same loop), and threading `ddp_rank`/`ddp_world_size` through from
the call site — the old (also broken) call passed neither, which would have made every rank
redundantly eval the full suite under real multi-GPU. Committed and pushed to `master` before any
GPU pod resumed.

**B300 SXM6 AC, EU-NL-1, $7.89/hr** (the pre-approved fallback — B200 read `unavailable` across
every catalog probe and a real create-then-terminate attempt at the time): the byte-identical
Stage 6 command (`llama_kvshare_win` d13, ratio 10, `--fp8`) hit two independent, hardware-specific
compiler gaps back to back, neither a nanochat bug:

1. `torch.compile` crashed outright — Triton's bundled `ptxas` cannot codegen for `sm_103a`
   (Blackwell Ultra's real compute capability, `(10, 3)`) at all: `PTXASError: Internal Triton PTX
   codegen error`, independent of FA3.
2. With `TORCHDYNAMO_DISABLE=1` past that, FA3 crashed too: `kernels-community/flash-attn3`'s own
   `has_kernel()` check (`modelcore/kernels/flash_attn.py`'s `_load_flash_attention_3`) reports the
   kernel available on this GPU, but the actual published binary has no compiled kernel image for
   sm_103 at all — `CUDA error: no kernel image is available for execution on the device` on the
   very first `flash_attn_func` call.

Forcing both off (`TORCHDYNAMO_DISABLE=1` env var + a one-off bootstrap script setting
`modelcore.kernels.flash_attn.USE_FA3 = False` after import, no code change) ran stably at
`--device-batch-size=128` — B300's 275GB made the 128-vs-64 batch-size question moot — but at only
**~152,300 tok/sec**, eager-mode SDPA with no compiled kernels at all. Projected **~159 min** for
the full 2,786-step run (vs. Stage 6's 26.52 min on 2x H100), not a representative speed number and
well outside budget — stopped after ~30 steps rather than let it complete in a crippled mode.
**Terminated the pod**; no checkpoint was ever written (`--save-every=-1`, only saves at the end).

**B200, US-NE-1, $6.79/hr** (found on a re-probe — stock across both Blackwell chips is genuinely
this volatile run to run, worth re-checking immediately before every attempt rather than trusting a
prior read). Neither B200-stocked datacenter this session (`US-TX-6`, then `US-NE-1`) supports
network volumes at all (confirmed by the create-network-volume API's own error, which lists every
volume-capable DC and neither appears) — the prepared dataset had to be re-downloaded and
re-prepared fresh on the pod's own local disk (`~35 min` on 36 vCPUs, reproducing **exactly** the
same numbers as the EU-NL-1 prep: 914,113 train / 20,414 val sequences — content-deterministic
again, as every prior re-prep in this doc has been) rather than reused from the EU-NL-1 volume, a
same-datacenter CPU relay having also failed (EU-NL-1 had zero CPU stock of any flavor at the time,
confirmed by five failed create attempts plus the catalog listing itself).

On B200 (real compute capability `(10, 0)`, plain Blackwell, not Ultra): `torch.compile` **worked
fine** — a cheap smoke test (`torch.compile` a bare `Linear`, run it) succeeded before committing to
the full launch. But FA3 has the identical gap Stage 6's own code comment already anticipated
("Blackwell (sm100) needs SDPA fallback until FA3 is recompiled") — `has_kernel()` reports true,
the real kernel call still hits "no kernel image is available," confirming this is not
Ultra-specific but affects **all of Blackwell** in this pinned `kernels-community/flash-attn3`
build. With FA3 forced off the same way (bootstrap script) but `torch.compile` left **on** this
time, the run was stable and gave a real, representative number:

| | Stage 6 (2x H100, FA3 on, compiled) | Stage 11 (1x B200, FA3 off/SDPA, compiled) |
|---|---|---|
| tok/sec | ~915,000 (2 GPUs) | ~391,000 (1 GPU) |
| bf16 MFU | ~49% | ~18.5% |
| val bpb (step 1,250/2,786) | not directly comparable (different step) | 0.9225 |
| val bpb trajectory | 1.09 @ step 250 | 1.09 @ step 250 (identical) |

The val-bpb trajectory matching Stage 6's step-250 value exactly is itself a correctness proof —
the whole point of this run — that Stage 14/15's `evaluate_bpb`/`token_bytes` code path produces
identical numbers on real GPU hardware as it does on this Mac's CPU/MPS tests. Stopped by request
at step 1,250/2,786 (45%) rather than complete the full run — the speed question was already
answered, and completing it would only have added cost without changing the conclusion.
**Terminated the pod**; again no checkpoint was written.

**Decision: defer Blackwell adoption, keep using H100 for production runs.** Losing FA3 alone costs
roughly the same ballpark as the entire H100-vs-B200 hardware generation gap would otherwise gain —
a wash at best, before counting the engineering cost of chasing a fix. An FA2 fallback was
considered and set aside: unclear whether FA2 even covers this gap (it may hit the identical
"no kernel image" wall, being similarly precompiled-binary-based), so the investigation cost isn't
obviously justified by a probable win. Native FP4 support was also considered and set aside for the
same reason from the other direction — real development work (`modelcore/precision/` has no FP4
path today) with no evidence yet that it would offset the FA3 gap on this hardware.

**Settled for good: the "~47M validation tokens" figure.** The real number, read directly from this
session's own prepared-dataset manifest (twice, identically, on two different pods): val split =
**41,828,286 tokens** (20,414 sequences × 2048), i.e. ~41.8M — close to `--eval-tokens`'s ~41.9M
default (which is why the two are easy to conflate) but not the dataset's own size. Reducing
`--eval-tokens` to 1,048,576 ran with no functional issues at either budget; the isolated
before/after eval-cost measurement itself (Step 2.5b of the session's plan) was never reached,
since the run was stopped for the Blackwell speed question first — a fair remaining experiment for
a future H100 run, now that it's cheap to ask (same command, `--eval-tokens=1048576` vs. omitted).

**Cost**: roughly $15-16 total across every probe, the two false-start CPU-pod attempts, the B300
run, and the B200 run — see this session's own transcript for the itemized breakdown; RunPod's
billing API lags real-time by several minutes, so live totals during a session are always
approximate on the high side of "at least."

**Corrections to this doc's own past entries**: Stage 6 and Stage 7's tables both label their wall
time column "wall time" — it is not. `total_training_time` (`scripts/base_train.py`) accumulates
only the fwd/bwd/optimizer window between two `synchronize()` calls, for `step > 10` onward
specifically excluding validation, CORE eval, sampling, and checkpoint saves. Both stages' printed
values are netto training time, not wall clock — there is no recorded pod wall-clock time for
either stage. The column headers above (Stage 11's own table) say `tok/sec`/`MFU` rather than
repeat the ambiguous label.
