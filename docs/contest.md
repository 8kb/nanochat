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
  of the rows) needs ≈3.16B tokens; this tokenizer's vocab averages ≈4.7 characters/token on this
  dataset, and the BOS-aligned dataloader keeps ≈65% of tokens after cropping (see
  `nanochat/dataloader.py`), so each ~253M-character shard yields ≈35M usable training tokens —
  ≈91 shards needed, 100 leaves a ~10% margin. Raise `NUM_SHARDS` if you raise `TARGET_FLOPS` or
  `--depth` significantly. The validation shard (always the last one, `shard_06542.parquet`) is
  identical across every row regardless of `NUM_SHARDS`, which is what makes the val-bpb numbers
  comparable to each other.
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
