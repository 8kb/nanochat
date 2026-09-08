#!/bin/bash

# FP8 sanity check + a new attention shape: one architecture, two base-training rows (bf16 vs.
# --fp8), nothing else. Not a multi-architecture contest like runs/contest.sh / contest_d12.sh --
# kept as its own file because its row grammar is "name|train_args" (both rows share one
# architecture) rather than "arch|depth|extra_args", and because it drops the chat_sft/chat_eval
# half entirely (base training only, no SFT, no metrics beyond val bpb).
#
# Why this matters: Float8Linear used to subclass torch.nn.Linear instead of
# modelcore.components.linear.Linear, so collect_param_roles raised ValueError and
# create_optimizer died at startup -- --fp8 was a hard crash on any CUDA box. Fixed on
# stage7-modelcore-extraction (commit 8e59911), but only ever exercised on CPU, where the
# _scaled_mm kernel never runs. This run's job is to prove the H100 path actually executes and
# converges -- not to prove it's faster. Upstream's own benchmarking (docs/upstream/LOG.md) found
# FP8 still slower than bf16 at d12; neutral-to-slower here is the expected, accepted outcome.
#
# The architecture: llama_kvshare_win at depth 13 (not 12) with an explicit per-layer window
# pattern and kv_share_frac tuned to leave exactly 4 KV-owning layers. Depth 13 is deliberate --
# compute_window_sizes (nanochat/architectures/derive.py) unconditionally forces the final layer
# to full context, which would have silently rewritten a 12-layer "...S S" tail into "...S L". At
# 13 layers the pattern already ends in L, so the override is a no-op:
#   layer:   1  2  3  4  5  6  7  8  9 10 11 12 13
#   window:  L  L  L  L  S  S  L  S  S  L  S  S  L
#   kv slot: 0  1  2  3  3  3  3  3  3  3  3  3  3
# Verified locally (scripts/model_info.py + a CPU forward pass) before ever touching a pod.
#
# Usage: bash runs/contest_fp8_d13.sh [label]
# Example: bash runs/contest_fp8_d13.sh fp8d13
#
# DRY_RUN=1 bash runs/contest_fp8_d13.sh    # print the budget and exit -- no training, no GPU
#                                             # required, no money spent. Always run this first.
#
# Env overrides (all optional):
#   TARGET_FLOPS        iso-FLOPs budget per row (default 1e18, same as runs/contest_d12.sh)
#   NPROC_PER_NODE       GPUs to use (default 4; set to 1 to rehearse on a single GPU/CPU/MPS box)
#   DEVICE_BATCH_SIZE     per-device micro-batch (default 64 -- sized for an H100's 80GB at this
#                            model's n_embd=896; must evenly divide the auto total_batch_size
#                            together with max_seq_len*NPROC_PER_NODE, so only 64 or 32 are valid
#                            at NPROC_PER_NODE=4 -- see docs/contest.md "Lessons" on sizing this
#                            per GPU class, not inheriting a previous card's value)
#   NUM_SHARDS             pretraining data shards to download during setup (default 45 -- this
#                            budget needs ~27M tokens/shard * 45 >> 940M tokens/row, wide margin)
#   GPU_NAME, MFU, PRICE_PER_GPU_HOUR   feed the GPU-hours/dollar estimate (defaults: "NVIDIA H100", 0.45, 3.49)
#   SKIP_SETUP=1            skip venv/data/tokenizer setup (same convention as runs/contest_d12.sh)
#   EXTRA_TRAIN_ARGS         appended (last) to every base_train.py invocation

set -euo pipefail

# RunPod injects account Secrets (e.g. WANDB_API_KEY) into /etc/rp_environment, but only sources
# it into *interactive* shells via .bashrc's interactive-shell guard -- a non-interactive launch
# never sees it otherwise. Sourcing it here makes it available regardless of how this script was
# launched; harmless no-op off RunPod. (Not that this run needs it -- WANDB_RUN defaults to
# "dummy" below, since everything being compared lands in results.csv and the logs anyway.)
[ -f /etc/rp_environment ] && source /etc/rp_environment

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
mkdir -p "$NANOCHAT_BASE_DIR"

# Fixed architecture, shared by both rows -- this is what makes "identical except --fp8"
# structural rather than a promise. Not passed to CONTEST_ROWS since --fp8 is a base_train-only
# flag that scripts.model_info doesn't understand; the preflight below uses $ARCH_ARGS alone.
ARCH_ARGS="--arch=llama_kvshare_win --depth=13 --window-pattern=LLLLSSLSSLSSL --arch-opt kv_share_frac=0.6923"

# One row per condition: "name|extra_train_args".
CONTEST_ROWS=(
    "bf16|"
    "fp8|--fp8"
)

LABEL="${1:-${LABEL:-$(date +%b%d | tr '[:upper:]' '[:lower:]')}}"
TARGET_FLOPS="${TARGET_FLOPS:-1e18}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-64}"
NUM_SHARDS="${NUM_SHARDS:-45}"
GPU_NAME="${GPU_NAME:-NVIDIA H100}"
MFU="${MFU:-0.45}"
PRICE_PER_GPU_HOUR="${PRICE_PER_GPU_HOUR:-3.49}"
EVAL_TOKENS=$((20 * 524288))
WANDB_RUN="${WANDB_RUN:-dummy}"
EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS:-}"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# -----------------------------------------------------------------------------
# Setup (skip with SKIP_SETUP=1): one NANOCHAT_BASE_DIR, one tokenizer, shared by both rows.
if [ -z "${SKIP_SETUP:-}" ]; then
    command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
    [ -d ".venv" ] || uv venv
    uv sync --extra gpu
    source .venv/bin/activate
    python -m nanochat.dataset -n "$NUM_SHARDS"
    # tok_train's own guard used to check only tokenizer.pkl -- token_bytes.pt is also required
    # (base_train.py loads it) and a partial copy of just the pickle would pass this guard and
    # then crash at training start. Check both.
    if [ ! -f "$NANOCHAT_BASE_DIR/tokenizer/tokenizer.pkl" ] || [ ! -f "$NANOCHAT_BASE_DIR/tokenizer/token_bytes.pt" ]; then
        if [ -f "nanochat/default_tokenizer/tokenizer.pkl" ] && [ -f "nanochat/default_tokenizer/token_bytes.pt" ]; then
            log "Using repo-committed tokenizer (nanochat/default_tokenizer/) -- skipping tok_train"
            mkdir -p "$NANOCHAT_BASE_DIR/tokenizer"
            cp nanochat/default_tokenizer/tokenizer.pkl nanochat/default_tokenizer/token_bytes.pt "$NANOCHAT_BASE_DIR/tokenizer/"
        else
            python -m scripts.tok_train --max-chars=2000000000 --vocab-size=32768
        fi
    fi
    # Prepare the pretokenized, packed base dataset once (CPU-only) -- no SFT prep here, this
    # script is base-training-only (see docs/roadmap.md's Stage 8 step 6 note). Skipped if already
    # prepared, so a resumed/rerun invocation doesn't redo this every time.
    dataset_name=$(python -c "
from nanochat.tokenizer import get_tokenizer
from scripts.data_prep import default_dataset_name
print(default_dataset_name('base', 2048, get_tokenizer()))
")
    if ! python -m scripts.data_prep --describe --dataset="$dataset_name" > /dev/null 2>&1; then
        python -m scripts.data_prep --kind=base --sequence-len=2048
    fi
else
    source .venv/bin/activate
fi

RESULTS_DIR="$NANOCHAT_BASE_DIR/contest_${LABEL}_results"
mkdir -p "$RESULTS_DIR"
RESULTS_FILE="$RESULTS_DIR/results.csv"
if [ ! -f "$RESULTS_FILE" ]; then
    echo "label,row,arch,depth,train_args,n_embd,num_kv_slots,params_total,params_scaling,flops_per_token,kv_mb,device_batch_size,total_batch_size,grad_accum,num_iterations,tokens_trained,val_bpb,tok_per_sec,bf16_mfu,peak_mem_mib,fp8_converted,train_time_sec" > "$RESULTS_FILE"
fi

# -----------------------------------------------------------------------------
# Preflight: one shape shared by both rows (only $ARCH_ARGS reaches model_info -- --fp8 would
# crash it, it's a base_train-only flag) -- see budget/GPU-hours before spending anything.
log "=============================================="
log "FP8 sanity contest '${LABEL}' preflight (no training yet)"
log "=============================================="

PLAN_FILE="$RESULTS_DIR/plan_kvshare4_win_d13.json"
python -m scripts.model_info \
    $ARCH_ARGS \
    --target-flops="$TARGET_FLOPS" --target-param-data-ratio=-1 \
    --gpu="$GPU_NAME" --num-gpus="$NPROC_PER_NODE" --mfu="$MFU" \
    --json > "$PLAN_FILE"

python - "$PRICE_PER_GPU_HOUR" "$PLAN_FILE" <<'PYEOF'
import json, sys
price_per_gpu_hour = float(sys.argv[1])
row = json.load(open(sys.argv[2]))[0]
p, f, k, t = row['params'], row['flops'], row['shape'], row['training_plan']
gpu_hours = t['gpu_hours'] or 0.0
wall_clock_hours = t['wall_clock_hours'] or 0.0
print(f"\n{'arch':20s} {'d':>3s} {'params(total)':>14s} {'params(scaling)':>16s} {'FLOPs/tok':>11s} {'KV slots':>9s} {'GPU-hours':>10s}")
print(f"{row['arch']:20s} {row['depth']:3d} {p['total']:14,d} {p['scaling']:16,d} {f['per_token']:11.3e} {k['num_kv_slots']:9d} {gpu_hours:10.2f}")
# Two rows, same shape -- double the single-row estimate. gpu_hours is total GPU-resource-hours
# (billed rate), it does NOT shrink with more GPUs; wall-clock is what shrinks.
print(f"\nTwo rows (bf16, fp8): {2*gpu_hours:.2f} GPU-hours  ~=  ${2*gpu_hours * price_per_gpu_hour:.2f} at ${price_per_gpu_hour:.2f}/GPU-hour"
      f"  |  ~{2*wall_clock_hours*60:.0f} min wall-clock training ({2*wall_clock_hours:.2f}h), plus setup")
PYEOF

if [ -n "${DRY_RUN:-}" ]; then
    log "DRY_RUN set -- stopping before training. Re-run without DRY_RUN to actually train."
    exit 0
fi

# -----------------------------------------------------------------------------
# Train both rows, skipping any already recorded in results.csv (resume after an interruption).
launch_module() {
    local module="$1"; shift
    if [ "$NPROC_PER_NODE" -eq 1 ]; then
        python -m "$module" "$@"
    else
        torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m "$module" -- "$@"
    fi
}

for row in "${CONTEST_ROWS[@]}"; do
    IFS='|' read -r name train_args <<< "$row"
    tag="contest_${LABEL}_${name}_d13"

    if [ "$WANDB_RUN" = "dummy" ]; then
        run_name="dummy"
    else
        run_name="${WANDB_RUN}_${name}_d13"
    fi

    if grep -q "^${LABEL},${name}," "$RESULTS_FILE" 2>/dev/null; then
        log "Skipping ${tag} (already in results)"
        continue
    fi

    log "=============================================="
    log "Training ${tag} (${train_args:-bf16})"
    log "=============================================="

    log_file="$RESULTS_DIR/${tag}_train.log"
    start_time=$(date +%s)
    launch_module scripts.base_train \
        $ARCH_ARGS \
        --target-flops="$TARGET_FLOPS" --target-param-data-ratio=-1 \
        --device-batch-size="$DEVICE_BATCH_SIZE" \
        --model-tag="$tag" --run="$run_name" \
        --eval-tokens="$EVAL_TOKENS" \
        --core-metric-every=-1 --sample-every=-1 --save-every=-1 \
        $train_args \
        $EXTRA_TRAIN_ARGS \
        2>&1 | tee "$log_file"
    train_time=$(( $(date +%s) - start_time ))

    read -r n_embd num_kv_slots params_total params_scaling flops_per_token kv_mb \
        total_batch_size grad_accum num_iterations val_bpb tok_per_sec bf16_mfu peak_mem_mib fp8_converted <<< "$(python - "$PLAN_FILE" "$log_file" <<'PYEOF'
import json, re, statistics, sys
plan = json.load(open(sys.argv[1]))[0]
log_text = open(sys.argv[2]).read()

def last_match(pattern, default="0.0"):
    matches = re.findall(pattern, log_text, re.MULTILINE)
    return matches[-1].replace(",", "") if matches else default

def median_match(pattern, default="0.0"):
    matches = re.findall(pattern, log_text, re.MULTILINE)
    if not matches:
        return default
    values = [float(m.replace(",", "")) for m in matches]
    return f"{statistics.median(values):.4f}"

total_batch_size = last_match(r'Total batch size ([\d,]+)', default="0")
grad_accum = last_match(r'gradient accumulation steps: (\d+)', default="0")
# Matches both "Calculated number of iterations from ...: N" (target-flops horizon) and "Using
# user-provided number of iterations: N".
num_iterations = last_match(r'number of iterations[^:\n]*:\s*([\d,]+)', default="0")
val_bpb = last_match(r'Validation bpb:\s*([\d.]+)')
# Median over all steps, not the last -- the final logged step carries eval overhead in its dt.
tok_per_sec = median_match(r'tok/sec:\s*([\d,]+)')
bf16_mfu = median_match(r'bf16_mfu:\s*([\d.]+)')
peak_mem_mib = last_match(r'Peak memory usage:\s*([\d.]+)MiB')
fp8_match = re.search(r'converted (\d+)/(\d+) linear layers', log_text)
fp8_converted = f"{fp8_match.group(1)}/{fp8_match.group(2)}" if fp8_match else "0/0"

print(plan["shape"]["n_embd"], plan["shape"]["num_kv_slots"], plan["params"]["total"],
      plan["params"]["scaling"], plan["flops"]["per_token"], plan["kv_cache"]["total_mb_at_seqlen"],
      total_batch_size, grad_accum, num_iterations, val_bpb, tok_per_sec, bf16_mfu, peak_mem_mib,
      fp8_converted)
PYEOF
)"
    tokens_trained=$((num_iterations * total_batch_size))

    log "  ${tag}: val_bpb=${val_bpb}, tok/sec=${tok_per_sec}, bf16_mfu=${bf16_mfu} (not a true ceiling for the fp8 row -- see script header), peak_mem=${peak_mem_mib}MiB, fp8_converted=${fp8_converted}, time=${train_time}s"
    echo "$LABEL,$name,llama_kvshare_win,13,\"$train_args\",$n_embd,$num_kv_slots,$params_total,$params_scaling,$flops_per_token,$kv_mb,$DEVICE_BATCH_SIZE,$total_batch_size,$grad_accum,$num_iterations,$tokens_trained,$val_bpb,$tok_per_sec,$bf16_mfu,$peak_mem_mib,$fp8_converted,$train_time" >> "$RESULTS_FILE"
done

# column isn't installed on every minimal image (confirmed missing on a real RunPod pod) --
# fall back to plain cat rather than fail the whole script on a purely cosmetic step.
print_csv() {
    if command -v column &> /dev/null; then
        column -t -s',' "$1"
    else
        cat "$1"
    fi
}

log "=============================================="
log "FP8 sanity contest '${LABEL}' complete"
log "=============================================="
log "Results: $RESULTS_FILE"
log "Checkpoints: $NANOCHAT_BASE_DIR/base_checkpoints/contest_${LABEL}_*"
echo ""
echo "Next: compare checkpoints locally with"
echo "  python -m scripts.model_info --checkpoints \$(ls $NANOCHAT_BASE_DIR/base_checkpoints | grep contest_${LABEL} | tr '\n' ',')"
echo ""
print_csv "$RESULTS_FILE"
