#!/bin/bash

# Train all three registered architectures (gpt, llama, llama_kvshare) on the same tokenizer and
# the same compute budget, so their results are actually comparable, and record everything needed
# to compare them locally afterwards (see docs/contest.md for the full runbook).
#
# Usage: bash runs/contest.sh [label]
# Example: bash runs/contest.sh jan26
#
# DRY_RUN=1 bash runs/contest.sh    # print the budget for every row and exit -- no training,
#                                    # no GPU required, no money spent. Always run this first.
#
# Env overrides (all optional):
#   TARGET_FLOPS        iso-FLOPs budget per row (default 5e18, ~2.8 GPU-hours/row on 4xA100)
#   NPROC_PER_NODE       GPUs to use (default 4; set to 1 to rehearse on a single GPU/CPU/MPS box)
#   DEVICE_BATCH_SIZE     per-device micro-batch (default 16, sized for 40GB A100s)
#   NUM_SHARDS             pretraining data shards to download during setup (default 100, a
#                            generous margin for a d16 row at TARGET_FLOPS=5e18 -- see docs/contest.md)
#   GPU_NAME, MFU, PRICE_PER_GPU_HOUR   feed the GPU-hours/dollar estimate (defaults: "NVIDIA A100", 0.4, 1.50)
#   SKIP_SETUP=1            skip venv/data/tokenizer setup (same convention as runs/miniseries.sh)
#   EXTRA_TRAIN_ARGS         appended (last) to every base_train.py invocation -- the local
#                             rehearsal path uses this to force a tiny run regardless of the row's
#                             own --depth/--target-flops, see docs/contest.md

set -euo pipefail

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
mkdir -p "$NANOCHAT_BASE_DIR"

# One row per architecture entered into the contest: "arch|depth|extra_args".
# extra_args reaches both the preflight (model_info) and the real run (base_train) -- this is
# where an architecture-specific --arch-opt (e.g. llama_kvshare's kv_share_frac) belongs. Edit
# this array to change what the contest compares (add a row, change a depth, try iso-params
# instead of iso-shape by tuning kv_share_frac until params match, etc).
CONTEST_ROWS=(
    "gpt|16|"
    "llama|16|"
    "llama_kvshare|16|--arch-opt kv_share_frac=0.5"
)

LABEL="${1:-${LABEL:-$(date +%b%d | tr '[:upper:]' '[:lower:]')}}"
TARGET_FLOPS="${TARGET_FLOPS:-5e18}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-16}"
NUM_SHARDS="${NUM_SHARDS:-100}"
GPU_NAME="${GPU_NAME:-NVIDIA A100}"
MFU="${MFU:-0.4}"
PRICE_PER_GPU_HOUR="${PRICE_PER_GPU_HOUR:-1.50}"
EVAL_TOKENS=$((20 * 524288))
WANDB_RUN="${WANDB_RUN:-contest_${LABEL}}"
EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS:-}"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# -----------------------------------------------------------------------------
# Setup (skip with SKIP_SETUP=1): one NANOCHAT_BASE_DIR, one tokenizer, shared by every row --
# this is what makes "same tokenizer for all three models" structural rather than a promise.
if [ -z "${SKIP_SETUP:-}" ]; then
    command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
    [ -d ".venv" ] || uv venv
    uv sync --extra gpu
    source .venv/bin/activate
    python -m nanochat.dataset -n "$NUM_SHARDS"
    if [ ! -f "$NANOCHAT_BASE_DIR/tokenizer/tokenizer.pkl" ]; then
        python -m scripts.tok_train --max-chars=2000000000 --vocab-size=32768
    fi
else
    source .venv/bin/activate
fi

RESULTS_DIR="$NANOCHAT_BASE_DIR/contest_${LABEL}_results"
mkdir -p "$RESULTS_DIR"
RESULTS_FILE="$RESULTS_DIR/results.csv"
if [ ! -f "$RESULTS_FILE" ]; then
    echo "label,arch,depth,extra_args,n_embd,num_kv_slots,params_total,params_scaling,flops_per_token,kv_mb,total_batch_size,num_iterations,tokens_trained,val_bpb,core_score,train_time_sec" > "$RESULTS_FILE"
fi

# -----------------------------------------------------------------------------
# Preflight: see every row's params/FLOPs/KV-cache/GPU-hours before spending anything.
# This is scripts/model_info.py's whole reason for existing (see docs/roadmap.md Stage 4/5).
log "=============================================="
log "Contest '${LABEL}' preflight (no training yet)"
log "=============================================="

PLAN_FILES=()
for row in "${CONTEST_ROWS[@]}"; do
    IFS='|' read -r arch depth extra_args <<< "$row"
    plan_file="$RESULTS_DIR/plan_${arch}_d${depth}.json"
    python -m scripts.model_info \
        --arch="$arch" --depth="$depth" $extra_args \
        --target-flops="$TARGET_FLOPS" --target-param-data-ratio=-1 \
        --gpu="$GPU_NAME" --num-gpus="$NPROC_PER_NODE" --mfu="$MFU" \
        --json > "$plan_file"
    PLAN_FILES+=("$plan_file")
done

python - "$PRICE_PER_GPU_HOUR" "${PLAN_FILES[@]}" <<'PYEOF'
import json, sys
price_per_gpu_hour = float(sys.argv[1])
rows = [json.load(open(f))[0] for f in sys.argv[2:]]
print(f"\n{'arch':16s} {'d':>3s} {'params(total)':>14s} {'params(scaling)':>16s} {'FLOPs/tok':>11s} {'KV slots':>9s} {'GPU-hours':>10s}")
total_gpu_hours = 0.0
total_wall_clock_hours = 0.0  # rows run sequentially, so summing each row's wall time is correct
for r in rows:
    p, f, k, t = r['params'], r['flops'], r['shape'], r['training_plan']
    gpu_hours = t['gpu_hours'] or 0.0
    total_gpu_hours += gpu_hours
    total_wall_clock_hours += t['wall_clock_hours'] or 0.0
    print(f"{r['arch']:16s} {r['depth']:3d} {p['total']:14,d} {p['scaling']:16,d} {f['per_token']:11.3e} {k['num_kv_slots']:9d} {gpu_hours:10.2f}")
# gpu_hours is total GPU-resource-hours (what you're billed at a per-GPU-hour rate); it does NOT
# shrink with more GPUs. wall-clock is how long the sequential run of all rows actually takes.
print(f"\nTotal: {total_gpu_hours:.2f} GPU-hours  ~=  ${total_gpu_hours * price_per_gpu_hour:.2f} at ${price_per_gpu_hour:.2f}/GPU-hour"
      f"  |  ~{total_wall_clock_hours*60:.0f} min wall-clock ({total_wall_clock_hours:.2f}h)")
PYEOF

if [ -n "${DRY_RUN:-}" ]; then
    log "DRY_RUN set -- stopping before training. Re-run without DRY_RUN to actually train."
    exit 0
fi

# -----------------------------------------------------------------------------
# Train every row, skipping any already recorded in results.csv (resume after an interruption).
if [ "$NPROC_PER_NODE" -eq 1 ]; then
    LAUNCH=(python -m scripts.base_train)
else
    LAUNCH=(torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.base_train --)
fi

for row in "${CONTEST_ROWS[@]}"; do
    IFS='|' read -r arch depth extra_args <<< "$row"
    tag="contest_${LABEL}_${arch}_d${depth}"

    if grep -q "^${LABEL},${arch},${depth}," "$RESULTS_FILE" 2>/dev/null; then
        log "Skipping ${tag} (already in results)"
        continue
    fi

    log "=============================================="
    log "Training ${tag}"
    log "=============================================="

    # base_train.py checks args.run == "dummy" (exact string) to skip wandb entirely -- suffixing
    # it unconditionally would silently require a real wandb login even when the caller asked for
    # dummy (WANDB_RUN=dummy, e.g. this script's own CPU rehearsal).
    if [ "$WANDB_RUN" = "dummy" ]; then
        run_name="dummy"
    else
        run_name="${WANDB_RUN}_${arch}_d${depth}"
    fi

    log_file="$RESULTS_DIR/${tag}_train.log"
    start_time=$(date +%s)
    "${LAUNCH[@]}" \
        --arch="$arch" --depth="$depth" $extra_args \
        --target-flops="$TARGET_FLOPS" --target-param-data-ratio=-1 \
        --device-batch-size="$DEVICE_BATCH_SIZE" \
        --model-tag="$tag" --run="$run_name" \
        --eval-tokens="$EVAL_TOKENS" \
        --core-metric-every=999999 --core-metric-max-per-task=-1 \
        --sample-every=-1 --save-every=-1 \
        $EXTRA_TRAIN_ARGS \
        2>&1 | tee "$log_file"
    train_time=$(( $(date +%s) - start_time ))

    # Static shape/params/FLOPs/KV columns come from the preflight JSON (shapes only, computed
    # once above); dynamic training-outcome columns are parsed from this row's own log. Plain
    # python re rather than `grep -oP`: -P is a GNU-grep extension, absent on macOS's BSD grep
    # (this script also runs locally for the CPU/MPS rehearsal in docs/contest.md).
    plan_file="$RESULTS_DIR/plan_${arch}_d${depth}.json"
    read -r n_embd num_kv_slots params_total params_scaling flops_per_token kv_mb \
        total_batch_size num_iterations val_bpb core_score <<< "$(python - "$plan_file" "$log_file" <<'PYEOF'
import json, re, sys
plan = json.load(open(sys.argv[1]))[0]
log_text = open(sys.argv[2]).read()

def last_match(pattern, default="0.0"):
    matches = re.findall(pattern, log_text, re.MULTILINE)
    return matches[-1].replace(",", "") if matches else default

total_batch_size = last_match(r'Total batch size ([\d,]+)', default="0")
# Matches both "Calculated number of iterations from ...: N" (target-flops/param-ratio horizon)
# and "Using user-provided number of iterations: N" (e.g. an EXTRA_TRAIN_ARGS override).
num_iterations = last_match(r'number of iterations[^:\n]*:\s*([\d,]+)', default="0")
val_bpb = last_match(r'Validation bpb:\s*([\d.]+)')
core_score = last_match(r'CORE metric:\s*([\d.]+)')

print(plan["shape"]["n_embd"], plan["shape"]["num_kv_slots"], plan["params"]["total"],
      plan["params"]["scaling"], plan["flops"]["per_token"], plan["kv_cache"]["total_mb_at_seqlen"],
      total_batch_size, num_iterations, val_bpb, core_score)
PYEOF
)"
    tokens_trained=$((num_iterations * total_batch_size))

    log "  ${tag}: params=${params_total}, iters=${num_iterations}, val_bpb=${val_bpb}, CORE=${core_score}, time=${train_time}s"
    echo "$LABEL,$arch,$depth,\"$extra_args\",$n_embd,$num_kv_slots,$params_total,$params_scaling,$flops_per_token,$kv_mb,$total_batch_size,$num_iterations,$tokens_trained,$val_bpb,$core_score,$train_time" >> "$RESULTS_FILE"
done

log "=============================================="
log "Contest '${LABEL}' complete"
log "=============================================="
log "Results: $RESULTS_FILE"
log "Checkpoints: $NANOCHAT_BASE_DIR/base_checkpoints/contest_${LABEL}_*"
echo ""
echo "Next: compare locally with"
echo "  python -m scripts.model_info --checkpoints \$(ls $NANOCHAT_BASE_DIR/base_checkpoints | grep contest_${LABEL} | tr '\n' ',')"
echo ""
echo "Results:"
column -t -s',' "$RESULTS_FILE"
