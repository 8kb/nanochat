#!/bin/bash

# Cheap shakedown of the architecture contest: same rows as runs/contest.sh (base training, then
# SFT + chat_eval per row), but at depth 12 and a 1e18 FLOPs budget instead of d16/5e18 -- about
# 1/5th the base GPU-hours, for proving the harness end-to-end on real cloud GPUs before
# committing to the full d16 run. Kept as its own file rather than a CONTEST_ROWS edit in
# runs/contest.sh so that script's committed defaults stay the real d16 contest (see
# docs/contest.md "Sizing"). Edit CONTEST_ROWS below to narrow this to one architecture for an
# even cheaper pipeline-validation run (e.g. just "llama_kvshare_win|12|--arch-opt kv_share_frac=0.5").
# Default rows omit llama_kvshare (already measured once against this exact pipeline -- see
# docs/contest.md's Stage 1 results) in favor of llama_kvshare_win, its sliding-window sibling.
# chat_eval runs under launch_module (torchrun on NPROC_PER_NODE>1) rather than a single-rank
# `python` -- it already shards problems across ranks and all_reduces the result (see
# scripts/chat_eval.py), so running it single-rank was leaving 3 of 4 billed GPUs idle during
# eval, the same mistake that made eval cost more than training once already.
#
# Usage: bash runs/contest_d12.sh [label]
# Example: bash runs/contest_d12.sh d12test
#
# DRY_RUN=1 bash runs/contest_d12.sh    # print the budget for every row and exit -- no training,
#                                        # no GPU required, no money spent. Always run this first.
#
# NUM_SHARDS matters more here than it looks: at a *fixed* FLOPs budget a shallower model is
# cheaper per token and so trains on MORE tokens, not fewer -- d12 needs more data than d16 at the
# same budget, not less. This script's default (45) is sized for its own 1e18 budget; raising
# TARGET_FLOPS without raising NUM_SHARDS silently exhausts the shards and starts repeating data
# (nanochat/dataloader.py cycles infinitely rather than erroring) -- see docs/contest.md "Sizing".
#
# Env overrides (all optional, same as runs/contest.sh):
#   TARGET_FLOPS        iso-FLOPs budget per row (default 1e18, ~0.56 GPU-hours/row on 4xA100)
#   NPROC_PER_NODE       GPUs to use (default 4; set to 1 to rehearse on a single GPU/CPU/MPS box)
#   DEVICE_BATCH_SIZE     per-device micro-batch (default 16, sized for 40GB A100s -- on an H100
#                            or larger card this leaves most of its memory unused and the run
#                            becomes overhead- rather than compute-bound; check `nvidia-smi` memory
#                            usage early and raise this if it's well under the card's total, see
#                            docs/contest.md "Lessons from the first real cloud run")
#   NUM_SHARDS             pretraining data shards to download during setup (default 45, sized for
#                            a d12 row at TARGET_FLOPS=1e18 with ~18% margin -- see docs/contest.md)
#   GPU_NAME, MFU, PRICE_PER_GPU_HOUR   feed the GPU-hours/dollar estimate (defaults: "NVIDIA A100", 0.4, 1.50)
#   SKIP_SETUP=1            skip venv/data/tokenizer setup (same convention as runs/miniseries.sh)
#   EXTRA_TRAIN_ARGS         appended (last) to every base_train.py invocation
#   EXTRA_SFT_ARGS           appended (last) to every scripts.chat_sft invocation
#   EXTRA_CHATEVAL_ARGS      appended (last) to every scripts.chat_eval invocation

set -euo pipefail

# RunPod injects account Secrets (e.g. WANDB_API_KEY) into /etc/rp_environment, but only sources
# it into *interactive* shells via .bashrc's interactive-shell guard -- a non-interactive launch
# (which is how this script gets run when kicked off over a scripted SSH command, not a human
# typing at a prompt) never sees it otherwise, and wandb then fails with "api_key not configured
# (no-tty)" even though the secret really is there. Sourcing it here makes it available
# automatically regardless of how this script was launched; harmless no-op off RunPod.
[ -f /etc/rp_environment ] && source /etc/rp_environment

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
mkdir -p "$NANOCHAT_BASE_DIR"

# One row per architecture entered into the contest: "arch|depth|extra_args". Depth matches
# runs/contest.sh's shape choice (d12 instead of d16); kv_share_frac unchanged.
CONTEST_ROWS=(
    "gpt|12|"
    "llama|12|"
    "llama_kvshare_win|12|--arch-opt kv_share_frac=0.5"
)

LABEL="${1:-${LABEL:-$(date +%b%d | tr '[:upper:]' '[:lower:]')}}"
TARGET_FLOPS="${TARGET_FLOPS:-1e18}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-16}"
NUM_SHARDS="${NUM_SHARDS:-45}"
GPU_NAME="${GPU_NAME:-NVIDIA A100}"
MFU="${MFU:-0.4}"
PRICE_PER_GPU_HOUR="${PRICE_PER_GPU_HOUR:-1.50}"
EVAL_TOKENS=$((20 * 524288))
WANDB_RUN="${WANDB_RUN:-contest_${LABEL}}"
EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS:-}"
EXTRA_SFT_ARGS="${EXTRA_SFT_ARGS:-}"           # appended to every scripts.chat_sft invocation
# chat_eval.py's --max-problems has no default cap (None = full test set) -- two of its five
# tasks (GSM8K ~1319 problems, HumanEval ~164) are generative (one autoregressive sample per
# problem, unbatched within a rank), so at full scale eval alone can badly dominate training cost
# for a small contest model: found the hard way when a single architecture's eval ran past 25
# minutes still partway through GSM8K, well past base+SFT training combined. Capped by default;
# set CHATEVAL_MAX_PROBLEMS="" (empty, not -1 -- chat_eval.py has no "uncapped" sentinel value,
# and -1 would evaluate zero problems instead) for the true uncapped full-scale eval.
CHATEVAL_MAX_PROBLEMS="${CHATEVAL_MAX_PROBLEMS-100}"
EXTRA_CHATEVAL_ARGS="${EXTRA_CHATEVAL_ARGS:-}" # appended (last) to every scripts.chat_eval invocation

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
    # tok_train's own guard used to check only tokenizer.pkl -- token_bytes.pt is also required
    # (base_train.py/base_eval.py/chat_sft.py all load it) and a partial copy of just the pickle
    # would pass this guard and then crash at training start. Check both.
    if [ ! -f "$NANOCHAT_BASE_DIR/tokenizer/tokenizer.pkl" ] || [ ! -f "$NANOCHAT_BASE_DIR/tokenizer/token_bytes.pt" ]; then
        if [ -f "nanochat/default_tokenizer/tokenizer.pkl" ] && [ -f "nanochat/default_tokenizer/token_bytes.pt" ]; then
            # Repo-committed tokenizer (small, content-derived, fully portable -- see
            # nanochat/default_tokenizer/) -- skips training one from scratch on every pod.
            log "Using repo-committed tokenizer (nanochat/default_tokenizer/) -- skipping tok_train"
            mkdir -p "$NANOCHAT_BASE_DIR/tokenizer"
            cp nanochat/default_tokenizer/tokenizer.pkl nanochat/default_tokenizer/token_bytes.pt "$NANOCHAT_BASE_DIR/tokenizer/"
        else
            python -m scripts.tok_train --max-chars=2000000000 --vocab-size=32768
        fi
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
CHAT_RESULTS_FILE="$RESULTS_DIR/chat_results.csv"
if [ ! -f "$CHAT_RESULTS_FILE" ]; then
    echo "label,arch,depth,base_tag,arc_easy,arc_challenge,mmlu,gsm8k,humaneval,chatcore_metric,sft_time_sec,eval_time_sec" > "$CHAT_RESULTS_FILE"
fi

# -----------------------------------------------------------------------------
# Preflight: see every row's params/FLOPs/KV-cache/GPU-hours before spending anything.
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
# launch_module lets both scripts.base_train and scripts.chat_sft share one launcher-selection
# rule (torchrun for multi-GPU, plain python for the NPROC_PER_NODE=1 CPU/MPS rehearsal path).
launch_module() {
    local module="$1"; shift
    if [ "$NPROC_PER_NODE" -eq 1 ]; then
        python -m "$module" "$@"
    else
        torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m "$module" -- "$@"
    fi
}

for row in "${CONTEST_ROWS[@]}"; do
    IFS='|' read -r arch depth extra_args <<< "$row"
    tag="contest_${LABEL}_${arch}_d${depth}"

    # base_train.py checks args.run == "dummy" (exact string) to skip wandb entirely -- suffixing
    # it unconditionally would silently require a real wandb login even when the caller asked for
    # dummy.
    if [ "$WANDB_RUN" = "dummy" ]; then
        run_name="dummy"
        sft_run_name="dummy"
    else
        run_name="${WANDB_RUN}_${arch}_d${depth}"
        sft_run_name="${WANDB_RUN}_${arch}_d${depth}_sft"
    fi

    if grep -q "^${LABEL},${arch},${depth}," "$RESULTS_FILE" 2>/dev/null; then
        log "Skipping ${tag} base training (already in results)"
    else
        log "=============================================="
        log "Training ${tag}"
        log "=============================================="

        log_file="$RESULTS_DIR/${tag}_train.log"
        start_time=$(date +%s)
        launch_module scripts.base_train \
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
        # python re rather than `grep -oP`: -P is a GNU-grep extension, absent on macOS's BSD grep.
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
    fi

    # -------------------------------------------------------------------------
    # Chat (SFT) step: fine-tune this row's base checkpoint, then chat_eval it. Reuses the same
    # tag as the base checkpoint -- chat_sft.py's output tag lands in a separate
    # chatsft_checkpoints/ namespace, so there's no collision, just a clean 1:1 base<->chat pairing.
    if grep -q "^${LABEL},${arch},${depth}," "$CHAT_RESULTS_FILE" 2>/dev/null; then
        log "Skipping ${tag} chat SFT (already in chat_results)"
        continue
    fi

    log "=============================================="
    log "SFT training ${tag}"
    log "=============================================="
    sft_log_file="$RESULTS_DIR/${tag}_sft.log"
    sft_start_time=$(date +%s)
    launch_module scripts.chat_sft \
        --arch="$arch" --model-tag="$tag" --run="$sft_run_name" \
        $EXTRA_SFT_ARGS \
        2>&1 | tee "$sft_log_file"
    sft_time=$(( $(date +%s) - sft_start_time ))

    log "=============================================="
    log "Chat eval ${tag}"
    log "=============================================="
    eval_log_file="$RESULTS_DIR/${tag}_chateval.log"
    eval_start_time=$(date +%s)
    MAX_PROBLEMS_ARG=()
    [ -n "$CHATEVAL_MAX_PROBLEMS" ] && MAX_PROBLEMS_ARG=(--max-problems="$CHATEVAL_MAX_PROBLEMS")
    launch_module scripts.chat_eval -i sft -g "$tag" \
        "${MAX_PROBLEMS_ARG[@]}" \
        $EXTRA_CHATEVAL_ARGS \
        2>&1 | tee "$eval_log_file"
    eval_time=$(( $(date +%s) - eval_start_time ))

    read -r arc_easy arc_challenge mmlu gsm8k humaneval chatcore <<< "$(python - "$eval_log_file" <<'PYEOF'
import re, sys
log_text = open(sys.argv[1]).read()

def acc(task):
    matches = re.findall(rf'{re.escape(task)} accuracy:\s*([\d.]+)%', log_text)
    return matches[-1] if matches else "0.0"

chatcore = re.findall(r'ChatCORE metric:\s*([\d.]+)', log_text)
print(acc("ARC-Easy"), acc("ARC-Challenge"), acc("MMLU"), acc("GSM8K"), acc("HumanEval"),
      chatcore[-1] if chatcore else "0.0")
PYEOF
)"

    log "  ${tag} SFT: ChatCORE=${chatcore}, ARC-Easy=${arc_easy}%, MMLU=${mmlu}%, GSM8K=${gsm8k}%, sft_time=${sft_time}s, eval_time=${eval_time}s"
    echo "$LABEL,$arch,$depth,$tag,$arc_easy,$arc_challenge,$mmlu,$gsm8k,$humaneval,$chatcore,$sft_time,$eval_time" >> "$CHAT_RESULTS_FILE"
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
log "Contest '${LABEL}' complete"
log "=============================================="
log "Base results: $RESULTS_FILE"
log "Chat (SFT) results: $CHAT_RESULTS_FILE"
log "Base checkpoints: $NANOCHAT_BASE_DIR/base_checkpoints/contest_${LABEL}_*"
log "Chat checkpoints: $NANOCHAT_BASE_DIR/chatsft_checkpoints/contest_${LABEL}_*"
echo ""
echo "Next: compare base checkpoints locally with"
echo "  python -m scripts.model_info --checkpoints \$(ls $NANOCHAT_BASE_DIR/base_checkpoints | grep contest_${LABEL} | tr '\n' ',')"
echo ""
echo "Base results:"
print_csv "$RESULTS_FILE"
echo ""
echo "Chat (SFT) results:"
print_csv "$CHAT_RESULTS_FILE"
