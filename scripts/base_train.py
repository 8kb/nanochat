"""
Train model. From root directory of the project, run as:

python -m scripts.base_train

or distributed as:

torchrun --nproc_per_node=8 -m scripts.base_train

If you are only on CPU/Macbook, you'll want to train a much much smaller LLM. Example:
python -m scripts.base_train --depth=4 --max-seq-len=512 --device-batch-size=1 --eval-tokens=512 --core-metric-every=-1 --total-batch-size=512 --num-iterations=20
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import ast
import gc
import json
import time
import math
import argparse

import wandb
import torch
import torch.distributed as dist

from datacore import DataManager, FileSystemDatasetStore
from modelcore import Model, ModelManager, OptimizerHparams
from nanochat.architectures import presets
from nanochat.scaling import derive_training_plan, B_REF
from nanochat.common import compute_init, compute_cleanup, print0, DummyWandb, print_banner, get_base_dir, autodetect_device_type, get_peak_flops, COMPUTE_DTYPE, COMPUTE_DTYPE_REASON, is_ddp_initialized
from nanochat.tokenizer import get_tokenizer, get_token_bytes
from nanochat.checkpoint_manager import save_checkpoint, load_checkpoint
from nanochat.loss_eval import evaluate_bpb
from nanochat.engine import Engine
from nanochat.flash_attention import HAS_FA3, FA3_LOAD_ERROR
from modelcore.kernels.flash_attn import build_doc_args
from scripts.base_eval import evaluate_core
print_banner()

# -----------------------------------------------------------------------------
# CLI arguments
parser = argparse.ArgumentParser(description="Pretrain base model")
# Logging
parser.add_argument("--run", type=str, default="dummy", help="wandb run name ('dummy' disables wandb logging)")
# Runtime
parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)")
# FP8 training
parser.add_argument("--fp8", action="store_true", help="enable FP8 training (requires H100+ GPU)")
parser.add_argument("--fp8-recipe", type=str, default="tensorwise", choices=["tensorwise"], help="FP8 scaling recipe (only tensorwise is implemented -- see modelcore.precision.fp8)")
# Model architecture
parser.add_argument("--arch", type=str, default="gpt", help="preset name (gpt, llama, llama_kvshare, llama_kvshare_win -- see nanochat.architectures.presets), used unless --model-config is also given")
parser.add_argument("--depth", type=int, default=20, help="depth of the Transformer model")
parser.add_argument("--aspect-ratio", type=int, default=64, help="model_dim = depth * aspect_ratio")
parser.add_argument("--head-dim", type=int, default=128, help="target head dimension for attention")
parser.add_argument("--max-seq-len", type=int, default=2048, help="max context length")
parser.add_argument("--window-pattern", type=str, default=None, help="sliding window pattern tiled across layers: L=full, S=half context (e.g. 'SSL'); default is the architecture's own from_depth default (GPT: SSSL, Llama/LlamaKVShare: L) rather than one arch's default overriding another's")
parser.add_argument("--arch-opt", action="append", default=None, metavar="KEY=VALUE", help="override an architecture-specific config field beyond from_depth's fixed kwargs, e.g. --arch-opt kv_share_frac=0.667 (repeatable)")
parser.add_argument("--model-config", type=str, default=None, help="overrides --arch: either a preset name (gpt, llama, llama_kvshare, llama_kvshare_win) or a path to a materialized JSON tree dumped by scripts/model_info.py --dump-config")
parser.add_argument("--d-ref-scaling-params", type=int, default=None, help="skip re-deriving the muP d12 scaling-law reference model and use this scaling-param count directly; only needed for a --model-config JSON tree with no 'reference' block")
# Training horizon (only one used, in order of precedence)
parser.add_argument("--num-iterations", type=int, default=-1, help="explicit number of optimization steps (-1 = disable)")
parser.add_argument("--target-flops", type=float, default=-1.0, help="calculate num_iterations to reach target_flops (-1 = disable)")
parser.add_argument("--target-param-data-ratio", type=float, default=12, help="calculate num_iterations to maintain data:param ratio (Chinchilla=20, -1 = disable)")
# Optimization
parser.add_argument("--device-batch-size", type=int, default=32, help="per-device batch size. good number to reduce to 16,8,4,... if you OOM on VRAM.")
parser.add_argument("--total-batch-size", type=int, default=-1, help="total batch size in tokens. decent numbers are e.g. 524288. (-1 = auto-compute optimal)")
parser.add_argument("--embedding-lr", type=float, default=0.3, help="learning rate for embedding parameters (Adam)")
parser.add_argument("--unembedding-lr", type=float, default=0.008, help="learning rate for unembedding parameters (Adam)")
parser.add_argument("--weight-decay", type=float, default=0.28, help="cautious weight decay for the Muon optimizer (for weights)")
parser.add_argument("--matrix-lr", type=float, default=0.02, help="learning rate for matrix parameters (Muon)")
parser.add_argument("--scalar-lr", type=float, default=0.5, help="learning rate for scalars (resid_lambdas, x0_lambdas)")
parser.add_argument("--warmup-steps", type=int, default=40, help="number of steps for LR warmup")
parser.add_argument("--warmdown-ratio", type=float, default=0.65, help="ratio of iterations for LR warmdown")
parser.add_argument("--final-lr-frac", type=float, default=0.05, help="final LR as fraction of initial LR")
parser.add_argument("--resume-from-step", type=int, default=-1, help="resume training from this step (-1 = disable)")
# Evaluation
parser.add_argument("--eval-every", type=int, default=250, help="evaluate val bpb every N steps (-1 = disable)")
parser.add_argument("--eval-tokens", type=int, default=80*524288, help="number of tokens to evaluate val loss on")
parser.add_argument("--core-metric-every", type=int, default=2000, help="evaluate CORE metric every N steps (-1 = disable)")
parser.add_argument("--core-metric-max-per-task", type=int, default=500, help="examples per task for CORE metric")
parser.add_argument("--sample-every", type=int, default=2000, help="sample from model every N steps (-1 = disable)")
parser.add_argument("--save-every", type=int, default=-1, help="save checkpoints every N steps (-1 = only at end)")
# Data
parser.add_argument("--dataset", type=str, default=None, help="prepared dataset name (see scripts/data_prep.py --kind=base); default: derived from --max-seq-len and the tokenizer fingerprint")
parser.add_argument("--ignore-dataloader-state", action="store_true", help="on --resume-from-step, restart the data stream from the beginning instead of refusing a pre-datacore checkpoint's dataloader state (model/optimizer weights load either way)")
parser.add_argument("--doc-masking", action="store_true", help="restrict attention to within each packed row's own document (BOS-delimited), instead of allowing attention across document boundaries within a row -- see modelcore.kernels.flash_attn.build_doc_args")
parser.add_argument("--doc-masking-max-docs-per-row", type=int, default=None, help="override build_doc_args's default per-row document budget (DEFAULT_MAX_DOCS_PER_ROW=64) used to size the FA3 varlen kernel's cu_seqlens; raise this if a run hits build_doc_args's 'exceeds max_docs' assertion for a dataset/sequence-length combination that packs unusually many documents per row")
# Output
parser.add_argument("--model-tag", type=str, default=None, help="override model tag for checkpoint directory name")
args = parser.parse_args()
user_config = vars(args).copy()  # for logging
# -----------------------------------------------------------------------------
# Compute init and wandb logging

device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None
get_max_memory = torch.cuda.max_memory_allocated if device_type == "cuda" else lambda: 0
if device_type == "cuda":
    gpu_device_name = torch.cuda.get_device_name(0)
    gpu_peak_flops = get_peak_flops(gpu_device_name)
    print0(f"GPU: {gpu_device_name} | Peak FLOPS (BF16): {gpu_peak_flops:.2e}")
else:
    gpu_peak_flops = float('inf')  # MFU not meaningful for CPU/MPS
print0(f"COMPUTE_DTYPE: {COMPUTE_DTYPE} ({COMPUTE_DTYPE_REASON})")

# wandb logging init
use_dummy_wandb = args.run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanochat", name=args.run, config=user_config)

# Flash Attention status
from nanochat.flash_attention import USE_FA3
using_fa3 = USE_FA3
if using_fa3:
    print0("✓ Using Flash Attention 3: efficient, new and awesome.")
else:
    print0("!" * 80)
    if HAS_FA3 and COMPUTE_DTYPE != torch.bfloat16:
        print0(f"WARNING: Flash Attention 3 only supports bf16, but COMPUTE_DTYPE={COMPUTE_DTYPE}. Using PyTorch SDPA fallback")
    else:
        print0(f"WARNING: Flash Attention 3 not available ({FA3_LOAD_ERROR}), using PyTorch SDPA fallback")
    print0("WARNING: Training will be less efficient without FA3")
    print0("!" * 80)
print0(f"Intra-document masking: {'ON' if args.doc_masking else 'off'}")

# -----------------------------------------------------------------------------
# Tokenizer will be useful for evaluation and also we need the vocab size to init the model
tokenizer = get_tokenizer()
token_bytes = get_token_bytes(device=device)
vocab_size = tokenizer.get_vocab_size()
tokenizer_fingerprint = tokenizer.fingerprint()
bos_token_id = tokenizer.get_bos_token_id() if args.doc_masking else None

print0(f"Vocab size: {vocab_size:,}")

# -----------------------------------------------------------------------------
# Open the prepared dataset (see scripts/data_prep.py -- tokenization/packing happens once,
# offline, CPU-only; training just reads rows). --max-seq-len must match the dataset's own
# sequence_len, and its tokenizer_fingerprint must match this run's tokenizer exactly -- both are
# hard errors, not warnings, since either mismatch produces silent garbage.
from scripts.data_prep import default_dataset_name, prepared_dir
dataset_name = args.dataset or default_dataset_name("base", args.max_seq_len, tokenizer)
dataset_dir = prepared_dir(dataset_name)
data_manager = DataManager()
data_store = FileSystemDatasetStore(dataset_dir)
try:
    dataset = data_manager.open(data_store)
except FileNotFoundError:
    raise SystemExit(
        f"No prepared dataset found at {dataset_dir}.\nPrepare one first (CPU-only -- do this "
        f"before starting a GPU run):\n"
        f"  python -m scripts.data_prep --kind=base --dataset={dataset_name} --sequence-len={args.max_seq_len}"
    )
if dataset.info.sequence_len != args.max_seq_len:
    raise SystemExit(
        f"Dataset {dataset_name!r} was prepared with sequence_len={dataset.info.sequence_len}, "
        f"but --max-seq-len={args.max_seq_len}. Re-prepare it at the matching length:\n"
        f"  python -m scripts.data_prep --kind=base --dataset={dataset_name} --sequence-len={args.max_seq_len}"
    )
if dataset.info.tokenizer_fingerprint != tokenizer_fingerprint:
    raise SystemExit(
        f"Dataset {dataset_name!r} was prepared against tokenizer fingerprint "
        f"{dataset.info.tokenizer_fingerprint}, but the local tokenizer's fingerprint is "
        f"{tokenizer_fingerprint} -- training on it would silently learn the wrong token "
        f"meanings. Re-prepare against the current tokenizer:\n"
        f"  python -m scripts.data_prep --kind=base --dataset={dataset_name} --sequence-len={args.max_seq_len}"
    )
print0(f"Dataset: {dataset_name} ({dataset.num_sequences('train'):,} train / "
      f"{dataset.num_sequences('val'):,} val sequences)")

# padding_id is None for the crop packer pretraining always uses -- build_doc_args's bos-run
# fold-in heuristic already handles that case correctly, so this is a no-op for base_train.py
# today, but a --dataset pointed at a pad-packed source (not currently a real use case for
# pretraining) picks up the dataset's own resolved padding_id automatically rather than silently
# using the heuristic against a real, distinct padding_id.
padding_id = dataset.info.padding_id if args.doc_masking else None

def make_doc_args(x):
    """None when --doc-masking is off; otherwise build_doc_args on x's actual batch size, honoring
    --doc-masking-max-docs-per-row if given (its own default, DEFAULT_MAX_DOCS_PER_ROW, is sized
    for ClimbMix at sequence_len=2048 -- see build_doc_args's docstring for why a wrong default
    here is a real memory bug, not just a style choice)."""
    if not args.doc_masking:
        return None
    max_docs = args.doc_masking_max_docs_per_row * x.size(0) if args.doc_masking_max_docs_per_row is not None else None
    return build_doc_args(x, bos_token_id, padding_id=padding_id, max_docs=max_docs)

# -----------------------------------------------------------------------------
# Initialize the Model

def _parse_arch_opts(opt_strings):
    """KEY=VALUE grammar for --arch-opt: returns a plain dict of constructor kwargs for a preset
    expander (nanochat.architectures.presets.expand). A preset function has no fixed field set to
    validate keys against, so a typo surfaces as that function's own TypeError."""
    opts = {}
    for raw in (opt_strings or []):
        assert "=" in raw, f"--arch-opt must be KEY=VALUE, got {raw!r}"
        key, _, value = raw.partition("=")
        opts[key] = ast.literal_eval(value)
    return opts


manager = ModelManager()


def build_model_meta(depth):
    """Build a model on meta device for a given depth (shapes/dtypes only, no data).

    --model-config overrides --arch: either a preset name (gpt, llama, llama_kvshare,
    llama_kvshare_win -- expanded fresh at `depth` via nanochat.architectures.presets) or a path
    to a materialized JSON tree (loaded once for the real depth; called again with depth=12 below
    for the muP scaling-law reference model, which re-expands via the tree's own config.reference
    block -- see presets.resolve_reference_config). Resolved once at the real --depth regardless
    of which `depth` was asked for here, so the d12 reference call re-expands via the resolved
    config's own `reference` block instead of re-reading --model-config at a depth it wasn't
    written for (a JSON file has its own fixed n_layer, unrelated to whatever `depth` is passed
    in). --window-pattern is only passed through when explicitly given, so each preset's own
    default (e.g. gpt's SSSL vs. llama's L) applies rather than one arch's default silently
    overriding another's. --arch-opt KEY=VALUE overrides reach fields the depth dial doesn't know
    about (e.g. kv_share_frac)."""
    model_config_selector = args.model_config or args.arch
    real_config = presets.resolve_model_config(
        model_config_selector, args.depth, aspect_ratio=args.aspect_ratio, head_dim=args.head_dim,
        max_seq_len=args.max_seq_len, vocab_size=vocab_size, window_pattern=args.window_pattern,
        arch_opts=_parse_arch_opts(args.arch_opt),
    )
    config = real_config if depth == args.depth else presets.resolve_reference_config(real_config, depth)
    with torch.device("meta"):
        return Model(config, runtime=manager.runtime)

# Build the model, move to device, init the weights
model = build_model_meta(args.depth) # 1) Build on meta device (only shapes/dtypes, no data)
model_config = model.config
model_config_kwargs = model_config.to_dict()
print0(f"Model config:\n{json.dumps(model_config_kwargs, indent=2)}")
model_stats = manager.stats(model_config)
if not using_fa3 and model_stats.has_sliding_window:
    print0("WARNING: SDPA's sliding window support falls back to an explicit attention mask instead of a fused kernel for at least one layer. Your GPU utilization will be terrible.")
    print0("WARNING: Recommend full-context attention (every layer's window >= sequence_len) without FA3.")
model.to_empty(device=device) # 2) All tensors get storage on target device but with uninitialized (garbage) data
model.init_weights() # 3) All tensors get initialized

# If we are resuming, overwrite the model parameters with those of the checkpoint
base_dir = get_base_dir()
if args.model_tag:
    output_dirname = args.model_tag
elif args.model_config:
    config_id = os.path.splitext(os.path.basename(args.model_config))[0] # preset name, or a JSON file's stem
    output_dirname = f"composed_{config_id}_d{model_config.n_layer}"
else:
    output_dirname = f"d{args.depth}" if args.arch == "gpt" else f"{args.arch}_d{args.depth}" # e.g. d12, or llama_d12
checkpoint_dir = os.path.join(base_dir, "base_checkpoints", output_dirname)
resuming = args.resume_from_step != -1
if resuming:
    print0(f"Resuming optimization from step {args.resume_from_step}")
    model_data, optimizer_data, meta_data = load_checkpoint(checkpoint_dir, args.resume_from_step, device, load_optimizer=True, rank=ddp_rank)
    model.load_state_dict(model_data, strict=True, assign=True)
    del model_data # free up this memory after the copy

# -----------------------------------------------------------------------------
# FP8 training (this has to be done before torch.compile) -- mechanism lives in
# modelcore.precision.fp8 / ModelManager.enable_fp8+fp8_disabled; see docs there for why
# Float8Linear must subclass modelcore's Linear rather than a bare nn.Linear.

if args.fp8:
    if device_type != "cuda":
        print0("Warning: FP8 training requires CUDA, ignoring --fp8 flag")
    else:
        fp8_report = manager.enable_fp8(model, recipe=args.fp8_recipe)
        print0(f"✓ FP8 training enabled ({args.fp8_recipe} scaling) - converted "
               f"{fp8_report.num_converted}/{fp8_report.num_linear} linear layers, "
               f"skipped {fp8_report.num_skipped} (too small)")

disable_fp8 = manager.fp8_disabled  # alias: every call site below reads `disable_fp8(model)`

# -----------------------------------------------------------------------------
# Compile the model

orig_model = model # original, uncompiled model, for saving raw model state_dict and for inference/evaluation (because the shapes may change shape)
model = torch.compile(model, dynamic=False) # the inputs to model will never change shape so dynamic=False is safe

# -----------------------------------------------------------------------------
# Scaling laws and muP extrapolations to determine the optimal training horizon, batch size,
# learning rates, weight decay. The math itself lives in nanochat/scaling.py (architecture-
# agnostic, no I/O) so scripts/model_info.py can report the same numbers without training
# anything; this script owns every print statement (some are grepped verbatim by
# runs/scaling_laws.sh and runs/miniseries.sh -- see AGENTS.md).

# Get the parameter counts of our model. Every architecture used to present a different key set
# (GPT's own six-key dict, everything else a generic role-named default); one materialized-tree
# system means one presentation, mapping modelcore's generic role names to the same six legacy
# keys runs/scaling_laws.sh greps verbatim out of this script's stdout (see AGENTS.md).
def _legacy_scaling_keys(role_counts):
    return {
        'wte': role_counts.get('embedding', 0),
        'value_embeds': role_counts.get('value_embedding', 0),
        'lm_head': role_counts.get('unembedding', 0),
        'transformer_matrices': role_counts.get('matrix', 0),
        'scalars': (role_counts.get('resid_scalar', 0) + role_counts.get('x0_scalar', 0)
                    + role_counts.get('smear', 0) + role_counts.get('backout_scalar', 0)),
        'total': sum(role_counts.values()),
    }
param_counts = _legacy_scaling_keys(model_stats.params_by_role)
print0(f"Parameter counts:")
for key, value in param_counts.items():
    print0(f"{key:24s}: {value:,}")
num_params = param_counts['total']
num_flops_per_token = model_stats.flops_per_token
print0(f"Estimated FLOPs per token: {num_flops_per_token:e}")

num_scaling_params = model_stats.num_scaling_params
print0(f"Number of parameters: {num_params:,} (scaling: {num_scaling_params:,})") # runs/miniseries.sh greps this exact line

# Our reference model is d12, this is where a lot of hyperparameters are tuned and then transfered to higher depths (muP style)
if args.d_ref_scaling_params is not None:
    d_ref_scaling_params = args.d_ref_scaling_params
else:
    d12_ref_config = build_model_meta(12).config # creates the config for the muP reference model
    d_ref_scaling_params = manager.stats(d12_ref_config).num_scaling_params

plan = derive_training_plan(
    num_scaling_params=num_scaling_params,
    d_ref_scaling_params=d_ref_scaling_params,
    num_flops_per_token=num_flops_per_token,
    target_param_data_ratio=args.target_param_data_ratio,
    target_flops=args.target_flops,
    num_iterations=args.num_iterations,
    total_batch_size=args.total_batch_size,
    weight_decay=args.weight_decay,
)
target_tokens = plan.target_tokens
total_batch_size = plan.total_batch_size
if plan.auto_batch_size:
    print0(f"Auto-computed optimal batch size: {total_batch_size:,} tokens")
batch_lr_scale = plan.batch_lr_scale
if batch_lr_scale != 1.0:
    print0(f"Scaling LRs by {batch_lr_scale:.4f} for batch size {total_batch_size:,} (reference: {B_REF:,})")
weight_decay_scaled = plan.weight_decay_scaled
if weight_decay_scaled != args.weight_decay:
    print0(f"Scaling weight decay from {args.weight_decay:.6f} to {weight_decay_scaled:.6f} for depth {args.depth}")
num_iterations = plan.num_iterations
if plan.horizon_source == "user":
    print0(f"Using user-provided number of iterations: {num_iterations:,}")
elif plan.horizon_source == "target_flops":
    print0(f"Calculated number of iterations from target FLOPs: {num_iterations:,}")
elif plan.horizon_source == "target_param_data_ratio":
    print0(f"Calculated number of iterations from target data:param ratio: {num_iterations:,}")
total_tokens = plan.total_tokens
print0(f"Total number of training tokens: {total_tokens:,}")
train_dataset_tokens = dataset.info.splits["train"]["num_tokens"]
print0(f"Training horizon is {total_tokens / train_dataset_tokens:.2f} epochs of the "
      f"{dataset_name!r} dataset ({dataset.num_sequences('train'):,} sequences) -- cycling past "
      f"1.0 is legal, just no longer invisible.")
print0(f"Tokens : Scaling params ratio: {total_tokens / num_scaling_params:.2f}") # e.g. Chinchilla was ~20
print0(f"Total training FLOPs estimate: {plan.total_flops:e}")

# -----------------------------------------------------------------------------
# Initialize the Optimizer (combined MuonAdamW: Muon for matrix params, AdamW for rest)
optimizer = manager.create_optimizer(orig_model, OptimizerHparams(
    unembedding_lr=args.unembedding_lr * batch_lr_scale,
    embedding_lr=args.embedding_lr * batch_lr_scale,
    scalar_lr=args.scalar_lr * batch_lr_scale,
    matrix_lr=args.matrix_lr * batch_lr_scale,
    weight_decay=weight_decay_scaled,
))

if resuming:
    optimizer.load_state_dict(optimizer_data)
    del optimizer_data

# -----------------------------------------------------------------------------
# GradScaler for fp16 training (bf16/fp32 don't need it — bf16 has the same exponent range as fp32)
scaler = torch.amp.GradScaler() if COMPUTE_DTYPE == torch.float16 else None
if scaler is not None:
    print0("GradScaler enabled for fp16 training")

# -----------------------------------------------------------------------------
# Initialize the DataLoaders for train/val
dataloader_resume_state_dict = None if not resuming else meta_data["dataloader_state_dict"]
if dataloader_resume_state_dict is not None and dataloader_resume_state_dict.get("format") != "datacore.v1":
    if not args.ignore_dataloader_state:
        raise SystemExit(
            f"{checkpoint_dir} step {args.resume_from_step} carries a pre-datacore dataloader "
            f"state ({sorted(dataloader_resume_state_dict)}); there is no faithful translation "
            f"into a prepared-dataset cursor. Model and optimizer weights load fine -- only the "
            f"data-stream position is affected. Pass --ignore-dataloader-state to restart the "
            f"data stream from the beginning."
        )
    print0("WARNING: ignoring a pre-datacore dataloader state -- restarting the data stream from cursor 0")
    dataloader_resume_state_dict = None
train_loader = data_manager.batches(dataset, "train", args.device_batch_size, device=device,
                                     rank=ddp_rank, world_size=ddp_world_size,
                                     resume=dataloader_resume_state_dict, infinite=True)
build_val_loader = lambda: data_manager.batches(dataset, "val", args.device_batch_size, device=device,
                                                 rank=ddp_rank, world_size=ddp_world_size, infinite=True)
x, y, dataloader_state_dict = next(train_loader) # kick off load of the very first batch of data

# -----------------------------------------------------------------------------
# Set up the LR/momentum/weight-decay schedulers (num_iterations was already derived above, via
# nanochat.scaling.derive_training_plan, before the optimizer was built)

# Learning rate schedule (linear warmup, constant, linear warmdown)
def get_lr_multiplier(it):
    warmup_iters = args.warmup_steps
    warmdown_iters = round(args.warmdown_ratio * num_iterations)
    if it < warmup_iters:
        return (it + 1) / warmup_iters
    elif it <= num_iterations - warmdown_iters:
        return 1.0
    else:
        progress = (num_iterations - it) / warmdown_iters
        return progress * 1.0 + (1 - progress) * args.final_lr_frac

# Momentum scheduler for Muon optimizer (warms up to 0.97, warms down to 0.90 during LR warmdown)
def get_muon_momentum(it):
    warmdown_iters = round(args.warmdown_ratio * num_iterations)
    warmdown_start = num_iterations - warmdown_iters
    if it < 400:
        frac = it / 400
        return (1 - frac) * 0.85 + frac * 0.97
    elif it >= warmdown_start:
        progress = (it - warmdown_start) / warmdown_iters
        return 0.97 * (1 - progress) + 0.90 * progress
    else:
        return 0.97

# Weight decay scheduler for Muon optimizer (cosine decay to zero over the course of training)
def get_weight_decay(it):
    return weight_decay_scaled * 0.5 * (1 + math.cos(math.pi * it / num_iterations))

# -----------------------------------------------------------------------------
# Training loop

# Loop state (variables updated by the training loop)
if not resuming:
    step = 0
    val_bpb = None # will be set if eval_every > 0
    min_val_bpb = float("inf")
    smooth_train_loss = 0 # EMA of training loss
    total_training_time = 0 # total wall-clock time of training
else:
    step = meta_data["step"]
    loop_state = meta_data["loop_state"]
    val_bpb = meta_data["val_bpb"]
    min_val_bpb = loop_state["min_val_bpb"]
    smooth_train_loss = loop_state["smooth_train_loss"]
    total_training_time = loop_state["total_training_time"]

# Figure out the needed gradient accumulation micro-steps to reach the desired total batch size per step
tokens_per_fwdbwd = args.device_batch_size * args.max_seq_len # tokens per iteration for a single rank
world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size # total tokens per iteration for all ranks
assert total_batch_size % world_tokens_per_fwdbwd == 0, f"total_batch_size ({total_batch_size}) must be a multiple of {world_tokens_per_fwdbwd}."
grad_accum_steps = total_batch_size // world_tokens_per_fwdbwd
print0(f"Tokens / micro-batch / rank: {args.device_batch_size} x {args.max_seq_len} = {tokens_per_fwdbwd:,}")
print0(f"Tokens / micro-batch: {world_tokens_per_fwdbwd:,}")
print0(f"Total batch size {total_batch_size:,} => gradient accumulation steps: {grad_accum_steps}")

# Go!
while True:
    last_step = step == num_iterations # loop runs num_iterations+1 times so that we can eval/save at the end
    flops_so_far = num_flops_per_token * total_batch_size * step

    # once in a while: evaluate the val bpb (all ranks participate)
    if args.eval_every > 0 and (last_step or step % args.eval_every == 0):
        model.eval()
        val_loader = build_val_loader()
        eval_steps = args.eval_tokens // (args.device_batch_size * args.max_seq_len * ddp_world_size)
        with disable_fp8(model):
            val_bpb = evaluate_bpb(model, val_loader, eval_steps, token_bytes, bos_token_id=bos_token_id,
                                    doc_masking_max_docs_per_row=args.doc_masking_max_docs_per_row,
                                    padding_id=padding_id)
        print0(f"Step {step:05d} | Validation bpb: {val_bpb:.6f}")
        if val_bpb < min_val_bpb:
            min_val_bpb = val_bpb
        wandb_run.log({
            "step": step,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "val/bpb": val_bpb,
        })
        model.train()

    # once in a while: estimate the CORE metric (all ranks participate)
    # use the original uncompiled model because the inputs keep changing shape
    # disable FP8 for evaluation to use BF16 for more consistent/accurate results
    results = {}
    if args.core_metric_every > 0 and (last_step or (step > 0 and step % args.core_metric_every == 0)):
        model.eval()
        with disable_fp8(orig_model):
            results = evaluate_core(orig_model, tokenizer, device, max_per_task=args.core_metric_max_per_task)
        print0(f"Step {step:05d} | CORE metric: {results['core_metric']:.4f}")
        wandb_run.log({
            "step": step,
            "total_training_flops": flops_so_far,
            "core_metric": results["core_metric"],
            "centered_results": results["centered_results"],
        })
        model.train()

    # once in a while: sample from the model (only on master process)
    # use the original uncompiled model because the inputs keep changing shape
    if args.sample_every > 0 and master_process and (last_step or (step > 0 and step % args.sample_every == 0)):
        model.eval()
        prompts = [
            "The capital of France is",
            "The chemical symbol of gold is",
            "If yesterday was Friday, then tomorrow will be",
            "The opposite of hot is",
            "The planets of the solar system are:",
            "My favorite color is",
            "If 5*x + 3 = 13, then x is",
        ]
        engine = Engine(orig_model, tokenizer) # use orig_model to avoid recompilation
        for prompt in prompts:
            tokens = tokenizer(prompt, prepend="<|bos|>")
            with disable_fp8(orig_model):
                sample, _ = engine.generate_batch(tokens, num_samples=1, max_tokens=16, temperature=0)
            print0(tokenizer.decode(sample[0]))
        model.train()

    # save checkpoint: at the end of the run, or every save_every steps, except at the first step or the resume step
    if last_step or (step > 0 and step != args.resume_from_step and args.save_every > 0 and step % args.save_every == 0):
        save_checkpoint(
            checkpoint_dir,
            step,
            orig_model.state_dict(), # model parameters
            optimizer.state_dict(), # optimizer state
            { # metadata saved as json
                "step": step,
                "val_bpb": val_bpb, # loss at last step
                "core_metric": results.get("core_metric"), # None unless a CORE eval ran this exact step
                "tokenizer_fingerprint": tokenizer_fingerprint,
                "model_config": model_config_kwargs,
                "user_config": user_config, # inputs to the training script
                "device_batch_size": args.device_batch_size,
                "max_seq_len": args.max_seq_len,
                "total_batch_size": total_batch_size,
                "dataloader_state_dict": dataloader_state_dict,
                "loop_state": { # all loop state (other than step) so that we can resume training
                    "min_val_bpb": min_val_bpb,
                    "smooth_train_loss": smooth_train_loss,
                    "total_training_time": total_training_time,
                },
            },
            rank=ddp_rank,
        )

    # termination conditions (TODO: possibly also add loss explosions etc.)
    if last_step:
        break

    # -------------------------------------------------------------------------
    # single training step
    # evaluate the gradient
    synchronize()
    t0 = time.time()
    for micro_step in range(grad_accum_steps):
        # doc_args is built here, outside the compiled model, and passed in as plain data --
        # deriving it from idx inside a torch.compile'd forward hits the recompile limit (see
        # modelcore.kernels.flash_attn.build_doc_args's docstring).
        doc_args = make_doc_args(x)
        loss = model(x, y, doc_args=doc_args)
        train_loss = loss.detach() # for logging
        loss = loss / grad_accum_steps # each .backward() is a grad sum => normalize loss here
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        x, y, dataloader_state_dict = next(train_loader) # prefetch the next batch while the GPU is busy with forward/backward
    # step the optimizer
    lrm = get_lr_multiplier(step)
    muon_momentum = get_muon_momentum(step)
    muon_weight_decay = get_weight_decay(step)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm
        if group['kind'] == 'muon':
            group["momentum"] = muon_momentum
            group["weight_decay"] = muon_weight_decay
    if scaler is not None:
        scaler.unscale_(optimizer)
        # In distributed training, all ranks must agree on whether to skip the step.
        # Each rank may independently encounter inf/nan gradients, so we all-reduce
        # the found_inf flag (MAX = if any rank found inf, all ranks skip).
        if is_ddp_initialized():
            for v in scaler._found_inf_per_device(optimizer).values():
                dist.all_reduce(v, op=dist.ReduceOp.MAX)
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    model.zero_grad(set_to_none=True)
    train_loss_f = train_loss.item() # .item() is a CPU-GPU sync point
    synchronize()
    t1 = time.time()
    dt = t1 - t0
    # -------------------------------------------------------------------------

    # logging (CPU action only)
    ema_beta = 0.9 # EMA decay factor for some smoothing just for nicer logging
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f # EMA the training loss
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1)) # debias the EMA
    pct_done = 100 * step / num_iterations
    tok_per_sec = int(total_batch_size / dt)
    flops_per_sec = num_flops_per_token * total_batch_size / dt
    mfu = 100 * flops_per_sec / (gpu_peak_flops * ddp_world_size)
    if step > 10:
        total_training_time += dt # only count the time after the first 10 steps
    # Calculate ETA based on average time per step (excluding first 10 steps)
    steps_done = step - 10
    if steps_done > 0:
        avg_time_per_step = total_training_time / steps_done
        remaining_steps = num_iterations - step
        eta_seconds = remaining_steps * avg_time_per_step
        eta_str = f" | eta: {eta_seconds/60:.1f}m"
    else:
        eta_str = ""
    frac_epoch = dataloader_state_dict["cursor"] / dataset.num_sequences("train")
    epoch = f"{frac_epoch:.2f} | cursor: {dataloader_state_dict['cursor']:,}"
    print0(f"step {step:05d}/{num_iterations:05d} ({pct_done:.2f}%) | loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f} | dt: {dt * 1000:.2f}ms | tok/sec: {tok_per_sec:,} | bf16_mfu: {mfu:.2f} | epoch: {epoch} | total time: {total_training_time/60:.2f}m{eta_str}")
    if step % 100 == 0:
        log_data = {
            "step": step,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "train/loss": debiased_smooth_loss,
            "train/lrm": lrm,
            "train/dt": dt,
            "train/tok_per_sec": tok_per_sec,
            "train/mfu": mfu,
            "train/epoch": epoch,
        }
        wandb_run.log(log_data)

    # state update
    first_step_of_run = (step == 0) or (resuming and step == args.resume_from_step)
    step += 1

    # The garbage collector is sadly a little bit overactive and for some poorly understood reason,
    # it spends ~500ms scanning for cycles quite frequently, just to end up cleaning up very few tiny objects each time.
    # So we manually manage and help it out here
    if first_step_of_run:
        gc.collect() # manually collect a lot of garbage from setup
        gc.freeze() # immediately freeze all currently surviving objects and exclude them from GC
        gc.disable() # nuclear intervention here: disable GC entirely except:
    elif step % 5000 == 0: # every 5000 steps...
        gc.collect() # manually collect, just to be safe for very, very long runs

# print a few more stats
print0(f"Peak memory usage: {get_max_memory() / 1024 / 1024:.2f}MiB")
print0(f"Total training time: {total_training_time/60:.2f}m")
if val_bpb is not None:
    print0(f"Minimum validation bpb: {min_val_bpb:.6f}")

# cleanup
wandb_run.finish() # wandb run finish
compute_cleanup()
