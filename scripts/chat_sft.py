"""
Supervised fine-tuning (SFT) the model.
Run as:

python -m scripts.chat_sft

Or torchrun for training:

torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft -- --device-batch-size=16
"""

import gc
import argparse
import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import time
import wandb
import torch
from datacore import DataManager, FileSystemDatasetStore
from nanochat.common import compute_init, compute_cleanup, print0, DummyWandb, get_base_dir, autodetect_device_type, get_peak_flops, COMPUTE_DTYPE, COMPUTE_DTYPE_REASON, is_ddp_initialized
from nanochat.checkpoint_manager import save_checkpoint, load_model, load_optimizer_state, arch_of
import torch.distributed as dist
from nanochat.flash_attention import HAS_FA3, FA3_LOAD_ERROR
from nanochat.engine import Engine
from nanochat.architectures import legacy
from modelcore import ModelManager, OptimizerHparams
from modelcore.kernels.flash_attn import build_doc_args
from scripts.chat_eval import run_chat_eval
from scripts.data_prep import default_dataset_name, prepared_dir

# -----------------------------------------------------------------------------
# CLI arguments
parser = argparse.ArgumentParser(description="Supervised fine-tuning (SFT) the model")
# Logging
parser.add_argument("--run", type=str, default="dummy", help="wandb run name ('dummy' disables wandb logging)")
# Runtime
parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)")
# Model loading
parser.add_argument("--arch", type=str, default=None, help="restrict base-checkpoint auto-discovery/output tag to this architecture (gpt|llama|llama_kvshare|llama_kvshare_win); default None picks any")
parser.add_argument("--model-tag", type=str, default=None, help="model tag to load from")
parser.add_argument("--model-step", type=int, default=None, help="model step to load from")
parser.add_argument("--load-optimizer", type=int, default=1, help="warm-start optimizer from pretrained checkpoint (0=no, 1=yes)")
# Training horizon
parser.add_argument("--num-iterations", type=int, default=-1, help="number of optimization steps (-1 = full epoch)")
# Batch sizes (default: inherit from pretrained checkpoint)
parser.add_argument("--max-seq-len", type=int, default=None, help="max context length (default: inherit from pretrain)")
parser.add_argument("--device-batch-size", type=int, default=None, help="per-device batch size (default: inherit from pretrain)")
parser.add_argument("--total-batch-size", type=int, default=None, help="total batch size in tokens (default: inherit from pretrain)")
# Optimization (default: inherit from pretrained checkpoint)
parser.add_argument("--embedding-lr", type=float, default=None, help="learning rate for embedding parameters (Adam) (default: inherit from pretrain)")
parser.add_argument("--unembedding-lr", type=float, default=None, help="learning rate for unembedding parameters (Adam) (default: inherit from pretrain)")
parser.add_argument("--matrix-lr", type=float, default=None, help="learning rate for matrix parameters (Muon) (default: inherit from pretrain)")
parser.add_argument("--init-lr-frac", type=float, default=0.8, help="initial LR as fraction of base LR")
parser.add_argument("--warmup-ratio", type=float, default=0.0, help="ratio of iterations for LR warmup")
parser.add_argument("--warmdown-ratio", type=float, default=0.5, help="ratio of iterations for LR warmdown")
parser.add_argument("--final-lr-frac", type=float, default=0.0, help="final LR as fraction of initial LR")
# Evaluation
parser.add_argument("--eval-every", type=int, default=200, help="evaluate val bpb every N steps (-1 = disable)")
parser.add_argument("--eval-tokens", type=int, default=40*524288, help="number of tokens to evaluate val loss on")
parser.add_argument("--chatcore-every", type=int, default=200, help="evaluate ChatCORE metric every N steps (-1 = disable)")
parser.add_argument("--chatcore-max-cat", type=int, default=-1, help="max problems per categorical task for ChatCORE")
parser.add_argument("--chatcore-max-sample", type=int, default=24, help="max problems per generative task for ChatCORE")
# Data
parser.add_argument("--dataset", type=str, default=None, help="prepared SFT dataset name (see scripts/data_prep.py --kind=sft, which also owns --mmlu-epochs/--gsm8k-epochs now); default: derived from --max-seq-len and the tokenizer fingerprint")
parser.add_argument("--doc-masking", action="store_true", help="restrict attention to within each packed row's own document (BOS-delimited), instead of allowing attention across document boundaries within a row -- see modelcore.kernels.flash_attn.build_doc_args. Independent of whether the base checkpoint was pretrained with masking on or off.")
parser.add_argument("--doc-masking-max-docs-per-row", type=int, default=None, help="override build_doc_args's default per-row document budget (DEFAULT_MAX_DOCS_PER_ROW=64) used to size the FA3 varlen kernel's cu_seqlens -- an SFT row packs many short conversations, so this plausibly needs raising; check with `python -m scripts.data_prep --describe --deep --dataset=<name>`'s documents/row max BEFORE launching a GPU run, not after hitting build_doc_args's assertion")
args = parser.parse_args()
user_config = vars(args).copy()
# -----------------------------------------------------------------------------

# Compute init
device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0
print0(f"COMPUTE_DTYPE: {COMPUTE_DTYPE} ({COMPUTE_DTYPE_REASON})")
synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None
get_max_memory = torch.cuda.max_memory_allocated if device_type == "cuda" else lambda: 0
if device_type == "cuda":
    gpu_device_name = torch.cuda.get_device_name(0)
    gpu_peak_flops = get_peak_flops(gpu_device_name)
    print0(f"GPU: {gpu_device_name} | Peak FLOPS (BF16): {gpu_peak_flops:.2e}")
else:
    gpu_peak_flops = float('inf')  # MFU not meaningful for CPU/MPS

# wandb logging init
use_dummy_wandb = args.run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanochat-sft", name=args.run, config=user_config)

# Flash Attention status
if not HAS_FA3:
    print0(f"WARNING: Flash Attention 3 not available ({FA3_LOAD_ERROR}), using PyTorch SDPA fallback. Training will be less efficient.")
print0(f"Intra-document masking: {'ON' if args.doc_masking else 'off'}")

# Load the model and tokenizer
model, tokenizer, meta = load_model("base", device, phase="train", model_tag=args.model_tag, step=args.model_step, arch=args.arch)

# Inherit training hyperparameters from pretrained checkpoint (None = inherit, explicit value = override)
pretrain_user_config = meta.get("user_config", {})
for name, fallback, source in [
    ("max_seq_len",       2048,  meta),
    ("device_batch_size", 32,    meta),
    ("total_batch_size",  524288, meta),
    ("embedding_lr",      0.3,   pretrain_user_config),
    ("unembedding_lr",    0.004, pretrain_user_config),
    ("matrix_lr",         0.02,  pretrain_user_config),
]:
    arg_val = getattr(args, name)
    pretrain_val = source.get(name)
    if arg_val is None:
        resolved = pretrain_val if pretrain_val is not None else fallback
        setattr(args, name, resolved)
        print0(f"Inherited {name}={resolved} from pretrained checkpoint")
    elif pretrain_val is not None and arg_val != pretrain_val:
        print0(f"NOTE: --{name.replace('_', '-')}={arg_val} overrides pretrained value of {pretrain_val}")
    else:
        print0(f"Using {name}={arg_val}")

manager = ModelManager()
orig_model = model
model = torch.compile(model, dynamic=False)
depth = orig_model.config.n_layer
num_flops_per_token = manager.stats(orig_model.config).flops_per_token
tokens_per_fwdbwd = args.device_batch_size * args.max_seq_len # tokens per iteration for a single rank
world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size # total tokens per iteration for all ranks
assert args.total_batch_size % world_tokens_per_fwdbwd == 0, f"total_batch_size ({args.total_batch_size}) must be a multiple of {world_tokens_per_fwdbwd}."
grad_accum_steps = args.total_batch_size // world_tokens_per_fwdbwd
print0(f"Tokens / micro-batch / rank: {args.device_batch_size} x {args.max_seq_len} = {tokens_per_fwdbwd:,}")
print0(f"Tokens / micro-batch: {world_tokens_per_fwdbwd:,}")
print0(f"Total batch size {args.total_batch_size:,} => gradient accumulation steps: {grad_accum_steps}")

# Initialize the Optimizer (combined MuonAdamW: Muon for matrix params, AdamW for rest)
# Note that pretraining ramps weight_decay to zero by end of pretraining, so SFT continues with zero
optimizer = manager.create_optimizer(orig_model, OptimizerHparams(
    unembedding_lr=args.unembedding_lr, embedding_lr=args.embedding_lr, matrix_lr=args.matrix_lr, weight_decay=0.0,
))

# Optionally warm-start optimizer from pretrained checkpoint (momentum buffers etc.)
# Note: load_state_dict overwrites param_group metadata (LRs, betas, etc.) with the
# pretrained values. Since pretraining warmdown brings LRs to ~0, we must save and
# restore our fresh SFT LRs after loading.
base_dir = get_base_dir()
if args.load_optimizer:
    optimizer_data = load_optimizer_state("base", device, rank=ddp_rank, model_tag=args.model_tag, step=args.model_step, arch=args.arch)
    if optimizer_data is not None:
        optimizer_data = legacy.migrate_optimizer_state_from_meta(optimizer_data, meta["model_config"], depth, log=print0)
        base_lrs = [group["lr"] for group in optimizer.param_groups]
        optimizer.load_state_dict(optimizer_data)
        del optimizer_data
        for group, base_lr in zip(optimizer.param_groups, base_lrs):
            group["lr"] = base_lr
        print0("Loaded optimizer state from pretrained checkpoint (momentum buffers only, LRs reset)")
    else:
        print0("WARNING: optimizer checkpoint not found, starting with fresh optimizer (slightly worse)")

# GradScaler for fp16 training (bf16/fp32 don't need it)
scaler = torch.amp.GradScaler() if COMPUTE_DTYPE == torch.float16 else None
if scaler is not None:
    print0("GradScaler enabled for fp16 training")

# Override the initial learning rate as a fraction of the base learning rate
for group in optimizer.param_groups:
    group["lr"] = group["lr"] * args.init_lr_frac
    group["initial_lr"] = group["lr"]

# Prepared SFT dataset (see scripts/data_prep.py --kind=sft -- tokenization/packing/masking
# happens once, offline, CPU-only; training just reads rows). Unlike the old inline generator,
# the dataset's size is known up front, so num_iterations and the LR schedule's progress no
# longer need generator-mutated globals or a per-step cross-rank all-reduce to agree on when to
# stop -- every rank derives the same num_iterations from the same manifest.
dataset_name = args.dataset or default_dataset_name("sft", args.max_seq_len, tokenizer)
data_store = FileSystemDatasetStore(prepared_dir(dataset_name))
data_manager = DataManager()
try:
    dataset = data_manager.open(data_store)
except FileNotFoundError:
    raise SystemExit(
        f"No prepared SFT dataset found for {dataset_name!r}.\nPrepare one first (CPU-only):\n"
        f"  python -m scripts.data_prep --kind=sft --dataset={dataset_name} --sequence-len={args.max_seq_len}"
    )
if dataset.info.sequence_len != args.max_seq_len:
    raise SystemExit(
        f"Dataset {dataset_name!r} was prepared with sequence_len={dataset.info.sequence_len}, "
        f"but --max-seq-len={args.max_seq_len}. Re-prepare it:\n"
        f"  python -m scripts.data_prep --kind=sft --dataset={dataset_name} --sequence-len={args.max_seq_len}"
    )
if dataset.info.tokenizer_fingerprint != tokenizer.fingerprint():
    raise SystemExit(
        f"Dataset {dataset_name!r} was prepared against tokenizer fingerprint "
        f"{dataset.info.tokenizer_fingerprint}, but the local tokenizer's fingerprint is "
        f"{tokenizer.fingerprint()}. Re-prepare against the current tokenizer:\n"
        f"  python -m scripts.data_prep --kind=sft --dataset={dataset_name} --sequence-len={args.max_seq_len}"
    )
print0(f"Dataset: {dataset_name} ({dataset.num_sequences('train'):,} train / "
      f"{dataset.num_sequences('val'):,} val sequences)")
token_bytes = data_manager.token_bytes(dataset)

bos_token_id = tokenizer.get_bos_token_id() if args.doc_masking else None
padding_id = dataset.info.padding_id if args.doc_masking else None

def make_doc_args(x):
    """None when --doc-masking is off; otherwise build_doc_args on x's actual batch size, honoring
    --doc-masking-max-docs-per-row if given. Mirrors scripts/base_train.py's make_doc_args --
    called outside the torch.compile'd model, in the micro-batch loop, same reason (see
    modelcore.kernels.flash_attn.build_doc_args's docstring)."""
    if not args.doc_masking:
        return None
    max_docs = args.doc_masking_max_docs_per_row * x.size(0) if args.doc_masking_max_docs_per_row is not None else None
    return build_doc_args(x, bos_token_id, padding_id=padding_id, max_docs=max_docs)

# --num-iterations now means optimizer STEPS (matching scripts/base_train.py), not micro-batches
# -- the old inline generator's `it` counted individual next() calls, i.e. micro-batches, which
# meant the same flag meant something different in every training script. A caller relying on the
# old micro-batch count should divide it by grad_accum_steps.
sequences_per_optimizer_step = args.device_batch_size * ddp_world_size * grad_accum_steps
if args.num_iterations > 0:
    num_iterations = args.num_iterations
else:
    num_iterations = max(1, dataset.num_sequences("train") // sequences_per_optimizer_step)
print0(f"Training horizon: {num_iterations:,} optimizer steps "
      f"({num_iterations * sequences_per_optimizer_step / dataset.num_sequences('train'):.2f} epochs "
      f"of {dataset.num_sequences('train'):,} sequences)")

train_loader = data_manager.batches(dataset, "train", args.device_batch_size, device=device,
                                    rank=ddp_rank, world_size=ddp_world_size, infinite=True)
build_val_loader = lambda: data_manager.batches(dataset, "val", args.device_batch_size, device=device,
                                                rank=ddp_rank, world_size=ddp_world_size, infinite=True)

# Learning rate schedule (linear warmup, constant, linear warmdown), driven by step/num_iterations
# -- exact now that num_iterations is known up front, rather than an approximation lagging behind
# a data-consumption buffer (SFT loss curves will not be bit-identical to before this change).
def get_lr_multiplier(progress):
    if progress < args.warmup_ratio:
        return (progress + 1e-8) / args.warmup_ratio
    elif progress <= 1.0 - args.warmdown_ratio:
        return 1.0
    else:
        decay = (progress - (1.0 - args.warmdown_ratio)) / args.warmdown_ratio
        return (1 - decay) * 1.0 + decay * args.final_lr_frac

# Momentum scheduler for Muon optimizer
def get_muon_momentum(it):
    frac = min(it / 300, 1)
    momentum = (1 - frac) * 0.85 + frac * 0.95
    return momentum

# -----------------------------------------------------------------------------
# Training loop
x, y, dataloader_state_dict = next(train_loader) # prefetch the very first batch of data
min_val_bpb = float("inf")
smooth_train_loss = 0 # EMA of training loss
ema_beta = 0.9 # EMA decay factor
total_training_time = 0 # total wall-clock time of training
step = 0
while True:
    flops_so_far = num_flops_per_token * args.total_batch_size * step
    last_step = step == num_iterations  # every rank derives this identically -- no all_reduce needed

    # once in a while: evaluate the val bpb (all ranks participate)
    if last_step or (args.eval_every > 0 and step % args.eval_every == 0):
        model.eval()
        val_loader = build_val_loader()
        eval_steps = args.eval_tokens // (args.device_batch_size * args.max_seq_len * ddp_world_size)
        val_bpb = manager.evaluate_bpb(model, val_loader, eval_steps, token_bytes, bos_token_id=bos_token_id,
                                        doc_masking_max_docs_per_row=args.doc_masking_max_docs_per_row,
                                        padding_id=padding_id)
        print0(f"Step {step:05d} | Validation bpb: {val_bpb:.4f}")
        if val_bpb < min_val_bpb:
            min_val_bpb = val_bpb
        wandb_run.log({
            "step": step,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "val/bpb": val_bpb,
        })
        model.train()

    # once in a while: estimate the ChatCORE metric (all ranks participate)
    # use the original uncompiled model because the inputs keep changing shape
    chatcore_results = {}
    if args.chatcore_every > 0 and (last_step or (step > 0 and step % args.chatcore_every == 0)):
        model.eval()
        engine = Engine(orig_model, tokenizer)
        all_tasks = ['ARC-Easy', 'ARC-Challenge', 'MMLU', 'GSM8K', 'HumanEval']
        categorical_tasks = {'ARC-Easy', 'ARC-Challenge', 'MMLU'}
        baseline_accuracies = {
            'ARC-Easy': 0.25, 'ARC-Challenge': 0.25, 'MMLU': 0.25,
            'GSM8K': 0.0, 'HumanEval': 0.0,
        }
        task_results = {}
        for task_name in all_tasks:
            limit = args.chatcore_max_cat if task_name in categorical_tasks else args.chatcore_max_sample
            max_problems = None if limit < 0 else limit  # -1 means no limit
            acc = run_chat_eval(task_name, orig_model, tokenizer, engine,
                                batch_size=args.device_batch_size, max_problems=max_problems)
            task_results[task_name] = acc
            print0(f"  {task_name}: {100*acc:.2f}%")
        # Compute ChatCORE metrics (mean centered accuracy, ranges from 0=random to 1=perfect)
        def centered_mean(tasks):
            return sum((task_results[t] - baseline_accuracies[t]) / (1.0 - baseline_accuracies[t]) for t in tasks) / len(tasks)
        chatcore = centered_mean(all_tasks)
        chatcore_cat = centered_mean(categorical_tasks)
        print0(f"Step {step:05d} | ChatCORE: {chatcore:.4f} | ChatCORE_cat: {chatcore_cat:.4f}")
        wandb_run.log({
            "step": step,
            "total_training_flops": flops_so_far,
            "chatcore_metric": chatcore,
            "chatcore_cat": chatcore_cat,
            **{f"chatcore/{task_name}": acc for task_name, acc in task_results.items()},
        })
        model.train()

    # save checkpoint at the end of the run (all ranks participate so each saves its optimizer shard)
    if last_step:
        # Arch-qualify the auto-generated tag the same way base_train.py does (base_train.py:168)
        # -- without this, an SFT run of two different architectures at the same depth (e.g. gpt
        # and llama both at d12, auto-discovered rather than given an explicit --model-tag) would
        # silently overwrite each other's chatsft_checkpoints/d12/ directory.
        arch = arch_of(meta["model_config"])
        output_dirname = args.model_tag if args.model_tag else (f"d{depth}" if arch == "gpt" else f"{arch}_d{depth}") # e.g. d12, or llama_d12
        checkpoint_dir = os.path.join(base_dir, "chatsft_checkpoints", output_dirname)
        save_checkpoint(
            checkpoint_dir,
            step,
            orig_model.state_dict(),
            optimizer.state_dict(),
            {
                "step": step,
                "val_bpb": val_bpb, # loss at last step
                "model_config": orig_model.config.to_dict(),
                "user_config": user_config, # inputs to the training script
                # Provenance: which base checkpoint this SFT run started from -- meta["model_tag"]
                # is always populated by load_model_from_dir (checkpoint_manager.py), whether the
                # tag was given explicitly or auto-discovered, so this is never missing.
                "base_model_tag": meta.get("model_tag"),
                "base_model_step": meta.get("step"),
            },
            rank=ddp_rank,
        )

    if last_step:
        break

    # -------------------------------------------------------------------------
    # single training step
    # evaluate the gradient
    synchronize()
    t0 = time.time()
    for micro_step in range(grad_accum_steps):
        doc_args = make_doc_args(x)  # built outside the compiled model -- see make_doc_args's docstring
        loss = model(x, y, doc_args=doc_args)
        train_loss = loss.detach() # for logging
        loss = loss / grad_accum_steps # each .backward() is a grad sum => normalize loss here
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        x, y, dataloader_state_dict = next(train_loader) # prefetch the next batch while the GPU is busy with forward/backward
    # step the optimizer
    lrm = get_lr_multiplier(step / num_iterations)
    muon_momentum = get_muon_momentum(step)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm
        if group['kind'] == 'muon':
            group["momentum"] = muon_momentum
    if scaler is not None:
        scaler.unscale_(optimizer)
        if is_ddp_initialized():
            for v in scaler._found_inf_per_device(optimizer).values():
                dist.all_reduce(v, op=dist.ReduceOp.MAX)
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    model.zero_grad(set_to_none=True)
    synchronize()
    t1 = time.time()
    dt = t1 - t0
    # -------------------------------------------------------------------------

    # State
    step += 1

    # logging
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss.item() # EMA the training loss
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1)) # debias the EMA
    pct_done = 100 * step / num_iterations
    tok_per_sec = int(args.total_batch_size / dt)
    flops_per_sec = num_flops_per_token * args.total_batch_size / dt
    mfu = 100 * flops_per_sec / (gpu_peak_flops * ddp_world_size)
    if step > 10:
        total_training_time += dt # only count the time after the first 10 steps
    frac_epoch = dataloader_state_dict["cursor"] / dataset.num_sequences("train")
    print0(f"step {step:05d} ({pct_done:.2f}%) | loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f} | dt: {dt * 1000:.2f}ms | tok/sec: {tok_per_sec:,} | mfu: {mfu:.2f} | epoch: {frac_epoch:.2f} | total time: {total_training_time/60:.2f}m")
    if step % 10 == 0:
        wandb_run.log({
            "step": step,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "train/loss": debiased_smooth_loss,
            "train/lrm": lrm,
            "train/dt": dt,
            "train/tok_per_sec": tok_per_sec,
            "train/mfu": mfu,
            "train/epoch": frac_epoch,
        })

    # The garbage collector spends ~500ms scanning for cycles quite frequently.
    # We manually manage it to avoid these pauses during training.
    if step == 1:
        gc.collect() # manually collect a lot of garbage from setup
        gc.freeze() # freeze all currently surviving objects and exclude them from GC
        gc.disable() # disable GC entirely except:
    elif step % 5000 == 0: # every 5000 steps...
        gc.collect() # manually collect, just to be safe for very long runs

# print a few more stats
print0(f"Peak memory usage: {get_max_memory() / 1024 / 1024:.2f}MiB")
print0(f"Total training time: {total_training_time/60:.2f}m")
print0(f"Minimum validation bpb: {min_val_bpb:.4f}")

# cleanup
wandb_run.finish() # wandb run finish
compute_cleanup()
