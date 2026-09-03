"""
Report parameters, FLOPs, KV-cache bytes, and the derived training horizon for one or more
(--arch, --depth) configs -- without a GPU, without cached training data, and without training
anything. This is the tool for picking matched configs before spending cloud GPU-hours on an
architecture comparison (see docs/roadmap.md's contest stage): every number here is computed on
torch.device("meta") (shapes/dtypes only, no real weight values -- see docs/architecture.md's
"The meta-device footgun"), the same way scripts/base_train.py sizes a run before it starts.

The --arch-opt/training-horizon flags mirror scripts/base_train.py's exactly (same defaults, same
math via nanochat.scaling.derive_training_plan) so this tool's numbers and a real base_train run's
printed numbers must agree for the same arguments -- that agreement is the tool's correctness
check, not a coincidence.

--checkpoints switches to the other half of the same job: inspecting models that have *already*
been trained (e.g. downloaded from a cloud contest run). No weights are loaded, just each
checkpoint's meta.json -- so this stays instant and GPU-free too -- and it additionally reports what
training actually produced (val bpb, CORE, wall time) and whether the checkpoint's tokenizer
fingerprint matches this machine's local tokenizer (see nanochat.tokenizer.RustBPETokenizer.fingerprint).

Examples:

    python -m scripts.model_info --arch gpt,llama,llama_kvshare --depth 12
    python -m scripts.model_info --arch llama_kvshare --depth 12,14,16 --arch-opt kv_share_frac=0.667
    python -m scripts.model_info --arch gpt,llama --depth 12 --gpu "NVIDIA A100" --num-gpus 4 --json
    python -m scripts.model_info --checkpoints                       # every checkpoint under base_checkpoints/
    python -m scripts.model_info --checkpoints d6,llama_d12 --json
"""
import os
import io
import json as json_module
import argparse
import contextlib

import torch

from nanochat.model import get_config_class, get_model_class, config_from_dict, apply_arch_opts
from nanochat.model.param_roles import collect_param_roles
from nanochat.common import get_peak_flops, get_base_dir
from nanochat.scaling import derive_training_plan
from nanochat.checkpoint_manager import find_last_step


def default_vocab_size():
    """Same vocab_size scripts/base_train.py would use: the local tokenizer's, if one has been
    trained on this machine, else the shared default (32768) both GPTConfig/LlamaConfig use."""
    try:
        from nanochat.tokenizer import get_tokenizer
        return get_tokenizer().get_vocab_size(), "local tokenizer"
    except FileNotFoundError:
        return 32768, "no local tokenizer found, using the architecture default"


def get_scaling_params(m):
    """Same definition as scripts/base_train.py's get_scaling_params: matrix + unembedding role
    params give the cleanest scaling laws (see dev/LOG.md Jan 27, 2026). Reads role names
    directly rather than m.num_scaling_params()'s dict keys, since role names ("matrix",
    "unembedding") are stable across architectures by construction."""
    roles = collect_param_roles(m)
    matrix = sum(p.numel() for p in roles.get('matrix', []))
    unembedding = sum(p.numel() for p in roles.get('unembedding', []))
    return matrix + unembedding


def build_meta(arch, depth, args, vocab_size):
    """Build a model on meta device: shapes/dtypes only, no real weight values -- and therefore
    no to_empty()/init_weights() needed either, since every number this script reports (param
    counts, FLOPs, KV-cache bytes) depends only on shapes, not values."""
    config_cls = get_config_class(arch)
    model_cls = get_model_class(arch)
    assert hasattr(config_cls, "from_depth"), (
        f"Architecture {arch!r} ({config_cls.__name__}) has no from_depth(...) classmethod."
    )
    from_depth_kwargs = dict(aspect_ratio=args.aspect_ratio, head_dim=args.head_dim, max_seq_len=args.max_seq_len, vocab_size=vocab_size)
    if args.window_pattern is not None:
        from_depth_kwargs["window_pattern"] = args.window_pattern
    config = config_cls.from_depth(depth, **from_depth_kwargs)
    config = apply_arch_opts(config, args.arch_opt)
    with torch.device("meta"):
        model = model_cls(config)
    return model


def inspect_model(model, arch, depth, args):
    """Shape/params/FLOPs/KV-cache block, shared by config-mode (a fresh meta-device build from
    --depth) and checkpoint-mode (a meta-device build reconstructed from a checkpoint's saved
    model_config) inspection -- every number here depends only on shapes, not on trained values."""
    config = model.config

    role_counts = model.num_scaling_params()
    num_matmul_params = model.num_matmul_params()
    scaling_params = get_scaling_params(model)

    flops_per_token = model.estimate_flops()
    prefill_flops = model.estimate_prefill_flops(config.sequence_len)
    decode_flops = model.estimate_decode_flops(config.sequence_len)

    kv_cache_spec = model.kv_cache_spec()
    kv_bytes_per_token = model.kv_bytes_per_token()
    kv_cache_bytes = kv_bytes_per_token * config.sequence_len * args.kv_batch_size

    return {
        "arch": arch,
        "depth": depth,
        "shape": {
            "n_layer": config.n_layer, "n_embd": config.n_embd,
            "n_head": config.n_head, "n_kv_head": config.n_kv_head,
            "sequence_len": config.sequence_len, "window_pattern": config.window_pattern,
            "num_kv_slots": kv_cache_spec["num_kv_slots"],
        },
        "params": {**role_counts, "matmul": num_matmul_params, "scaling": scaling_params},
        "flops": {
            "per_token": flops_per_token, "prefill_at_seqlen": prefill_flops, "decode_at_seqlen": decode_flops,
        },
        "kv_cache": {
            "bytes_per_token": kv_bytes_per_token,
            "total_mb_at_seqlen": kv_cache_bytes / 1e6,
            "kv_batch_size": args.kv_batch_size,
        },
    }


def inspect_one(arch, depth, args, vocab_size):
    """Config mode: a hypothetical (arch, depth) that hasn't been trained -- reports the static
    block plus the training plan a real base_train.py run would derive for it."""
    model = build_meta(arch, depth, args, vocab_size)
    row = inspect_model(model, arch, depth, args)
    scaling_params = row["params"]["scaling"]

    d12_ref = build_meta(arch, 12, args, vocab_size)
    plan = derive_training_plan(
        num_scaling_params=scaling_params,
        d_ref_scaling_params=get_scaling_params(d12_ref),
        num_flops_per_token=row["flops"]["per_token"],
        target_param_data_ratio=args.target_param_data_ratio,
        target_flops=args.target_flops,
        num_iterations=args.num_iterations,
        total_batch_size=args.total_batch_size,
        weight_decay=args.weight_decay,
    )

    gpu_hours = None
    if args.gpu is not None:
        peak_flops = get_peak_flops(args.gpu)
        gpu_hours = plan.total_flops / (peak_flops * args.mfu * args.num_gpus) / 3600

    row["training_plan"] = {
        "target_tokens": plan.target_tokens, "total_batch_size": plan.total_batch_size,
        "auto_batch_size": plan.auto_batch_size, "num_iterations": plan.num_iterations,
        "horizon_source": plan.horizon_source, "total_tokens": plan.total_tokens,
        "total_flops": plan.total_flops, "gpu_hours": gpu_hours,
    }
    return row


def local_tokenizer_fingerprint():
    try:
        from nanochat.tokenizer import get_tokenizer
        return get_tokenizer().fingerprint()
    except FileNotFoundError:
        return None


def list_checkpoint_tags(explicit):
    """Resolve --checkpoints into a list of model tags: an explicit comma-separated list, or (if
    given with no value) every directory under base_checkpoints/. Empty entries (a trailing comma,
    e.g. from `tr '\\n' ','` on a shell one-liner) are dropped rather than resolving to the
    checkpoints directory itself."""
    if explicit:
        return [t.strip() for t in explicit.split(",") if t.strip()]
    checkpoints_dir = os.path.join(get_base_dir(), "base_checkpoints")
    if not os.path.isdir(checkpoints_dir):
        return []
    return sorted(t for t in os.listdir(checkpoints_dir) if os.path.isdir(os.path.join(checkpoints_dir, t)))


def inspect_checkpoint(tag, args, local_fingerprint):
    """Checkpoint mode: a model that has actually been trained -- read only its meta.json (no
    weights loaded, so this stays instant and GPU-free) and report the same static block as config
    mode, reconstructed from the checkpoint's own saved model_config (not from --arch/--depth),
    plus what training actually produced."""
    checkpoint_dir = os.path.join(get_base_dir(), "base_checkpoints", tag)
    step = find_last_step(checkpoint_dir)
    meta_path = os.path.join(checkpoint_dir, f"meta_{step:06d}.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json_module.load(f)

    config = config_from_dict(meta["model_config"])
    model_cls = get_model_class(config.arch)
    with torch.device("meta"):
        model = model_cls(config)
    depth = meta.get("user_config", {}).get("depth")
    row = inspect_model(model, config.arch, depth, args)

    total_batch_size = meta.get("total_batch_size")
    ckpt_fingerprint = meta.get("tokenizer_fingerprint")
    if ckpt_fingerprint is None or local_fingerprint is None:
        fingerprint_status = "unknown"
    elif ckpt_fingerprint == local_fingerprint:
        fingerprint_status = "match"
    else:
        fingerprint_status = "MISMATCH"

    row["trained"] = {
        "tag": tag, "step": step,
        "tokens_trained": step * total_batch_size if total_batch_size is not None else None,
        "val_bpb": meta.get("val_bpb"),
        "core_metric": meta.get("core_metric"),
        "train_time_sec": meta.get("loop_state", {}).get("total_training_time"),
        "tokenizer_fingerprint_status": fingerprint_status,
    }
    return row


def print_human(row):
    shape, params, flops, kv = row["shape"], row["params"], row["flops"], row["kv_cache"]
    label = row["trained"]["tag"] if "trained" in row else f"{row['arch']} d{row['depth']}"
    print(f"\n{'='*80}\n{label}\n{'='*80}")
    print(f"  n_layer={shape['n_layer']} n_embd={shape['n_embd']} n_head={shape['n_head']} n_kv_head={shape['n_kv_head']} "
          f"sequence_len={shape['sequence_len']} window_pattern={shape['window_pattern']!r} num_kv_slots={shape['num_kv_slots']}")
    print("  Params by role:")
    for key, value in params.items():
        print(f"    {key:16s}: {value:,}")
    print(f"  FLOPs/token: {flops['per_token']:.3e}  |  prefill@seqlen: {flops['prefill_at_seqlen']:.3e}  |  decode@seqlen: {flops['decode_at_seqlen']:.3e}")
    print(f"  KV cache: {kv['bytes_per_token']:,} bytes/token  |  {kv['total_mb_at_seqlen']:.2f} MB @ seqlen x batch={kv['kv_batch_size']}")
    if "training_plan" in row:
        plan = row["training_plan"]
        print(f"  Training plan: {plan['total_tokens']:,} tokens ({plan['horizon_source']}) over {plan['num_iterations']:,} iters "
              f"@ batch={plan['total_batch_size']:,}{' (auto)' if plan['auto_batch_size'] else ''}  |  {plan['total_flops']:.3e} FLOPs")
        if plan["gpu_hours"] is not None:
            print(f"  Estimated GPU-hours: {plan['gpu_hours']:.2f}")
    if "trained" in row:
        t = row["trained"]
        tokens_str = f"{t['tokens_trained']:,}" if t["tokens_trained"] is not None else "?"
        bpb_str = f"{t['val_bpb']:.6f}" if t["val_bpb"] is not None else "?"
        core_str = f"{t['core_metric']:.4f}" if t["core_metric"] is not None else "not evaluated"
        time_str = f"{t['train_time_sec']/60:.2f}m" if t["train_time_sec"] is not None else "?"
        print(f"  Trained: step {t['step']:,}  |  {tokens_str} tokens  |  val bpb: {bpb_str}  |  CORE: {core_str}  |  wall time: {time_str}")
        print(f"  Tokenizer fingerprint: {t['tokenizer_fingerprint_status']}")


def main():
    parser = argparse.ArgumentParser(description="Inspect model configs without training them")
    parser.add_argument("--arch", type=str, default="gpt", help="comma-separated architecture names")
    parser.add_argument("--depth", type=str, default="20", help="comma-separated depths")
    parser.add_argument("--aspect-ratio", type=int, default=64)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--window-pattern", type=str, default=None, help="default: each architecture's own from_depth default")
    parser.add_argument("--arch-opt", action="append", default=None, metavar="KEY=VALUE", help="repeatable; same as scripts/base_train.py's --arch-opt")
    parser.add_argument("--vocab-size", type=int, default=None, help="default: the local tokenizer's, if trained, else 32768")
    # Training horizon: same flags/defaults as scripts/base_train.py, so the two agree exactly given the same arguments
    parser.add_argument("--target-param-data-ratio", type=float, default=12)
    parser.add_argument("--target-flops", type=float, default=-1.0)
    parser.add_argument("--num-iterations", type=int, default=-1)
    parser.add_argument("--total-batch-size", type=int, default=-1)
    parser.add_argument("--weight-decay", type=float, default=0.28)
    # Cloud budgeting
    parser.add_argument("--gpu", type=str, default=None, help="GPU name for a GPU-hours estimate, e.g. 'NVIDIA A100' (see nanochat.common.get_peak_flops)")
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--mfu", type=float, default=0.4, help="assumed model FLOPs utilization for the GPU-hours estimate")
    parser.add_argument("--kv-batch-size", type=int, default=1, help="batch size for the reported total KV-cache MB")
    parser.add_argument("--checkpoints", type=str, nargs="?", const="", default=None,
                         help="inspect already-trained checkpoints instead of hypothetical configs: comma-separated "
                              "model tags, or bare --checkpoints for every tag under base_checkpoints/ (--arch/--depth "
                              "and the training-horizon flags are ignored in this mode -- shape comes from the "
                              "checkpoint's own saved config)")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    # Model construction (via print0) prints diagnostics like Llama/GPT's "Padding vocab_size..."
    # or LlamaKVShare's "KV sharing: ..." -- fine for human output, but would pollute --json's
    # stdout, so swallow them there.
    sink = io.StringIO() if args.json else None

    if args.checkpoints is not None:
        tags = list_checkpoint_tags(args.checkpoints)
        if not tags:
            raise SystemExit("No checkpoints found (base_checkpoints/ is empty or --checkpoints named none that exist).")
        local_fingerprint = local_tokenizer_fingerprint()
        with contextlib.redirect_stdout(sink) if sink is not None else contextlib.nullcontext():
            rows = [inspect_checkpoint(tag, args, local_fingerprint) for tag in tags]
    else:
        archs = [a.strip() for a in args.arch.split(",")]
        depths = [int(d.strip()) for d in args.depth.split(",")]
        if args.vocab_size is not None:
            vocab_size, vocab_source = args.vocab_size, "--vocab-size"
        else:
            vocab_size, vocab_source = default_vocab_size()
        if not args.json:
            print(f"Vocab size: {vocab_size:,} ({vocab_source})")
        with contextlib.redirect_stdout(sink) if sink is not None else contextlib.nullcontext():
            rows = [inspect_one(arch, depth, args, vocab_size) for arch in archs for depth in depths]

    if args.json:
        print(json_module.dumps(rows, indent=2))
    else:
        for row in rows:
            print_human(row)


if __name__ == "__main__":
    main()
