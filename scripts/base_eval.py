"""
Unified evaluation script for base models.

Supports three evaluation modes (comma-separated):
  --eval core    : CORE metric (accuracy on ICL tasks)
  --eval bpb     : Bits per byte on train/val splits
  --eval sample  : Generate samples from the model

Default is all three: --eval core,bpb,sample

Examples:

    # Evaluate a nanochat model (e.g. d24) using 8 GPUs
    torchrun --nproc_per_node=8 -m scripts.base_eval --model-tag d24 --device-batch-size=16

    # Quick/approximate evaluation using a single GPU
    python -m scripts.base_eval --model-tag d24 --device-batch-size=16 --max-per-task=100 --split-tokens=524288
"""
import os
import time
import argparse
import torch

from datacore import DataManager, FileSystemDatasetStore
from modelcore import ModelManager
from benchcore import BenchManager, center, load_core_suite

from nanochat.common import compute_init, compute_cleanup, print0, get_base_dir, autodetect_device_type
from nanochat.checkpoint_manager import load_model
from nanochat.engine import Engine
from scripts.data_prep import default_dataset_name, prepared_dir

# -----------------------------------------------------------------------------
# evaluate_core: base_train.py's periodic in-training CORE check imports this directly (a
# thin wrapper around BenchManager.core -- main()'s own 'core' eval mode below duplicates the
# per-task loop instead, to add CLI-only timing prints and the CSV write).

def evaluate_core(model, tokenizer, device, max_per_task=-1, rank=0, world_size=1):
    """Evaluate a base model on the CORE benchmark. Returns a dict with results,
    centered_results, and core_metric -- the same shape base_train.py has always expected."""
    suite = load_core_suite(get_base_dir(), max_per_task=max_per_task)
    bench_manager = BenchManager()
    report = bench_manager.core(model, tokenizer, suite, device=device, rank=rank, world_size=world_size)
    return {"results": report.results, "centered_results": report.centered_results, "core_metric": report.core_metric}

# -----------------------------------------------------------------------------
# Main

def main():
    parser = argparse.ArgumentParser(description="Base model evaluation")
    parser.add_argument('--eval', type=str, default='core,bpb,sample', help='Comma-separated evaluations to run: core,bpb,sample (default: all)')
    parser.add_argument('--arch', type=str, default='gpt', help='preset name to filter auto-discovery by when --model-tag is not given (see nanochat.architectures.presets); ignored if --model-tag is set')
    parser.add_argument('--model-tag', type=str, default=None, help='nanochat model tag to identify the checkpoint directory')
    parser.add_argument('--step', type=int, default=None, help='Model step to load (default = last)')
    parser.add_argument('--max-per-task', type=int, default=-1, help='Max examples per CORE task (-1 = all)')
    parser.add_argument('--device-batch-size', type=int, default=32, help='Per-device batch size for BPB evaluation')
    parser.add_argument('--split-tokens', type=int, default=40*524288, help='Number of tokens to evaluate per split for BPB')
    parser.add_argument('--device-type', type=str, default='', help='cuda|cpu|mps (empty = autodetect)')
    parser.add_argument('--dataset', type=str, default=None, help='prepared dataset name for --eval bpb (default: derived from the checkpoint\'s sequence_len and the tokenizer fingerprint)')
    args = parser.parse_args()

    # Parse evaluation modes
    eval_modes = set(mode.strip() for mode in args.eval.split(','))
    valid_modes = {'core', 'bpb', 'sample'}
    invalid = eval_modes - valid_modes
    if invalid:
        parser.error(f"Invalid eval modes: {invalid}. Valid: {valid_modes}")

    # Distributed / precision setup
    device_type = autodetect_device_type() if args.device_type == '' else args.device_type
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    # Load model and tokenizer
    model, tokenizer, meta = load_model("base", device, phase="eval", model_tag=args.model_tag, step=args.step, arch=args.arch)
    sequence_len = meta["model_config"]["sequence_len"]
    manager = ModelManager()
    model_name = f"{meta['model_tag']} (step {meta['step']})"
    model_slug = f"{meta['model_tag']}_{meta['step']:06d}" # includes the tag so two architectures evaluated in the same run don't overwrite each other's CSV

    print0(f"Evaluating model: {model_name}")
    print0(f"Eval modes: {', '.join(sorted(eval_modes))}")

    # Results to log
    core_report = None
    bpb_results = {}
    samples = []
    unconditioned_samples = []

    # --- Sampling ---
    if 'sample' in eval_modes:
        print0("\n" + "="*80)
        print0("Model Samples")
        print0("="*80)
        if ddp_rank == 0:
            prompts = [
                "The capital of France is",
                "The chemical symbol of gold is",
                "If yesterday was Friday, then tomorrow will be",
                "The opposite of hot is",
                "The planets of the solar system are:",
                "My favorite color is",
                "If 5*x + 3 = 13, then x is",
            ]
            engine = Engine(model, tokenizer)
            print0("\nConditioned samples:")
            for prompt in prompts:
                tokens = tokenizer(prompt, prepend="<|bos|>")
                sample, _ = engine.generate_batch(tokens, num_samples=1, max_tokens=16, temperature=0)
                sample_str = tokenizer.decode(sample[0])
                print0("-" * 80)
                print0(sample_str)
                samples.append(sample_str)

            print0("\nUnconditioned samples:")
            tokens = tokenizer("", prepend="<|bos|>")
            uncond, _ = engine.generate_batch(tokens, num_samples=8, max_tokens=128, temperature=1.0)
            for sample in uncond:
                sample_str = tokenizer.decode(sample)
                print0("-" * 80)
                print0(sample_str)
                unconditioned_samples.append(sample_str)

    # --- BPB evaluation ---
    if 'bpb' in eval_modes:
        print0("\n" + "="*80)
        print0("BPB Evaluation")
        print0("="*80)
        tokens_per_step = args.device_batch_size * sequence_len * ddp_world_size
        if args.split_tokens % tokens_per_step != 0:
            # Adjust to nearest multiple
            args.split_tokens = (args.split_tokens // tokens_per_step) * tokens_per_step
            print0(f"Adjusted split_tokens to {args.split_tokens} (must be divisible by {tokens_per_step})")
        steps = args.split_tokens // tokens_per_step

        dataset_name = args.dataset or default_dataset_name("base", sequence_len, tokenizer)
        data_store = FileSystemDatasetStore(prepared_dir(dataset_name))
        data_manager = DataManager()
        try:
            dataset = data_manager.open(data_store)
        except FileNotFoundError:
            raise SystemExit(
                f"No prepared dataset found for {dataset_name!r}. Run:\n"
                f"  python -m scripts.data_prep --kind=base --dataset={dataset_name} --sequence-len={sequence_len}"
            )
        if dataset.info.sequence_len != sequence_len:
            raise SystemExit(
                f"Dataset {dataset_name!r} has sequence_len={dataset.info.sequence_len}, but this "
                f"checkpoint's model was trained at sequence_len={sequence_len}. Re-prepare it:\n"
                f"  python -m scripts.data_prep --kind=base --dataset={dataset_name} --sequence-len={sequence_len}"
            )
        token_bytes = data_manager.token_bytes(dataset)

        for split_name in ["train", "val"]:
            loader = data_manager.batches(dataset, split_name, args.device_batch_size, device=device,
                                          rank=ddp_rank, world_size=ddp_world_size, infinite=True)
            bpb = manager.evaluate_bpb(model, loader, steps, token_bytes)
            bpb_results[split_name] = bpb
            print0(f"{split_name} bpb: {bpb:.6f}")

    # --- CORE evaluation ---
    if 'core' in eval_modes:
        print0("\n" + "="*80)
        print0("CORE Evaluation")
        print0("="*80)
        suite = load_core_suite(get_base_dir(), max_per_task=args.max_per_task)
        bench_manager = BenchManager()
        results, centered_results = {}, {}
        for task in suite.tasks:
            start_time = time.time()
            print0(f"Evaluating: {task.label} ({task.num_fewshot}-shot, type: {task.task_type})... ", end='')
            accuracy = bench_manager.core_task(model, tokenizer, task.data, task.task_meta, device=device, rank=ddp_rank, world_size=ddp_world_size)
            results[task.label] = accuracy
            centered_results[task.label] = center(accuracy, suite.random_baselines[task.label])
            elapsed = time.time() - start_time
            print0(f"accuracy: {accuracy:.4f} | centered: {centered_results[task.label]:.4f} | time: {elapsed:.2f}s")
        core_metric = sum(centered_results.values()) / len(centered_results)
        core_report = {"results": results, "centered_results": centered_results, "core_metric": core_metric}

        # Write CSV output
        if ddp_rank == 0:
            base_dir = get_base_dir()
            output_csv_path = os.path.join(base_dir, "base_eval", f"{model_slug}.csv")
            os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
            with open(output_csv_path, 'w', encoding='utf-8', newline='') as f:
                f.write(f"{'Task':<35}, {'Accuracy':<10}, {'Centered':<10}\n")
                for label in core_report["results"]:
                    acc = core_report["results"][label]
                    centered = core_report["centered_results"][label]
                    f.write(f"{label:<35}, {acc:<10.6f}, {centered:<10.6f}\n")
                f.write(f"{'CORE':<35}, {'':<10}, {core_report['core_metric']:<10.6f}\n")
            print0(f"\nResults written to: {output_csv_path}")
            print0(f"CORE metric: {core_report['core_metric']:.4f}")

    compute_cleanup()


if __name__ == "__main__":
    main()
