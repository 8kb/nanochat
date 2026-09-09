"""
Evaluate the Chat model.
All the generic eval-loop code lives in benchcore now (ARC/MMLU/GSM8K/HumanEval and the
categorical/generative loops); this script is a thin CLI wiring a loaded checkpoint to it.

Example runs:
python -m scripts.chat_eval -i sft -a ARC-Easy
torchrun --nproc_per_node=8 -m scripts.chat_eval -- -i sft -a ARC-Easy
"""

import argparse

from benchcore import ARC, GSM8K, MMLU, ALL_CHAT_TASKS, HumanEval, BenchManager, chatcore_metric

from nanochat.common import compute_init, compute_cleanup, get_base_dir, print0, autodetect_device_type
from nanochat.checkpoint_manager import load_model
from nanochat.engine import Engine

# -----------------------------------------------------------------------------
if __name__ == "__main__":

    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--source', type=str, required=True, help="Source of the model: sft|rl")
    parser.add_argument('-a', '--task-name', type=str, default=None, help="Task name. Default = all tasks. Use | to split multiple tasks.")
    parser.add_argument('-t', '--temperature', type=float, default=0.0)
    parser.add_argument('-m', '--max-new-tokens', type=int, default=512)
    parser.add_argument('-n', '--num-samples', type=int, default=1)
    parser.add_argument('-k', '--top-k', type=int, default=50)
    parser.add_argument('-b', '--batch-size', type=int, default=8, help='Batch size for categorical evaluation')
    parser.add_argument('-g', '--model-tag', type=str, default=None, help='Model tag to load')
    parser.add_argument('-s', '--step', type=int, default=None, help='Step to load')
    parser.add_argument('-x', '--max-problems', type=int, default=None, help='Max problems to evaluate')
    parser.add_argument('--device-type', type=str, default='', choices=['cuda', 'cpu', 'mps'], help='Device type for evaluation: cuda|cpu|mps. empty => autodetect')
    args = parser.parse_args()

    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)

    model, tokenizer, meta = load_model(args.source, device, phase="eval", model_tag=args.model_tag, step=args.step)
    engine = Engine(model, tokenizer)

    cache_dir = get_base_dir()
    task_builders = {
        'ARC-Easy': lambda: ARC(subset="ARC-Easy", split="test", cache_dir=cache_dir),
        'ARC-Challenge': lambda: ARC(subset="ARC-Challenge", split="test", cache_dir=cache_dir),
        'MMLU': lambda: MMLU(subset="all", split="test", cache_dir=cache_dir),
        'GSM8K': lambda: GSM8K(subset="main", split="test", cache_dir=cache_dir),
        'HumanEval': lambda: HumanEval(cache_dir=cache_dir),
    }
    task_names = list(ALL_CHAT_TASKS) if args.task_name is None else args.task_name.split('|')

    # Run all the task evaluations sequentially
    manager = BenchManager()
    results = {}
    for task_name in task_names:
        task = task_builders[task_name]()
        acc = manager.chat(
            task, model, tokenizer, generator=engine,
            batch_size=args.batch_size,
            num_samples=args.num_samples,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            max_problems=args.max_problems,
            device=device, rank=ddp_rank, world_size=ddp_world_size,
        )
        results[task_name] = acc
        print0(f"{task_name} accuracy: {100 * acc:.2f}%")

    # calculate the ChatCORE metric if we can (similar to CORE, it's the mean centered accuracy)
    # this way, ChatCORE ranges from 0 (at random baseline) to 1 (peak performance)
    if set(ALL_CHAT_TASKS) <= set(results):
        print0(f"ChatCORE metric: {chatcore_metric(results):.4f}")

    compute_cleanup()
