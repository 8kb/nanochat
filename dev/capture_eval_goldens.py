"""
Captures byte-exact reference output of today's (pre-benchcore-extraction) jinja2-based CORE
prompt rendering (nanochat/core_eval.py) and chat-task conversation rendering (tasks/*.py), before
either is touched. Frozen after this commit, like dev/capture_data_goldens.py: a point-in-time
snapshot proving the benchcore move (in particular, dropping the jinja2 dependency in favor of
plain Python string building) is behavior-preserving, not meant to track future refactors.

Requires the real cached eval bundle (~/.cache/nanochat/eval_bundle/) and task_data
(~/.cache/nanochat/task_data/{allenai--ai2_arc,cais--mmlu,openai--gsm8k,openai--openai_humaneval})
and tokenizer -- all already present on this machine, no network needed.

Writes tests/goldens/eval_core_prompts.json (rendered prompts + token ids/start/end indices for
one multiple_choice, one schema, and one language_modeling task, 5 examples each, num_fewshot as
configured in eval_bundle/core.yaml) and tests/goldens/eval_render_for_completion.json (20
render_for_completion() token-id outputs across ARC-Easy/MMLU/GSM8K/HumanEval).

Like dev/capture_model_goldens.py and dev/capture_data_goldens.py, this can no longer actually run
once the benchcore move lands (nanochat.core_eval and tasks/ are deleted) -- it's kept as a record
of how tests/goldens/eval_core_prompts.json and eval_render_for_completion.json were produced, not
a live tool. benchcore/tests/test_prompts.py cross-checks its own plain-Python rendering against
these same goldens' values (hand-crafted fixtures there, verified against these at extraction time).

python -m dev.capture_eval_goldens          # (re)writes the goldens
python -m dev.capture_eval_goldens --check  # recomputes and asserts identical to what's on disk
"""
import argparse
import csv
import json
import os
import random

import yaml

from nanochat.core_eval import (
    batch_sequences_lm,
    batch_sequences_mc,
    batch_sequences_schema,
    render_prompts_lm,
    render_prompts_mc,
    render_prompts_schema,
)
from nanochat.common import get_base_dir
from nanochat.tokenizer import get_tokenizer

GOLDENS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests", "goldens")

# One task per icl_task_type in eval_bundle/core.yaml, chosen for being small/fast to load.
CORE_TASK_LABELS = {
    "multiple_choice": "hellaswag_zeroshot",
    "schema": "winograd",
    "language_modeling": "lambada_openai",
}
NUM_EXAMPLES = 5


def _load_core_task_meta(label):
    base_dir = get_base_dir()
    eval_bundle_dir = os.path.join(base_dir, "eval_bundle")
    config_path = os.path.join(eval_bundle_dir, "core.yaml")
    data_base_path = os.path.join(eval_bundle_dir, "eval_data")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    task = next(t for t in config["icl_tasks"] if t["label"] == label)
    task_meta = {
        "task_type": task["icl_task_type"],
        "dataset_uri": task["dataset_uri"],
        "num_fewshot": task["num_fewshot"][0],
        "continuation_delimiter": task.get("continuation_delimiter", " "),
    }
    data_path = os.path.join(data_base_path, task_meta["dataset_uri"])
    with open(data_path, "r", encoding="utf-8") as f:
        data = [json.loads(line.strip()) for line in f]
    shuffle_rng = random.Random(1337)
    shuffle_rng.shuffle(data)
    return task_meta, data[:NUM_EXAMPLES]


def capture_core_prompts(tokenizer):
    out = {}
    for task_type, label in CORE_TASK_LABELS.items():
        task_meta, data = _load_core_task_meta(label)
        num_fewshot = task_meta["num_fewshot"]
        cd = task_meta["continuation_delimiter"]
        examples = []
        for idx in range(len(data)):
            rng = random.Random(1234 + idx)
            available = [i for i in range(len(data)) if i != idx]
            fewshot_idxs = rng.sample(available, min(num_fewshot, len(available)))
            fewshot = [data[i] for i in fewshot_idxs]
            item = data[idx]
            if task_type == "multiple_choice":
                prompts = render_prompts_mc(item, cd, fewshot)
                tokens, start_idxs, end_idxs = batch_sequences_mc(tokenizer, prompts)
            elif task_type == "schema":
                prompts = render_prompts_schema(item, cd, fewshot)
                tokens, start_idxs, end_idxs = batch_sequences_schema(tokenizer, prompts)
            else:
                prompts = render_prompts_lm(item, cd, fewshot)
                tokens, start_idxs, end_idxs = batch_sequences_lm(tokenizer, prompts)
            examples.append({
                "prompts": prompts,
                "tokens": tokens,
                "start_idxs": start_idxs,
                "end_idxs": end_idxs,
            })
        out[task_type] = {"label": label, "examples": examples}
    return out


def capture_render_for_completion(tokenizer):
    from tasks.arc import ARC
    from tasks.mmlu import MMLU
    from tasks.gsm8k import GSM8K
    from tasks.humaneval import HumanEval

    out = []
    task_specs = [
        ("ARC-Easy", ARC(subset="ARC-Easy", split="test")),
        ("MMLU", MMLU(subset="all", split="test")),
        ("GSM8K", GSM8K(subset="main", split="test")),
        ("HumanEval", HumanEval()),
    ]
    for name, task in task_specs:
        for i in range(5):
            conversation = task[i]
            ids = tokenizer.render_for_completion(conversation)
            out.append({"task": name, "index": i, "ids": ids})
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    tokenizer = get_tokenizer()
    core_prompts = capture_core_prompts(tokenizer)
    render_outputs = capture_render_for_completion(tokenizer)

    prompts_path = os.path.join(GOLDENS_DIR, "eval_core_prompts.json")
    render_path = os.path.join(GOLDENS_DIR, "eval_render_for_completion.json")

    if args.check:
        with open(prompts_path) as f:
            assert json.load(f) == core_prompts, "eval_core_prompts.json mismatch!"
        with open(render_path) as f:
            assert json.load(f) == render_outputs, "eval_render_for_completion.json mismatch!"
        print("OK: goldens match current code")
    else:
        os.makedirs(GOLDENS_DIR, exist_ok=True)
        with open(prompts_path, "w") as f:
            json.dump(core_prompts, f, indent=2)
        with open(render_path, "w") as f:
            json.dump(render_outputs, f, indent=2)
        print(f"Wrote {prompts_path}")
        print(f"Wrote {render_path}")


if __name__ == "__main__":
    main()
