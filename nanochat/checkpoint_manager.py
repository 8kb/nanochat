"""
Utilities for saving and loading model/optim/state checkpoints.

This is naming policy only -- which directory, which step, which tag, plus meta.json's extra
fields (val_bpb, user_config, tokenizer_fingerprint, dataloader_state, loop_state, ...). The
actual model/optimizer artifact format belongs to modelcore (see modelcore.manager.ModelManager);
this module hands it a modelcore.store.FileSystemStore over the right directory+step, routing an
old (pre-modelcore) checkpoint through nanochat.architectures.legacy first (see
LegacyCheckpointStore below).
"""
import os
import re
import json
import logging
import torch

from modelcore import ModelManager
from modelcore.store import FileSystemStore

from nanochat.architectures import legacy
from nanochat.common import get_base_dir
from nanochat.tokenizer import get_tokenizer
from nanochat.common import setup_default_logging

# Set up logging
setup_default_logging()
logger = logging.getLogger(__name__)
def log0(message):
    if int(os.environ.get('RANK', 0)) == 0:
        logger.info(message)

_manager = ModelManager()

def save_checkpoint(checkpoint_dir, step, model_data, optimizer_data, meta_data, rank=0):
    store = FileSystemStore(checkpoint_dir, step)
    if rank == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)
        # Save the model state parameters
        store.write_model_state(model_data)
        logger.info(f"Saved model parameters to: {os.path.join(checkpoint_dir, f'model_{step:06d}.pt')}")
        # meta_data bundles modelcore's "model_config" key alongside nanochat's own sibling keys
        # (val_bpb, user_config, dataloader_state, ...) in one dict -- write nanochat's keys
        # directly (naming/training-loop policy, which this module owns), then let write_config
        # own the model_config key specifically, so a save always goes through the same code
        # ModelManager.save_model does.
        meta_path = os.path.join(checkpoint_dir, f"meta_{step:06d}.json")
        existing = {}
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
        existing.update({k: v for k, v in meta_data.items() if k != "model_config"})
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2)
        if "model_config" in meta_data:
            store.write_config(meta_data["model_config"])
        logger.info(f"Saved metadata to: {meta_path}")
    # Note that optimizer state is sharded across ranks, so each rank must save its own.
    if optimizer_data is not None:
        os.makedirs(checkpoint_dir, exist_ok=True)
        store.write_optimizer_state(optimizer_data, rank=rank)
        logger.info(f"Saved optimizer state to: {os.path.join(checkpoint_dir, f'optim_{step:06d}_rank{rank}.pt')}")

def load_checkpoint(checkpoint_dir, step, device, load_optimizer=False, rank=0):
    store = FileSystemStore(checkpoint_dir, step)
    # Load the model state
    model_data = store.read_model_state(map_location=device)
    # Load the optimizer state if requested
    optimizer_data = None
    if load_optimizer:
        optimizer_data = store.read_optimizer_state(rank=rank, map_location=device)
        if optimizer_data is None:
            raise FileNotFoundError(
                f"No optimizer state found for step {step} in {checkpoint_dir} (rank {rank})"
            )
    # Load the metadata
    meta_path = os.path.join(checkpoint_dir, f"meta_{step:06d}.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta_data = json.load(f)
    return model_data, optimizer_data, meta_data


class LegacyCheckpointStore(FileSystemStore):
    """A FileSystemStore that migrates an old (pre-modelcore) checkpoint's config and state dict
    through nanochat.architectures.legacy before handing them to ModelManager -- so ModelManager
    (and a fresh, post-modelcore checkpoint) never has to know legacy formats exist. Migrates once,
    on first access to either read method, and memoizes: migrate_checkpoint needs the raw config to
    migrate the state dict, so the two must happen together.

    Note: the *raw* config dict in meta.json is left completely untouched on disk -- build_model
    reads it separately (via load_checkpoint) for its own purposes, because arch_of()/tag naming
    needs to keep seeing a legacy checkpoint's original "arch" key, not the migrated tree's (always
    unset) `reference` field."""

    def __init__(self, checkpoint_dir, step, *, device, log=lambda msg: None):
        super().__init__(checkpoint_dir, step)
        self._device = device
        self._log = log
        self._migrated = None  # (ModelConfig, state dict), computed once

    def _migrate(self):
        if self._migrated is None:
            raw_config = super().read_config()
            raw_state = super().read_model_state(map_location=self._device)
            if self._device.type in {"cpu", "mps"}:
                # Convert bfloat16 tensors to float for CPU/MPS inference
                raw_state = {k: v.float() if v.dtype == torch.bfloat16 else v for k, v in raw_state.items()}
            # Hack: fix torch compile issue, which prepends all keys with _orig_mod.
            raw_state = {k.removeprefix("_orig_mod."): v for k, v in raw_state.items()}
            self._migrated = legacy.migrate_checkpoint(dict(raw_config), raw_state, log=self._log)
        return self._migrated

    def read_config(self):
        config, _ = self._migrate()
        return config.to_dict()

    def read_model_state(self, map_location=None):
        _, state = self._migrate()
        return state


def build_model(checkpoint_dir, step, device, phase, config_override=None):
    """
    A bunch of repetitive code to build a model from a given checkpoint.

    config_override, when given, replaces the checkpoint's own stored model_config for the build
    -- e.g. a modelcore.ModelConfig with a hand-edited `adapters`/`frozen` list, or a fresh one
    built by nanochat.architectures.adapters.expand_adapters. This is the caller-facing lever for
    "add/disable an adapter on an existing checkpoint": ModelManager.load_model does the actual
    reconciling load (see its docstring) -- this function only has to pass the override through.

    Returns:
    - base model - uncompiled, not wrapped in DDP
    - tokenizer
    - meta data saved during base model training
    """
    assert phase in ["train", "eval"], f"Invalid phase: {phase}"
    store = LegacyCheckpointStore(checkpoint_dir, step, device=device, log=log0)
    model = _manager.load_model(store, device=device, train=(phase == "train"), config=config_override)
    log0(f"Building model with config: {_manager.config_to_dict(model.config)}")
    # Load the metadata (kept as a separate, untouched read -- see LegacyCheckpointStore's
    # docstring for why this must stay the checkpoint's raw model_config, not the migrated one)
    meta_path = os.path.join(checkpoint_dir, f"meta_{step:06d}.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta_data = json.load(f)
    # Load the Tokenizer
    tokenizer = get_tokenizer()
    # Sanity check: compatibility between model and tokenizer
    assert tokenizer.get_vocab_size() == model.config.vocab_size, f"Tokenizer vocab size {tokenizer.get_vocab_size()} does not match model config vocab size {model.config.vocab_size}"
    # Same vocab_size doesn't mean same vocab (e.g. a checkpoint trained on a different machine's
    # tokenizer): checkpoints saved before this fingerprint existed have no key to check, so this
    # only warns, and only when there's something to compare.
    checkpoint_fingerprint = meta_data.get("tokenizer_fingerprint")
    if checkpoint_fingerprint is not None and checkpoint_fingerprint != tokenizer.fingerprint():
        log0(f"WARNING: tokenizer fingerprint mismatch -- this checkpoint was trained with a "
             f"different tokenizer than the one loaded here ({checkpoint_fingerprint} != "
             f"{tokenizer.fingerprint()}). Vocab size matches, but token ids may mean different "
             f"things; expect garbage output.")
    return model, tokenizer, meta_data


def find_largest_model(checkpoints_dir, arch=None):
    # attempt to guess the model tag: take the biggest model available
    model_tags = [f for f in os.listdir(checkpoints_dir) if os.path.isdir(os.path.join(checkpoints_dir, f))]
    if not model_tags:
        raise FileNotFoundError(f"No checkpoints found in {checkpoints_dir}")
    if arch is not None:
        # Filter to tags whose latest saved checkpoint is actually this architecture, so two
        # architectures trained at the same --depth (e.g. gpt's "d12" and llama's "llama_d12")
        # don't get confused for each other by callers that don't pass an explicit model_tag.
        model_tags = [t for t in model_tags if _checkpoint_arch(checkpoints_dir, t) == arch]
        if not model_tags:
            raise FileNotFoundError(f"No {arch!r} checkpoints found in {checkpoints_dir}")
    # 1) normally all model tags are of the form d<number>, try that first:
    candidates = []
    for model_tag in model_tags:
        match = re.match(r"d(\d+)", model_tag)
        if match:
            model_depth = int(match.group(1))
            candidates.append((model_depth, model_tag))
    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]
    # 2) if that failed, take the most recently updated model:
    model_tags.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoints_dir, x)), reverse=True)
    return model_tags[0]


def find_last_step(checkpoint_dir):
    # Look into checkpoint_dir and find model_<step>.pt with the highest step
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if re.search(r'model_(\d+)\.pt$', f)]
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    last_step = max(int(f.split("_")[-1].split(".")[0]) for f in checkpoint_files)
    return last_step


def arch_of(model_config: dict) -> str:
    """Best-effort preset/architecture name for a raw model_config dict (as read from a
    checkpoint's meta.json), for tag naming and auto-discovery filtering. A current-format
    (post-modelcore) model_config has no "arch" key at all -- its `reference.preset` (stamped by
    nanochat.architectures.presets) is the closest equivalent, defaulting to "custom" for a
    hand-written tree with no reference block. A legacy model_config (no "format" key) still
    carries the old "arch" key directly, defaulting to "gpt" for checkpoints predating even that."""
    if "format" in model_config:
        return (model_config.get("reference") or {}).get("preset", "custom")
    return model_config.get("arch", "gpt")


def _checkpoint_arch(checkpoints_dir, model_tag):
    """Best-effort: arch_of() applied to a checkpoint tag's latest saved step, read straight off
    disk. Returns None if the tag has no valid checkpoint at all (an empty or malformed
    directory), so it never matches a real arch filter."""
    checkpoint_dir = os.path.join(checkpoints_dir, model_tag)
    try:
        step = find_last_step(checkpoint_dir)
    except FileNotFoundError:
        return None
    meta_path = os.path.join(checkpoint_dir, f"meta_{step:06d}.json")
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    return arch_of(meta.get("model_config", {}))

# -----------------------------------------------------------------------------
# convenience functions that take into account nanochat's directory structure

def load_model_from_dir(checkpoints_dir, device, phase, model_tag=None, step=None, arch=None, config_override=None):
    if model_tag is None:
        # guess the model tag by defaulting to the largest model (of the given arch, if any)
        model_tag = find_largest_model(checkpoints_dir, arch=arch)
        log0(f"No model tag provided, guessing model tag: {model_tag}")
    checkpoint_dir = os.path.join(checkpoints_dir, model_tag)
    if step is None:
        # guess the step by defaulting to the last step
        step = find_last_step(checkpoint_dir)
    assert step is not None, f"No checkpoints found in {checkpoint_dir}"
    # build the model
    log0(f"Loading model from {checkpoint_dir} with step {step}")
    model, tokenizer, meta_data = build_model(checkpoint_dir, step, device, phase, config_override=config_override)
    meta_data["model_tag"] = model_tag # so a caller with no explicit --model-tag still knows which checkpoint was picked
    return model, tokenizer, meta_data

def load_model(source, *args, arch=None, config_override=None, **kwargs):
    model_dir = {
        "base": "base_checkpoints",
        "sft": "chatsft_checkpoints",
        "rl": "chatrl_checkpoints",
    }[source]
    base_dir = get_base_dir()
    checkpoints_dir = os.path.join(base_dir, model_dir)
    return load_model_from_dir(checkpoints_dir, *args, arch=arch, config_override=config_override, **kwargs)

def load_optimizer_state(source, device, rank, model_tag=None, step=None, arch=None):
    """Load just the optimizer shard for a given rank, without re-loading the model."""
    model_dir = {
        "base": "base_checkpoints",
        "sft": "chatsft_checkpoints",
        "rl": "chatrl_checkpoints",
    }[source]
    base_dir = get_base_dir()
    checkpoints_dir = os.path.join(base_dir, model_dir)
    if model_tag is None:
        model_tag = find_largest_model(checkpoints_dir, arch=arch)
    checkpoint_dir = os.path.join(checkpoints_dir, model_tag)
    if step is None:
        step = find_last_step(checkpoint_dir)
    store = FileSystemStore(checkpoint_dir, step)
    optimizer_data = store.read_optimizer_state(rank=rank, map_location=device)
    if optimizer_data is None:
        log0(f"Optimizer checkpoint not found: optim_{step:06d}_rank{rank}.pt in {checkpoint_dir}")
        return None
    log0(f"Loading optimizer state from {checkpoint_dir} (step {step}, rank {rank})")
    return optimizer_data
