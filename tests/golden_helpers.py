"""
Digest helpers shared by dev/capture_model_goldens.py (which wrote tests/goldens/*.json once,
before the Stage 7 modelcore/ extraction -- see docs/roadmap.md) and tests/test_goldens.py (which
replays them against current code). Split out at Stage 8 so these stay importable as a live
library independent of dev/capture_model_goldens.py's own capture_*/main functions, which can no
longer run at all (they import nanochat.model and tests.conftest.TINY_KWARGS_BY_ARCH, both deleted
along with nanochat/model/ at Stage 7 step 5) but are kept there as a record of how the goldens
were originally produced.
"""
import hashlib
import os

import torch

from nanochat.checkpoint_manager import load_checkpoint

GOLDENS_DIR = os.path.join(os.path.dirname(__file__), "goldens")
TINY_DIR = os.path.join(GOLDENS_DIR, "tiny")
DEVICE = torch.device("cpu")  # pinned: mps/cpu logits differ, goldens must not depend on which
                                # device autodetection happens to pick (this machine has mps).
PROMPT_TEXT = "The chemical formula of water is"
GEN_TOKENS = 32


class _FakeTokenizer:
    """Stub for the synthetic tiny sources, which don't share a vocabulary with any real BPE
    tokenizer. Special-token ids sit at/above vocab_size (matching tests/test_generate.py's
    pattern) so an untrained model's logits (always < vocab_size) can never produce one --
    generation is deterministic and independent of random init."""
    def __init__(self, vocab_size):
        self._vocab_size = vocab_size
        self._specials = {
            "<|python_start|>": vocab_size + 0,
            "<|python_end|>": vocab_size + 1,
            "<|output_start|>": vocab_size + 2,
            "<|output_end|>": vocab_size + 3,
            "<|assistant_end|>": vocab_size + 4,
        }
        self._bos = vocab_size + 5

    def get_vocab_size(self):
        return self._vocab_size

    def fingerprint(self):
        return "synthetic0000000"

    def encode_special(self, s):
        return self._specials[s]

    def get_bos_token_id(self):
        return self._bos

    def decode(self, ids):
        return "".join(str(i) for i in ids)


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def tensor_hash(t: torch.Tensor) -> str:
    return sha256_hex(t.detach().to("cpu").contiguous().numpy().tobytes())


def state_dict_fingerprint(sd: dict) -> dict:
    """Two views of a state dict: exact key -> [shape, dtype, hash] (for human debugging; key
    names are expected to change across the refactor, so not asserted verbatim), and a sorted
    multiset of [shape, dtype, hash] triples (the actual regression invariant -- robust to any
    key rename as long as no tensor's shape/dtype/values change)."""
    by_key = {}
    multiset = []
    for k, v in sd.items():
        shape = list(v.shape)
        dtype = str(v.dtype)
        h = tensor_hash(v)
        by_key[k] = [shape, dtype, h]
        multiset.append([shape, dtype, h])
    multiset.sort(key=lambda x: (x[0], x[1], x[2]))
    return {"by_key": by_key, "multiset": multiset, "num_tensors": len(sd),
            "num_params": sum(v.numel() for v in sd.values())}


def accounting(model) -> dict:
    """Reused by tests/test_goldens.py to recompute the same numbers post-refactor. Handles both
    a legacy nanochat.model.BaseModel-based model (the shape every number here was originally
    captured against) and a modelcore.Model (post Stage 7 step 5, once checkpoint_manager itself
    was rewired) -- see ModelManager.stats() for the latter. The one field whose *shape* legitimately
    differs between the two -- num_scaling_params was GPT's own legacy six-key dict
    (wte/value_embeds/lm_head/transformer_matrices/scalars/total), now modelcore's generic
    role-keyed dict for every architecture uniformly -- is why tests/test_goldens.py compares its
    *total* rather than the dict itself; every other field here is unaffected and compared exactly."""
    if hasattr(model, "num_scaling_params"):
        return {
            "num_scaling_params": model.num_scaling_params(),
            "num_matmul_params": model.num_matmul_params(),
            "estimate_flops": model.estimate_flops(),
            "estimate_decode_flops_256": model.estimate_decode_flops(256),
            "estimate_prefill_flops_256": model.estimate_prefill_flops(256),
            "kv_bytes_per_token": model.kv_bytes_per_token(),
            "kv_read_bytes_256": model.kv_read_bytes(256),
            "kv_cache_spec": model.kv_cache_spec(),
            "layer_specs": [
                {"n_head": s.n_head, "n_kv_head": s.n_kv_head, "head_dim": s.head_dim,
                 "window": s.window, "kv_slot": s.kv_slot}
                for s in model.layer_specs()
            ],
            "shape_summary": model.shape_summary(),
        }
    from modelcore import ModelManager
    stats = ModelManager().stats(model.config)
    return {
        "num_scaling_params": stats.params_by_role,
        "num_matmul_params": stats.num_matmul_params,
        "estimate_flops": stats.flops_per_token,
        "estimate_decode_flops_256": stats.decode_flops(256),
        "estimate_prefill_flops_256": stats.prefill_flops(256),
        "kv_bytes_per_token": stats.kv_bytes_per_token(),
        "kv_read_bytes_256": stats.kv_read_bytes(256),
        "kv_cache_spec": stats.kv_cache_spec,
        "layer_specs": [
            {"n_head": s.n_head, "n_kv_head": s.n_kv_head, "head_dim": s.head_dim,
             "window": s.window, "kv_slot": s.kv_slot}
            for s in stats.layer_specs
        ],
        "shape_summary": stats.shape_summary,
    }


def fixed_logits_hash(model, vocab_size: int, seq_len: int) -> str:
    T = min(8, seq_len)
    idx = (torch.arange(T) % vocab_size).long().unsqueeze(0)
    model.eval()
    with torch.no_grad():
        logits = model.forward(idx)
    return tensor_hash(logits.float())


def _jsonify(d):
    """Normalize hparam values to exactly what a JSON round-trip produces (tuples -> lists),
    so a value compared live (never serialized) matches one read back from a golden file."""
    out = {}
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.item() if v.numel() == 1 else v.tolist()
        elif isinstance(v, tuple):
            out[k] = list(v)
        else:
            out[k] = v
    return out


def _build_optimizer(model):
    """Dual-path, like accounting() above: a legacy model builds its own optimizer; a
    modelcore.Model's optimizer comes from ModelManager.create_optimizer() instead (Model itself
    carries no setup_optimizer method -- see modelcore/model.py)."""
    if hasattr(model, "setup_optimizer"):
        return model.setup_optimizer()
    from modelcore import ModelManager
    return ModelManager().create_optimizer(model)


def optimizer_layout(model) -> list:
    """Freshly-built optimizer groups (default hparams), independent of whether a saved shard
    exists -- captures role -> policy -> group order/hparams/shapes. NOTE: for a gpt-arch model,
    the post-refactor group *count* legitimately differs by one from a pre-refactor golden's
    recorded layout (backout_lambda moves from inside the "smear" role to its own "backout_scalar"
    role -- see nanochat.architectures.legacy._split_backout_lambda_from_smear); every other
    architecture's layout is unaffected. tests/test_goldens.py accounts for this."""
    optimizer = _build_optimizer(model)
    groups = []
    for g in optimizer.param_groups:
        hparams = {k: v for k, v in g.items() if k != "params"}
        shapes = sorted([list(p.shape) for p in g["params"]])
        groups.append({"hparams": _jsonify(hparams), "shapes": shapes, "num_params": len(g["params"])})
    return groups


def _find_optimizer_step(checkpoint_dir) -> int | None:
    steps = [int(f.split("_")[1]) for f in os.listdir(checkpoint_dir) if f.startswith("optim_")]
    return max(steps) if steps else None


def optimizer_digest_from_dir(model, checkpoint_dir, raw_config_dict=None) -> dict | None:
    """Load whatever optimizer shard is saved in `checkpoint_dir` (if any), migrate it for this
    model's config, and load it into a fresh optimizer -- records success plus a digest of every
    state tensor, so migration (positional flat-index splitting, role renaming) and plain
    save/load round-tripping are both provably unchanged after the refactor. `raw_config_dict`
    (the checkpoint's raw, pre-migration meta.json "model_config") selects the modelcore path via
    nanochat.architectures.legacy; omit it for a legacy nanochat.model.BaseModel-based model."""
    step = _find_optimizer_step(checkpoint_dir)
    if step is None:
        return None
    _, optimizer_data, _ = load_checkpoint(checkpoint_dir, step, DEVICE, load_optimizer=True)
    optimizer = _build_optimizer(model)
    if raw_config_dict is not None:
        from nanochat.architectures import legacy
        patched = legacy.migrate_optimizer_state_from_meta(optimizer_data, raw_config_dict, model.config.n_layer, log=lambda m: None)
    else:
        patched = type(model).patch_optimizer_state_dict(optimizer_data, model.config, log=lambda m: None)
    optimizer.load_state_dict(patched)
    state = optimizer.state_dict()
    state_digest = {
        str(flat_idx): {k: (tensor_hash(v) if isinstance(v, torch.Tensor) else v) for k, v in entry.items()}
        for flat_idx, entry in state["state"].items()
    }
    group_shapes = [[list(p.shape) for p in g["params"]] for g in optimizer.param_groups]
    return {"step": step, "num_groups": len(state["param_groups"]), "num_state_entries": len(state["state"]),
            "group_param_shapes": group_shapes, "state_digest": state_digest}
