"""
Capture golden artifacts describing today's `nanochat/model/` + `nanochat/checkpoint_manager.py`
behavior, BEFORE the Stage 7 `modelcore/` extraction (see the Stage 7 plan / docs/roadmap.md).

Run once, against unmodified code:
    python -m dev.capture_model_goldens

Writes:
    tests/goldens/<name>.json    -- small digests (committed): config, state-dict fingerprint,
                                     accounting numbers, greedy-generation token ids, a logits
                                     hash, and the optimizer layout + a full state-tensor digest.
    tests/goldens/tiny/<name>/   -- real checkpoint directories for small, synthetic, seeded
                                     models (model + optimizer + meta.json, a few hundred KB
                                     each, committed) -- written through save_checkpoint/
                                     load_model_from_dir exactly like a real training run, so the
                                     post-refactor replay test exercises the exact same
                                     checkpoint_manager façade for synthetic and real sources
                                     alike, and doesn't depend on init-RNG order staying stable
                                     (docs/architecture.md notes it does not, across Stage 2).

Nothing here survives the refactor -- it's a one-time snapshot of "what the code does right now",
so tests/test_goldens.py can prove the rewrite doesn't change any of it.
"""
import hashlib
import json
import os

import torch

from nanochat import checkpoint_manager
from nanochat.checkpoint_manager import load_checkpoint, load_model_from_dir, save_checkpoint
from nanochat.common import get_base_dir
from nanochat.engine import Engine, generate_naive
# NOTE: capture_synthetic() and main() below import nanochat.model / tests.conftest.TINY_KWARGS_BY_ARCH
# lazily, inside the functions that use them -- both were deleted along with nanochat/model/ at
# Stage 7 step 5, so this module can still be imported (for its utility functions, reused by
# tests/test_goldens.py and tests/test_modelcore.py) even though main()/capture_synthetic() can
# no longer actually run; they only ever ran once, against the pre-Stage-7 commit.

GOLDENS_DIR = os.path.join(os.path.dirname(__file__), "..", "tests", "goldens")
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


def capture_checkpoint_dir(checkpoints_dir, tag, out_name, prompt_tokens=None):
    """Shared capture path for both real and synthetic checkpoint directories -- both go through
    the exact same checkpoint_manager façade (load_model_from_dir), so this doubles as a
    save/load round-trip test for the synthetic sources."""
    print(f"--- {out_name} ({tag}) ---")
    model, tokenizer, meta = load_model_from_dir(checkpoints_dir, DEVICE, phase="eval", model_tag=tag)
    model.eval()
    config = model.config
    result = {
        "source": out_name,
        "model_tag": tag,
        "meta_model_config": meta["model_config"],
        "state_dict_fingerprint": state_dict_fingerprint(model.state_dict()),
        "accounting": accounting(model),
        "logits_hash": fixed_logits_hash(model, config.vocab_size, config.sequence_len),
        "optimizer_layout": optimizer_layout(model),
    }

    if prompt_tokens is not None:
        prompt = list(prompt_tokens)
    else:
        prompt = tokenizer.encode(PROMPT_TEXT, prepend=tokenizer.get_bos_token_id())
    naive_tokens = list(generate_naive(model, prompt, max_tokens=GEN_TOKENS, temperature=0.0))
    result["generate_naive_tokens"] = naive_tokens

    engine = Engine(model, tokenizer)
    batch_tokens, _ = engine.generate_batch(prompt, num_samples=1, max_tokens=GEN_TOKENS, temperature=0.0)
    result["generate_batch_tokens"] = batch_tokens[0][len(prompt):]

    checkpoint_dir = os.path.join(checkpoints_dir, tag)
    result["optimizer_shard"] = optimizer_digest_from_dir(model, checkpoint_dir)

    out_path = os.path.join(GOLDENS_DIR, f"{out_name}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1)
    print(f"    wrote {out_path}")


def capture_synthetic(arch, out_name, preset_kwargs=None):
    print(f"=== building synthetic {out_name} ({arch}) ===")
    torch.manual_seed(0)
    if arch == "composed":
        from nanochat.model.composed.model import ComposedModel
        from nanochat.model.composed.presets import expand_preset
        preset, kwargs = preset_kwargs
        config = expand_preset(preset, **kwargs)
        with torch.device("meta"):
            model = ComposedModel(config)
    else:
        from nanochat.model import get_config_class, get_model_class
        from tests.conftest import TINY_KWARGS_BY_ARCH
        config_cls = get_config_class(arch)
        model_cls = get_model_class(arch)
        kwargs = TINY_KWARGS_BY_ARCH[arch]
        config = config_cls(**kwargs)
        with torch.device("meta"):
            model = model_cls(config)
    model.to_empty(device=DEVICE)
    torch.manual_seed(0)
    model.init_weights()
    model.eval()

    optimizer = model.setup_optimizer()
    checkpoint_dir = os.path.join(TINY_DIR, out_name)
    save_checkpoint(
        checkpoint_dir, step=0,
        model_data=model.state_dict(), optimizer_data=optimizer.state_dict(),
        meta_data={"model_config": config.to_dict()},
    )

    checkpoint_manager.get_tokenizer = lambda: _FakeTokenizer(config.vocab_size)
    prompt_tokens = [i % config.vocab_size for i in range(1, 5)]
    capture_checkpoint_dir(TINY_DIR, out_name, out_name, prompt_tokens=prompt_tokens)


def capture_real(checkpoints_dir, tag, out_name):
    if not os.path.isdir(os.path.join(checkpoints_dir, tag)):
        print(f"SKIP {out_name}: {checkpoints_dir}/{tag} not found on this machine")
        return
    capture_checkpoint_dir(checkpoints_dir, tag, out_name)


def main():
    from tests.conftest import TINY_KWARGS_BY_ARCH

    os.makedirs(GOLDENS_DIR, exist_ok=True)
    os.makedirs(TINY_DIR, exist_ok=True)

    base_dir = get_base_dir()
    base_checkpoints = os.path.join(base_dir, "base_checkpoints")
    chatsft_checkpoints = os.path.join(base_dir, "chatsft_checkpoints")

    capture_real(base_checkpoints, "d6", "d6")
    capture_real(base_checkpoints, "smoketest_kvshare_win_d2", "smoketest_kvshare_win_d2")
    capture_real(base_checkpoints, "contest_h100run_gpt_d12", "contest_h100run_gpt_d12")
    capture_real(base_checkpoints, "contest_h100run_llama_d12", "contest_h100run_llama_d12")
    capture_real(base_checkpoints, "contest_h100run_llama_kvshare_win_d12", "contest_h100run_llama_kvshare_win_d12")
    capture_real(base_checkpoints, "contest_d12test_llama_kvshare_d12", "contest_d12test_llama_kvshare_d12")
    capture_real(chatsft_checkpoints, "contest_h100run_gpt_d12", "chatsft_contest_h100run_gpt_d12")

    for arch in TINY_KWARGS_BY_ARCH:
        capture_synthetic(arch, f"tiny_{arch}")

    composed_presets = [
        ("gpt", dict(depth=4, aspect_ratio=16, head_dim=32, max_seq_len=32, vocab_size=128, window_pattern="SSSL")),
        ("llama", dict(depth=4, aspect_ratio=16, head_dim=32, max_seq_len=32, vocab_size=128, window_pattern="L")),
        ("llama_kvshare", dict(depth=4, aspect_ratio=16, head_dim=32, max_seq_len=32, vocab_size=128, window_pattern="L", kv_share_frac=0.5)),
        ("llama_kvshare_win", dict(depth=4, aspect_ratio=16, head_dim=32, max_seq_len=32, vocab_size=128, window_pattern="SSSL", kv_share_frac=0.5)),
    ]
    for preset, kwargs in composed_presets:
        capture_synthetic("composed", f"tiny_composed_{preset}", preset_kwargs=(preset, kwargs))

    print("Done.")


if __name__ == "__main__":
    main()
