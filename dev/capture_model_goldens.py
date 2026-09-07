"""
Capture golden artifacts describing today's `nanochat/model/` + `nanochat/checkpoint_manager.py`
behavior, BEFORE the Stage 7 `modelcore/` extraction (see docs/roadmap.md).

Ran once, against unmodified pre-Stage-7 code:
    python -m dev.capture_model_goldens

Wrote:
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
                                     across Stage 2 (docs/architecture.md notes it does not).

FROZEN: this module can no longer actually run. capture_synthetic()'s non-composed branch and
main() both import nanochat.model / tests.conftest.TINY_KWARGS_BY_ARCH, and both were deleted
along with nanochat/model/ at Stage 7 step 5 (the composed branch used nanochat.model.composed,
also deleted). It is kept as a record of exactly how tests/goldens/*.json and tests/goldens/tiny/
were produced -- nothing here needs to run again unless the goldens themselves are ever
recaptured from scratch.

The digest helpers this module used (accounting, optimizer_layout, state_dict_fingerprint, ...)
moved to tests/golden_helpers.py at Stage 8 (see docs/roadmap.md), since tests/test_goldens.py
needs them as a live library and this module does not -- imported back here so the capture
functions below still read exactly as they did when this last ran.
"""
import json
import os

import torch

from nanochat import checkpoint_manager
from nanochat.checkpoint_manager import load_model_from_dir, save_checkpoint
from nanochat.common import get_base_dir
from nanochat.engine import Engine, generate_naive

from tests.golden_helpers import (
    DEVICE, GOLDENS_DIR, GEN_TOKENS, PROMPT_TEXT, TINY_DIR,
    _FakeTokenizer, accounting, fixed_logits_hash, optimizer_digest_from_dir, optimizer_layout,
    state_dict_fingerprint,
)


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
