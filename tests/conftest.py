"""
Shared pytest fixtures for tests/. All fixtures here are CPU-only and hermetic (no network, no
cached data, no GPU) so they run on a MacBook without CUDA.
"""

import torch
import pytest

from nanochat.model import get_config_class, get_model_class


TINY_GPT_KWARGS = dict(
    sequence_len=32,
    vocab_size=128,
    n_layer=4,
    n_head=2,
    n_kv_head=2,
    n_embd=64,
    window_pattern="L",
)

TINY_LLAMA_KWARGS = dict(
    sequence_len=32,
    vocab_size=128,
    n_layer=4,
    n_head=2,
    n_kv_head=2,
    n_embd=64,
    window_pattern="L",
)

TINY_KVSHARE_KWARGS = dict(
    sequence_len=32,
    vocab_size=128,
    n_layer=4,
    n_head=2,
    n_kv_head=2,
    n_embd=64,
    window_pattern="L",
    kv_share_frac=0.5,  # 4 layers -> 2 KV slots
)

TINY_KVSHARE_WIN_KWARGS = dict(
    sequence_len=32,
    vocab_size=128,
    n_layer=4,
    n_head=2,
    n_kv_head=2,
    n_embd=64,
    window_pattern="SL",  # a real window (not "L") so the generic suite exercises the windowed path
    kv_share_frac=0.5,  # 4 layers -> 2 KV slots
)

TINY_KWARGS_BY_ARCH = {
    "gpt": TINY_GPT_KWARGS, "llama": TINY_LLAMA_KWARGS, "llama_kvshare": TINY_KVSHARE_KWARGS,
    "llama_kvshare_win": TINY_KVSHARE_WIN_KWARGS,
}

# Every architecture the tiny_model fixture parametrizes over, i.e. every architecture
# tests/test_model_common.py's generic suite covers. Deliberately kept separate from
# TINY_KWARGS_BY_ARCH -- other test files (test_model_info.py, test_model_registry.py) import
# TINY_KWARGS_BY_ARCH directly and assume a flat kwargs dict, which "composed" (a materialized
# tree, not flat kwargs -- see _tiny_composed_config below) doesn't fit.
TINY_MODEL_ARCHS = [*TINY_KWARGS_BY_ARCH, "composed"]


def _tiny_composed_config(preset="gpt", **overrides):
    """A composed-architecture tree, same shape as TINY_GPT_KWARGS (n_embd=64, n_head=2, n_layer=4)
    by construction: expand_preset("gpt", depth=4, aspect_ratio=16, head_dim=32, ...) derives the
    same muP dims TINY_GPT_KWARGS hardcodes -- see nanochat.model.composed.presets.expand_gpt."""
    from nanochat.model.composed.presets import expand_preset
    kwargs = dict(depth=4, aspect_ratio=16, head_dim=32, max_seq_len=32, vocab_size=128, window_pattern="L")
    kwargs.update(overrides)
    depth = kwargs.pop("depth")
    return expand_preset(preset, depth, **kwargs)


def build_tiny_model(arch="gpt", **overrides):
    """Build a small model of the given (registered) architecture on CPU with real (initialized)
    weights, mirroring the meta-device -> to_empty -> init_weights sequence used by
    checkpoint_manager.build_model and scripts/base_train.py. For arch="composed", pass either
    config=<a ComposedConfig> directly, or preset kwargs (see _tiny_composed_config)."""
    model_cls = get_model_class(arch)
    if arch == "composed":
        config = overrides.pop("config", None)
        if config is None:
            config = _tiny_composed_config(**overrides)
        else:
            assert not overrides, "pass either config=... or preset kwargs to build_tiny_model('composed', ...), not both"
    else:
        kwargs = {**TINY_KWARGS_BY_ARCH[arch], **overrides}
        config_cls = get_config_class(arch)
        config = config_cls(**kwargs)
    with torch.device("meta"):
        model = model_cls(config)
    model.to_empty(device="cpu")
    model.init_weights()
    model.eval()
    return model


def build_tiny_gpt(**overrides):
    """Backward-compat alias for build_tiny_model("gpt", **overrides) -- several existing tests
    import this directly."""
    return build_tiny_model("gpt", **overrides)


@pytest.fixture
def tiny_gpt():
    return build_tiny_gpt()


@pytest.fixture
def tiny_llama():
    return build_tiny_model("llama")


@pytest.fixture
def tiny_llama_kvshare():
    return build_tiny_model("llama_kvshare")


@pytest.fixture
def tiny_llama_kvshare_win():
    return build_tiny_model("llama_kvshare_win")


@pytest.fixture
def tiny_composed():
    return build_tiny_model("composed")


@pytest.fixture(params=TINY_MODEL_ARCHS)
def tiny_model(request):
    """Parametrized over every registered architecture in TINY_KWARGS_BY_ARCH -- a test taking
    this fixture runs once per architecture (test ids get a [gpt]/[llama]/[llama_kvshare]/
    [llama_kvshare_win]/[composed] suffix). Use this for anything that should hold for any
    architecture; use tiny_gpt/tiny_llama/tiny_llama_kvshare/tiny_llama_kvshare_win/tiny_composed
    for architecture-specific behavior."""
    return build_tiny_model(request.param)
