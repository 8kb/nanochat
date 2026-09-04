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


def build_tiny_model(arch="gpt", **overrides):
    """Build a small model of the given (registered) architecture on CPU with real (initialized)
    weights, mirroring the meta-device -> to_empty -> init_weights sequence used by
    checkpoint_manager.build_model and scripts/base_train.py."""
    kwargs = {**TINY_KWARGS_BY_ARCH[arch], **overrides}
    config_cls = get_config_class(arch)
    model_cls = get_model_class(arch)
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


@pytest.fixture(params=list(TINY_KWARGS_BY_ARCH.keys()))
def tiny_model(request):
    """Parametrized over every registered architecture in TINY_KWARGS_BY_ARCH -- a test taking
    this fixture runs once per architecture (test ids get a [gpt]/[llama]/[llama_kvshare]/
    [llama_kvshare_win] suffix). Use this for anything that should hold for any architecture; use
    tiny_gpt/tiny_llama/tiny_llama_kvshare/tiny_llama_kvshare_win for architecture-specific
    behavior."""
    return build_tiny_model(request.param)
