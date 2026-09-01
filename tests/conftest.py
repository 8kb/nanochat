"""
Shared pytest fixtures for tests/. All fixtures here are CPU-only and hermetic (no network, no
cached data, no GPU) so they run on a MacBook without CUDA.
"""

import torch
import pytest

from nanochat.model import GPT, GPTConfig


TINY_GPT_KWARGS = dict(
    sequence_len=32,
    vocab_size=128,
    n_layer=4,
    n_head=2,
    n_kv_head=2,
    n_embd=64,
    window_pattern="L",
)


def build_tiny_gpt(**overrides):
    """Build a small GPT on CPU with real (initialized) weights, mirroring the meta-device ->
    to_empty -> init_weights sequence used by checkpoint_manager.build_model and
    scripts/base_train.py."""
    kwargs = {**TINY_GPT_KWARGS, **overrides}
    config = GPTConfig(**kwargs)
    with torch.device("meta"):
        model = GPT(config)
    model.to_empty(device="cpu")
    model.init_weights()
    model.eval()
    return model


@pytest.fixture
def tiny_gpt():
    return build_tiny_gpt()
