"""
Test the architecture registry (nanochat/model/registry.py): the seam that lets
checkpoint_manager and training scripts reconstruct a model/config without importing any
specific architecture.

python -m pytest tests/test_model_registry.py -v
"""

import pytest

from nanochat.model import GPT, GPTConfig, get_model_class, get_config_class, config_from_dict
from tests.conftest import TINY_GPT_KWARGS


def test_gpt_is_registered_under_arch_name():
    assert GPTConfig.arch == "gpt"
    assert get_model_class("gpt") is GPT
    assert get_config_class("gpt") is GPTConfig


def test_unknown_arch_raises():
    with pytest.raises(ValueError):
        get_model_class("does-not-exist")
    with pytest.raises(ValueError):
        get_config_class("does-not-exist")


def test_to_dict_config_from_dict_roundtrip():
    config = GPTConfig(**TINY_GPT_KWARGS)
    d = config.to_dict()
    assert d["arch"] == "gpt"
    rebuilt = config_from_dict(d)
    assert rebuilt == config


def test_config_from_dict_defaults_missing_arch_to_gpt():
    """Checkpoints saved before architectures were pluggable have no "arch" key."""
    legacy_dict = dict(TINY_GPT_KWARGS)  # no "arch" key, like an old checkpoint's model_config
    assert "arch" not in legacy_dict
    rebuilt = config_from_dict(legacy_dict)
    assert isinstance(rebuilt, GPTConfig)
    assert rebuilt == GPTConfig(**TINY_GPT_KWARGS)


def test_config_from_dict_does_not_mutate_input():
    d = GPTConfig(**TINY_GPT_KWARGS).to_dict()
    d_copy = dict(d)
    config_from_dict(d)
    assert d == d_copy
