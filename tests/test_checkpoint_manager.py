"""
Test nanochat/checkpoint_manager.py's naming-policy helpers directly: arch_of() (the
architecture/preset name for tag naming and auto-discovery filtering) and find_largest_model's
arch filter built on top of it.

python -m pytest tests/test_checkpoint_manager.py -v
"""
import json
import os

from nanochat.checkpoint_manager import arch_of, find_largest_model


def test_arch_of_current_format_reads_reference_preset():
    model_config = {"format": "modelcore.v1", "reference": {"preset": "llama_kvshare"}}
    assert arch_of(model_config) == "llama_kvshare"


def test_arch_of_current_format_defaults_to_custom_without_reference():
    model_config = {"format": "modelcore.v1", "reference": None}
    assert arch_of(model_config) == "custom"
    assert arch_of({"format": "modelcore.v1"}) == "custom"


def test_arch_of_legacy_format_reads_arch_key():
    assert arch_of({"arch": "llama"}) == "llama"


def test_arch_of_legacy_format_defaults_to_gpt_without_arch_key():
    """A checkpoint predating even the "arch" key (e.g. d6) defaults to gpt."""
    assert arch_of({}) == "gpt"


def _write_checkpoint(base_dir, tag, step, model_config):
    checkpoint_dir = os.path.join(base_dir, tag)
    os.makedirs(checkpoint_dir, exist_ok=True)
    with open(os.path.join(checkpoint_dir, f"model_{step:06d}.pt"), "w") as f:
        f.write("")  # find_last_step only checks the filename exists
    with open(os.path.join(checkpoint_dir, f"meta_{step:06d}.json"), "w") as f:
        json.dump({"model_config": model_config}, f)


def test_find_largest_model_filters_by_arch(tmp_path):
    base_dir = str(tmp_path)
    _write_checkpoint(base_dir, "d12", 100, {"format": "modelcore.v1", "reference": {"preset": "gpt"}})
    _write_checkpoint(base_dir, "llama_d16", 100, {"format": "modelcore.v1", "reference": {"preset": "llama"}})

    assert find_largest_model(base_dir, arch="gpt") == "d12"
    assert find_largest_model(base_dir, arch="llama") == "llama_d16"
    # No filter: only "d<N>"-shaped tags (re.match anchors at the start) compete on depth, so
    # "llama_d16" isn't a candidate here at all -- "d12" wins by default, not by depth comparison.
    assert find_largest_model(base_dir) == "d12"


def test_find_largest_model_arch_filter_raises_when_none_match(tmp_path):
    base_dir = str(tmp_path)
    _write_checkpoint(base_dir, "d12", 100, {"format": "modelcore.v1", "reference": {"preset": "gpt"}})

    import pytest
    with pytest.raises(FileNotFoundError):
        find_largest_model(base_dir, arch="llama")
