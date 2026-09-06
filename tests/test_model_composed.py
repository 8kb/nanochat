"""
Equivalence tests: for each of the four native architectures, the composed preset that expands to
it must build a model identical in every accounting number, and -- after remapping state-dict keys
(the composed body wraps what used to be top-level: blocks.N.* -> body.blocks.N.*, and GPT's
backout_lambda -> body.backout_lambda) -- bit-identical in forward output. This is the actual proof
that StackComposer/BackoutComposer reproduce Llama.forward / GPT._forward_trunk exactly, not just
approximately. See docs/architecture.md's "Composed architectures".

python -m pytest tests/test_model_composed.py -v
"""
import dataclasses

import torch
import pytest

from nanochat.model import get_config_class, get_model_class
from nanochat.model.composed.model import ComposedModel
from nanochat.model.composed.presets import expand_preset
from nanochat.model.composed.spec import ComposedConfig


TINY_DIMS = dict(depth=4, aspect_ratio=16, head_dim=32, max_seq_len=32, vocab_size=128)

PRESET_CASES = [
    ("gpt", dict(window_pattern="SSSL")),
    ("llama", dict(window_pattern="L")),
    ("llama_kvshare", dict(window_pattern="L", kv_share_frac=0.5)),
    ("llama_kvshare_win", dict(window_pattern="SSSL", kv_share_frac=0.5)),
]


def _build_native(arch, **extra):
    config_cls = get_config_class(arch)
    model_cls = get_model_class(arch)
    kwargs = {**TINY_DIMS, **extra}
    depth = kwargs.pop("depth")
    kv_share_frac = kwargs.pop("kv_share_frac", None)
    config = config_cls.from_depth(depth, **kwargs)
    if kv_share_frac is not None:
        config = dataclasses.replace(config, kv_share_frac=kv_share_frac)
    with torch.device("meta"):
        model = model_cls(config)
    model.to_empty(device="cpu")
    model.init_weights()
    model.eval()
    return model


def _build_composed(preset, **extra):
    kwargs = {**TINY_DIMS, **extra}
    depth = kwargs.pop("depth")
    config = expand_preset(preset, depth, **kwargs)
    with torch.device("meta"):
        model = ComposedModel(config)
    model.to_empty(device="cpu")
    model.init_weights()
    model.eval()
    return model


def _remap_key(key):
    """Native state_dict key -> composed state_dict key: the composer wraps what used to be
    top-level. Applies uniformly to all four presets (only gpt has backout_lambda)."""
    if key == "backout_lambda":
        return "body.backout_lambda"
    if key.startswith("blocks."):
        return "body." + key
    return key


@pytest.mark.parametrize("arch,extra", PRESET_CASES)
def test_composed_preset_matches_native_accounting(arch, extra):
    native = _build_native(arch, **extra)
    composed = _build_composed(arch, **extra)
    assert composed.layer_specs() == native.layer_specs()
    assert composed.kv_cache_spec() == native.kv_cache_spec()
    assert composed.num_matmul_params() == native.num_matmul_params()
    assert composed.estimate_flops() == native.estimate_flops()
    assert composed.kv_bytes_per_token() == native.kv_bytes_per_token()
    native_total = sum(p.numel() for p in native.parameters())
    composed_total = sum(p.numel() for p in composed.parameters())
    assert composed_total == native_total


@pytest.mark.parametrize("arch,extra", PRESET_CASES)
def test_composed_preset_matches_native_forward(arch, extra):
    """Copies the native model's weights onto the composed model (via the documented state-dict
    key remap) and asserts bit-identical forward output -- the real proof that
    StackComposer/BackoutComposer reproduce the native trunk loop exactly, not just its shape."""
    native = _build_native(arch, **extra)
    composed = _build_composed(arch, **extra)

    remapped = {_remap_key(k): v for k, v in native.state_dict().items()}
    composed_keys = set(composed.state_dict().keys())
    assert set(remapped.keys()) == composed_keys, set(remapped.keys()) ^ composed_keys
    composed.load_state_dict(remapped, strict=True)

    torch.manual_seed(0)
    idx = torch.randint(0, 128, (2, 8))
    with torch.no_grad():
        native_logits = native.forward(idx)
        composed_logits = composed.forward(idx)
    assert torch.equal(native_logits, composed_logits)


def test_composed_config_to_dict_from_dict_roundtrip():
    config = expand_preset("gpt", **TINY_DIMS, window_pattern="SSSL")
    rebuilt = ComposedConfig.from_dict(config.to_dict())
    assert rebuilt == config
    assert rebuilt.n_layer == config.n_layer == TINY_DIMS["depth"]


def test_composed_registered_under_arch_name():
    assert get_config_class("composed") is ComposedConfig
    assert get_model_class("composed") is ComposedModel
    assert ComposedConfig.arch == "composed"
