"""
LlamaKVShareWin-specific tests (nanochat/model/llama_kvshare_win/). Architecture-agnostic
structural tests live in tests/test_model_common.py, and cross-backend correctness (naive vs.
KV-cached generation) lives in tests/test_generate.py, both parametrized to include
"llama_kvshare_win". This file covers what's specific to this architecture: that it really is
just llama_kvshare with a different window_pattern default, and that windowing strictly lowers
FLOPs/token at an identical shape/param count (the plan's central claim).

python -m pytest tests/test_model_kvshare_win.py -v
"""

from nanochat.model import get_config_class, get_model_class
from nanochat.model.components.windows import compute_window_sizes
from tests.conftest import build_tiny_model, TINY_KVSHARE_KWARGS


def test_llama_kvshare_win_is_registered_under_arch_name():
    config_cls = get_config_class("llama_kvshare_win")
    assert config_cls.arch == "llama_kvshare_win"
    assert get_model_class("llama_kvshare_win").__name__ == "LlamaKVShareWin"


def test_from_depth_defaults_to_windowed_pattern():
    """The whole point of LlamaKVShareWinConfig's from_depth override -- LlamaConfig.from_depth
    hardcodes window_pattern="L" in its own signature default, so without overriding from_depth
    (not just the dataclass field) a --depth run would silently build a full-context model."""
    config = get_config_class("llama_kvshare_win").from_depth(12)
    assert config.window_pattern == "SSSL"
    assert config.kv_share_frac == 0.5  # inherited from LlamaKVShareConfig, unchanged


def test_layer_windows_match_compute_window_sizes(tiny_llama_kvshare_win):
    n_layer = tiny_llama_kvshare_win.config.n_layer
    pattern = tiny_llama_kvshare_win.config.window_pattern
    seq_len = tiny_llama_kvshare_win.config.sequence_len
    expected = compute_window_sizes(pattern, n_layer, seq_len)
    actual = [block.attn.window for block in tiny_llama_kvshare_win.blocks]
    assert actual == [w for w, _ in expected]
    assert actual[-1] == seq_len  # final layer always forced to full context


def test_same_params_but_strictly_fewer_flops_than_full_context_kvshare():
    """Windowing changes nothing about parameter count (it's a mask, not a shape change) but
    strictly lowers FLOPs/token relative to plain llama_kvshare at the same shape -- the
    "trade attention span for more tokens at fixed FLOPs budget" trade this architecture exists
    to make. sequence_len is bumped to 512 (vs. TINY_KVSHARE_KWARGS's 32) because
    compute_window_sizes's short-window formula rounds up to a 128-token tile and floors at 128,
    so at sequence_len=32 the "short" window is already >= the sequence length and has no effect
    once _effective_window caps it -- 512 is the smallest size where "S" actually differs from "L"."""
    kwargs = {**TINY_KVSHARE_KWARGS, "sequence_len": 512}
    kvshare = build_tiny_model("llama_kvshare", **{**kwargs, "window_pattern": "L"})
    kvshare_win = build_tiny_model("llama_kvshare_win", **{**kwargs, "window_pattern": "SL"})
    assert kvshare_win.num_scaling_params()["total"] == kvshare.num_scaling_params()["total"]
    assert kvshare_win.estimate_flops() < kvshare.estimate_flops()
