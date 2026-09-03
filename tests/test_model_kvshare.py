"""
LlamaKVShare-specific model tests (nanochat/model/llama_kvshare/). Architecture-agnostic
structural tests live in tests/test_model_common.py; the real cross-backend correctness check
(naive forward vs. KV-cached Engine agreement) lives in tests/test_generate.py, parametrized to
include "llama_kvshare". This file covers what's specific to cross-layer KV sharing: the slot
map, the absence of c_k/c_v on consumer layers, and that sharing actually shrinks params/KV bytes
relative to plain Llama at the same shape.

python -m pytest tests/test_model_kvshare.py -v
"""

from nanochat.model.components.kv_sharing import compute_kv_slots
from tests.conftest import build_tiny_model, TINY_KVSHARE_KWARGS


def test_kv_slots_match_compute_kv_slots(tiny_llama_kvshare):
    n_layer = tiny_llama_kvshare.config.n_layer
    kv_share_frac = tiny_llama_kvshare.config.kv_share_frac
    expected = compute_kv_slots(n_layer, kv_share_frac)
    actual = [block.attn.kv_slot for block in tiny_llama_kvshare.blocks]
    assert actual == expected


def test_consumer_layers_have_no_own_kv_projection(tiny_llama_kvshare):
    n_own = max(block.attn.kv_slot for block in tiny_llama_kvshare.blocks) + 1
    for i, block in enumerate(tiny_llama_kvshare.blocks):
        if i < n_own:
            assert block.attn.produces_kv
            assert block.attn.c_k is not None and block.attn.c_v is not None
        else:
            assert not block.attn.produces_kv
            assert block.attn.c_k is None and block.attn.c_v is None


def test_kv_cache_spec_num_slots_equals_num_owning_layers(tiny_llama_kvshare):
    n_own = max(block.attn.kv_slot for block in tiny_llama_kvshare.blocks) + 1
    assert tiny_llama_kvshare.kv_cache_spec()["num_kv_slots"] == n_own
    assert n_own < tiny_llama_kvshare.config.n_layer  # actually sharing at kv_share_frac=0.5


def test_sharing_strictly_shrinks_params_and_kv_bytes_vs_plain_llama():
    """Same shape (n_layer/n_embd/n_head/...), only kv_share_frac differs from plain Llama's
    implicit 0 -- sharing should have strictly fewer params (dropped c_k/c_v) and strictly fewer
    KV-cache bytes/token (fewer distinct slots) than nanochat.model.llama.Llama."""
    llama = build_tiny_model("llama", **{k: v for k, v in TINY_KVSHARE_KWARGS.items() if k != "kv_share_frac"})
    kvshare = build_tiny_model("llama_kvshare")
    assert kvshare.num_scaling_params()["total"] < llama.num_scaling_params()["total"]
    assert kvshare.kv_bytes_per_token() < llama.kv_bytes_per_token()


def test_kv_share_frac_zero_reproduces_plain_llama_slotting():
    """kv_share_frac=0.0 means every layer owns its own slot -- structurally identical slotting
    to nanochat.model.llama.Llama (all kv_slot == layer_idx, no sharing)."""
    model = build_tiny_model("llama_kvshare", kv_share_frac=0.0)
    n_layer = model.config.n_layer
    assert [block.attn.kv_slot for block in model.blocks] == list(range(n_layer))
    assert all(block.attn.produces_kv for block in model.blocks)
    assert model.kv_cache_spec()["num_kv_slots"] == n_layer
