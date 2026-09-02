"""
GPT-specific model tests (nanochat/model/gpt/). Architecture-agnostic structural tests (forward
shape, backward grads, matmul-param marker, optimizer partition, layer_specs/kv_cache_spec) live
in tests/test_model_common.py, parametrized over every registered architecture. This file is only
for behavior specific to GPT: its num_scaling_params() six-key legacy contract.

python -m pytest tests/test_model_gpt.py -v
"""


def test_num_scaling_params_six_key_legacy_contract(tiny_gpt):
    """GPT overrides BaseModel's generic num_scaling_params() default to preserve this exact
    six-key shape -- runs/scaling_laws.sh greps these key names out of scripts/base_train.py's
    stdout (see nanochat/model/gpt/model.py and docs/architecture.md)."""
    counts = tiny_gpt.num_scaling_params()
    assert counts["total"] == sum(p.numel() for p in tiny_gpt.parameters())
    group_sum = counts["wte"] + counts["value_embeds"] + counts["lm_head"] + counts["transformer_matrices"] + counts["scalars"]
    assert group_sum == counts["total"]
