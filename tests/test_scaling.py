"""
nanochat/scaling.py's derive_training_plan is scripts/base_train.py's training-horizon math
(batch size, LR/weight-decay scaling, num_iterations), pulled out into pure functions so
scripts/model_info.py can report identical numbers without training anything. This test checks
the math directly, by hand, against the formulas documented in nanochat/scaling.py and (before
this stage) inlined in base_train.py.

python -m pytest tests/test_scaling.py -v
"""

import math

from nanochat.scaling import derive_training_plan, B_REF


def test_user_provided_num_iterations_is_used_verbatim():
    plan = derive_training_plan(
        num_scaling_params=1_000_000, d_ref_scaling_params=1_000_000, num_flops_per_token=1e6,
        target_param_data_ratio=12, target_flops=-1.0, num_iterations=100,
        total_batch_size=1024, weight_decay=0.1,
    )
    assert plan.horizon_source == "user"
    assert plan.num_iterations == 100
    assert plan.total_batch_size == 1024
    assert plan.auto_batch_size is False
    assert plan.total_tokens == 1024 * 100
    assert plan.total_flops == 1e6 * 1024 * 100


def test_target_flops_horizon_matches_hand_computation():
    num_flops_per_token = 2e6
    total_batch_size = 2048
    target_flops = 5e13
    plan = derive_training_plan(
        num_scaling_params=1_000_000, d_ref_scaling_params=1_000_000, num_flops_per_token=num_flops_per_token,
        target_param_data_ratio=12, target_flops=target_flops, num_iterations=-1,
        total_batch_size=total_batch_size, weight_decay=0.1,
    )
    assert plan.horizon_source == "target_flops"
    assert plan.num_iterations == round(target_flops / (num_flops_per_token * total_batch_size))


def test_target_param_data_ratio_horizon_matches_hand_computation():
    num_scaling_params = 500_000
    target_param_data_ratio = 20.0
    total_batch_size = 4096
    plan = derive_training_plan(
        num_scaling_params=num_scaling_params, d_ref_scaling_params=num_scaling_params, num_flops_per_token=1e6,
        target_param_data_ratio=target_param_data_ratio, target_flops=-1.0, num_iterations=-1,
        total_batch_size=total_batch_size, weight_decay=0.1,
    )
    assert plan.horizon_source == "target_param_data_ratio"
    target_tokens = int(target_param_data_ratio * num_scaling_params)
    assert plan.target_tokens == target_tokens
    assert plan.num_iterations == target_tokens // total_batch_size


def test_auto_batch_size_uses_power_lines_scaling_and_snaps_to_power_of_two():
    num_scaling_params = 2_000_000
    d_ref_scaling_params = 1_000_000  # bigger than d12 reference -> larger predicted batch
    target_param_data_ratio = 12.0
    plan = derive_training_plan(
        num_scaling_params=num_scaling_params, d_ref_scaling_params=d_ref_scaling_params, num_flops_per_token=1e6,
        target_param_data_ratio=target_param_data_ratio, target_flops=-1.0, num_iterations=1,
        total_batch_size=-1, weight_decay=0.1,
    )
    assert plan.auto_batch_size is True
    target_tokens = int(target_param_data_ratio * num_scaling_params)
    D_REF = target_param_data_ratio * d_ref_scaling_params
    predicted = B_REF * (target_tokens / D_REF) ** 0.383
    expected_batch_size = 2 ** round(math.log2(predicted))
    assert plan.total_batch_size == expected_batch_size
    # a power of two
    assert expected_batch_size & (expected_batch_size - 1) == 0


def test_batch_lr_scale_is_one_at_reference_batch_size():
    plan = derive_training_plan(
        num_scaling_params=1_000_000, d_ref_scaling_params=1_000_000, num_flops_per_token=1e6,
        target_param_data_ratio=12, target_flops=-1.0, num_iterations=1,
        total_batch_size=B_REF, weight_decay=0.1,
    )
    assert plan.batch_lr_scale == 1.0


def test_batch_lr_scale_matches_sqrt_ratio_away_from_reference():
    total_batch_size = B_REF * 4
    plan = derive_training_plan(
        num_scaling_params=1_000_000, d_ref_scaling_params=1_000_000, num_flops_per_token=1e6,
        target_param_data_ratio=12, target_flops=-1.0, num_iterations=1,
        total_batch_size=total_batch_size, weight_decay=0.1,
    )
    assert plan.batch_lr_scale == (total_batch_size / B_REF) ** 0.5


def test_weight_decay_scaled_matches_t_epoch_formula():
    num_scaling_params = 1_000_000
    d_ref_scaling_params = 1_000_000
    target_param_data_ratio = 12.0
    total_batch_size = 2048
    weight_decay = 0.28
    plan = derive_training_plan(
        num_scaling_params=num_scaling_params, d_ref_scaling_params=d_ref_scaling_params, num_flops_per_token=1e6,
        target_param_data_ratio=target_param_data_ratio, target_flops=-1.0, num_iterations=1,
        total_batch_size=total_batch_size, weight_decay=weight_decay,
    )
    target_tokens = int(target_param_data_ratio * num_scaling_params)
    D_REF = target_param_data_ratio * d_ref_scaling_params
    expected = weight_decay * math.sqrt(total_batch_size / B_REF) * (D_REF / target_tokens)
    assert plan.weight_decay_scaled == expected


def test_no_horizon_specified_raises():
    import pytest
    with pytest.raises(ValueError):
        derive_training_plan(
            num_scaling_params=1_000_000, d_ref_scaling_params=1_000_000, num_flops_per_token=1e6,
            target_param_data_ratio=-1, target_flops=-1.0, num_iterations=-1,
            total_batch_size=1024, weight_decay=0.1,
        )
