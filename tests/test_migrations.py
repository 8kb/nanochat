"""
Test nanochat/model/gpt/migrations.py's Stage 2 renames: the state-dict layout migration
(flat/top-level keys -> per-module keys) and the optimizer-state split (one [n_layer] scalar
parameter -> n_layer independent per-block scalars). Both must be no-ops on already-new-layout
input.

python -m pytest tests/test_migrations.py -v
"""

import torch

from nanochat.model.gpt import migrations
from nanochat.model.gpt.config import GPTConfig


def _old_layout_state_dict(model):
    """Given a real (new-layout) tiny GPT, hand-construct the OLD (pre-Stage-2) layout state
    dict an equivalent checkpoint would have had, reusing the same tensor values -- so migrating
    it forward can be checked against the model's own real state dict exactly."""
    new_sd = model.state_dict()
    # resid_lambda/x0_lambda collapse from n_layer separate keys to 1 combined [n_layer] tensor
    # each in the old layout, so old_sd legitimately has fewer keys than new_sd.
    old_sd = {
        "transformer.wte.weight": new_sd["embedding.wte.weight"],
        "smear_gate.weight": new_sd["embedding.smear.gate.weight"],
        "smear_lambda": new_sd["embedding.smear.lambda_"],
        "lm_head.weight": new_sd["unembedding.lm_head.weight"],
        "backout_lambda": new_sd["backout_lambda"],
    }
    n_layer = model.config.n_layer
    old_sd["resid_lambdas"] = torch.stack([new_sd[f"blocks.{i}.resid_lambda"] for i in range(n_layer)])
    old_sd["x0_lambdas"] = torch.stack([new_sd[f"blocks.{i}.x0_lambda"] for i in range(n_layer)])
    for key, value in new_sd.items():
        if not key.startswith("blocks."):
            continue
        _, i, rest = key.split(".", 2)
        if rest in ("resid_lambda", "x0_lambda"):
            continue
        if rest.startswith("attn.value_embed."):
            old_sd[f"value_embeds.{i}.{rest.removeprefix('attn.value_embed.')}"] = value
        else:
            old_sd[f"transformer.h.{i}.{rest}"] = value
    return old_sd


def test_patch_state_dict_layout_renames_old_keys_and_preserves_values(tiny_gpt):
    old_sd = _old_layout_state_dict(tiny_gpt)
    n_layer = tiny_gpt.config.n_layer
    assert len(old_sd) == len(tiny_gpt.state_dict()) - 2 * (n_layer - 1)  # see comment above
    new_sd = migrations.patch_state_dict_layout(dict(old_sd), tiny_gpt.config, log=lambda msg: None)
    real_sd = tiny_gpt.state_dict()
    assert set(new_sd.keys()) == set(real_sd.keys())
    for key, value in real_sd.items():
        assert torch.equal(new_sd[key], value), f"value mismatch for {key}"


def test_patch_state_dict_layout_is_idempotent_on_new_layout(tiny_gpt):
    new_sd = tiny_gpt.state_dict()
    patched = migrations.patch_state_dict_layout(dict(new_sd), tiny_gpt.config, log=lambda msg: None)
    assert patched.keys() == new_sd.keys()
    for key in new_sd:
        assert torch.equal(patched[key], new_sd[key])


def _fake_config(n_layer):
    return GPTConfig(sequence_len=8, vocab_size=16, n_layer=n_layer, n_head=1, n_kv_head=1, n_embd=4, window_pattern="L")


def _old_optimizer_data(n_layer, n_value_embedding=3, n_smear=3, matrix_shape_counts=(2, 3)):
    """Build a synthetic OLD-layout optimizer_data dict matching MuonAdamW's flat-index
    state_dict() shape, with groups in GPT.setup_optimizer's policy order:
    [unembedding(1), embedding(1), value_embedding(n_value_embedding),
     resid_scalar(1, an [n_layer]-shaped tensor), x0_scalar(1, ditto), smear(n_smear),
     *matrix(one group per shape)]."""
    idx = 0
    groups = []
    state = {}

    def add_group(kind, count):
        nonlocal idx
        params = list(range(idx, idx + count))
        for p in params:
            state[p] = {"step": 7, "exp_avg": torch.tensor(float(p)), "exp_avg_sq": torch.tensor(float(p) ** 2)}
        idx += count
        groups.append({"kind": kind, "lr": 0.1, "params": params})
        return params

    add_group("adamw", 1)  # 0: unembedding
    add_group("adamw", 1)  # 1: embedding
    add_group("adamw", n_value_embedding)  # 2: value_embedding
    resid_params = add_group("adamw", 1)  # 3: resid_scalar
    state[resid_params[0]] = {"step": 7, "exp_avg": torch.arange(n_layer, dtype=torch.float32), "exp_avg_sq": torch.arange(n_layer, dtype=torch.float32) ** 2}
    x0_params = add_group("adamw", 1)  # 4: x0_scalar
    state[x0_params[0]] = {"step": 7, "exp_avg": torch.arange(n_layer, dtype=torch.float32) * 10, "exp_avg_sq": torch.arange(n_layer, dtype=torch.float32) * 100}
    add_group("adamw", n_smear)  # 5: smear
    for count in matrix_shape_counts:  # 6+: matrix, one group per shape
        add_group("muon", count)
    return {"state": state, "param_groups": groups}


def test_patch_optimizer_state_dict_splits_resid_and_x0_groups():
    n_layer = 4
    config = _fake_config(n_layer)
    data = _old_optimizer_data(n_layer)
    resid_old_idx = data["param_groups"][3]["params"][0]
    x0_old_idx = data["param_groups"][4]["params"][0]
    resid_exp_avg_old = data["state"][resid_old_idx]["exp_avg"].clone()
    x0_exp_avg_old = data["state"][x0_old_idx]["exp_avg"].clone()
    later_group_old_params = [p for g in data["param_groups"][5:] for p in g["params"]]
    original_later_state = {p: data["state"][p] for p in later_group_old_params}

    patched = migrations.patch_optimizer_state_dict(data, config, log=lambda msg: None)

    resid_group = patched["param_groups"][3]
    x0_group = patched["param_groups"][4]
    assert len(resid_group["params"]) == n_layer
    assert len(x0_group["params"]) == n_layer
    for i, p in enumerate(resid_group["params"]):
        assert patched["state"][p]["exp_avg"].item() == resid_exp_avg_old[i].item()
        assert patched["state"][p]["step"] == 7
    for i, p in enumerate(x0_group["params"]):
        assert patched["state"][p]["exp_avg"].item() == x0_exp_avg_old[i].item()
        assert patched["state"][p]["step"] == 7

    # Groups after x0_scalar keep their param count, and their state is untouched (just moved to
    # renumbered flat indices) -- migration must not corrupt anything beyond the split groups.
    later_group_new_params = [p for g in patched["param_groups"][5:] for p in g["params"]]
    assert len(later_group_new_params) == len(later_group_old_params)
    for old_p, new_p in zip(later_group_old_params, later_group_new_params):
        assert patched["state"][new_p] is original_later_state[old_p]

    # Groups before resid_scalar are untouched entirely (same flat indices).
    assert patched["param_groups"][0]["params"] == data["param_groups"][0]["params"]
    assert patched["param_groups"][1]["params"] == data["param_groups"][1]["params"]
    assert patched["param_groups"][2]["params"] == data["param_groups"][2]["params"]


def test_patch_optimizer_state_dict_is_idempotent():
    n_layer = 4
    config = _fake_config(n_layer)
    data = _old_optimizer_data(n_layer)
    once = migrations.patch_optimizer_state_dict(data, config, log=lambda msg: None)
    twice = migrations.patch_optimizer_state_dict(once, config, log=lambda msg: None)
    assert twice is once  # already-new-layout early-return path returns the same object
