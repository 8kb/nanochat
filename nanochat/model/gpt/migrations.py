"""
Backward-compatibility patches for GPT checkpoints saved before a config field, parameter, or
module layout existed. Dispatched by nanochat.checkpoint_manager via GPT.patch_config_dict /
GPT.patch_state_dict, and by scripts/base_train.py + scripts/chat_sft.py via
GPT.patch_optimizer_state_dict (see nanochat.model.base.BaseModel for the no-op defaults other
architectures get).
"""

import torch


def patch_missing_config_keys(model_config_kwargs, log=lambda msg: None):
    """Add default values for new config keys missing in old checkpoints."""
    # Old models were trained with full context (no sliding window)
    if "window_pattern" not in model_config_kwargs:
        model_config_kwargs["window_pattern"] = "L"
        log("Patching missing window_pattern in model config to 'L'")
    return model_config_kwargs


def patch_missing_state_keys(model_data, model_config, log=lambda msg: None):
    """Add default values for new parameters that may be missing in old (pre-Stage-2, flat
    layout) checkpoints. Runs before patch_state_dict_layout, which does the rename to per-block
    keys. A state dict already in the new layout has nothing to backfill here -- a checkpoint
    predating resid_lambda/x0_lambda entirely necessarily also predates the Stage 2 rename."""
    if "blocks.0.resid_lambda" in model_data:
        return model_data  # already new layout
    n_layer = model_config.n_layer
    # resid_lambdas defaults to 1.0 (identity scaling)
    if "resid_lambdas" not in model_data:
        model_data["resid_lambdas"] = torch.ones(n_layer)
        log("Patching missing resid_lambdas in model data to 1.0")
    # x0_lambdas defaults to 0.0 (disabled)
    if "x0_lambdas" not in model_data:
        model_data["x0_lambdas"] = torch.zeros(n_layer)
        log("Patching missing x0_lambdas in model data to 0.0")
    return model_data


def patch_state_dict_layout(model_data, model_config, log=lambda msg: None):
    """Stage 2: GPT's parameters moved from a flat, mostly-top-level layout into the modules that
    now own them (embedding/unembedding/per-block). Renames old keys to their new locations; a
    no-op if the state dict is already in the new layout.

        transformer.wte.weight        -> embedding.wte.weight
        smear_gate.weight             -> embedding.smear.gate.weight
        smear_lambda                  -> embedding.smear.lambda_
        lm_head.weight                -> unembedding.lm_head.weight
        resid_lambdas[i]              -> blocks.{i}.resid_lambda
        x0_lambdas[i]                 -> blocks.{i}.x0_lambda
        value_embeds.{i}.weight       -> blocks.{i}.attn.value_embed.weight
        transformer.h.{i}.*           -> blocks.{i}.*
        backout_lambda                -> unchanged
    """
    if "transformer.wte.weight" not in model_data:
        return model_data  # already new layout

    n_layer = model_config.n_layer
    renamed = {
        "embedding.wte.weight": model_data.pop("transformer.wte.weight"),
        "embedding.smear.gate.weight": model_data.pop("smear_gate.weight"),
        "embedding.smear.lambda_": model_data.pop("smear_lambda"),
        "unembedding.lm_head.weight": model_data.pop("lm_head.weight"),
    }
    resid_lambdas = model_data.pop("resid_lambdas")
    x0_lambdas = model_data.pop("x0_lambdas")
    for i in range(n_layer):
        # .clone() so each per-block scalar owns independent storage rather than being a view
        # into the original [n_layer] tensor -- load_state_dict(assign=True) would otherwise
        # leave every block's parameter aliased to the same underlying storage.
        renamed[f"blocks.{i}.resid_lambda"] = resid_lambdas[i].clone()
        renamed[f"blocks.{i}.x0_lambda"] = x0_lambdas[i].clone()
    for key in [k for k in model_data if k.startswith("transformer.h.")]:
        renamed[key.replace("transformer.h.", "blocks.", 1)] = model_data.pop(key)
    for key in [k for k in model_data if k.startswith("value_embeds.")]:
        _, i, rest = key.split(".", 2)
        renamed[f"blocks.{i}.attn.value_embed.{rest}"] = model_data.pop(key)

    log(f"Patching state dict layout: renamed {len(renamed)} keys for the Stage 2 module restructure")
    model_data.update(renamed)
    return model_data


def patch_optimizer_state_dict(optimizer_data, model_config, log=lambda msg: None):
    """Stage 2: resid_lambdas and x0_lambdas were each a single [n_layer] parameter (their
    optimizer param_group held exactly one flat parameter index); they are now n_layer
    independent per-block scalars. Splits that group's per-index optimizer state
    (exp_avg/exp_avg_sq, carrying step through unchanged) into n_layer entries and renumbers
    every later flat parameter index to make room. A no-op if already in the new layout.

    Relies on GPT.setup_optimizer's fixed policy order: groups are
    [unembedding, embedding, value_embedding, resid_scalar, x0_scalar, smear, *matrix_by_shape].
    """
    n_layer = model_config.n_layer
    groups = optimizer_data["param_groups"]
    resid_group, x0_group = groups[3], groups[4]
    if len(resid_group["params"]) != 1:
        return optimizer_data  # already new layout
    assert len(x0_group["params"]) == 1, (
        f"expected the old layout's x0_scalar group to hold exactly one parameter, "
        f"got {len(x0_group['params'])}"
    )
    old_state = optimizer_data["state"]
    resid_old_idx, x0_old_idx = resid_group["params"][0], x0_group["params"][0]
    assert x0_old_idx == resid_old_idx + 1, (
        f"expected resid_scalar ({resid_old_idx}) and x0_scalar ({x0_old_idx}) to be adjacent "
        "flat indices"
    )

    resid_new_indices = list(range(resid_old_idx, resid_old_idx + n_layer))
    x0_new_indices = list(range(resid_old_idx + n_layer, resid_old_idx + 2 * n_layer))
    grow_by = 2 * (n_layer - 1)  # each of the two groups grows from 1 index to n_layer

    def remap(old_idx):
        if old_idx < resid_old_idx:
            return old_idx
        assert old_idx > x0_old_idx, f"unexpected flat index {old_idx} between resid ({resid_old_idx}) and x0 ({x0_old_idx})"
        return old_idx + grow_by

    def split(st, new_indices):
        # exp_avg/exp_avg_sq are per-element moments (same shape as the parameter, i.e. [n_layer]
        # here); step is a plain python counter shared identically by every split-off scalar.
        split_state = {}
        for i, new_idx in enumerate(new_indices):
            split_state[new_idx] = {
                k: (v[i].clone() if torch.is_tensor(v) and tuple(v.shape) == (n_layer,) else v)
                for k, v in st.items()
            }
        return split_state

    new_state = {}
    for old_idx, st in old_state.items():
        if old_idx == resid_old_idx:
            new_state.update(split(st, resid_new_indices))
        elif old_idx == x0_old_idx:
            new_state.update(split(st, x0_new_indices))
        else:
            new_state[remap(old_idx)] = st

    new_groups = []
    for gi, g in enumerate(groups):
        new_g = dict(g)
        if gi == 3:
            new_g["params"] = resid_new_indices
        elif gi == 4:
            new_g["params"] = x0_new_indices
        else:
            new_g["params"] = [remap(p) for p in g["params"]]
        new_groups.append(new_g)

    log(f"Patching optimizer state dict: split resid/x0 scalar groups from 1 to {n_layer} entries each")
    return {"state": new_state, "param_groups": new_groups}
