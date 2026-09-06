"""
Migrates an old-format checkpoint (any generation before modelcore -- flat per-architecture
configs, pre-Stage-2 module layouts, missing fields) into modelcore's materialized ModelConfig +
state dict. Every hand-written architecture class (GPT, Llama, LlamaKVShare, LlamaKVShareWin) that
used to interpret a flat config directly is gone (see docs/upstream-sync.md); this module is where
that interpretation now happens instead, once, at load time -- nothing in modelcore or in a fresh
training run ever needs it.

A config dict with no "format" key predates modelcore entirely and always goes through the full
chain below; migrate_checkpoint() is a no-op pass-through for a config dict that already has one.
"""
import torch

from modelcore import ModelConfig

from nanochat.architectures.presets import assemble_gpt, assemble_plain

_FLAT_ARCH_EXPANDERS = {
    "gpt": lambda cfg: _materialize_gpt(cfg),
    "llama": lambda cfg: _materialize_plain(cfg),
    "llama_kvshare": lambda cfg: _materialize_plain(cfg),
    "llama_kvshare_win": lambda cfg: _materialize_plain(cfg),
}


# -----------------------------------------------------------------------------
# Config: flat legacy dict -> modelcore.ModelConfig

def patch_missing_config_keys(model_config_kwargs, log=lambda msg: None):
    """Backfill fields missing from an old flat config dict, before it's read for materialization
    below. Old models were trained with full context (no sliding window)."""
    if "window_pattern" not in model_config_kwargs:
        model_config_kwargs["window_pattern"] = "L"
        log("Patching missing window_pattern in model config to 'L'")
    return model_config_kwargs


def _materialize_gpt(cfg: dict) -> ModelConfig:
    """Reconstructs the exact tree expand_gpt(depth=cfg['n_layer'], ...) would have produced, but
    from the checkpoint's own *stored* per-model fields (n_layer, n_head, n_embd, window_pattern)
    rather than re-deriving them from --depth/--aspect-ratio/--head-dim -- lossless even for a run
    that used --arch-opt to diverge from the muP dial's defaults. `reference` is left unset: it
    can't be reconstructed with certainty from stored fields alone, and a wrong guess is worse
    than the existing --d-ref-scaling-params fallback (see presets.resolve_reference_config)."""
    head_dim = cfg["n_embd"] // cfg["n_head"]
    return assemble_gpt(cfg["n_layer"], cfg["n_head"], cfg["n_kv_head"], cfg["n_embd"], head_dim,
                         cfg["sequence_len"], cfg["vocab_size"], cfg["window_pattern"])


def _materialize_plain(cfg: dict) -> ModelConfig:
    head_dim = cfg["n_embd"] // cfg["n_head"]
    return assemble_plain(cfg["n_layer"], cfg["n_head"], cfg["n_kv_head"], cfg["n_embd"], head_dim,
                           cfg["sequence_len"], cfg["vocab_size"], cfg["window_pattern"],
                           kv_share_frac=cfg.get("kv_share_frac"))


def migrate_config(config_dict: dict) -> ModelConfig:
    """Any config dict with no "format" key -- every checkpoint saved before modelcore existed,
    regardless of which of the four flat architectures produced it (missing "arch" defaults to
    "gpt", the only architecture old enough to predate that key too)."""
    if "format" in config_dict:
        return ModelConfig.from_dict(config_dict)  # already current; nothing to migrate
    cfg = patch_missing_config_keys(dict(config_dict))
    arch = cfg.pop("arch", "gpt")
    if arch not in _FLAT_ARCH_EXPANDERS:
        raise ValueError(f"Unknown legacy architecture {arch!r}; cannot migrate. Known: {sorted(_FLAT_ARCH_EXPANDERS)}")
    return _FLAT_ARCH_EXPANDERS[arch](cfg)


# -----------------------------------------------------------------------------
# State dict: pre-Stage-2 flat layout, then new-layout key prefixing

def patch_missing_state_keys(model_data, n_layer, log=lambda msg: None):
    """Backfill parameters missing in a pre-Stage-2, flat-layout checkpoint that predates
    resid_lambda/x0_lambda entirely. Runs before the layout rename below -- a checkpoint missing
    these necessarily also predates the module restructure that renamed everything else."""
    if "blocks.0.resid_lambda" in model_data:
        return model_data  # already new layout
    if "resid_lambdas" not in model_data:
        model_data["resid_lambdas"] = torch.ones(n_layer)
        log("Patching missing resid_lambdas in model data to 1.0")
    if "x0_lambdas" not in model_data:
        model_data["x0_lambdas"] = torch.zeros(n_layer)
        log("Patching missing x0_lambdas in model data to 0.0")
    return model_data


def patch_gpt_state_dict_layout(model_data, n_layer, log=lambda msg: None):
    """Stage 2: GPT's parameters moved from a flat, mostly-top-level layout into the modules that
    now own them. A no-op if already in the new (pre-modelcore) layout.

        transformer.wte.weight   -> embedding.wte.weight
        smear_gate.weight        -> embedding.smear.gate.weight
        smear_lambda             -> embedding.smear.lambda_
        lm_head.weight           -> unembedding.lm_head.weight
        resid_lambdas[i]         -> blocks.{i}.resid_lambda
        x0_lambdas[i]            -> blocks.{i}.x0_lambda
        value_embeds.{i}.weight  -> blocks.{i}.attn.value_embed.weight
        transformer.h.{i}.*      -> blocks.{i}.*
        backout_lambda           -> unchanged (renamed to body.backout_lambda below instead)
    """
    if "transformer.wte.weight" not in model_data:
        return model_data  # already new layout

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


def patch_body_prefix(model_data, arch: str, log=lambda msg: None):
    """The materialized tree wraps what used to be top-level under `body.` (a real composer
    submodule, not a bare list) -- matches the remap tests/test_modelcore.py's golden cross-check
    verifies. A no-op if already prefixed."""
    if any(k.startswith("body.") for k in model_data):
        return model_data  # already current
    remapped = {}
    for key, value in model_data.items():
        if key == "backout_lambda":
            remapped["body.backout_lambda"] = value
        elif key.startswith("blocks."):
            remapped[f"body.{key}"] = value
        else:
            remapped[key] = value
    log(f"Patching state dict: prefixed {sum(1 for k in model_data if k.startswith('blocks.') or k == 'backout_lambda')} keys under body.")
    return remapped


def migrate_state_dict(model_data: dict, config_dict: dict, arch: str, n_layer: int, log=lambda msg: None) -> dict:
    if "format" in config_dict:
        return model_data  # already current; nothing to migrate
    if arch == "gpt":
        model_data = patch_missing_state_keys(model_data, n_layer, log=log)
        model_data = patch_gpt_state_dict_layout(model_data, n_layer, log=log)
    model_data = patch_body_prefix(model_data, arch, log=log)
    return model_data


# -----------------------------------------------------------------------------
# Optimizer state: resid/x0 scalar split, and gpt's backout_lambda role rename

def _patch_resid_x0_split(optimizer_data, n_layer, log=lambda msg: None):
    """Stage 2: resid_lambdas/x0_lambdas were each a single [n_layer] parameter (one flat
    optimizer index); they are now n_layer independent per-block scalars. Splits that group's
    per-index state (exp_avg/exp_avg_sq, carrying step through unchanged) into n_layer entries and
    renumbers every later flat index to make room. A no-op if already in the new layout.

    Relies on the old GPT policy's fixed order: groups are
    [unembedding, embedding, value_embedding, resid_scalar, x0_scalar, smear, *matrix_by_shape]."""
    groups = optimizer_data["param_groups"]
    resid_group, x0_group = groups[3], groups[4]
    if len(resid_group["params"]) != 1:
        return optimizer_data  # already new layout
    assert len(x0_group["params"]) == 1
    old_state = optimizer_data["state"]
    resid_old_idx, x0_old_idx = resid_group["params"][0], x0_group["params"][0]
    assert x0_old_idx == resid_old_idx + 1, "expected resid_scalar and x0_scalar to be adjacent flat indices"

    resid_new_indices = list(range(resid_old_idx, resid_old_idx + n_layer))
    x0_new_indices = list(range(resid_old_idx + n_layer, resid_old_idx + 2 * n_layer))
    grow_by = 2 * (n_layer - 1)

    def remap(old_idx):
        if old_idx < resid_old_idx:
            return old_idx
        assert old_idx > x0_old_idx
        return old_idx + grow_by

    def split(st, new_indices):
        return {
            new_idx: {k: (v[i].clone() if torch.is_tensor(v) and tuple(v.shape) == (n_layer,) else v)
                      for k, v in st.items()}
            for i, new_idx in enumerate(new_indices)
        }

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


def _split_backout_lambda_from_smear(optimizer_data, log=lambda msg: None):
    """The old GPT.PARAM_ROLES mapped backout_lambda to role "smear" (grouped alongside
    embedding.smear's gate/lambda_); modelcore's BackoutComposer gives it its own role,
    "backout_scalar", positioned immediately after "smear" in ModelManager.create_optimizer's
    policy order. Because flat optimizer indices are assigned purely by walking every group's
    params in order, backout_lambda's flat index doesn't move at all (still directly after
    embedding.smear.lambda_, directly before "matrix"'s first param) -- only the *group boundary*
    does, so this only needs to split one group's param list into two, with no state renumbering.
    A no-op if the smear group doesn't have the old 3-member shape (gate, lambda_, backout_lambda)."""
    groups = optimizer_data["param_groups"]
    smear_group = groups[5]
    if len(smear_group["params"]) != 3:
        return optimizer_data  # already new layout (or no backout_lambda to split out)
    gate_idx, lambda_idx, backout_idx = smear_group["params"]
    new_smear = dict(smear_group)
    new_smear["params"] = [gate_idx, lambda_idx]
    # Old "smear" and new "backout_scalar" policies share identical numeric hyperparameters
    # (lr=0.2, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0) by construction -- copying the
    # group's own saved hparams is exactly correct, not a coincidental shortcut.
    backout_group = dict(smear_group)
    backout_group["params"] = [backout_idx]
    new_groups = groups[:5] + [new_smear, backout_group] + groups[6:]
    log("Patching optimizer state dict: split backout_lambda out of the smear group into its own backout_scalar group")
    return {"state": optimizer_data["state"], "param_groups": new_groups}


def migrate_optimizer_state(optimizer_data: dict, config_dict: dict, arch: str, n_layer: int,
                             log=lambda msg: None) -> dict:
    if "format" in config_dict or arch != "gpt":
        return optimizer_data  # only gpt-arch checkpoints ever need either fix below
    optimizer_data = _patch_resid_x0_split(optimizer_data, n_layer, log=log)
    optimizer_data = _split_backout_lambda_from_smear(optimizer_data, log=log)
    return optimizer_data


# -----------------------------------------------------------------------------
# Entry point combining the config + state-dict migrations

def migrate_checkpoint(config_dict: dict, model_data: dict, log=lambda msg: None) -> tuple[ModelConfig, dict]:
    """config_dict is the raw dict as read from a checkpoint's meta.json "model_config" key;
    model_data is the raw model state dict. Returns (ModelConfig, migrated state dict). See
    migrate_optimizer_state for the separate optimizer-state migration (loaded independently, so
    called separately by whoever resumes optimizer state)."""
    arch = config_dict.get("arch", "gpt") if "format" not in config_dict else None
    config = migrate_config(config_dict)
    model_data = migrate_state_dict(model_data, config_dict, arch or "gpt", config.n_layer, log=log)
    return config, model_data
