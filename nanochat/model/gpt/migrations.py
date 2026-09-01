"""
Backward-compatibility patches for GPT checkpoints saved before a config field or parameter
existed. Dispatched by nanochat.checkpoint_manager via GPT.patch_config_dict /
GPT.patch_state_dict (see nanochat.model.base.BaseModel for the no-op default other
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
    """Add default values for new parameters that may be missing in old checkpoints."""
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
