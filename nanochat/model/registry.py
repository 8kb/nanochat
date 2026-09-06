"""
Architecture registry: maps an "arch" name to its (config class, model class) pair.

Checkpoints store their architecture name in meta["model_config"]["arch"] (via
BaseModelConfig.to_dict()). checkpoint_manager.build_model uses this registry to reconstruct the
right classes without importing any specific architecture. Old checkpoints saved before this
registry existed have no "arch" key; config_from_dict defaults that case to "gpt".
"""

import ast
import dataclasses

_MODEL_REGISTRY = {}  # arch name -> (config_cls, model_cls)


def register_model(name, config_cls):
    """Class decorator: registers model_cls under `name`, and stamps config_cls.arch = name."""
    def decorator(model_cls):
        config_cls.arch = name
        _MODEL_REGISTRY[name] = (config_cls, model_cls)
        return model_cls
    return decorator


def _lookup(arch):
    if arch not in _MODEL_REGISTRY:
        raise ValueError(f"Unknown architecture {arch!r}. Registered architectures: {sorted(_MODEL_REGISTRY)}")
    return _MODEL_REGISTRY[arch]


def get_model_class(arch):
    return _lookup(arch)[1]


def get_config_class(arch):
    return _lookup(arch)[0]


def config_from_dict(d):
    """Reconstruct a BaseModelConfig subclass instance from a dict (as produced by
    BaseModelConfig.to_dict() or loaded from checkpoint meta json). Missing "arch" defaults to
    "gpt" for checkpoints saved before architectures were pluggable. Dispatches to the resolved
    class's own from_dict (default: a flat kwargs splat; nanochat.model.composed.spec.ComposedConfig
    overrides it for its nested tree shape)."""
    d = dict(d)  # don't mutate the caller's dict
    arch = d.pop("arch", "gpt")
    config_cls = get_config_class(arch)
    return config_cls.from_dict(d)


def apply_arch_opts(config, opt_strings):
    """Apply CLI --arch-opt KEY=VALUE overrides onto a BaseModelConfig instance, returning a new
    instance. Lets a script's fixed from_depth(...) kwarg set (--depth/--aspect-ratio/--head-dim/
    ...) stay generic while still reaching architecture-specific fields it doesn't know about
    (e.g. LlamaKVShareConfig.kv_share_frac) -- see scripts/base_train.py and
    scripts/model_info.py, the two callers. Values are parsed with ast.literal_eval so numbers/
    bools/strings all work without extra per-field CLI plumbing. An unknown key raises rather
    than silently no-op'ing a typo."""
    if not opt_strings:
        return config
    field_names = {f.name for f in dataclasses.fields(config)}
    opts = {}
    for raw in opt_strings:
        assert "=" in raw, f"--arch-opt must be KEY=VALUE, got {raw!r}"
        key, _, value = raw.partition("=")
        assert key in field_names, (
            f"--arch-opt {key!r} is not a field of {type(config).__name__}; "
            f"valid fields: {sorted(field_names)}"
        )
        opts[key] = ast.literal_eval(value)
    return dataclasses.replace(config, **opts)
