"""
Architecture registry: maps an "arch" name to its (config class, model class) pair.

Checkpoints store their architecture name in meta["model_config"]["arch"] (via
BaseModelConfig.to_dict()). checkpoint_manager.build_model uses this registry to reconstruct the
right classes without importing any specific architecture. Old checkpoints saved before this
registry existed have no "arch" key; config_from_dict defaults that case to "gpt".
"""

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
    """Reconstruct a BaseModelConfig subclass instance from a flat dict (as produced by
    BaseModelConfig.to_dict() or loaded from checkpoint meta json). Missing "arch" defaults to
    "gpt" for checkpoints saved before architectures were pluggable."""
    d = dict(d)  # don't mutate the caller's dict
    arch = d.pop("arch", "gpt")
    config_cls = get_config_class(arch)
    return config_cls(**d)
