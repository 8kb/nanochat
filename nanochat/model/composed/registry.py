"""
Component registry for composed architectures (nanochat.model.composed): maps a type name -- the
"#type" field of a nanochat.model.composed.spec.ComponentSpec -- to the nn.Module class that
implements it, plus which build-context values (nanochat.model.composed.model.ComposedModel's
ctx: derived globals like n_embd/padded_vocab_size, and built `shared` components like rope) it
needs injected as constructor kwargs. One flat namespace (no "kind" of embedding/block/composer/
shared) -- the parent deciding what a slot means is enough, and type names are globally unique by
convention. Separate from nanochat.model.registry, which maps a top-level *architecture* name
("gpt", "composed", ...) to a (config, model) class pair.
"""

_COMPONENT_REGISTRY = {}  # type name -> (cls, needs: tuple[str, ...])


def register_component(name, needs=()):
    def decorator(cls):
        assert name not in _COMPONENT_REGISTRY, f"component type {name!r} already registered"
        _COMPONENT_REGISTRY[name] = (cls, tuple(needs))
        return cls
    return decorator


def build_component(spec, ctx):
    """Build one ComponentSpec into a real nn.Module. Any ComponentSpec-valued param (or list of
    them) is resolved into a built module first -- so e.g. a composer receives its `blocks` param
    already built -- then the component class is constructed via cls(**resolved_params, **needed).
    Because a spec's params are already materialized to the component's exact constructor kwarg
    names, no per-component adapter layer is needed."""
    from nanochat.model.composed.spec import ComponentSpec  # local import: spec.py has no need of us

    if spec.type not in _COMPONENT_REGISTRY:
        raise ValueError(f"Unknown component type {spec.type!r}. Registered: {sorted(_COMPONENT_REGISTRY)}")
    cls, needs = _COMPONENT_REGISTRY[spec.type]

    kwargs = {}
    for key, value in spec.params.items():
        if isinstance(value, ComponentSpec):
            kwargs[key] = build_component(value, ctx)
        elif isinstance(value, list) and value and all(isinstance(v, ComponentSpec) for v in value):
            kwargs[key] = [build_component(v, ctx) for v in value]
        else:
            kwargs[key] = value
    for name in needs:
        assert name in ctx, f"component {spec.type!r} needs {name!r}, not present in the build context"
        kwargs[name] = ctx[name]
    return cls(**kwargs)
