"""
Low-code -> materialized adapter expansion: turns a compact "attach LoRA/DoRA here" request into
the concrete modelcore.AdapterSpec list (and frozen-FQN list) that modelcore.ModelConfig.adapters/
frozen actually hold. Same split as nanochat.architectures.presets: the derivation rule ("every
attn.c_q across every layer") lives here, host-side; modelcore itself only ever consumes an
already-materialized list of concrete FQNs (see modelcore/AGENTS.md's "a config tree carries only
concrete values, never a derivation rule" invariant, and llmllab/docs/subsystem-conventions.md).

This is also where the family's move toward a config-first "DSL" for data prep and training plans
is headed: expand_adapters(model, request) is a low-code -> materialized expansion step in exactly
the shape a future preset registry, CLI flag, or UI would call -- request in, a concrete list of
already-decided values out. --adapters on scripts/base_train.py and scripts/chat_sft.py accepts
this request form directly; a fully materialized adapters/frozen pair (e.g. hand-edited JSON) is
passed straight through unchanged, the same "preset name OR a materialized JSON tree" duality
nanochat.architectures.presets.resolve_model_config already established for --model-config.

A hand-written config never needs this module at all -- it can always set
ModelConfig.adapters/.frozen directly with modelcore.AdapterSpec.
"""
from modelcore import AdapterSpec
from modelcore.components.linear import Linear


def list_linear_targets(model) -> list[str]:
    """Every module FQN in a built (or meta-device) model that resolves to a modelcore Linear --
    i.e. every legal AdapterSpec.target for this model. For discovery: a caller editing a config
    by hand needs to know what's there before it can target it (see scripts/model_info.py's
    --list-targets)."""
    return [name for name, module in model.named_modules() if isinstance(module, Linear)]


def expand_adapters(model, request: dict) -> tuple[list, list]:
    """Turns a low-code request into (adapters, frozen) ready to assign onto a ModelConfig's own
    fields. `model` must already be built from the *base* config (no adapters yet, meta-device is
    fine) -- enumerating "every layer" and checking a target is really a Linear both need the real
    tree, the same reason modelcore.config.validate's adapter checks need one (see
    modelcore/config/validate.py's _validate_adapters).

    `request` keys:
      name: str -- the adapter's stable handle (AdapterSpec.name), required.
      method: "lora" | "dora" (a modelcore.peft.registry name) -- default "lora".
      r, alpha, dropout: passed straight through as the delta's own params (only the ones given).
      targets: submodule names *relative to a block* (e.g. ["attn.c_q", "attn.c_v"]) -- not full
        FQNs; this is what lets one request cover every layer without enumerating
        body.blocks.0..N-1 by hand. Default: every attention Linear (c_q/c_k/c_v/c_proj), the
        conventional LoRA target set.
      layers: "all" (default) or an explicit list of block indices.
      freeze_base: bool, default True -- adds "body" to the returned `frozen` list.

    Assumes model.config.body's per-layer list lives at the "blocks" attribute name -- true for
    both cataloged composers today (modelcore.composers.stack.StackComposer,
    modelcore.composers.backout.BackoutComposer); a future composer with a different layout would
    need its own expansion helper, same as modelcore.config.spec._count_blocks's own convention."""
    name = request["name"]
    method = request.get("method", "lora")
    params = {k: request[k] for k in ("r", "alpha", "dropout") if k in request and request[k] is not None}
    targets = request.get("targets", ["attn.c_q", "attn.c_k", "attn.c_v", "attn.c_proj"])
    layers = request.get("layers", "all")
    freeze_base = request.get("freeze_base", True)

    n_layer = model.config.n_layer
    layer_indices = range(n_layer) if layers == "all" else list(layers)
    existing = set(list_linear_targets(model))

    adapters = []
    for i in layer_indices:
        for rel_target in targets:
            fqn = f"body.blocks.{i}.{rel_target}"
            if fqn not in existing:
                raise ValueError(
                    f"adapter target {fqn!r} does not resolve to a Linear in this model "
                    f"(from relative target {rel_target!r} at layer {i}) -- see "
                    f"scripts/model_info.py --list-targets for what's actually there"
                )
            adapters.append(AdapterSpec(target=fqn, name=name, type=method, params=dict(params)))

    frozen = ["body"] if freeze_base else []
    return adapters, frozen


def expand_adapters_for_config(config, request: dict) -> tuple[list, list]:
    """expand_adapters, but for a caller that has a modelcore.ModelConfig (e.g. read straight off
    an existing checkpoint's meta.json) and no built Model yet -- builds its own throwaway
    meta-device probe (shapes only, no real weights allocated) rather than requiring the caller to
    build one first. See scripts/chat_sft.py's --adapters wiring: the base checkpoint's model is
    already loaded with real weights by the time an adapter request needs expanding, but
    re-deriving (adapters, frozen) only ever needs shapes, so a second real load is wasteful --
    this is what lets that stay meta-device-cheap instead."""
    import torch
    from modelcore.model import Model

    with torch.device("meta"):
        probe = Model(config)
    return expand_adapters(probe, request)
