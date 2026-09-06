"""
Composed architectures: a materialized tree of components (embedding / body-composer / unembedding)
instead of one hardcoded Python class per architecture. See docs/architecture.md's "Composed
architectures" section.

Importing this module registers "composed" with nanochat.model.registry (via
nanochat.model.composed.model) and populates the component catalog (via
nanochat.model.composed.catalog) that nanochat.model.composed.registry.build_component resolves
"#type" names against.
"""
from nanochat.model.composed.spec import ComponentSpec, ComposedConfig
from nanochat.model.composed.model import ComposedModel
from nanochat.model.composed.presets import (
    expand_preset, resolve_composed_config, resolve_composed_reference_config, PRESETS,
)

__all__ = [
    "ComponentSpec", "ComposedConfig", "ComposedModel",
    "expand_preset", "resolve_composed_config", "resolve_composed_reference_config", "PRESETS",
]
