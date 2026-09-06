"""
The materialized-tree config for composed architectures. See docs/architecture.md's "Composed
architectures" section.

Every component (embedding, block, unembedding, composer, shared component like RoPE) is one
ComponentSpec: its type under a "#type" key, its already-concrete constructor kwargs as siblings.
The "#" sigil can't collide with a parameter name (no Python identifier contains it), so no
namespacing wrapper is needed. A composer is a component like any other -- config.body's own
"#type" names the composer, and its per-layer block list is just one of its params (conventionally
named "blocks") -- so nothing in this schema privileges "a stack of blocks": a composer with
several block lists, or one nesting another composer, is expressible without a schema change.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from nanochat.model.base import BaseModelConfig

TYPE_KEY = "#type"


@dataclass
class ComponentSpec:
    type: str
    params: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {TYPE_KEY: self.type, **{k: _unresolve(v) for k, v in self.params.items()}}

    @classmethod
    def from_dict(cls, d: dict) -> "ComponentSpec":
        d = dict(d)
        type_ = d.pop(TYPE_KEY)
        return cls(type=type_, params={k: _resolve(v) for k, v in d.items()})


def _resolve(value):
    """Recursively turn any dict carrying "#type" -- at any depth, including inside a list -- into
    a ComponentSpec. This one rule is what makes nested composers and multiple block lists work
    without the schema knowing about them in advance."""
    if isinstance(value, dict) and TYPE_KEY in value:
        return ComponentSpec.from_dict(value)
    if isinstance(value, list):
        return [_resolve(v) for v in value]
    return value


def _unresolve(value):
    if isinstance(value, ComponentSpec):
        return value.to_dict()
    if isinstance(value, list):
        return [_unresolve(v) for v in value]
    return value


def _count_blocks(spec) -> int:
    """Total number of block ComponentSpecs reachable under `spec`, however the composer(s)
    arrange them: sums every list bound to a "blocks" param, recursing into nested composers.
    Backs ComposedConfig.n_layer below -- a convention (composers name their per-layer list
    "blocks"), not a schema requirement."""
    if not isinstance(spec, ComponentSpec):
        return 0
    total = 0
    for key, value in spec.params.items():
        if key == "blocks" and isinstance(value, list):
            total += len(value)
        elif isinstance(value, ComponentSpec):
            total += _count_blocks(value)
        elif isinstance(value, list):
            total += sum(_count_blocks(v) for v in value)
    return total


@dataclass
class ComposedConfig(BaseModelConfig):
    """A materialized architecture tree. `reference` optionally records how this config was
    produced (`{"preset": name, "kwargs": {...}}`, stamped by nanochat.model.composed.presets) so
    scripts/base_train.py's muP d12 scaling-law reference (nanochat.scaling.derive_training_plan)
    can be re-derived at a different depth without re-deriving the whole tree by hand; a config
    with no `reference` (e.g. a from-scratch hand-written tree) needs --d-ref-scaling-params
    instead -- see nanochat.model.composed.presets.resolve_composed_reference_config.

    Only truly global, uniform-across-layers values live at this level: n_embd is the residual-
    stream width every block reads from and writes to, so it can't vary per layer without breaking
    the residual connection itself. Everything genuinely per-layer lives inside `body`."""
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_embd: int = 768
    pad_vocab_size_to: int = 64
    reference: dict | None = None
    shared: dict = field(default_factory=dict)   # name -> ComponentSpec, e.g. {"rope": ...}
    input: "ComponentSpec | None" = None
    body: "ComponentSpec | None" = None
    output: "ComponentSpec | None" = None

    @property
    def padded_vocab_size(self) -> int:
        p = self.pad_vocab_size_to
        return ((self.vocab_size + p - 1) // p) * p

    @property
    def n_layer(self) -> int:
        """Total block count, however config.body arranges them -- a derived read-only property
        (not a dataclass field) so existing code that reads model.config.n_layer
        (scripts/chat_sft.py, scripts/chat_rl.py, tests/test_model_common.py) keeps working
        unchanged; --arch-opt n_layer=... still correctly rejects since it isn't a real field."""
        return _count_blocks(self.body)

    def to_dict(self) -> dict:
        return {
            "sequence_len": self.sequence_len, "vocab_size": self.vocab_size, "n_embd": self.n_embd,
            "pad_vocab_size_to": self.pad_vocab_size_to, "reference": self.reference,
            "shared": {k: v.to_dict() for k, v in self.shared.items()},
            "input": self.input.to_dict(), "body": self.body.to_dict(), "output": self.output.to_dict(),
            "arch": type(self).arch,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ComposedConfig":
        d = dict(d)
        d.pop("arch", None)
        return cls(
            sequence_len=d["sequence_len"], vocab_size=d["vocab_size"], n_embd=d["n_embd"],
            pad_vocab_size_to=d.get("pad_vocab_size_to", 64), reference=d.get("reference"),
            shared={k: ComponentSpec.from_dict(v) for k, v in d.get("shared", {}).items()},
            input=ComponentSpec.from_dict(d["input"]),
            body=ComponentSpec.from_dict(d["body"]),
            output=ComponentSpec.from_dict(d["output"]),
        )
