# Roadmap

This fork's goal: turn nanochat from a single-architecture speedrun codebase into a playground
for trying *different architectures*, not just different hyperparameters — while staying mergeable
with upstream (see [upstream-sync.md](upstream-sync.md)) and runnable end-to-end on a MacBook with
no CUDA (see the root [README.md](../README.md)).

Each stage below is deliberately scoped to one session's worth of work. Do not start a stage
until the previous one is merged and verified.

## Stage 1 — Extract `nanochat/model/` and open the architecture seam (done)

Split `nanochat/gpt.py` into `nanochat/model/`: a `BaseModel`/`BaseModelConfig` interface, an
architecture registry, reusable components (`Linear`, `norm`, RoPE, attention, MLP, block,
sliding-window patterns) separated from GPT-specific assembly, and generic FLOPs/KV-cache-bytes
accounting built on a per-layer `layer_specs()` descriptor. `nanochat/checkpoint_manager.py` and
`nanochat/engine.py` no longer import `GPT`/`GPTConfig` directly. See
[architecture.md](architecture.md) for the resulting contract.

## Stage 2 — push model state into its owning modules (done)

Stage 1 opened the seam but left most of `GPT`'s state (resid/x0 lambdas, value embeddings,
smear, the lm_head) sitting at the top level, hand-partitioned into optimizer groups by module
path. Stage 2 introduced three module contracts — `BaseEmbedding`, `BaseBlock`, `BaseUnembedding`
— so each concern is owned by the module responsible for it, plus a parameter-role protocol
(`nanochat/model/param_roles.py`) so `setup_optimizer()`/`num_scaling_params()` no longer depend
on a hand-maintained (and easily wrong, silently) partition of `self.parameters()`. Also gave
`RotaryEmbedding` its own module (shared across attention layers, injected rather than
GPT-internal) and made `CausalSelfAttention`/`MLP` take explicit dims instead of a config object,
so they're reusable by an architecture with a different config's field names. See
[architecture.md](architecture.md) for the resulting contracts and
[upstream-sync.md](upstream-sync.md) for the full mapping of what moved where.

## Stage 3 — prove the seam with a second architecture (done)

Added `nanochat/model/llama/`: SwiGLU MLP, plain pre-norm blocks (`PlainBlock`), no value
embeddings / smear / backout / per-layer resid-x0 lambdas — reusing `CausalSelfAttention`,
`RotaryEmbedding`, `TokenEmbedding` (smear disabled) and `LMHead` from `nanochat/model/components/`
completely unmodified. Confirms `BaseModel`/`BaseEmbedding`/`BaseBlock`/`BaseUnembedding` are real
interfaces: Llama needed zero `PARAM_ROLES` declarations anywhere (every parameter it owns is
either a `Linear` weight, defaulting to role `"matrix"`, or reused directly from a GPT component
that already declares its own roles), and `BaseModel` gained a generic `num_scaling_params()`
default (`{role: numel, ..., "total": ...}`, built on `collect_param_roles`) that Llama just
inherits — GPT overrides it to keep its legacy six-key dict.

`--arch=llama` is wired through `scripts/base_train.py` end to end (train, checkpoint, eval,
generate); `scripts/base_eval.py` gained a matching `--arch` flag. Fixed the checkpoint tag
collision the previous version of this stage description flagged (`d<depth>` regardless of
architecture): the default save tag is now arch-qualified (`{arch}_d{depth}` for anything but
`gpt`, which keeps its original naming), and `checkpoint_manager.find_largest_model` gained an
optional `arch=` filter (peeks at each candidate's `meta.json`, no directory renaming). Also fixed
a real bug found while wiring this up: `scripts/base_train.py`'s `get_scaling_params` indexed
`num_scaling_params()` by GPT's legacy dict keys, which `KeyError`'d for any architecture using
the generic default — it now reads `collect_param_roles`'s stable role names directly. The 5 (of
7) architecture-generic model tests moved from `tests/test_model_gpt.py` into
`tests/test_model_common.py`, parametrized over `["gpt", "llama"]`.

**Out of scope, intentionally**: the SFT/RL/serving pipeline (`chat_sft.py`, `chat_rl.py`,
`chat_cli.py`, `infer_bench.py`, `chat_eval.py`) does not have `--arch` flags yet and still relies
on unfiltered checkpoint auto-discovery — fine for now since there's normally only one architecture
"in flight" through that pipeline at a time, but a future stage that actually SFTs a second
architecture will need to revisit this.

## Stage 4 — attention variants

Per-layer attention and position-encoding selection in the config (mixing local/global, or
different attention types per layer). Position encoding becomes its own swappable component
(RoPE / NoPE / ALiBi) — Stage 2 already extracted `RotaryEmbedding` as an injected module rather
than wiring it straight into `CausalSelfAttention`, so this is mostly about adding alternatives
and a way to pick one, not further extraction. Add MLA (DeepSeek-style latent attention) and
differential attention as reference implementations. Generalize `nanochat.engine.KVCache` from one
fixed k/v tensor pair per layer into a per-layer state object the layer itself allocates and
manages — MLA's compressed latent cache in particular doesn't fit the current
`(n_layers, B, T, H, D)` k/v tensor shape.

## Stage 5 — depth and residual topology

`GPT._forward_trunk` (Stage 1, refined in Stage 2 to own `x0` and the block loop) is the seed for
this: weight tying across layers, looped/universal transformers, layer skipping,
multi-token-prediction (MTP) heads. Muon's shape-bucketed param grouping (now
`nanochat/model/param_roles.py:build_param_groups`, driven by `GPT.setup_optimizer`'s policy
table) needs generalizing for an architecture with tied or ragged-shaped matrix params — Stage 2's
role protocol makes this more tractable than before (a tied parameter is already a solved case at
the role-collection level, just not yet exercised by any real architecture), but the shape-based
Muon stacking itself still assumes independent, per-layer-shaped matrices.

## Stage 6 — experiment ergonomics

Config files as an alternative to pure argparse CLI flags (useful once there are several
architectures with different field sets). An architecture-comparison harness on top of `runs/`
(train N architectures at matched depth/FLOPs, compare val_bpb/CORE). A `docs/experiments/` log
in the spirit of `docs/upstream/LOG.md`, but for architecture ablations specifically.

## Explicitly deferred, not scheduled

- `jinja2` / `pyyaml` / `requests` are imported (`nanochat/core_eval.py`, `scripts/base_eval.py`,
  `nanochat/dataset.py`) but undeclared in `pyproject.toml`, resolving only transitively through
  `torch`/`wandb`. Worth a standalone dependency-hygiene commit whenever convenient.
