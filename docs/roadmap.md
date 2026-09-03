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

## Stage 4 — cross-layer KV sharing + a config inspector (done)

Added `nanochat/model/llama_kvshare/`: Llama, but the last `kv_share_frac` fraction of layers
reuse an earlier layer's K/V (Gemma-3n-style) instead of computing their own — fewer parameters
(no `c_k`/`c_v` on sharing layers), less prefill compute, and a smaller KV cache at the same depth.
This is the first architecture to break the "one KV-cache slot per layer" assumption that used to
be baked into three places, fixed generically so GPT/Llama (still one-slot-per-layer) are
unaffected: `AttentionLayerSpec` gained `kv_slot` (decoupling a layer's position from its cache
slot); `kv_cache.advance()` moved out of `CausalSelfAttention` (it used to fire on the last
`layer_idx`, which breaks the moment slot count != layer count) into each model's own `forward`;
`nanochat.engine.KVCache` was renamed `num_layers`/`n_layers`/`get_layer_cache` ->
`num_kv_slots`/`n_slots`/`get_slot_cache`; `flops.kv_bytes_per_token` now sums per distinct slot,
not per layer. `CausalSelfAttention` gained `kv_slot`/`produces_kv` constructor kwargs and a
`kv_bus` forward kwarg so a consumer layer can reuse a producer layer's already-RoPE'd/normed K/V
from the same forward pass — see [architecture.md](architecture.md#cross-layer-kv-sharing) for why
the consumer re-passes the producer's own tensors rather than `k=None` (a real FA3-vs-SDPA
semantics divergence). `SwiGLUMLP`/`PlainBlock` moved from `nanochat/model/llama/` into
`nanochat/model/components/`, shared by both `llama` and `llama_kvshare`. Also added `--arch-opt
KEY=VALUE` (`nanochat.model.registry.apply_arch_opts`) to `scripts/base_train.py` so an
architecture-specific config field like `kv_share_frac` is reachable from the CLI without a new
flag per field, and fixed `--window-pattern`'s default (was always `"SSSL"`, silently overriding
`llama`/`llama_kvshare`'s own `"L"` default) to only apply when explicitly passed.

Also added `scripts/model_info.py`: prints parameters (by role), FLOPs/token, KV-cache bytes, and
the derived training horizon for any `--arch`/`--depth` combination purely from a meta-device
build — no GPU, no cached data, no training. The training-horizon math (`--target-param-data-ratio`
etc.) was pulled out of `scripts/base_train.py`'s module body (which has no `main()` and starts a
real run on import) into `nanochat/scaling.py:derive_training_plan`, so both the inspector and the
real training script compute identical numbers — the inspector's output was checked against a real
`base_train.py` run's stdout for the same arguments as part of this stage's verification. This is
the prerequisite for Stage 5's architecture contest: choosing matched configs was guesswork before
this existed.

## Stage 5 — architecture contest

Train all three architectures (`gpt`, `llama`, `llama_kvshare`) at a matched medium size (d12–d16
range) on rented multi-GPU cloud hardware (e.g. 4x A100 40GB on RunPod/Lambda), using
`scripts/model_info.py` (Stage 4) to pick configs that land within a few percent of each other on
whichever axis matters (params, FLOPs, or shape — the inspector reports all three). One shared
tokenizer across all three runs (`NANOCHAT_BASE_DIR` shared, only `--arch`/`--model-tag` differ).
Download checkpoints and compare locally (val bpb, CORE, qualitative samples via `chat_cli.py`).
Needs: an arch-aware version of `runs/scaling_laws.sh`/`runs/miniseries.sh` (or a new
`runs/contest.sh`) that records which architecture produced each row, and provisioning notes for
whichever cloud provider ends up used.

## Stage 6 — attention variants

Per-layer attention and position-encoding selection in the config (mixing local/global, or
different attention types per layer). Position encoding becomes its own swappable component
(RoPE / NoPE / ALiBi) — Stage 2 already extracted `RotaryEmbedding` as an injected module rather
than wiring it straight into `CausalSelfAttention`, so this is mostly about adding alternatives
and a way to pick one, not further extraction. Add MLA (DeepSeek-style latent attention) and
differential attention as reference implementations. Stage 4 already generalized
`nanochat.engine.KVCache` from a strict one-slot-per-layer allocation to a slot-indexed one and
moved `advance()` out of the attention layer; MLA's compressed latent cache will likely still need
KVCache generalized further (a per-layer state object the layer itself allocates and manages,
rather than a fixed `(n_slots, B, T, H, D)` k/v tensor pair).

## Stage 7 — depth and residual topology

`GPT._forward_trunk` (Stage 1, refined in Stage 2 to own `x0` and the block loop) is the seed for
this: weight tying across layers, looped/universal transformers, layer skipping,
multi-token-prediction (MTP) heads. Muon's shape-bucketed param grouping (now
`nanochat/model/param_roles.py:build_param_groups`, driven by `GPT.setup_optimizer`'s policy
table) needs generalizing for an architecture with tied or ragged-shaped matrix params — Stage 2's
role protocol makes this more tractable than before (a tied parameter is already a solved case at
the role-collection level, just not yet exercised by any real architecture), but the shape-based
Muon stacking itself still assumes independent, per-layer-shaped matrices.

## Stage 8 — experiment ergonomics

Config files as an alternative to pure argparse CLI flags (useful once there are several
architectures with different field sets). A `docs/experiments/` log in the spirit of
`docs/upstream/LOG.md`, but for architecture ablations specifically — Stage 5's contest is this
stage's first real entry.

## Explicitly deferred, not scheduled

- `jinja2` / `pyyaml` / `requests` are imported (`nanochat/core_eval.py`, `scripts/base_eval.py`,
  `nanochat/dataset.py`) but undeclared in `pyproject.toml`, resolving only transitively through
  `torch`/`wandb`. Worth a standalone dependency-hygiene commit whenever convenient.
