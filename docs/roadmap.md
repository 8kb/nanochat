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

## Stage 5 — architecture contest (persistent-volume pipeline verified on real cloud GPUs; H100 d12 contest and the real d16 contest still ahead)

Train all four architectures (`gpt`, `llama`, `llama_kvshare`, `llama_kvshare_win`) on the same
tokenizer and the same iso-FLOPs compute budget, so the comparison is real — and now also SFT
(chat) fine-tune and `chat_eval` each resulting base checkpoint, so the contest compares base *and*
chat models. `llama_kvshare_win` (added after the persistent-volume pipeline's first real run
below) is `llama_kvshare` plus sliding-window attention — a config-only subclass (one field default
changed, `model.py` is `class LlamaKVShareWin(LlamaKVShare): pass`) that composes the KV-sharing
and windowing axes without any new model logic, since the window is a mask applied at attention
time and KV sharing is a slot assignment, and neither interferes with the other. See
[docs/architecture.md](architecture.md)'s "Worked example: `llama_kvshare_win`".
`runs/contest.sh` (architecture-aware, unlike `runs/scaling_laws.sh`/`runs/miniseries.sh`, which
grep GPT-only stdout keys) declares one row per architecture, previews every row's
params/FLOPs/KV-cache/GPU-hours via `scripts/model_info.py` (Stage 4) before training anything
(`DRY_RUN=1`, always run first), trains, SFTs, chat-evals, records results into two CSVs
(`results.csv` for base, `chat_results.csv` for chat) keyed by architecture, and prints exactly
what to `rsync` home. `scripts/model_info.py --checkpoints` inspects an *already-trained*
checkpoint from its own saved meta.json (no weights loaded), reporting the same params/FLOPs/KV
block plus what training actually produced. Full runbook, RunPod pod spec, and cost table:
[docs/contest.md](contest.md).

Correctness traps found and fixed across this stage's plan-mode research *and* its first two real
runs (the second, a real 4x A100 pod, found bugs plan-mode research couldn't have — see
docs/contest.md's "Lessons from the first real cloud run" for the full list):
- **Two checkpoints could share a vocab *size* but not a vocab.** Fixed with
  `RustBPETokenizer.fingerprint()` (a content hash of the vocab), written into checkpoint meta and
  checked (warn, not raise) by `checkpoint_manager.build_model`.
- **`base_eval.py`'s CORE-eval CSV was named only after the step number**, overwriting one file
  three times when evaluating all three contest checkpoints. Fixed via the resolved model tag now
  stamped into `meta["model_tag"]`.
- **The GPU-hours/cost estimate was wrong by a factor of `num_gpus` (4x)**, found running the d12
  shakedown for real: a wall-clock-hours value got mislabeled "GPU-hours" and fed into a per-GPU
  price. Fixed — `gpu_hours` is now true GPU-resource-hours.
- **wandb credentials didn't reach a non-interactively-launched training script**, even though the
  RunPod Secret genuinely resolved on the pod — an interactive-shell-only `.bashrc` guard was the
  gap. Now sourced automatically by the scripts themselves.
- **FA3 silently discarded its own failure reason**, making a fixable environment issue
  indistinguishable from genuinely unsupported hardware. Now captured into
  `nanochat.flash_attention.FA3_LOAD_ERROR` and printed in the fallback warning. On the first real
  pod where this diagnostic actually ran, it found the real cause in one shot: RunPod's pytorch
  image sets `HF_HUB_ENABLE_HF_TRANSFER=1` but doesn't install the `hf_transfer` package, so the
  `kernels` hub download raised before reaching the network. Fixed by adding `hf_transfer` to
  `pyproject.toml`; confirmed FA3 active on a real A100 run afterward.
- **`chat_eval` cost more than training.** Its `--max-problems` has no default cap, and two of its
  five tasks (GSM8K, HumanEval) are generative and unbatched — a single architecture's eval ran
  past 25 minutes, more than base+SFT training combined. Fixed with a default cap
  (`CHATEVAL_MAX_PROBLEMS=100`) in the contest scripts. Separately, `chat_eval` already shards
  across DDP ranks but was launched with plain `python`, wasting 3 of 4 billed GPUs on that step —
  now runs through the same `launch_module` (`torchrun`) helper as `base_train`/`chat_sft`.

Verified in stages, each on real infrastructure where it matters:
1. **Locally, no GPU rented**: a `DRY_RUN=1` dry run reproduced a hand-computed reference cost
   table to the GPU-hour, and a full end-to-end CPU/MPS rehearsal (`NPROC_PER_NODE=1`,
   `EXTRA_TRAIN_ARGS` forcing a 3-step toy run) trained all three architectures then existing,
   produced distinct checkpoints and a real results CSV, and `scripts/model_info.py --checkpoints`
   correctly read back each checkpoint's true trained shape and a `match` tokenizer fingerprint.
2. **On a real 4x A100-SXM4-80GB pod** (`runs/contest_d12.sh`, d12/`TARGET_FLOPS=1e18`, ~$10.60):
   all three architectures trained end to end with real val bpb/CORE scores (gpt won on quality,
   0.1509 CORE, but was slowest at 33% MFU vs. llama/llama_kvshare's 57-58% — SDPA fallback
   penalizes gpt's sliding-window pattern more). This run is what found the four real-infra bugs
   above.
3. **Locally again, for the SFT/chat extension**: after fixing those bugs and adding the
   base→SFT→chat_eval chain, a second full CPU/MPS rehearsal proved arch-qualified chat checkpoint
   tags don't collide and base→chat provenance (`base_model_tag`/`base_model_step`) is stamped
   correctly, before spending any more real money on it.
4. **On a real 4x A100-SXM4-80GB pod again, against the new persistent-volume pipeline**
   (`llama_kvshare` only, deliberately narrowed to cheapen a pipeline-validation run): a 50GB
   RunPod Network Volume (`nanochat-contest-archive`, US-KS-2), pre-staged once from a cheap CPU
   pod (data shards, tokenizer, `.venv`), mounted read/write by the GPU pod so "sync" is just
   "already there." Base training: val bpb 0.8716, CORE 0.1307, 23.03 min, **FA3 active** —
   validating the `hf_transfer` fix above on real infrastructure. SFT: val bpb 0.3831, 9.35 min.
   `chat_eval` (capped at 100 problems/task after the cost bug above was hit and fixed mid-run):
   ChatCORE 0.0900. Full numbers and the incidents hit running it for real:
   [docs/contest.md](contest.md)'s "Stage 1 results" and "Lessons from the first real cloud run".
5. **On a fresh 2x H100 SXM 80GB pod** (4x wasn't orderable despite showing live "LOW" stock; a
   second Network Volume was needed since US-KS-2 carries no H100/H200 stock at all): the full d12
   contest, `gpt`/`llama`/`llama_kvshare_win` (`llama_kvshare` itself skipped — already measured in
   step 4). FA3 confirmed active directly and in every log — the first real exercise of the Hopper
   (`major==9`) code path. Results: gpt val bpb 0.8434/CORE 0.1553/ChatCORE 0.0833; llama val bpb
   0.8791/CORE 0.1109/ChatCORE 0.0507; **llama_kvshare_win val bpb 0.8693/CORE 0.1359/ChatCORE
   0.0620**, 2nd on both metrics (behind gpt, ahead of llama) with the fewest scaling params (103M)
   and smallest KV cache (37.7MB) of the three — the KV-sharing + windowing combination pays off on
   quality-per-byte. A mid-run tuning fix (batch size sized for A100 was leaving 80% of the H100's
   memory idle, capping MFU at ~30-35% despite FA3) doubled MFU to ~42-44% once caught and fixed —
   see docs/contest.md's "Lessons" for the full diagnosis and numbers.

Reference numbers at the defaults (d16, `TARGET_FLOPS=5e18`, 4x A100, `--mfu 0.4` — corrected after
the GPU-hours fix above; docs/contest.md also gives a more conservative `--mfu 0.33` estimate
matching what was actually measured):

| arch | scaling params | FLOPs/token | KV slots | GPU-hours |
|---|---|---|---|---|
| gpt | 234.9M | 1.585e9 | 16 | ~11.13 |
| llama | 239.1M | 1.837e9 | 16 | ~11.13 |
| llama_kvshare | 222.3M | 1.736e9 | 8 | ~11.13 |
| llama_kvshare_win | 222.3M | 1.510e9 | 8 | ~11.13 |

Total ≈44.5 GPU-hours ≈11.1h wall clock ≈$62-71 — a real, half-to-full-day run, not the ~$12-15 the
pre-fix estimate said.

**What's left**: the real d16 contest, re-costed for H100 (this stage's numbers, once
`DEVICE_BATCH_SIZE` is sized correctly for the card, are the best available reference) — see
docs/contest.md for the exact cost table and sequence.

## Stage 6 — composed architectures (materialized config tree) (done)

Added `nanochat/model/composed/`: a second, additive way to get a model, alongside (not replacing)
gpt/llama/llama_kvshare/llama_kvshare_win. Instead of one hardcoded Python class per architecture,
`--arch composed` builds from a materialized JSON tree — `ComponentSpec` nodes (`{"#type": ...,
...params}`) for the embedding, the per-layer blocks, and a **composer** (`StackComposer` /
`BackoutComposer`) that owns how blocks connect, replacing the near-duplicated trunk loop that used
to be hardcoded per model class. Per-layer values (window, `has_value_embed`, KV-slot assignment,
the resid/x0-lambda init schedule) are concrete in the tree, not re-derived by a rule at build
time — editing one block's entry is now how you get a custom per-layer window or extra FFN width
on shared-KV layers, no new architecture class required. `nanochat/model/composed/presets.py` is
the compatibility layer: `expand_preset("gpt"/"llama"/"llama_kvshare"/"llama_kvshare_win", depth,
...)` reproduces each native architecture's own `from_depth` + `__init__` derivation exactly (the
same `compute_window_sizes`/`compute_kv_slots`/`has_ve` calls, run once at expansion time instead
of on every build), verified in `tests/test_model_composed.py` by comparing accounting numbers and
copying weights across the documented state-dict key remap to assert bit-identical forward output
against the real native model.

`scripts/base_train.py`/`scripts/model_info.py` gained `--model-config <preset-name|json-file>`;
`model_info.py --dump-config` emits any architecture's (native or composed) materialized tree, so
the intended workflow is dump → hand-edit → `--arch composed --model-config <file>`. Kept
deliberately additive: the four native architectures, their checkpoints, and their optimizer state
are untouched — no migration, no format change, no shared code behavior change for them (verified:
`BaseModel.shape_summary()`'s default reproduces `scripts/model_info.py`'s old inline shape block
byte-for-byte, and `Block`'s two new optional kwargs default to exactly today's behavior).
Stage 7 below generalizes this additive tree into the *only* format the model subsystem knows;
Stages 8 and 9 build as new component/composer types under that system, rather than new bespoke
model classes — see [architecture.md](architecture.md).

## Stage 7 — extract `modelcore/`: one entrypoint, one config format (done)

Total redesign of the model subsystem's boundary. Stage 6 made the materialized config tree
*additive*, alongside four hand-written architecture classes; Stage 7 makes it the *only* format,
splitting the subsystem in two:

- **`modelcore/`** — a new, standalone package (zero `nanochat` imports) holding everything that
  only ever needs to know about the materialized tree: components, composers, the catalog,
  parameter roles, FLOPs/param accounting, the optimizer, the KV cache, the flash-attention
  kernel interface. `modelcore.ModelManager` is its one public entrypoint — create/load/save a
  model or its optimizer, validate a config (returning every error found, not just the first),
  and compute a config's stats, all without the caller ever touching a model class, a registry, or
  the meta-device dance directly. `Model` itself carries no accounting or optimizer methods
  (`layer_specs`, `kv_cache_spec`, `estimate_flops`, `setup_optimizer` are all gone from its
  surface) — those need a model only to read shapes/roles, which `ModelManager` does from outside.
- **`nanochat/architectures/`** — everything that knows an architecture *by name*: `presets.py`
  turns a `--depth` dial into a tree (what the four deleted classes' own `from_depth` + `__init__`
  used to do), `legacy.py` migrates an old checkpoint (any generation, including the four native
  formats and Stage 6's `"composed"`) into current shape, `derive.py` holds the actual derivation
  rules (`has_value_embed`'s parity policy, window-pattern tiling, KV-slot sharing, the muP depth
  dial) exactly once each — previously duplicated up to three times, or embedded inside a
  component as a leak (`has_ve(layer_idx, n_layer)` no longer exists anywhere near a component;
  a config tree only ever carries the already-decided output of that rule).

`GPT`/`Llama`/`LlamaKVShare`/`LlamaKVShareWin` and `nanochat/model/` (2300 LOC) are deleted
entirely, along with `nanochat/gpt.py` (the upstream-compat shim — recorded as the third
intentional upstream deviation in [upstream-sync.md](upstream-sync.md)). `nanochat.checkpoint_manager`/
`nanochat.engine` and every training/eval script were rewired onto the new API; the on-disk
checkpoint format is unchanged (an old checkpoint just migrates through `legacy.py` on load now,
same as it always migrated through `GPT.patch_state_dict` before). See
[architecture.md](architecture.md) for the full new contract.

Verified via `tests/goldens/*.json` — state-dict fingerprints, every accounting number,
greedy-generation token ids, forward-logits hashes, and optimizer layout/state, captured from
every real checkpoint on the dev machine (including a genuinely pre-Stage-2 one, `d6`) plus a
seeded synthetic model of every architecture/preset, **before** any code changed — replayed
against the finished refactor by `tests/test_goldens.py`, with exactly one documented,
minimal presentation-layer exception (a native gpt checkpoint's `num_scaling_params` key names and
optimizer group count, both retired along with the `GPT` class itself). `tests/test_modelcore.py`
and `tests/test_architectures.py` cross-check the same numbers directly through the new API.

## Stage 8 — `modelcore` stands alone (done)

Stage 7 made `modelcore/` a standalone *package* (zero `nanochat` imports); it was not yet a
standalone *project* — its tests, docs, and packaging metadata all lived outside it. Stage 8 closes
that gap and fixes three real defects the boundary review turned up along the way:

- **`--fp8` was broken.** `nanochat/fp8.py`'s `Float8Linear` subclassed `torch.nn.Linear`, not
  `modelcore.components.linear.Linear` — so after conversion, `collect_param_roles` raised
  (`Float8Linear.weight has no declared role`) the moment `ModelManager.create_optimizer` tried to
  build param groups. FP8 is now `modelcore/precision/fp8.py`, `Float8Linear` subclasses core's
  `Linear`, and `ModelManager.enable_fp8`/`fp8_disabled` replace the ~75 lines of conversion/
  eval-swap logic `scripts/base_train.py` used to carry inline (including the only production
  import that reached past the `ModelManager` seam, `modelcore.components.linear.Linear`, which
  existed solely for fp8's eval swap-back).
- **Generation primitives moved into core.** `sample_next_token`/`generate_naive` were already
  pure token-id math with no tokenizer dependency; `modelcore/generate.py` also gained `Decoder`
  (a cached prefill+decode primitive, reached via `ModelManager.new_decoder`), so `nanochat.engine.
  Engine` now drives a `Decoder` instead of allocating/cloning a `KVCache` inline, and modelcore
  can prove its own KV cache against its own naive reference with no host application present.
- **`ArtifactStore` became a real code path.** `modelcore/store.py`'s `FileSystemStore` and
  `ModelManager.{load,save}_model`/`{load,save}_optimizer` had no production caller —
  `checkpoint_manager` did raw `torch.save`/`torch.load` despite its own docstring's claim
  otherwise. `save_checkpoint`/`load_checkpoint`/`build_model` now genuinely route through the
  store; a new `LegacyCheckpointStore(FileSystemStore)` adapts an old checkpoint onto
  `ModelManager.load_model` by migrating on first read (memoized) — `modelcore` never learns
  legacy formats exist.

`modelcore/runtime.py`'s env var is now `MODELCORE_DTYPE` (`NANOCHAT_DTYPE` kept as a back-compat
alias), and every docstring inside `modelcore/` that named `nanochat` or a repo-root `docs/` path
now describes the *role* instead ("the host application"), pointing at `modelcore/docs/`.

`modelcore/tests/` is now a complete, self-contained suite (moved from `tests/test_modelcore*.py`/
`test_optim.py`/`test_attention_fallback.py`, plus new `test_precision.py`/`test_generate.py`/
`test_standalone.py` — the last AST-scans every file under `modelcore/` for a host-application
import), with its own goldens (`tiny_composed_*`, `modelcore`'s own baseline) and a
`modelcore/docs/architecture.md` carrying the core contract (the repo root's
[architecture.md](architecture.md) now covers only the nanochat-app side: presets, legacy
migration, checkpoint naming, and how the app consumes `ModelManager`). `modelcore/README.md` and
`modelcore/pyproject.toml` (declaring `torch` as modelcore's only hard dependency, `kernels` as an
optional extra for FA3) round out the package identity, without being wired into this repo's own
`uv` workspace — nothing about how nanochat itself installs changed.

Verified at every step: `python -m pytest -q` (237 passed, 14 skipped, one pre-existing unrelated
macOS sandbox failure, throughout); real end-to-end runs (`scripts.base_train` with
`--resume-from-step` exercising the optimizer save+load path through the store,
`scripts.base_eval`/`scripts.chat_cli`/`scripts.model_info` against the real, genuinely
pre-Stage-2 `d6` checkpoint through `LegacyCheckpointStore`). The actual proof this stage exists
for: copying `modelcore/` to a fresh directory with nothing else alongside it and running its
suite there — 97 passed, 14 skipped (CUDA-only), zero failures, no `nanochat` on the path at all.

## Stage 9 — the repo split

The actual extraction: `modelcore/` becomes its own git repository (a `git subtree split`
preserving history), and this repo depends on it as a path or VCS dependency instead of a
same-repo directory. What Stage 8 leaves for this stage specifically: deciding whether the one
remaining cross-boundary test dependency (`tests/test_architectures.py` reading
`modelcore/tests/goldens/tiny_composed_*` directly, to prove `nanochat.architectures.presets.expand`
reproduces modelcore's own baseline) vendors a copy of those goldens or narrows to a
config-tree-equality assertion that doesn't need modelcore's test data at all; wiring
`modelcore`'s `pyproject.toml` into an actual installable dependency (a git URL or a local path
override) rather than the aspirational, unwired file it is today; and re-verifying the standalone
guard (`modelcore/tests/test_standalone.py`) still passes against the split repo's own history,
not just a directory copy.

## Stage 10 — attention variants

Per-layer attention and position-encoding selection in the config (mixing local/global, or
different attention types per layer). Position encoding becomes its own swappable component
(RoPE / NoPE / ALiBi) — already an injected module (`RotaryEmbedding`) rather than wired straight
into `CausalSelfAttention`, so this is mostly about adding alternatives and a way to pick one, not
further extraction. Add MLA (DeepSeek-style latent attention) and differential attention as
reference implementations. `KVCache` already generalized from a strict one-slot-per-layer
allocation to a slot-indexed one (Stage 4) with `advance()` owned by `Model.forward` (Stage 7);
MLA's compressed latent cache will likely still need it generalized further (a per-layer state
object the layer itself allocates and manages, rather than a fixed `(n_slots, B, T, H, D)` k/v
tensor pair).

## Stage 11 — depth and residual topology

Weight tying across layers, looped/universal transformers, layer skipping, multi-token-prediction
(MTP) heads — new composers under `modelcore/composers/` (`BackoutComposer`/`StackComposer` are
the seed: Stage 7 made every composer a real, swappable component). Muon's shape-bucketed param
grouping (`modelcore/roles.py:build_param_groups`, driven by `ModelManager.create_optimizer`'s
policy table) needs generalizing for an architecture with tied or ragged-shaped matrix params —
the role protocol makes this more tractable than before (a tied parameter is already a solved case
at the role-collection level, just not yet exercised by any real architecture), but the
shape-based Muon stacking itself still assumes independent, per-layer-shaped matrices.

## Stage 12 — experiment ergonomics

Config files as an alternative to pure argparse CLI flags — `--model-config` is this for model
architecture specifically; this stage is the rest of a run's configuration (data, optimizer,
eval). A `docs/experiments/` log in the spirit of `docs/upstream/LOG.md`, but for architecture
ablations specifically — Stage 5's contest is this stage's first real entry.

## Explicitly deferred, not scheduled

- `jinja2` / `pyyaml` / `requests` are imported (`nanochat/core_eval.py`, `scripts/base_eval.py`,
  `nanochat/dataset.py`) but undeclared in `pyproject.toml`, resolving only transitively through
  `torch`/`wandb`. Worth a standalone dependency-hygiene commit whenever convenient.
