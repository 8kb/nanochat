# Syncing with upstream (karpathy/nanochat)

This fork restructures `nanochat/gpt.py` into `nanochat/model/` to support multiple
architectures side by side (see [architecture.md](architecture.md)). We still intend to pull
changes from upstream, so this page exists to make that tractable: where upstream code ended up,
what changed on the way, and how to bring in a new upstream commit.

There is currently no `upstream` git remote configured. Add one with:

```bash
git remote add upstream https://github.com/karpathy/nanochat.git
git fetch upstream
```

## Where `nanochat/gpt.py` went

Every symbol below was moved **verbatim** (same body, same comments) unless noted. Line numbers
are from the pre-refactor `nanochat/gpt.py` (555 lines).

| Upstream (`nanochat/gpt.py`) | Now lives in |
|---|---|
| `GPTConfig` (28-39) | `nanochat/model/gpt/config.py` — now subclasses `BaseModelConfig`; gained a `from_depth()` classmethod (moved out of `scripts/base_train.py`'s `build_model_meta`, not upstream code) |
| `norm()` (42-43) | `nanochat/model/components/norm.py` |
| `Linear` (45-50) | `nanochat/model/components/linear.py` |
| `has_ve()` (53-55) | `nanochat/model/components/attention.py` |
| `apply_rotary_emb()` (57-65) | `nanochat/model/components/rope.py` |
| `CausalSelfAttention` (67-128) | `nanochat/model/components/attention.py` |
| `MLP` (131-141) | `nanochat/model/components/mlp.py` |
| `Block` (144-153) | `nanochat/model/components/block.py` |
| `GPT.__init__` (157-201) | `nanochat/model/gpt/model.py`, `GPT.__init__` |
| `GPT.init_weights` (203-268) | same, `GPT.init_weights` |
| `GPT._precompute_rotary_embeddings` (270-285) | `nanochat/model/components/rope.py`, `precompute_rotary_embeddings()` — turned into a free function; call sites now pass `device`/`dtype` explicitly instead of the method inferring them from `self` |
| `GPT._compute_window_sizes` (287-314) | `nanochat/model/components/windows.py`, `compute_window_sizes()` — free function, same signature shape (`pattern, n_layer, sequence_len`) |
| `GPT.get_device` (316-317) | `nanochat/model/base.py`, `BaseModel.get_device` — generalized to `next(self.parameters()).device` (same value, not GPT-specific) |
| `GPT.estimate_flops` / `num_matmul_params` / `estimate_decode_flops` / `estimate_prefill_flops` / `kv_bytes_per_token` / `kv_read_bytes` (319-388) | `nanochat/model/flops.py`, plus thin wrapper methods on `BaseModel` — reworked to operate on `layer_specs()` (a list of `AttentionLayerSpec`) instead of `self.config`/`self.window_sizes` directly. Produces identical numbers for GPT (checked against a real checkpoint, see the "Verification" note below) |
| `GPT.num_scaling_params` (390-417) | `nanochat/model/gpt/model.py`, `GPT.num_scaling_params` — unchanged, GPT-specific (declared abstract on `BaseModel`) |
| `GPT.setup_optimizer` (419-457) | same file, unchanged |
| `GPT.forward` (459-524) | same file, mostly unchanged; the trunk loop (494-507) is factored out into `GPT._forward_trunk` so future depth/residual-topology architectures can override just that piece. `kv_cache.prev_embedding` reads/writes became `kv_cache.state["prev_embedding"]` (see `nanochat/engine.py` below) |
| `GPT.generate` (526-555) | `nanochat/engine.py`, `generate_naive(model, tokens, ...)` — now a free function (was a model method), reuses `sample_next_token` instead of duplicating temperature/top-k logic. **The one intentional behavior change**: top-k sampling now draws from the renormalized top-k distribution via `multinomial` (same as `Engine.generate` always did) instead of masking to `-inf` and sampling over the full vocab. Same distribution, different draw for a given seed when `top_k > 0`. Greedy (`temperature=0`) is bit-identical. |

`nanochat/gpt.py` itself still exists as a **compatibility shim** re-exporting `GPT`, `GPTConfig`,
`Linear`, `norm`, `apply_rotary_emb`, `has_ve`, `CausalSelfAttention`, `MLP`, `Block` — so an
upstream diff that touches `nanochat/gpt.py` and does `from nanochat.gpt import GPT` elsewhere
still resolves.

## Stage 2: parameters and per-layer logic moved into their owning modules

Stage 1 moved `gpt.py`'s code into `nanochat/model/`, but `GPT` still owned most of the *state* at
the top level (resid/x0 lambdas as `[n_layer]` tensors, `value_embeds` as a `ModuleDict`,
`smear_gate`/`smear_lambda`, `lm_head`, and an `init_weights()` that reached into every submodule's
parameters by path). Stage 2 pushes that down into three module contracts
(`BaseEmbedding`/`BaseBlock`/`BaseUnembedding` — see [architecture.md](architecture.md)) and a
parameter-role protocol (`nanochat/model/param_roles.py`) replacing the hand-maintained optimizer
grouping. Mapping from Stage 1's locations to Stage 2's:

| Stage 1 location | Now lives in |
|---|---|
| `GPT.__init__`'s `self.transformer.wte` / smear params | `nanochat/model/components/embedding.py`, `TokenEmbedding` (+ `Smear`) |
| `GPT.__init__`'s `self.lm_head` | `nanochat/model/components/unembedding.py`, `LMHead` |
| `GPT.__init__`'s `self.cos`/`self.sin` buffers, `_precompute_rotary_embeddings` call | `nanochat/model/components/rotary.py`, `RotaryEmbedding` — one shared instance, injected into every attention layer |
| `GPT.__init__`'s `self.resid_lambdas`/`self.x0_lambdas` (`[n_layer]` tensors) | `nanochat/model/components/block.py`, `Block.resid_lambda`/`Block.x0_lambda` (per-block scalars); the per-layer schedule is still computed by `GPT.__init__` and passed to each `Block` |
| `GPT.__init__`'s `self.value_embeds` (`ModuleDict`) | `nanochat/model/components/attention.py`, `CausalSelfAttention.value_embed` (per-layer, constructor-injected `has_value_embed` bool) |
| `GPT.__init__`'s `self.backout_lambda` | unchanged (still a top-level `GPT` parameter — backout is trunk-level, not per-layer) |
| `GPT.init_weights`'s per-submodule init logic | distributed to each submodule's own `init_weights()` (`TokenEmbedding`, `Smear`, `RotaryEmbedding`, `CausalSelfAttention`, `MLP`, `Block`, `LMHead`); `GPT.init_weights` now just calls them in order plus the `backout_lambda` constant. **One behavior change**: the sequence of RNG calls during a from-scratch init differs from before Stage 2 (same calls, different order) — see "The meta-device footgun" in [architecture.md](architecture.md) |
| `CausalSelfAttention(config, layer_idx)` | same file, now `CausalSelfAttention(n_embd, n_head, n_kv_head, layer_idx, window, rope, padded_vocab_size, has_value_embed)` — explicit dims instead of a config object, so a different architecture's config (different field names) can still reuse it; owns `layer_spec()` |
| `MLP(config)` | same file, now `MLP(n_embd)` |
| `Block(config, layer_idx)` | same file, now `Block(n_embd, n_head, n_kv_head, layer_idx, n_layer, window, rope, padded_vocab_size, resid_lambda_init, x0_lambda_init)`; owns the resid/x0 mixing (moved out of `GPT._forward_trunk`) |
| `GPT._forward_trunk`'s `ve = self.value_embeds[...]` lookup, `self.window_sizes[i]` indexing | gone — each `Block`/`CausalSelfAttention` already knows its own value-embed table and window; `_forward_trunk` now only loops, and computes `x0 = x` itself (moved out of `GPT.forward`, since `x0` is trunk-level state, not embedding-level) |
| `GPT.forward`'s smear branch, lm_head/softcap/loss code | moved into `TokenEmbedding`/`Smear` and `LMHead` respectively; `GPT.forward` is now three lines: `embedding` -> `_forward_trunk` -> `unembedding` |
| `GPT.num_scaling_params` / `GPT.setup_optimizer`'s hand-partitioned `self.parameters()` | `nanochat/model/param_roles.py`'s `collect_param_roles`/`build_param_groups`, driven by each module's `PARAM_ROLES` (see "Parameter roles" in [architecture.md](architecture.md)); `setup_optimizer` is now a `{role: hyperparameters}` policy table, same six output keys and same on-disk group layout as before |

`nanochat/model/gpt/migrations.py` gained two new patches for checkpoints saved before this
restructure: `patch_state_dict_layout` (renames every moved state-dict key — see its docstring for
the full table) and `patch_optimizer_state_dict` (splits the resid/x0 scalar groups' optimizer
moments from one `[n_layer]`-shaped entry into `n_layer` per-block entries). Both are no-ops on an
already-new-layout checkpoint. `scripts/base_train.py`'s `--resume-from-step` path and
`scripts/chat_sft.py`'s `--load-optimizer` path both now call `patch_optimizer_state_dict`
explicitly right before `optimizer.load_state_dict(...)` (it isn't wired into
`checkpoint_manager.build_model` like the other two hooks, since optimizer state loads through a
separate path — see `load_optimizer_state`). `scripts/base_train.py`'s `--resume-from-step` model
load also now calls `patch_state_dict` (previously it called `model.load_state_dict` directly,
bypassing migrations — a pre-existing gap, fixed alongside this).

## Other call sites that changed

- `nanochat/checkpoint_manager.py` no longer imports `GPT`/`GPTConfig` directly. It looks up the
  architecture via `nanochat.model.get_model_class` / `config_from_dict`, keyed on an `"arch"`
  field in `meta["model_config"]` (defaults to `"gpt"` if absent, so old checkpoints keep
  loading). The old module-private `_patch_missing_config_keys` / `_patch_missing_keys` became
  `GPT.patch_config_dict` / `GPT.patch_state_dict` classmethods, backed by
  `nanochat/model/gpt/migrations.py`.
- `nanochat/engine.py`'s `KVCache.prev_embedding` became a generic `KVCache.state: dict`; GPT's
  smear reads/writes `state["prev_embedding"]`. `Engine.generate` gets KV-cache geometry from
  `model.kv_cache_spec()` instead of reading `model.config.n_kv_head` / `n_embd` / `n_head` /
  `n_layer` directly.
- `scripts/base_train.py` gained an `--arch` flag (default `"gpt"`); `build_model_meta` looks up
  the config/model classes via the registry instead of importing `GPT`/`GPTConfig`; checkpoint
  serialization uses `model_config.to_dict()` instead of `dataclasses.asdict(model_config)` (the
  only difference is the added `"arch"` key).
- `scripts/chat_sft.py` and `scripts/chat_rl.py` had their hand-rolled `model_config` dicts
  (`chat_sft.py` built one field-by-field; `chat_rl.py` used `model.config.__dict__`) replaced
  with `model.config.to_dict()`. `chat_rl.py`'s save block also gained a `"step"` key that was
  previously missing (a pre-existing bug: `scripts/base_eval.py` and `scripts/infer_bench.py`
  both read `meta["step"]` and would `KeyError` on an RL checkpoint).
- `scripts/base_train.py` gained back a `Number of parameters: N (scaling: M)` print line
  (Stage 2): `runs/miniseries.sh` greps that exact text and it had gone missing at some point
  before Stage 1, silently producing empty CSV columns — a pre-existing bug, fixed here since it's
  adjacent to the `num_scaling_params()` changes.
- **Stage 3** (second architecture, `nanochat/model/llama/`): `nanochat/checkpoint_manager.py`'s
  `find_largest_model` gained an optional `arch=` filter (peeks at each candidate tag's
  `meta_*.json`), threaded through as an optional kwarg on `load_model`/`load_model_from_dir`/
  `load_optimizer_state` (default `None` everywhere — no existing call site's behavior changes).
  `scripts/base_train.py`'s default checkpoint tag became architecture-aware (`d<depth>` for
  `gpt`, unchanged; `<arch>_d<depth>` otherwise) to prevent two architectures at the same
  `--depth` from writing into the same directory. `scripts/base_eval.py` gained a matching
  `--arch` flag. `scripts/base_train.py`'s `get_scaling_params` (a training-horizon helper, not
  upstream code) was changed to read parameter-role sums directly via
  `nanochat.model.param_roles.collect_param_roles` instead of indexing `num_scaling_params()` by
  GPT's legacy dict keys, which doesn't generalize to other architectures' key sets — see
  "Parameter roles" in [architecture.md](architecture.md). `nanochat/model/base.py`'s
  `num_scaling_params()` went from abstract to a generic role-summing default, which GPT overrides
  to keep its legacy six-key dict (no GPT behavior change).
- **Stage 4** (cross-layer KV sharing, `nanochat/model/llama_kvshare/`): three renames/relocations
  worth knowing if a future upstream diff touches `nanochat/engine.py` or
  `nanochat/model/components/attention.py`'s KV-cache path:
  - `nanochat/engine.py`'s `KVCache.__init__` kwarg `num_layers` -> `num_kv_slots`, attribute
    `n_layers` -> `n_slots`, method `get_layer_cache(layer_idx)` -> `get_slot_cache(slot)`.
    `BaseModel.kv_cache_spec()`'s dict key `num_layers` -> `num_kv_slots` to match (splatted
    straight into `KVCache.__init__`, so the two names are one contract).
  - `CausalSelfAttention.forward`'s `if self.layer_idx == kv_cache.n_layers - 1:
    kv_cache.advance(T)` was deleted; `kv_cache.advance(idx.size(1))` is now called once by each
    model's own `forward`, after its whole block loop (`GPT.forward`/`Llama.forward`/
    `LlamaKVShare.forward`) — see "Cross-layer KV sharing" in
    [architecture.md](architecture.md#cross-layer-kv-sharing) for why (a same-layer-count
    assumption broke once a layer's KV slot can differ from its position).
  - `nanochat/model/llama/mlp.py` (`SwiGLUMLP`) and `nanochat/model/llama/block.py` (`PlainBlock`)
    moved into `nanochat/model/components/mlp.py`/`block.py` (alongside GPT's `MLP`/`Block`), since
    `llama_kvshare` needed `PlainBlock` too — a second consumer is this repo's bar for promoting
    something into `components/`.
  - New `nanochat/scaling.py:derive_training_plan`, extracted from `scripts/base_train.py`'s
    module body (the batch-size/LR-scale/weight-decay/num_iterations derivation, previously
    ~120 lines inline) so `scripts/model_info.py` can compute the same numbers without training
    anything; `base_train.py` still owns every print statement. `scripts/base_train.py` also
    gained `--arch-opt KEY=VALUE` (`nanochat.model.registry.apply_arch_opts`) and changed
    `--window-pattern`'s default from `"SSSL"` to `None` (only passed to `from_depth` when given,
    so each architecture's own default applies instead of GPT's silently overriding it).
- **Stage 5** (architecture contest harness, `runs/contest.sh`, [contest.md](contest.md)): four
  small touch points, all upstream-adjacent:
  - `nanochat/tokenizer.py`'s `RustBPETokenizer` gained `fingerprint()` (a content hash of the
    vocab, sha256 over every token's bytes in id order).
  - `scripts/base_train.py`'s checkpoint-save metadata dict gained two keys:
    `tokenizer_fingerprint` (from the tokenizer already loaded for training) and `core_metric`
    (from `results.get("core_metric")` — `None` unless a CORE eval happened to run on that exact
    step, which it always does at the final step when `--core-metric-every > 0`).
  - `nanochat/checkpoint_manager.py`'s `build_model` gained a fingerprint check after its existing
    vocab-size assert: if `meta_data["tokenizer_fingerprint"]` is present and disagrees with the
    currently-loaded tokenizer's, it logs a warning (not an assert — loading still succeeds; old
    checkpoints without the key are silently skipped). `load_model_from_dir` now also stamps the
    resolved `model_tag` into the returned meta dict (`meta["model_tag"]`), since callers that
    don't pass an explicit `--model-tag` previously had no way to know which checkpoint got picked.
  - `scripts/base_eval.py`'s CORE-eval CSV filename changed from `base_model_<step>.csv` to
    `<model_tag>_<step>.csv` (using the newly-stamped `meta["model_tag"]`), since three
    architectures evaluated in one `NANOCHAT_BASE_DIR` previously all wrote the same filename.

## Merge procedure for a new upstream commit

1. `git fetch upstream && git log HEAD..upstream/master -- nanochat/gpt.py` to see what changed.
2. For a change inside one of the functions/classes in the table above: find the new home via
   the table, apply the diff there by hand (the code is verbatim, so upstream's diff context
   should still line up almost exactly).
3. For a change to `nanochat/checkpoint_manager.py`, `nanochat/engine.py`, or the training
   scripts: check "Other call sites that changed" above first — the surrounding code moved, so a
   textual patch may not apply, but the same edit intent almost always still makes sense.
4. For a genuinely new file or a change elsewhere in the repo: apply directly, no mapping needed.
5. After merging, re-run the golden-checkpoint regression check (see the model card in
   [architecture.md](architecture.md#verifying-a-change-is-behavior-preserving)) before trusting
   the result.
