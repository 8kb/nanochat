# Architecture contract

`nanochat/model/` holds every transformer architecture this fork supports, side by side. The
rest of the codebase — `checkpoint_manager`, `Engine`, the training scripts — talks to a model
only through the interface described here, so adding an architecture never requires touching
those call sites.

```
nanochat/model/
├── __init__.py            public API: BaseModel, BaseModelConfig, AttentionLayerSpec,
│                          register_model, get_model_class, get_config_class, config_from_dict,
│                          GPT, GPTConfig
├── base.py                BaseModelConfig + BaseModel (this contract)
├── registry.py            arch name -> (config class, model class)
├── flops.py                FLOPs / KV-cache-bytes accounting, generic over layer_specs()
├── components/            reusable building blocks any architecture can import
│   ├── linear.py            Linear      -- weight-casting nn.Linear (see "Precision" below)
│   ├── norm.py               norm()      -- parameter-free RMSNorm
│   ├── rope.py                apply_rotary_emb(), precompute_rotary_embeddings()
│   ├── attention.py           has_ve(), CausalSelfAttention  (FA3 / SDPA, GQA, value residual)
│   ├── mlp.py                  MLP         -- relu² MLP
│   ├── block.py                 Block       -- pre-norm attn + MLP residual block
│   └── windows.py               compute_window_sizes()  -- sliding-window pattern tiling
└── gpt/                    the default architecture
    ├── config.py             GPTConfig(BaseModelConfig)
    ├── model.py                GPT(BaseModel)
    └── migrations.py            old-checkpoint backward-compat patches
```

## The contract (`nanochat/model/base.py`)

```python
class BaseModelConfig:            # a dataclass
    sequence_len: int
    vocab_size: int
    arch: ClassVar[str]           # stamped by @register_model
    def to_dict(self) -> dict     # -> {**dataclasses.asdict(self), "arch": ...}

class AttentionLayerSpec:         # a dataclass, one per transformer layer
    n_head: int
    n_kv_head: int
    head_dim: int
    window: int = -1              # -1 = full/unlimited context

class BaseModel(nn.Module):
    # you implement these:
    def init_weights(self): ...
    def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean'): ...
    def setup_optimizer(self, **kwargs) -> torch.optim.Optimizer: ...
    def layer_specs(self) -> list[AttentionLayerSpec]: ...
    def num_scaling_params(self) -> dict: ...

    # optional, no-op by default (see "Old-checkpoint migrations" below):
    @classmethod
    def patch_config_dict(cls, model_config_kwargs, log=...): return model_config_kwargs
    @classmethod
    def patch_state_dict(cls, model_data, model_config, log=...): return model_data

    # you get these for free, built on layer_specs():
    def get_device(self)
    def kv_cache_spec(self) -> dict          # {num_heads, head_dim, num_layers}
    def num_matmul_params(self)
    def estimate_flops(self)
    def estimate_decode_flops(self, context_len)
    def estimate_prefill_flops(self, num_tokens)
    def kv_bytes_per_token(self)
    def kv_read_bytes(self, context_len)
```

`layer_specs()` is the pivot the free methods are built on: it is one `AttentionLayerSpec` per
layer, in forward-pass order. `kv_cache_spec()` (what `nanochat.engine.Engine` uses to size its
`KVCache`) and every FLOPs/KV-bytes estimate in `nanochat/model/flops.py` are expressed purely in
terms of it — a new architecture gets all of that accounting for free by implementing
`layer_specs()` correctly, no matter how its per-layer attention geometry varies.

`kv_cache_spec()`'s default implementation requires uniform `n_kv_head`/`head_dim` across layers
(true for GPT). An architecture with heterogeneous per-layer KV geometry (e.g. mixed local/global
attention with different head dims) should override `kv_cache_spec()` directly.

### Implicit requirements not spelled out above

- The model must set `self.config = config` in `__init__` (used by `estimate_flops()` via
  `self.config.sequence_len`).
- `__init__` should assume it may run under `torch.device("meta")` (shapes/dtypes only, no real
  data) — see "The meta-device footgun" below. Real initialization goes in `init_weights()`.
- Every matmul-participating parameter should go through `nanochat.model.components.linear.Linear`
  rather than a raw `nn.Linear` or `nn.Parameter`, or `num_matmul_params()` (and everything built
  on it: `estimate_flops`, `estimate_decode_flops`, `estimate_prefill_flops`) will silently
  undercount it.

## Registering an architecture

```python
# nanochat/model/<arch>/model.py
from nanochat.model.base import BaseModel
from nanochat.model.registry import register_model
from nanochat.model.<arch>.config import MyConfig

@register_model("my_arch", MyConfig)
class MyModel(BaseModel):
    ...
```

The decorator stamps `MyConfig.arch = "my_arch"` and registers the (config, model) pair.
Register it by importing the module somewhere that always runs — either add
`from nanochat.model.<arch> import MyModel, MyConfig` to `nanochat/model/__init__.py` (this is
what `gpt/` does), or import it explicitly wherever you invoke it (e.g. a script's own imports).
Only architectures that get imported are registered — if `get_model_class("my_arch")` raises
`ValueError: Unknown architecture`, the import never happened.

`BaseModelConfig.to_dict()` is what goes into `meta["model_config"]` in every checkpoint, and
`nanochat.model.registry.config_from_dict()` is the inverse — pop the `"arch"` key, look up the
config class, and construct it from the rest. This round-trip is how
`nanochat/checkpoint_manager.py` reconstructs the right model class without importing any
specific architecture. A checkpoint's `model_config` with no `"arch"` key (saved before this
registry existed) defaults to `"gpt"`.

### Adding an architecture, step by step

1. `mkdir nanochat/model/<arch>` with `config.py`, `model.py`, `__init__.py` (mirror `gpt/`).
2. `config.py`: a `@dataclass class MyConfig(BaseModelConfig)` with your architecture's fields.
   If you want it to work with `scripts/base_train.py`'s `--depth` dial, add a
   `from_depth(depth, aspect_ratio, head_dim, max_seq_len, vocab_size, window_pattern)`
   classmethod (see `GPTConfig.from_depth` for the muP-style derivation GPT uses) — otherwise
   `base_train.py` raises a clear assertion telling you it's missing.
3. `model.py`: implement `BaseModel`, reusing whatever fits from `nanochat/model/components/`.
   Reach for `Block`/`CausalSelfAttention`/`MLP` directly if you're only changing depth/residual
   topology; write your own attention module if you're changing attention itself.
4. Add the import to `nanochat/model/__init__.py` so `@register_model` actually runs.
5. Add a fixture + a couple of tests mirroring `tests/conftest.py`'s `tiny_gpt` /
   `tests/test_model_gpt.py` (forward shape, loss finite, backward populates every parameter's
   grad, `setup_optimizer` partitions all parameters exactly once).
6. Try it: `python -m scripts.base_train --arch=my_arch --depth=2 --num-iterations=3 ...` (see
   the CPU smoke-test invocation in "Verifying a change is behavior-preserving" below).

## Old-checkpoint migrations

Checkpoints predate fields. `BaseModel.patch_config_dict` / `patch_state_dict` are the hook: a
no-op by default, overridden per-architecture. GPT's overrides live in
`nanochat/model/gpt/migrations.py` and backfill `window_pattern` (config) and `resid_lambdas` /
`x0_lambdas` (state dict) for checkpoints saved before those existed.
`nanochat/checkpoint_manager.py:build_model` calls them automatically:

```python
model_cls = get_model_class(arch)
model_config_kwargs = model_cls.patch_config_dict(model_config_kwargs, log=log0)
model_config = config_from_dict(model_config_kwargs)
model_data = model_cls.patch_state_dict(model_data, model_config, log=log0)
```

## The meta-device footgun

`GPT.__init__` (and any architecture's `__init__`) may run under `torch.device("meta")` —
shapes and dtypes only, no real storage. `scripts/base_train.py`'s `build_model_meta` and
`checkpoint_manager.build_model` both do this so they can inspect a model's config/param counts
before committing to allocate it. Any real data (weight init, RNG-dependent buffers) must live in
`init_weights()`, called after `model.to_empty(device=...)` materializes real storage. This is
why `checkpoint_manager.build_model` calls `init_weights()` even when *loading* a checkpoint
(right before `load_state_dict(..., assign=True)` overwrites everything) — GPT's rotary `cos`/
`sin` buffers are `persistent=False` (not saved in the checkpoint) and only get populated with
real values inside `init_weights()`.

## Precision

There's no `torch.amp.autocast` anywhere in this codebase. Precision is controlled by one global,
`COMPUTE_DTYPE` (`nanochat/common.py`, overridable via the `NANOCHAT_DTYPE` env var): model
weights stay fp32 (for optimizer precision), and `nanochat.model.components.linear.Linear` casts
them to `COMPUTE_DTYPE` in `forward()`. On CPU/MPS (this repo's day-to-day dev machine) that's
`float32` by default. Any new architecture should route its matmul weights through `Linear`
rather than a raw `nn.Linear`, both for this precision policy and so FLOPs accounting sees it
(see "Implicit requirements" above).

## Verifying a change is behavior-preserving

There's a trained checkpoint at `~/.cache/nanochat/base_checkpoints/d6` on this machine (from
`runs/runcpu.sh`). Before and after a change to shared code (`nanochat/model/base.py`,
`nanochat/model/flops.py`, `nanochat/engine.py`, `nanochat/checkpoint_manager.py`), compare:

```python
from nanochat.checkpoint_manager import load_model
from nanochat.common import compute_init, autodetect_device_type
from nanochat.engine import generate_naive

_, _, _, _, device = compute_init(autodetect_device_type())
model, tokenizer, meta = load_model("base", device, phase="eval", model_tag="d6")
prompt = tokenizer.encode("The chemical formula of water is", prepend=tokenizer.get_bos_token_id())
tokens = list(generate_naive(model, prompt, max_tokens=64, temperature=0.0))
# also worth checking: model.estimate_flops(), .num_matmul_params(), .num_scaling_params(),
# .kv_bytes_per_token(), .estimate_decode_flops(256), .estimate_prefill_flops(256)
```

Every value should match exactly. This is how Stage 1's `nanochat/gpt.py` -> `nanochat/model/`
extraction was checked end to end.

For an end-to-end smoke test of the training path on CPU/MPS:

```bash
python -m scripts.base_train --depth=2 --head-dim=32 --window-pattern=L --max-seq-len=128 \
  --device-batch-size=1 --total-batch-size=256 --num-iterations=3 \
  --eval-every=-1 --core-metric-every=-1 --sample-every=-1 --model-tag=smoke --run=dummy
```

Delete `~/.cache/nanochat/base_checkpoints/smoke` afterward — it's a throwaway.
