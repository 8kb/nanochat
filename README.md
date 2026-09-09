# nanochat (architecture playground fork)

![nanochat logo](dev/nanochat.png)

This is a fork of [karpathy/nanochat](https://github.com/karpathy/nanochat) — "the simplest
experimental harness for training LLMs" — repurposed as a playground for trying **different model
architectures**, not just different hyperparameters. Upstream nanochat is deliberately
single-architecture (one GPT variant, one `--depth` dial that derives everything else); this fork
opens that up while keeping the rest of the stack — tokenization, pretraining, SFT, RL, eval,
inference — exactly as upstream built it.

Original upstream documentation (README, dev log, leaderboard) is preserved at
[`docs/upstream/`](docs/upstream/). This page covers what's different here. This repo is one part
of a small family — see [`../llmllab/AGENTS.md`](../llmllab/AGENTS.md) for the map and shared
conventions (RunPod ops, the standalone-subsystem pattern).

## What's different from upstream

`nanochat/gpt.py`'s single GPT model became two separate, standalone repos plus a thin
architecture-selection layer inside this one. **[`modelcore`](https://github.com/8kb/modelcore)**
is the model subsystem: a materialized-config-tree format, a component/composer catalog (RoPE,
attention with cross-layer KV sharing, MLP, embedding/unembedding, ...), and `ModelManager` — the
one entrypoint that creates, loads, saves, and validates a model and computes its FLOPs/param/
KV-cache stats. **[`datacore`](https://github.com/8kb/datacore)** is the data subsystem: prepares a
raw corpus into a pretokenized, packed, on-disk dataset and reads it back as flexible-batch-size,
DDP-shardable, exactly-resumable batches through `DataManager`. Both are zero-dependency on this
repo (or each other) and pinned here as git dependencies by tag.

`nanochat/architectures/` is what turns a `--depth` dial (or an old checkpoint) into a concrete
tree for `modelcore` to build — see [`docs/architecture.md`](docs/architecture.md) for the full
contract. Four presets exist today: `gpt` (the original architecture), `llama` (SwiGLU MLP, plain
pre-norm blocks), `llama_kvshare` (Llama, but the last fraction of layers reuse an earlier layer's
K/V instead of computing their own — Gemma-3n-style cross-layer KV sharing — for fewer params, less
prefill compute, and a smaller KV cache at the same depth), and `llama_kvshare_win`
(`llama_kvshare` plus sliding-window attention — one config subclass, no new model code).
`--arch=gpt` (default), `--arch=llama`, `--arch=llama_kvshare`, or `--arch=llama_kvshare_win`
selects which one `scripts/base_train.py` trains; `scripts/model_info.py` reports any
architecture's parameters, FLOPs, KV-cache bytes, and training horizon without training anything,
for picking matched configs before spending GPU-hours. We still track upstream — see
[`docs/upstream-sync.md`](docs/upstream-sync.md) for where every piece of `gpt.py` ended up and how
to merge a new upstream commit.

## Setup (this fork's dev machine: Apple Silicon, no CUDA)

```bash
uv sync --extra cpu --group dev
source .venv/bin/activate
```

(`uv sync --extra gpu --group dev` on a CUDA machine — see upstream's
[precision/dtype notes](docs/upstream/README.md#precision--dtype) for what that changes.)

`uv sync` also fetches [`modelcore`](https://github.com/8kb/modelcore) and
[`datacore`](https://github.com/8kb/datacore) — this fork's standalone model and data subsystems,
each its own public repo, pinned by tag — so it needs network access to github.com the first time
(or after bumping either pin).

Run the test suite:

```bash
python -m pytest tests -q
```

Everything passes here except one pre-existing, unrelated failure — `test_memory_limit` (macOS's
memory-limit enforcement behaves differently from Linux's). The CUDA-gated tests
(`modelcore/tests/test_optim.py`, `modelcore/tests/test_kernels.py`'s `TestFA3VsSDPA` class) live
in `modelcore`'s own suite, not this repo's — see
[`AGENTS.md`](AGENTS.md#what-runs-on-this-mac) for the full picture of what does and doesn't run
here.

Training reads a prepared dataset (Stage 9), not raw parquet directly — prepare a small one before
the smoke run below, or before any `base_train.py`/`chat_sft.py` invocation:

```bash
python -m scripts.data_prep --kind=base --dataset=smoke --sequence-len=1024 --max-shards=2
```

Small end-to-end smoke run on CPU/MPS (tokenizer → pretrain → SFT → chat), same as upstream's:

```bash
bash runs/runcpu.sh
```

## `nanochat/architectures/` layout

```
nanochat/architectures/
├── derive.py           the depth-dial derivation rules: mup_dims, compute_window_sizes,
│                        compute_kv_slots, has_value_embed, gpt_lambda_schedule
├── presets.py            expand(name, depth, **kwargs) -> modelcore.ModelConfig; assemble_gpt/
│                        assemble_plain factor out the actual tree assembly; PRESETS registry
└── legacy.py              migrate_checkpoint/migrate_optimizer_state -- old checkpoint -> current
```

The rest of `nanochat/` adapts `modelcore`/`datacore` onto this fork's own conventions — see the
repo map in [`AGENTS.md`](AGENTS.md#repo-map) for `checkpoint_manager.py` (naming policy +
`ArtifactStore` adapter), `engine.py` (inference on top of `modelcore.generate.Decoder`), and
`scaling.py` (muP training-plan math, unchanged since before Stage 7).

## Adding an architecture, briefly

An architecture is a **preset**, not a class — there's no model class to register and no registry
to add it to beyond `presets.py`'s own `PRESETS` dict:

1. Any new derivation rule it needs goes in `nanochat/architectures/derive.py` (e.g. a new
   window-pattern policy, a new KV-sharing fraction rule).
2. An `expand_<name>(depth, **kwargs) -> ModelConfig` function in `nanochat/architectures/presets.py`
   — reuse `assemble_gpt`/`assemble_plain` if its tree shape already fits one of them (looping over
   layers, building `ComponentSpec`s); write the tree by hand otherwise.
3. Add it to `PRESETS`.
4. If it needs a new `modelcore` component (a new attention pattern, a new MLP), add that in
   `modelcore` first, following [its own component-addition recipe](https://github.com/8kb/modelcore/blob/main/docs/architecture.md#adding-a-component-step-by-step)
   — declare a `PARAM_ROLES` (or `param_roles()`) for every parameter it introduces.
5. `python -m scripts.model_info --arch=<name> --depth=2` to sanity-check the tree before training
   anything; `python -m scripts.base_train --arch=<name> --depth=2 --num-iterations=3 ...` for an
   actual smoke run (a config field a preset's own kwargs don't expose is still reachable via
   `--arch-opt field=value`).

Full contract, the meta-device gotcha, precision policy, and a verification recipe:
[`docs/architecture.md`](docs/architecture.md).

To compare architectures/depths before spending any compute:

```bash
python -m scripts.model_info --arch gpt,llama,llama_kvshare,llama_kvshare_win --depth 12
```

Prints parameter counts (by role), FLOPs/token, KV-cache bytes, and the derived training horizon
for each — no GPU, no cached training data, no training. `--json` for scripted comparisons,
`--gpu "NVIDIA A100" --num-gpus 4` for a GPU-hours estimate. `--checkpoints` switches to
inspecting *already-trained* checkpoints instead (params/FLOPs plus val bpb/CORE/wall-clock and a
tokenizer-fingerprint match check, read from each checkpoint's own meta.json — still no weights
loaded): `python -m scripts.model_info --checkpoints` inspects every checkpoint under
`base_checkpoints/`, or pass a comma-separated list of tags.

`runs/contest.sh` runs all four architectures on the same tokenizer and the same iso-FLOPs
compute budget on rented cloud GPUs, then SFT (chat) fine-tunes and evaluates each resulting base
checkpoint too — see [`docs/contest.md`](docs/contest.md) for the full RunPod runbook (always
`DRY_RUN=1` first; nothing rents anything on its own — and read
[`../llmllab/docs/runpod-ops.md`](../llmllab/docs/runpod-ops.md) before creating any billed pod).

## Docs index

| Doc | What's in it |
|---|---|
| [`../AGENTS.md`](../AGENTS.md), [`../llmllab/AGENTS.md`](../llmllab/AGENTS.md) | Family map, this-laptop facts, shared conventions and RunPod ops |
| [`AGENTS.md`](AGENTS.md) | Repo map, invariants that will bite you, what runs on this Mac |
| [`docs/architecture.md`](docs/architecture.md) | Consuming `ModelManager`/`DataManager`, the `nanochat/architectures/` contract, old-checkpoint migration |
| [`docs/roadmap.md`](docs/roadmap.md) | Staged plan: what's done, what's next |
| [`docs/contest.md`](docs/contest.md) | Running the architecture contest on RunPod: pod spec, budgeting, comparing results |
| [`docs/upstream-sync.md`](docs/upstream-sync.md) | Where `gpt.py` code went, how to merge upstream |
| [`docs/upstream/README.md`](docs/upstream/README.md) | Original nanochat README (speedrun, leaderboard, research workflow) |
| [`docs/upstream/LOG.md`](docs/upstream/LOG.md) | Upstream's running experiment log |
| [`docs/upstream/LEADERBOARD.md`](docs/upstream/LEADERBOARD.md) | Upstream's "time to GPT-2" leaderboard rules |
| [modelcore](https://github.com/8kb/modelcore) | Standalone model subsystem — own README/AGENTS.md/docs/architecture.md |
| [datacore](https://github.com/8kb/datacore) | Standalone data subsystem — own README/AGENTS.md/docs/architecture.md |

## License

MIT (unchanged from upstream — see [`LICENSE`](LICENSE)).
