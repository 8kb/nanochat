# nanochat (architecture playground fork)

![nanochat logo](dev/nanochat.png)

This is a fork of [karpathy/nanochat](https://github.com/karpathy/nanochat) — "the simplest
experimental harness for training LLMs" — repurposed as a playground for trying **different model
architectures**, not just different hyperparameters. Upstream nanochat is deliberately
single-architecture (one GPT variant, one `--depth` dial that derives everything else); this fork
opens that up while keeping the rest of the stack — tokenization, pretraining, SFT, RL, eval,
inference — exactly as upstream built it.

Original upstream documentation (README, dev log, leaderboard) is preserved at
[`docs/upstream/`](docs/upstream/). This page covers what's different here.

## What's different from upstream

`nanochat/gpt.py`'s single GPT model became `nanochat/model/`: a `BaseModel` interface, an
architecture registry, and reusable pieces (attention, MLP, RoPE, norm, sliding-window patterns,
FLOPs accounting) that a second architecture reuses instead of forking the whole file — proven by
`nanochat/model/llama/` (SwiGLU MLP, plain pre-norm blocks, reuses GPT's attention/RoPE/embedding/
unembedding components verbatim), `nanochat/model/llama_kvshare/` (Llama, but the last fraction
of layers reuse an earlier layer's K/V instead of computing their own — Gemma-3n-style cross-layer
KV sharing — for fewer params, less prefill compute, and a smaller KV cache at the same depth), and
`nanochat/model/llama_kvshare_win/` (llama_kvshare plus sliding-window attention — one config
subclass, no new model code). `--arch=gpt` (default), `--arch=llama`, `--arch=llama_kvshare`, or
`--arch=llama_kvshare_win` selects which one `scripts/base_train.py` trains; `scripts/model_info.py`
reports any architecture's parameters,
FLOPs, KV-cache bytes, and training horizon without training anything, for picking matched configs
before spending GPU-hours. See [`docs/architecture.md`](docs/architecture.md) for the contract and
how to add an architecture, and [`docs/roadmap.md`](docs/roadmap.md) for what's built so far versus
planned. We still track upstream — see [`docs/upstream-sync.md`](docs/upstream-sync.md) for where
every piece of `gpt.py` ended up and how to merge a new upstream commit.

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

Everything passes here except tests gated on CUDA (`tests/test_optim.py`, the `TestFA3VsSDPA`
class in `tests/test_attention_fallback.py`) which skip, and one pre-existing unrelated failure
(`test_memory_limit` — macOS's memory-limit enforcement behaves differently from Linux's). See
[`AGENTS.md`](AGENTS.md#what-runs-on-this-mac) for the full picture of what does and doesn't run
here.

Small end-to-end smoke run on CPU/MPS (tokenizer → pretrain → SFT → chat), same as upstream's:

```bash
bash runs/runcpu.sh
```

## `nanochat/model/` layout

```
nanochat/model/
├── base.py          BaseModel/BaseModelConfig/AttentionLayerSpec + BaseEmbedding/BaseBlock/
│                    BaseUnembedding — the contracts
├── registry.py       arch name -> (config class, model class); apply_arch_opts() for --arch-opt
├── param_roles.py      parameter-role protocol backing setup_optimizer()/num_scaling_params()
├── flops.py              FLOPs / KV-cache-bytes accounting, generic over any architecture
├── components/             reusable pieces: Linear, norm, RoPE, rotary, attention (incl.
│                           cross-layer KV sharing), MLP, block, embedding (+smear), unembedding,
│                           windows, kv_sharing
├── gpt/                      the original architecture
├── llama/                      SwiGLU MLP, plain pre-norm blocks
├── llama_kvshare/               Llama + cross-layer KV sharing (Gemma-3n style)
└── llama_kvshare_win/            llama_kvshare + sliding-window attention (config-only subclass)
```

`nanochat/scaling.py` holds the training-horizon math shared by `scripts/base_train.py` and
`scripts/model_info.py`.

## Adding an architecture, briefly

1. `nanochat/model/<arch>/{config.py,model.py,__init__.py}`, mirroring `gpt/`.
2. `config.py`: a `BaseModelConfig` subclass with your fields.
3. `model.py`: a `BaseModel` subclass, `@register_model("<arch>", YourConfig)`-decorated, reusing
   whatever fits from `nanochat/model/components/` (attention/MLP take explicit dims, not a
   config object, so they're reusable regardless of your config's field names). Declare a
   `PARAM_ROLES` (or `param_roles()`) for every parameter you introduce.
4. Import it from `nanochat/model/__init__.py` so the decorator runs.
5. `python -m scripts.base_train --arch=<arch> --depth=2 --num-iterations=3 ...` (a config field
   `from_depth` doesn't take directly is still reachable via `--arch-opt field=value`).

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
`DRY_RUN=1` first; nothing rents anything on its own).

## Docs index

| Doc | What's in it |
|---|---|
| [`AGENTS.md`](AGENTS.md) | Repo map, invariants that will bite you, what runs on this Mac |
| [`docs/architecture.md`](docs/architecture.md) | The `BaseModel`/registry contract, how to add an architecture, cross-layer KV sharing |
| [`docs/roadmap.md`](docs/roadmap.md) | Staged plan: what's done, what's next |
| [`docs/contest.md`](docs/contest.md) | Running the architecture contest on RunPod: pod spec, budgeting, comparing results |
| [`docs/upstream-sync.md`](docs/upstream-sync.md) | Where `gpt.py` code went, how to merge upstream |
| [`docs/upstream/README.md`](docs/upstream/README.md) | Original nanochat README (speedrun, leaderboard, research workflow) |
| [`docs/upstream/LOG.md`](docs/upstream/LOG.md) | Upstream's running experiment log |
| [`docs/upstream/LEADERBOARD.md`](docs/upstream/LEADERBOARD.md) | Upstream's "time to GPT-2" leaderboard rules |

## License

MIT (unchanged from upstream — see [`LICENSE`](LICENSE)).
