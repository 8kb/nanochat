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
architecture registry, and GPT's reusable pieces (attention, MLP, RoPE, norm, sliding-window
patterns, FLOPs accounting) split out so a second architecture can reuse them instead of
forking the whole file. See [`docs/architecture.md`](docs/architecture.md) for the contract and
how to add an architecture, and [`docs/roadmap.md`](docs/roadmap.md) for what's built so far
versus planned. We still track upstream — see [`docs/upstream-sync.md`](docs/upstream-sync.md)
for where every piece of `gpt.py` ended up and how to merge a new upstream commit.

## Setup (this fork's dev machine: Apple Silicon, no CUDA)

```bash
uv sync --extra cpu --group dev
source .venv/bin/activate
```

(`uv sync --extra gpu --group dev` on a CUDA machine — see upstream's
[precision/dtype notes](docs/upstream/README.md#precision--dtype) for what that changes.)

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
├── base.py          BaseModel / BaseModelConfig / AttentionLayerSpec — the contract
├── registry.py       arch name -> (config class, model class)
├── flops.py            FLOPs / KV-cache-bytes accounting, generic over any architecture
├── components/          reusable pieces: Linear, norm, RoPE, attention, MLP, block, windows
└── gpt/                  the default (and currently only) architecture
```

## Adding an architecture, briefly

1. `nanochat/model/<arch>/{config.py,model.py,__init__.py}`, mirroring `gpt/`.
2. `config.py`: a `BaseModelConfig` subclass with your fields.
3. `model.py`: a `BaseModel` subclass, `@register_model("<arch>", YourConfig)`-decorated, reusing
   whatever fits from `nanochat/model/components/`.
4. Import it from `nanochat/model/__init__.py` so the decorator runs.
5. `python -m scripts.base_train --arch=<arch> --depth=2 --num-iterations=3 ...`

Full contract, the meta-device gotcha, precision policy, and a verification recipe:
[`docs/architecture.md`](docs/architecture.md).

## Docs index

| Doc | What's in it |
|---|---|
| [`AGENTS.md`](AGENTS.md) | Repo map, invariants that will bite you, what runs on this Mac |
| [`docs/architecture.md`](docs/architecture.md) | The `BaseModel`/registry contract, how to add an architecture |
| [`docs/roadmap.md`](docs/roadmap.md) | Staged plan: what's done, what's next |
| [`docs/upstream-sync.md`](docs/upstream-sync.md) | Where `gpt.py` code went, how to merge upstream |
| [`docs/upstream/README.md`](docs/upstream/README.md) | Original nanochat README (speedrun, leaderboard, research workflow) |
| [`docs/upstream/LOG.md`](docs/upstream/LOG.md) | Upstream's running experiment log |
| [`docs/upstream/LEADERBOARD.md`](docs/upstream/LEADERBOARD.md) | Upstream's "time to GPT-2" leaderboard rules |

## License

MIT (unchanged from upstream — see [`LICENSE`](LICENSE)).
