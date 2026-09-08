# datacore

A standalone data subsystem: prepares a raw text/token corpus into a pretokenized, packed,
multipart on-disk dataset (`DataManager.prepare`), and reads it back as flexible-batch-size,
DDP-shardable, exactly-resumable batches (`DataManager.batches`). Zero dependency on any host
application — only `numpy`; `torch` (for `batches()`) and `pyarrow` (for
`ParquetDirectorySource`) are optional extras.

It has exactly one public entrypoint, `DataManager`, and understands exactly one on-disk format —
a manifest plus `.npy` volumes (`datacore.v1`). It knows nothing about a corpus's *identity*
(which URL, which shard count, which task mixture is validation): that's a host application's
job, sitting on top and handing `prepare()` a source.

See [docs/architecture.md](docs/architecture.md) for the full contract.

## Quickstart

```python
from datacore import BestFitCropPacker, CharTokenizer, DataManager, FileSystemDatasetStore

class ListTextSource:
    def __init__(self, batches):
        self._batches = batches  # [(source_name, [text, text, ...]), ...]
    def text_batches(self):
        return iter(self._batches)

tokenizer = CharTokenizer(" abcdefghijklmnopqrstuvwxyz.\n")
manager = DataManager()
store = FileSystemDatasetStore("/tmp/my_dataset")

manifest = manager.prepare(
    store,
    sources={"train": ListTextSource([("shard0", ["hello world.\n"] * 100)])},
    tokenizer=tokenizer,
    sequence_len=128,
    sequences_per_volume=1024,
    packer=BestFitCropPacker(),
)

dataset = manager.open(store)
for inputs, targets, state in manager.batches(dataset, "train", batch_size=8, infinite=False):
    ...  # inputs: int32 (8, 128), targets: int64 (8, 128), state: the resumable cursor
```

## Tests

```bash
python -m pytest datacore/tests -v
```

No real tokenizer or GPU required — `CharTokenizer` and `torch`'s CPU path cover the whole suite;
`pyarrow`-dependent tests skip automatically if it isn't installed.
`datacore/tests/test_standalone.py` mechanically checks that nothing under `datacore/` imports a
host application (or `modelcore`, a sibling standalone component); see
[docs/architecture.md](docs/architecture.md#verifying-a-change-is-behavior-preserving) for the
full verification recipe, including a from-scratch standalone-copy check.
