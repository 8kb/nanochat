# datacore: architecture contract

datacore prepares a raw corpus into a pretokenized, packed, multipart on-disk dataset, and reads
it back as flexible-batch-size, DDP-shardable, exactly-resumable batches. It is the data-side
counterpart to `modelcore`: same shape (zero host-application imports, one Manager entrypoint, its
own tests/docs/packaging, an AST guard proving standalone-ness), different concern.

## `DataManager`: the one entrypoint

Everything a caller needs — prepare a dataset, open one, or read batches from one — goes through
`DataManager`. Nothing else in `datacore` (`packing`, `writer`, `reader` internals, `sources`,
`download`) is meant to be reached directly from outside the package, except the value types and
protocols `datacore/__init__.py` re-exports (`Tokenizer`, `CharTokenizer`, `Packer` and its two
implementations, `TextSource`/`TokenSource`/`ParquetDirectorySource`, `DatasetStore`/
`FileSystemDatasetStore`, `Dataset`/`DatasetInfo`).

```python
manager = DataManager()
manifest = manager.prepare(store, sources={"train": ..., "val": ...}, tokenizer=...,
                            sequence_len=2048, sequences_per_volume=16384, packer=...)
dataset  = manager.open(store)
for inputs, targets, state in manager.batches(dataset, "train", batch_size=32, device=dev,
                                               rank=r, world_size=W, resume=saved_state):
    ...
```

## The on-disk format (`datacore.v1`)

A prepared dataset is one directory: `manifest.json` plus `.npy` volumes per split.

```
<dataset_dir>/
  manifest.json
  train_000000.npy   train_000000.mask.npy   train_000001.npy   ...
  val_000000.npy     ...
```

Plain `.npy`, not a hand-rolled binary header — `np.load(path, mmap_mode="r")` gives a
self-describing dtype and shape with nothing to version. Tokens and mask are separate files, so
the mask's presence is a directory fact (and a `manifest["has_mask"]` flag), not a format variant.

- Tokens: `(rows, sequence_len + 1)`, `uint16` when `vocab_size <= 65535` else `uint32`
  (`store.token_dtype`).
- Mask (only when the packer emits one): `(rows, sequence_len + 1)`, `uint8`.

Row width is `sequence_len + 1` so `inputs = row[:-1]`, `targets = row[1:]`. `-1` (ignore_index)
is never stored on disk — the reader derives it from the mask at read time, which is what keeps an
unsigned token dtype always safe (a real invariant: `torch.from_numpy` on a `uint16` array
succeeds and yields a `torch.uint16` tensor, but `nn.Embedding` rejects it outright — widen to
`int32`/`int64` in numpy before handing anything to torch, which `reader.batches` always does).

Every volume is written to a `.tmp` sibling then `os.replace`d into place, and **the manifest is
written last** — a half-prepared directory can never be mistaken for a complete one; `open()`
raises `FileNotFoundError` if no manifest exists yet.

`sequences_per_volume` acts as a **cap**, not an exact count: a volume is also flushed at every
source-file boundary (see `writer.write_split`), so the last volume from each source file is
short. This is what makes `prepare()` incremental (topping up a corpus with new source files
appends volumes instead of rebuilding), parallelizable per source file with byte-identical output
regardless of worker count, and what keeps a split's earlier volumes bit-identical across
re-preps. The cost is at most one short, partial volume per source file. The reader never assumes
uniform volume length — it builds a prefix sum from each volume's manifest-recorded row count.

## Packing

`Packer` (duck-typed, not an ABC) turns a stream of `EncodedDoc` into fixed-width `PackedRow`.
Two implementations, in `packing.py`:

- `BestFitCropPacker` — every row filled to exactly `row_capacity`; the largest buffered document
  that fits wins, and when nothing fits, the *shortest* buffered document (not the longest — a
  deliberately frozen tie-break) is cropped to fill the remainder. 100% utilization, some tokens
  always dropped.
- `BestFitPadPacker` — same search, but pads the tail with the BOS token (mask=0) instead of
  cropping, so no token is ever discarded. A document longer than `row_capacity` can never fit a
  row at all once padding never crops it — such a document is dropped at refill time (counted in
  `num_documents_dropped`/`num_tokens_dropped`), not left stuck in the packer's buffer forever
  (which, left unfixed, degenerates into an infinite empty-padded-row generator once every other
  document in the stream has drained).

`pack()` consumes its `documents` iterable to exhaustion once and never wraps back on itself —
a caller wanting one continuous stream passes an iterable that itself cycles; a caller wanting
per-source-file volume boundaries (i.e. `DataManager.prepare`) calls `pack()` once per file.

## Read order, DDP, and resume

The entire iterator state is one integer, `cursor` — the count of sequences consumed by all ranks
across all epochs so far. For a split of length `N`, rank `r`, batch size `B`, world size `W`:

```
this rank's B indices this step:  arange(cursor + r*B, cursor + r*B + B) % N
after the step:                   cursor += B * W
```

- Rank-disjoint by construction within a step; batch size is free (never baked into the on-disk
  format).
- A batch may straddle the end of the split, mixing rows from epoch `e` and `e+1` — deliberate,
  and what keeps the state a single scalar.
- The state dict is `{"format": "datacore.v1", "cursor", "epoch", "num_sequences", "batch_size",
  "world_size"}`. Resuming at a *different* `world_size` than the run that saved the state still
  produces a gap-free, duplicate-free continuation of the global stream, since `cursor` counts
  sequences, not steps or row groups — only bit-exact reproduction of batch *contents* needs a
  matching `batch_size`/`world_size`.
- No shuffling: read order is the manifest's volume order, which is the source's own order. A
  full permutation would destroy memmap locality and change training dynamics from whatever the
  source's own ordering produces — an intentional, documented non-feature, not an oversight.

## `DatasetStore`: how a dataset's bytes are addressed

Mirrors `modelcore.store.ArtifactStore`: a store is deliberately narrow (read/write the manifest,
read/write one volume by filename), and it's a real code path — `FileSystemDatasetStore` is the
only implementation datacore ships, a plain directory convention any host application can point at
its own prepared-data directory.

## Sources and the tokenizer interface

`TextSource.text_batches()` yields `(source_name, texts)`; `prepare()` calls
`tokenizer.encode(texts, prepend=bos_id, num_threads=...)` on each batch directly — no adapter
class, encoding is one call. `TokenSource.token_batches()` yields `(source_name,
Iterable[EncodedDoc])` already tokenized, which is how a host application feeds its own
already-rendered documents (e.g. a conversation renderer's `(ids, mask)` output) without datacore
ever learning what produced them.

`Tokenizer` is a duck-typed protocol: `encode(text, prepend=None, num_threads=...)`,
`get_bos_token_id()`, `get_vocab_size()`, `fingerprint()`. Any tokenizer satisfying this shape
works unmodified — no adapter needed. `CharTokenizer` is the dependency-free implementation
datacore's own tests use: id `0` is `<unk>`, ids `1..N` a fixed character list, id `N+1` is BOS.

## Runtime shape: what needs torch

`prepare()`/`writer.write_split()`/`packing.py`/`sources.py`/`download.py` never import torch —
forking a worker pool after torch has touched CUDA is a real hazard, and the write path has no use
for a tensor. `reader.py` is the only module that imports torch, and only inside `batches()`,
lazily. `rank`/`world_size`/`device` are always explicit parameters to `batches()`, never read
from the environment — the same "no ambient globals" rule `modelcore.runtime` applies to compute
dtype.

## Verifying a change is behavior-preserving

```bash
python -m pytest datacore/tests -v
```

No GPU, no real tokenizer, no GPU-side dependency required. For a from-scratch standalone-copy
check (the actual proof `cp -r datacore /somewhere/else` is a real, testable claim):

```bash
mkdir -p /tmp/dc && cp -r datacore /tmp/dc/datacore && cd /tmp/dc && python -m pytest datacore/tests -v
```

For anything touching the packing algorithms specifically, cross-check against
`tests/goldens/data_bestfit_{crop,pad}.json` (captured from the host application's pre-datacore
packing code by `dev/capture_data_goldens.py`, frozen) via `tests/test_data_packing_parity.py` in
the host repo.
