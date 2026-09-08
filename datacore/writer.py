"""
writer.py: rolls a stream of PackedRow into volumes and writes them through a DatasetStore.

A volume is flushed at `sequences_per_volume` (the cap) OR at every source-file boundary,
whichever comes first -- never carrying a packer's document buffer across source files. This is
what makes DataManager.prepare incremental (topping up a corpus with new source files appends
volumes rather than rebuilding), parallelizable per source file with byte-identical output
regardless of worker count, and what keeps a split's earlier volumes bit-identical across re-preps
(the val split's stability across runs depends on this). The cost: at most one short, partial
volume per source file -- a handful of tokens, not a meaningful loss. See
datacore/docs/architecture.md.
"""
from dataclasses import dataclass, field

import numpy as np

from datacore.store import token_dtype


@dataclass
class VolumeRecord:
    """One row in the manifest's per-split volume list."""
    file: str
    rows: int
    source: str
    mask_file: str | None = None


@dataclass
class SplitTotals:
    volumes: list = field(default_factory=list)
    num_sequences: int = 0
    num_documents: int = 0
    num_documents_dropped: int = 0
    num_tokens: int = 0
    num_tokens_encoded: int = 0
    num_tokens_dropped: int = 0


class _VolumeAccumulator:
    """Owns exactly one in-progress volume's preallocated buffer. add() appends a row and reports
    whether the cap was hit; flush() returns what was written (trimmed to the actual row count)
    and resets for the next volume."""

    def __init__(self, row_capacity: int, cap: int, dtype: np.dtype, emits_mask: bool):
        self.row_capacity = row_capacity
        self.cap = cap
        self.dtype = dtype
        self.emits_mask = emits_mask
        self._reset()

    def _reset(self):
        self._tokens = np.empty((self.cap, self.row_capacity), dtype=self.dtype)
        self._mask = np.empty((self.cap, self.row_capacity), dtype=np.uint8) if self.emits_mask else None
        self._n = 0

    def add(self, row) -> bool:
        assert len(row.ids) == self.row_capacity
        self._tokens[self._n] = row.ids
        if self.emits_mask:
            self._mask[self._n] = row.mask if row.mask is not None else 1
        self._n += 1
        return self._n >= self.cap

    def is_empty(self) -> bool:
        return self._n == 0

    def flush(self):
        assert self._n > 0, "flush() called on an empty accumulator -- caller should check is_empty() first"
        tokens = self._tokens[: self._n].copy()
        mask = self._mask[: self._n].copy() if self.emits_mask else None
        n = self._n
        self._reset()
        return tokens, mask, n


def write_split(store, split, packer, sequence_len, sequences_per_volume, vocab_size,
                 named_document_batches):
    """Packs and writes one split's worth of volumes.

    named_document_batches: iterable of (source_name, Iterable[EncodedDoc]) -- one entry per
    source file/chunk. The packer's buffer is reset (and any leftover documents dropped) at every
    source_name boundary; the volume accumulator is flushed at the boundary too, unless it's
    already empty. `source_name` is recorded per volume for provenance/debugging.

    Returns SplitTotals, with `volumes` ready to drop straight into the manifest.
    `num_tokens_encoded` is every token pulled from the source (the raw corpus size);
    `num_tokens` is what actually landed on disk; `num_tokens_dropped` is their difference --
    tokens lost to cropping (BestFitCropPacker) or to an oversized document being dropped whole
    (BestFitPadPacker). This is what replaces docs/contest.md's hand-estimated retention ratio
    with a measured one.
    """
    row_capacity = sequence_len + 1
    dtype = token_dtype(vocab_size)
    acc = _VolumeAccumulator(row_capacity, sequences_per_volume, dtype, packer.emits_mask)
    totals = SplitTotals()
    volume_index = 0

    def flush_volume(source_name):
        nonlocal volume_index
        if acc.is_empty():
            return
        tokens, mask, n = acc.flush()
        tokens_file = store.volume_filename(split, volume_index, mask=False)
        store.write_volume(tokens_file, tokens)
        mask_file = None
        if mask is not None:
            mask_file = store.volume_filename(split, volume_index, mask=True)
            store.write_volume(mask_file, mask)
        totals.volumes.append(VolumeRecord(file=tokens_file, rows=n, source=source_name, mask_file=mask_file))
        totals.num_sequences += n
        totals.num_tokens += n * row_capacity
        volume_index += 1

    for source_name, documents in named_document_batches:
        doc_iter = _CountingIterator(documents)
        for row in packer.pack(doc_iter, row_capacity):
            if acc.add(row):
                flush_volume(source_name)
        totals.num_documents += doc_iter.count
        totals.num_tokens_encoded += doc_iter.token_count
        totals.num_documents_dropped += getattr(packer, "num_documents_dropped", 0)
        flush_volume(source_name)  # source-file boundary: flush whatever is buffered, even if short

    totals.num_tokens_dropped = max(0, totals.num_tokens_encoded - totals.num_tokens)
    return totals


class _CountingIterator:
    """Wraps a document iterable to count documents and their total token length as they pass
    through, without changing what the packer sees."""

    def __init__(self, documents):
        self._it = iter(documents)
        self.count = 0
        self.token_count = 0

    def __iter__(self):
        return self

    def __next__(self):
        doc = next(self._it)
        self.count += 1
        self.token_count += len(doc.ids)
        return doc
