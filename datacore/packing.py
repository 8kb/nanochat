"""
Packing: turns a stream of encoded documents into fixed-width rows for a prepared dataset.

Packer is duck-typed, not an ABC -- datacore has no business enforcing what a caller's packer
subclasses from (same rule modelcore.store.ArtifactStore states for stores). Two implementations,
each lifted behavior-for-behavior from where it lives today in the host application:

- BestFitCropPacker: nanochat/dataloader.py's original algorithm (100% utilization, crops to fit).
- BestFitPadPacker: scripts/chat_sft.py's original algorithm (never crops, pads + masks instead) --
  plus one real fix: a document longer than row_capacity can never fit a row at all once padding
  never crops it, so it is dropped (and counted in num_documents_dropped/num_tokens_dropped)
  rather than left stuck in the buffer forever, which is what today's code actually does (a
  silent, permanent buffer-slot leak that becomes an infinite empty-padded-row generator in the
  degenerate case where every other document has drained).

`Packer.pack(documents, row_capacity)` consumes `documents` (an arbitrary, possibly-finite
iterable of EncodedDoc) to exhaustion and yields as many full-width PackedRow as it can build.
Any documents pulled into an in-progress row when the iterable runs dry are dropped along with it
(the crop packer) or the row is padded out and yielded as the final one (the pad packer) --
`documents` never wraps back on itself. A caller wanting per-source-file volume boundaries (see
datacore/writer.py) simply calls `pack()` once per file; a caller wanting one continuous stream
passes an iterable that itself cycles.
"""
from dataclasses import dataclass
from typing import Iterable, Iterator, Optional, Protocol, runtime_checkable


@dataclass(frozen=True)
class EncodedDoc:
    """One already-tokenized unit to pack: `ids` includes any leading BOS the source chose to
    prepend (packing itself never adds one). `mask`, if given, is a per-token supervision mask
    the same length as `ids` (1 = train on this token, 0 = don't) -- None means "supervise
    everything", which is what a plain pretraining document implies."""
    ids: list
    mask: Optional[list] = None


@dataclass(frozen=True)
class PackedRow:
    """ids/mask are always exactly `row_capacity` long. mask is None when the packer never emits
    one (BestFitCropPacker)."""
    ids: list
    mask: Optional[list] = None


@runtime_checkable
class Packer(Protocol):
    name: str
    emits_mask: bool

    def pack(self, documents: Iterable[EncodedDoc], row_capacity: int) -> Iterator[PackedRow]:
        ...


class BestFitCropPacker:
    """BOS-aligned bestfit crop packing. Every row is filled to exactly row_capacity: the largest
    buffered document that still fits wins; when nothing fits, the SHORTEST document in the
    buffer (not the longest -- this is the actual tie-break, frozen deliberately, not "fixed")
    is cropped to fill the remainder. 100% utilization, no padding; some tokens are always
    dropped (typically ~35% at T=2048 on real text)."""

    name = "bestfit_crop"
    emits_mask = False

    def __init__(self, buffer_size: int = 1000):
        self.buffer_size = buffer_size

    def pack(self, documents, row_capacity):
        buf = []
        docs = iter(documents)
        exhausted = False

        def refill():
            nonlocal exhausted
            if exhausted:
                return
            while len(buf) < self.buffer_size:
                doc = next(docs, None)
                if doc is None:
                    exhausted = True
                    return
                buf.append(list(doc.ids))

        refill()
        while buf:
            row = []
            pos = 0
            row_complete = True
            while pos < row_capacity:
                if len(buf) < self.buffer_size:
                    refill()
                if not buf:
                    row_complete = False  # source ran dry mid-row: drop the partial row
                    break
                remaining = row_capacity - pos
                best_idx, best_len = -1, 0
                for i, doc in enumerate(buf):
                    doc_len = len(doc)
                    if doc_len <= remaining and doc_len > best_len:
                        best_idx, best_len = i, doc_len
                if best_idx >= 0:
                    doc = buf.pop(best_idx)
                    row.extend(doc)
                    pos += len(doc)
                else:
                    shortest_idx = min(range(len(buf)), key=lambda i: len(buf[i]))
                    doc = buf.pop(shortest_idx)
                    row.extend(doc[:remaining])
                    pos += remaining
            if row_complete:
                yield PackedRow(ids=row, mask=None)
            refill()


class BestFitPadPacker:
    """Same best-fit search as BestFitCropPacker, but pads the tail with `padding_id` (mask=0)
    instead of cropping when nothing fits -- no token is ever discarded. Used for SFT, where
    dropping half a conversation would be worse than a little padding.

    padding_id defaults to None, which resolves to `bos_token_id` -- this packer's original,
    only-ever behavior before this parameter existed, and what every already-prepared dataset on
    disk was built with. Passing a distinct value (any valid token id other than bos_token_id) is
    for a future consumer that wants an unambiguous pad tail: with bos_token_id reused as filler,
    the tail looks like a document (BOS-started) to BOS-based document-boundary logic, even though
    it's pure padding; modelcore.kernels.flash_attn.build_doc_args has a matching padding_id
    parameter that knows how to handle either choice -- its own None default applies a fallback
    heuristic (a document made entirely of bos_token_id can only be this pad tail, since a real
    document always has non-BOS content after its own leading BOS) so an already-prepared
    bos_token_id-padded dataset still gets correct document boundaries without re-preparing."""

    name = "bestfit_pad"
    emits_mask = True

    def __init__(self, bos_token_id: int, padding_id: int | None = None, buffer_size: int = 1000):
        self.bos_token_id = bos_token_id
        self.padding_id = padding_id if padding_id is not None else bos_token_id
        self.buffer_size = buffer_size
        # Reset at the start of every pack() call -- see the oversized-conversation note there.
        self.num_documents_dropped = 0
        self.num_tokens_dropped = 0

    def pack(self, documents, row_capacity):
        self.num_documents_dropped = 0
        self.num_tokens_dropped = 0
        buf = []
        docs = iter(documents)
        exhausted = False

        def refill():
            nonlocal exhausted
            if exhausted:
                return
            while len(buf) < self.buffer_size:
                doc = next(docs, None)
                if doc is None:
                    exhausted = True
                    return
                if len(doc.ids) > row_capacity:
                    # can never fit as a whole (padding never crops) -- left in the buffer, it
                    # would sit there forever once every other document drains, since the
                    # best-fit search below never pops anything wider than the row. Drop it now
                    # and count it, rather than block forever or silently leak a buffer slot.
                    self.num_documents_dropped += 1
                    self.num_tokens_dropped += len(doc.ids)
                    continue
                mask = doc.mask if doc.mask is not None else [1] * len(doc.ids)
                buf.append((list(doc.ids), list(mask)))

        refill()
        while buf:
            row, mask_row = [], []
            while len(row) < row_capacity:
                if len(buf) < self.buffer_size:
                    refill()
                remaining = row_capacity - len(row)
                best_idx, best_len = -1, 0
                for i, (ids, _) in enumerate(buf):
                    ids_len = len(ids)
                    if ids_len <= remaining and ids_len > best_len:
                        best_idx, best_len = i, ids_len
                if best_idx >= 0:
                    ids, mask = buf.pop(best_idx)
                    row.extend(ids)
                    mask_row.extend(mask)
                else:
                    # nothing fits -- either genuinely too big, or the source ran dry: pad either way
                    row.extend([self.padding_id] * remaining)
                    mask_row.extend([0] * remaining)
                    break
            yield PackedRow(ids=row[:row_capacity], mask=mask_row[:row_capacity])
            refill()
