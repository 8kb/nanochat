"""
sources.py: what DataManager.prepare consumes to build an EncodedDoc stream -- a TextSource (raw
strings, tokenized here) or a TokenSource (already-encoded documents, e.g. a host application's
own conversation renderer). Both are duck-typed protocols, not ABCs -- datacore has no business
enforcing what a caller's source subclasses from.

ParquetDirectorySource is the one concrete TextSource datacore ships. Which directory, which URL
a corpus was downloaded from, and which shard is held out as validation are host-application
identity, not a datacore concern -- see nanochat/dataset.py.
"""
from dataclasses import dataclass, field
from typing import Iterable, Iterator, Protocol, runtime_checkable

from datacore.packing import EncodedDoc


@runtime_checkable
class TextSource(Protocol):
    def text_batches(self) -> Iterator:
        """Yields (source_name, texts: Iterable[str]) -- one entry per logical file/chunk. Each
        yield is a source-boundary flush point (see datacore/writer.py): a packer's buffer is
        never carried from one entry to the next."""
        ...


@runtime_checkable
class TokenSource(Protocol):
    def token_batches(self) -> Iterator:
        """Yields (source_name, documents: Iterable[EncodedDoc]) -- already tokenized, e.g. a
        conversation renderer's own (ids, mask) output. datacore never learns what produced
        them."""
        ...


def named_document_batches(source, tokenizer, *, num_threads: int = 8):
    """Adapts a TextSource or TokenSource into the (source_name, Iterable[EncodedDoc]) stream
    datacore.writer.write_split needs. A TextSource's text is encoded here in one call per batch,
    BOS-prepended -- exactly nanochat/dataloader.py's original `tokenizer.encode(doc_batch,
    prepend=bos_token, num_threads=...)` call. Deliberately not a class: encoding is one call, not
    an abstraction worth its own type."""
    bos_id = tokenizer.get_bos_token_id()
    if hasattr(source, "text_batches"):
        for name, texts in source.text_batches():
            texts = list(texts)
            ids_batch = tokenizer.encode(texts, prepend=bos_id, num_threads=num_threads)
            yield name, [EncodedDoc(ids=ids) for ids in ids_batch]
    elif hasattr(source, "token_batches"):
        for name, docs in source.token_batches():
            yield name, docs
    else:
        raise TypeError(f"{source!r} is neither a TextSource (text_batches) nor a TokenSource (token_batches)")


@dataclass
class ParquetDirectorySource:
    """Reads text documents from an explicit, ordered list of parquet files -- one file is one
    source_name/volume-flush boundary (see datacore/writer.py). `paths` order and which files
    belong to this split (e.g. "last shard is val") is the caller's policy, not this source's --
    see nanochat.dataset.list_parquet_files. Reads a whole file's row groups into memory before
    yielding (simpler than a per-row-group boundary, and what keeps one file's tokens in one
    packer pass); a ~100MB compressed shard is a modest amount of memory for a CPU-only prep step,
    not something run on a billed GPU pod anyway."""
    paths: list = field(default_factory=list)
    column: str = "text"

    def text_batches(self):
        import pyarrow.parquet as pq
        for path in self.paths:
            pf = pq.ParquetFile(path)
            texts = []
            for rg_idx in range(pf.num_row_groups):
                rg = pf.read_row_group(rg_idx)
                texts.extend(rg.column(self.column).to_pylist())
            yield path, texts
