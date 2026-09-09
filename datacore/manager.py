"""
DataManager: the one entrypoint datacore exposes. Everything a caller needs to prepare a
pretokenized, packed, multipart dataset (write) or read exactly-resumable, DDP-shardable batches
from one (read) goes through here. Nothing else in datacore (packing, writer, reader, sources,
download internals) is meant to be used directly from outside the package -- see the module
docstrings for why each exists, but DataManager is the seam (mirrors modelcore.ModelManager).
"""
from datacore.reader import Dataset, batches as _batches, open_dataset
from datacore.sources import named_document_batches
from datacore.store import FORMAT, token_dtype
from datacore.writer import write_split


class DataManager:
    def prepare(self, store, *, sources, tokenizer, sequence_len, sequences_per_volume, packer, num_threads=8):
        """sources: dict[str, TextSource | TokenSource] -- one source per split (e.g.
        {"train": ..., "val": ...}). Which source produces which split is caller policy: a
        pretraining corpus's val split is usually a slice of the same corpus (held out by the
        caller before building the source), while an SFT val split is typically a wholly
        different task mixture -- datacore doesn't assume either shape.

        num_threads: forwarded to a TextSource's tokenizer.encode() call (tiktoken-style
        tokenizers release the GIL for batch encoding, so this is real parallelism -- see
        datacore/docs/architecture.md's "Preparation throughput" note). Ignored for a TokenSource,
        which arrives already encoded.

        Writes every split's volumes, then the manifest last (so a half-prepared directory can
        never be mistaken for a complete one). Returns the manifest dict.
        """
        vocab_size = tokenizer.get_vocab_size()
        bos_id = tokenizer.get_bos_token_id()
        manifest_splits = {}
        for split, source in sources.items():
            named_batches = named_document_batches(source, tokenizer, num_threads=num_threads)
            totals = write_split(store, split, packer, sequence_len, sequences_per_volume, vocab_size, named_batches)
            manifest_splits[split] = _split_totals_to_dict(totals)
        packer_params = {"buffer_size": packer.buffer_size}
        # padding_id is duck-typed (only BestFitPadPacker has it) -- record its resolved value
        # (never None: the packer itself resolves a None constructor arg to bos_token_id) so a
        # manifest fully documents what a prepared dataset's pad tail actually contains, whether
        # defaulted or explicit.
        packer_padding_id = getattr(packer, "padding_id", None)
        if packer_padding_id is not None:
            packer_params["padding_id"] = packer_padding_id
        manifest = {
            "format": FORMAT,
            "sequence_len": sequence_len,
            "stride": sequence_len + 1,
            "dtype": str(token_dtype(vocab_size)),
            "has_mask": bool(packer.emits_mask),
            "vocab_size": vocab_size,
            "bos_token_id": bos_id,
            "tokenizer_fingerprint": tokenizer.fingerprint(),
            "packer": {"name": packer.name, "params": packer_params},
            "sequences_per_volume": sequences_per_volume,
            "splits": manifest_splits,
        }
        store.write_manifest(manifest)
        return manifest

    def open(self, store) -> Dataset:
        return open_dataset(store)

    def batches(self, dataset: Dataset, split: str, batch_size: int, **kwargs):
        return _batches(dataset, split, batch_size, **kwargs)

    def read_rows(self, dataset: Dataset, split: str, start: int, count: int):
        return dataset.read_rows(split, start, count)


def _split_totals_to_dict(totals):
    volumes = []
    for v in totals.volumes:
        record = {"file": v.file, "rows": v.rows, "source": v.source}
        if v.mask_file:
            record["mask_file"] = v.mask_file
        volumes.append(record)
    return {
        "volumes": volumes,
        "num_sequences": totals.num_sequences,
        "num_tokens": totals.num_tokens,
        "num_documents": totals.num_documents,
        "num_documents_dropped": totals.num_documents_dropped,
        "num_tokens_encoded": totals.num_tokens_encoded,
        "num_tokens_dropped": totals.num_tokens_dropped,
    }
