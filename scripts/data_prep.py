"""
Prepares a pretokenized, packed, multipart dataset for training. Replaces realtime tokenization:
run this once (CPU-only -- never on a billed GPU pod), then scripts/base_train.py and
scripts/chat_sft.py just read rows.

Two kinds:

    python -m scripts.data_prep --kind=base --sequence-len=2048
        Pretraining corpus (nanochat.dataset's ClimbMix shards). Last shard is held out as val,
        same rule as the old realtime dataloader used.

    python -m scripts.data_prep --kind=sft --sequence-len=2048
        The SFT task mixture (SmolTalk + MMLU + GSM8K), rendered via
        RustBPETokenizer.render_conversation. --mmlu-epochs/--gsm8k-epochs live here now (they
        used to live on scripts/chat_sft.py, back when SFT data was assembled at train time).

Both write to $NANOCHAT_BASE_DIR/prepared/<name>/ (see --dataset to override the name), and both
are CPU-only work -- see docs/architecture.md and AGENTS.md.

    python -m scripts.data_prep --describe --dataset=<name>
        Prints an existing prepared dataset's stats and exits.
"""
import argparse
import os
import time

import numpy as np

from datacore import BestFitCropPacker, BestFitPadPacker, DataManager, EncodedDoc, FileSystemDatasetStore, ParquetDirectorySource

from nanochat.common import get_base_dir, print0
from nanochat.dataset import list_parquet_files
from nanochat.tokenizer import get_tokenizer

# -----------------------------------------------------------------------------

def prepared_dir(name: str) -> str:
    return os.path.join(get_base_dir(), "prepared", name)


def default_dataset_name(kind: str, sequence_len: int, tokenizer) -> str:
    stem = "climbmix" if kind == "base" else "sft"
    return f"{stem}_t{sequence_len}_{tokenizer.fingerprint()}"


# -----------------------------------------------------------------------------
# --kind=base: the pretraining corpus

def prepare_base(args, tokenizer):
    manager = DataManager()
    dataset_name = args.dataset or default_dataset_name("base", args.sequence_len, tokenizer)
    dataset_dir = prepared_dir(dataset_name)
    store = FileSystemDatasetStore(dataset_dir)

    parquet_paths = list_parquet_files()
    assert len(parquet_paths) >= 2, (
        f"need at least 2 parquet shards (1 train + 1 val), found {len(parquet_paths)} -- "
        f"run `python -m nanochat.dataset -n N` first"
    )
    if args.max_shards is not None:
        parquet_paths = parquet_paths[: args.max_shards]
        assert len(parquet_paths) >= 2, "--max-shards must leave at least 2 shards"
    train_paths, val_paths = parquet_paths[:-1], parquet_paths[-1:]
    print0(f"Preparing base dataset {dataset_name!r}: {len(train_paths)} train shard(s), "
          f"{len(val_paths)} val shard(s) -> {dataset_dir}")

    sources = {
        "train": ParquetDirectorySource(paths=train_paths),
        "val": ParquetDirectorySource(paths=val_paths),
    }
    packer = BestFitCropPacker(buffer_size=args.buffer_size)
    t0 = time.time()
    manifest = manager.prepare(
        store, sources=sources, tokenizer=tokenizer,
        sequence_len=args.sequence_len, sequences_per_volume=args.sequences_per_volume, packer=packer,
        num_threads=args.tokenizer_threads,
    )
    print0(f"Prepared in {time.time() - t0:.1f}s")
    _print_manifest_summary(manifest)
    return dataset_dir


# -----------------------------------------------------------------------------
# --kind=sft: the SFT task mixture

class TaskMixtureTokenSource:
    """Adapts a tasks.common.TaskMixture into datacore's TokenSource protocol: each conversation
    is rendered via RustBPETokenizer.render_conversation (ids + per-token loss mask) here, in
    chunks, so DataManager.prepare gets a volume flush boundary every `chunk_size` conversations
    rather than one giant flush at the very end."""

    def __init__(self, task_mixture, tokenizer, name, max_tokens=2048, chunk_size=2000):
        self.task_mixture = task_mixture
        self.tokenizer = tokenizer
        self.name = name
        self.max_tokens = max_tokens
        self.chunk_size = chunk_size
        self.num_dropped_over_max_tokens = 0

    def token_batches(self):
        n = len(self.task_mixture)
        for start in range(0, n, self.chunk_size):
            end = min(start + self.chunk_size, n)
            docs = []
            for i in range(start, end):
                conversation = self.task_mixture[i]
                ids, mask = self.tokenizer.render_conversation(conversation, max_tokens=self.max_tokens)
                docs.append(EncodedDoc(ids=ids, mask=mask))
            yield f"{self.name}[{start}:{end}]", docs


def _build_sft_mixtures(args):
    from tasks.common import TaskMixture
    from tasks.gsm8k import GSM8K
    from tasks.mmlu import MMLU
    from tasks.smoltalk import SmolTalk

    train_tasks = [
        SmolTalk(split="train"),
        *[MMLU(subset="all", split="auxiliary_train") for _ in range(args.mmlu_epochs)],
        *[GSM8K(subset="main", split="train") for _ in range(args.gsm8k_epochs)],
    ]
    train_mixture = TaskMixture(train_tasks)
    val_mixture = TaskMixture([
        SmolTalk(split="test"),
        MMLU(subset="all", split="test", stop=5200),
        GSM8K(subset="main", split="test", stop=420),
    ])
    if args.max_conversations is not None:
        train_mixture = _Truncated(train_mixture, args.max_conversations)
        val_mixture = _Truncated(val_mixture, min(args.max_conversations, len(val_mixture)))
    return train_mixture, val_mixture


class _Truncated:
    """Caps a Task/TaskMixture's apparent length for smoke tests, without touching tasks/."""
    def __init__(self, task, limit):
        self.task = task
        self.limit = min(limit, len(task))
    def __len__(self):
        return self.limit
    def __getitem__(self, index):
        return self.task[index]


def prepare_sft(args, tokenizer):
    manager = DataManager()
    dataset_name = args.dataset or default_dataset_name("sft", args.sequence_len, tokenizer)
    dataset_dir = prepared_dir(dataset_name)
    store = FileSystemDatasetStore(dataset_dir)

    train_mixture, val_mixture = _build_sft_mixtures(args)
    print0(f"Preparing SFT dataset {dataset_name!r}: {len(train_mixture):,} train conversations "
          f"(MMLU x{args.mmlu_epochs}, GSM8K x{args.gsm8k_epochs}), {len(val_mixture):,} val -> {dataset_dir}")

    bos_id = tokenizer.get_bos_token_id()
    sources = {
        "train": TaskMixtureTokenSource(train_mixture, tokenizer, "train", max_tokens=args.max_tokens_per_conversation),
        "val": TaskMixtureTokenSource(val_mixture, tokenizer, "val", max_tokens=args.max_tokens_per_conversation),
    }
    packer = BestFitPadPacker(bos_token_id=bos_id, padding_id=args.sft_padding_id, buffer_size=args.buffer_size)
    t0 = time.time()
    manifest = manager.prepare(
        store, sources=sources, tokenizer=tokenizer,
        sequence_len=args.sequence_len, sequences_per_volume=args.sequences_per_volume, packer=packer,
    )
    print0(f"Prepared in {time.time() - t0:.1f}s")
    _print_manifest_summary(manifest)
    return dataset_dir


# -----------------------------------------------------------------------------

def _print_manifest_summary(manifest):
    for split, data in manifest["splits"].items():
        ratio = 100.0 * data["num_tokens"] / max(data["num_tokens_encoded"], 1)
        # >100% is expected for a pad packer (padding adds tokens rather than dropping any);
        # <100% is expected for a crop packer (cropping drops tokens to hit 100% utilization).
        print0(f"  {split}: {data['num_sequences']:,} sequences, {data['num_tokens']:,} tokens on disk "
              f"({ratio:.1f}% of {data['num_tokens_encoded']:,} encoded tokens), "
              f"{data['num_documents']:,} documents ({data['num_documents_dropped']:,} dropped as "
              f"oversized), {len(data['volumes'])} volume(s)")


def describe(dataset_dir):
    store = FileSystemDatasetStore(dataset_dir)
    manager = DataManager()
    dataset = manager.open(store)
    info = dataset.info
    print0(f"{dataset_dir}")
    print0(f"  format: datacore.v1 | sequence_len: {info.sequence_len} | vocab_size: {info.vocab_size}")
    print0(f"  dtype: {info.dtype} | has_mask: {info.has_mask} | packer: {info.packer_name}")
    print0(f"  tokenizer_fingerprint: {info.tokenizer_fingerprint}")
    for split, data in info.splits.items():
        ratio = 100.0 * data["num_tokens"] / max(data["num_tokens_encoded"], 1)
        print0(f"  {split}: {data['num_sequences']:,} sequences ({ratio:.1f}% of encoded tokens on disk), "
              f"{data['num_volumes']} volume(s)")


# -----------------------------------------------------------------------------
# --describe --deep: row-level statistics, computed by scanning every row already on disk.
#
# Nothing here is recorded at prepare() time -- it's derived purely from a dataset's existing
# tokens/mask volumes plus its manifest's bos_token_id/padding_id, so it works uniformly on any
# prepared dataset, including ones prepared before this existed. See docs/contest.md's Stage 9 for
# why this exists: --doc-masking-max-docs-per-row was a guess (DEFAULT_MAX_DOCS_PER_ROW=64) that
# OOM'd a real run, and a padding_id dataset's "did this change anything real" question was
# answered by an ad-hoc one-off script instead of a repeatable tool.

def _pad_suffix_lengths(tokens, padding_id):
    """Per-row length of the trailing run of `padding_id` (0 if none). A packer only ever fills
    *unused remaining capacity* after the last real document, left-to-right, so the pad tail -- if
    any -- is always contiguous and at the very end of the row; this does not need to look inside
    the row at all."""
    is_pad = tokens == padding_id
    reversed_nonpad = ~is_pad[:, ::-1]
    has_nonpad = reversed_nonpad.any(axis=1)
    first_nonpad_from_end = np.argmax(reversed_nonpad, axis=1)
    T = tokens.shape[1]
    return np.where(has_nonpad, first_nonpad_from_end, T)


def _document_boundaries(tokens, bos_token_id, content_len):
    """Per-row document start columns within [0, content_len[row]) (the pad tail, if any, is
    excluded before this is called -- a pad tail can itself start with bos_token_id when
    padding_id defaults to it, and must not be counted as a document). Returns (rows_idx, cols_idx)
    -- one entry per document, row-major order -- ready to feed into length computation."""
    R, T = tokens.shape
    is_bos = tokens == bos_token_id
    is_start = is_bos.copy()
    is_start[:, 1:] = is_bos[:, 1:] & ~is_bos[:, :-1]
    col_idx = np.arange(T)[None, :]
    is_start &= col_idx < content_len[:, None]
    rows_idx, cols_idx = np.nonzero(is_start)
    docs_per_row = np.bincount(rows_idx, minlength=R)
    # A row with content but no bos at all (shouldn't happen for a real dataset, but the scan must
    # not silently drop it) -> treat its whole content as one document starting at column 0.
    no_start_rows = np.nonzero((docs_per_row == 0) & (content_len > 0))[0]
    if no_start_rows.size:
        rows_idx = np.concatenate([rows_idx, no_start_rows])
        cols_idx = np.concatenate([cols_idx, np.zeros_like(no_start_rows)])
        order = np.lexsort((cols_idx, rows_idx))
        rows_idx, cols_idx = rows_idx[order], cols_idx[order]
    return rows_idx, cols_idx


def _scan_split(manager, dataset, split, bos_token_id, padding_id, chunk_rows=8192):
    n = dataset.num_sequences(split)
    doc_lengths_chunks, docs_per_row_chunks, pad_len_chunks = [], [], []
    total_tokens = mask1 = mask0 = 0
    start = 0
    while start < n:
        count = min(chunk_rows, n - start)
        tokens, mask = manager.read_rows(dataset, split, start, count)
        tokens = np.asarray(tokens)
        R, T = tokens.shape
        total_tokens += R * T
        if mask is not None:
            mask = np.asarray(mask)
            mask1 += int(mask.sum())
            mask0 += int((mask == 0).sum())

        pad_len = _pad_suffix_lengths(tokens, padding_id) if padding_id is not None else np.zeros(R, dtype=np.int64)
        pad_len_chunks.append(pad_len)
        content_len = T - pad_len

        rows_idx, cols_idx = _document_boundaries(tokens, bos_token_id, content_len)
        docs_per_row_chunks.append(np.bincount(rows_idx, minlength=R))
        # next_rows/next_cols shifted by one WITHOUT wraparound: np.roll(..., -1) would compare the
        # very last document in the chunk against the FIRST one, which spuriously looks like "same
        # row" (and so computes a bogus length via next_cols - cols_idx) whenever the whole chunk
        # holds documents from only a single row -- a real case, not just a tiny-test artifact: any
        # dataset's final partial chunk can shrink to exactly one row. -1 is a row index that never
        # occurs for real, so the last document always correctly falls through to "last in its row".
        next_rows = np.append(rows_idx[1:], -1)
        next_cols = np.append(cols_idx[1:], 0)
        same_row = next_rows == rows_idx
        lengths = np.where(same_row, next_cols - cols_idx, content_len[rows_idx] - cols_idx)
        doc_lengths_chunks.append(lengths)

        start += count

    doc_lengths = np.concatenate(doc_lengths_chunks) if doc_lengths_chunks else np.zeros(0, dtype=np.int64)
    docs_per_row = np.concatenate(docs_per_row_chunks) if docs_per_row_chunks else np.zeros(0, dtype=np.int64)
    pad_lens = np.concatenate(pad_len_chunks) if pad_len_chunks else np.zeros(0, dtype=np.int64)
    has_padding = pad_lens > 0
    return {
        "num_documents": int(doc_lengths.size),
        "doc_len_min": int(doc_lengths.min()) if doc_lengths.size else 0,
        "doc_len_mean": float(doc_lengths.mean()) if doc_lengths.size else 0.0,
        "doc_len_p50": float(np.percentile(doc_lengths, 50)) if doc_lengths.size else 0.0,
        "doc_len_p90": float(np.percentile(doc_lengths, 90)) if doc_lengths.size else 0.0,
        "doc_len_p99": float(np.percentile(doc_lengths, 99)) if doc_lengths.size else 0.0,
        "doc_len_max": int(doc_lengths.max()) if doc_lengths.size else 0,
        "docs_per_row_min": int(docs_per_row.min()) if docs_per_row.size else 0,
        "docs_per_row_mean": float(docs_per_row.mean()) if docs_per_row.size else 0.0,
        "docs_per_row_p99": float(np.percentile(docs_per_row, 99)) if docs_per_row.size else 0.0,
        "docs_per_row_max": int(docs_per_row.max()) if docs_per_row.size else 0,
        "pad_tokens_total": int(pad_lens.sum()),
        "pad_token_share": float(pad_lens.sum()) / max(total_tokens, 1),
        "rows_with_padding": int(has_padding.sum()),
        "pad_tail_mean_when_present": float(pad_lens[has_padding].mean()) if has_padding.any() else 0.0,
        "mask1_tokens": mask1,
        "mask0_tokens": mask0,
        "total_tokens_scanned": total_tokens,
    }


def deep_scan(dataset_dir):
    """Returns (DatasetInfo, {split: stats_dict}) for a prepared dataset already on disk -- no
    re-prep, works on any dataset regardless of when it was prepared."""
    store = FileSystemDatasetStore(dataset_dir)
    manager = DataManager()
    dataset = manager.open(store)
    info = dataset.info
    # info.padding_id is None in two DIFFERENT situations that must not be conflated: (1) the
    # packer has no padding concept at all (bestfit_crop -- there is genuinely nothing to detect),
    # or (2) the manifest predates the padding_id field (a bestfit_pad dataset prepared before that
    # commit), in which case the packer still padded every row -- with bos_token_id, its own
    # documented default (datacore/packing.py's BestFitPadPacker.__init__) -- it just never
    # recorded that choice. Treating (2) as "no padding" undercounts real padding to exactly 0%
    # and, worse, miscounts each padded row's bos-valued tail as a spurious extra one-token
    # document, since it's then included in the "real content" region doc-boundary detection scans.
    effective_padding_id = info.padding_id
    if effective_padding_id is None and info.packer_name == "bestfit_pad":
        effective_padding_id = info.bos_token_id
    splits = {split: _scan_split(manager, dataset, split, info.bos_token_id, effective_padding_id)
              for split in info.splits} if info.bos_token_id is not None else {}
    return info, splits


def _print_deep_stats(label, info, splits):
    print0(f"-- deep stats: {label} --")
    if info.bos_token_id is None:
        print0("  (manifest predates bos_token_id -- cannot compute document stats)")
        return
    for split, s in splits.items():
        print0(f"  [{split}]")
        print0(f"    documents: {s['num_documents']:,} | length min/mean/p50/p90/p99/max: "
              f"{s['doc_len_min']}/{s['doc_len_mean']:.1f}/{s['doc_len_p50']:.1f}/"
              f"{s['doc_len_p90']:.1f}/{s['doc_len_p99']:.1f}/{s['doc_len_max']}")
        print0(f"    documents/row: min/mean/p99/max: {s['docs_per_row_min']}/{s['docs_per_row_mean']:.2f}/"
              f"{s['docs_per_row_p99']:.1f}/{s['docs_per_row_max']}  "
              f"(--doc-masking-max-docs-per-row should be set from this max)")
        print0(f"    padding: {s['pad_tokens_total']:,} tokens ({100 * s['pad_token_share']:.2f}% of "
              f"{s['total_tokens_scanned']:,}), {s['rows_with_padding']:,} rows with any padding, "
              f"mean pad tail when present: {s['pad_tail_mean_when_present']:.1f}")
        if s["mask1_tokens"] or s["mask0_tokens"]:
            total_mask = s["mask1_tokens"] + s["mask0_tokens"]
            print0(f"    loss mask: {s['mask1_tokens']:,} supervised "
                  f"({100 * s['mask1_tokens'] / max(total_mask, 1):.1f}%), {s['mask0_tokens']:,} masked out")


_COMPARE_KEYS = ("num_documents", "doc_len_mean", "docs_per_row_max", "pad_tokens_total",
                  "pad_token_share", "mask1_tokens", "mask0_tokens", "total_tokens_scanned")


def _print_deep_stats_compare(label_a, splits_a, label_b, splits_b):
    print0(f"-- comparing {label_a} vs {label_b} --")
    for split in splits_a:
        if split not in splits_b:
            print0(f"  [{split}]: only present in {label_a}")
            continue
        a, b = splits_a[split], splits_b[split]
        print0(f"  [{split}]")
        for key in _COMPARE_KEYS:
            va, vb = a[key], b[key]
            delta = f" (delta {vb - va:+.4g})" if vb != va else " (identical)"
            print0(f"    {key}: {va:,.4g} vs {vb:,.4g}{delta}" if isinstance(va, float) else
                  f"    {key}: {va:,} vs {vb:,}{delta}")


# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--kind", type=str, default="base", choices=["base", "sft"], help="which corpus to prepare")
    parser.add_argument("--dataset", type=str, default=None, help="dataset name (default: derived from kind/sequence-len/tokenizer fingerprint)")
    parser.add_argument("--sequence-len", type=int, default=2048, help="fixed sequence length for every row (must match --max-seq-len at train time)")
    parser.add_argument("--sequences-per-volume", type=int, default=16384, help="cap on sequences per volume file (a volume also flushes at every source-file boundary)")
    parser.add_argument("--buffer-size", type=int, default=1000, help="packer lookback buffer size")
    parser.add_argument("--tokenizer-threads", type=int, default=os.cpu_count() or 4, help="--kind=base: threads for tokenizer.encode() batch calls (tiktoken releases the GIL, so this is real parallelism -- see docs)")
    parser.add_argument("--max-shards", type=int, default=None, help="--kind=base: limit to this many parquet shards (smoke tests)")
    parser.add_argument("--max-conversations", type=int, default=None, help="--kind=sft: limit each mixture to this many conversations (smoke tests)")
    parser.add_argument("--mmlu-epochs", type=int, default=3, help="--kind=sft: MMLU auxiliary_train epochs in the training mixture (matches scripts/chat_sft.py's pre-datacore default)")
    parser.add_argument("--gsm8k-epochs", type=int, default=4, help="--kind=sft: GSM8K train epochs in the training mixture (matches scripts/chat_sft.py's pre-datacore default)")
    parser.add_argument("--max-tokens-per-conversation", type=int, default=2048, help="--kind=sft: truncate a rendered conversation to this many tokens (matches RustBPETokenizer.render_conversation's own default)")
    parser.add_argument("--sft-padding-id", type=int, default=None, help="--kind=sft: token id BestFitPadPacker uses to fill unused row capacity (default: None, which resolves to bos_token_id -- this packer's original behavior). A distinct id avoids the pad tail looking like a document to BOS-based document-boundary logic; see modelcore.kernels.flash_attn.build_doc_args's matching padding_id parameter")
    parser.add_argument("--describe", action="store_true", help="print an existing prepared dataset's info and exit (needs --dataset)")
    parser.add_argument("--deep", action="store_true", help="--describe: also scan every row for document-length/documents-per-row/padding/loss-mask statistics -- no re-prep, works on any prepared dataset regardless of when it was made; can take a while on a large dataset, run it on a cheap CPU pod rather than skipping it")
    parser.add_argument("--compare-to", type=str, default=None, help="--describe --deep: also deep-scan this other dataset name and print a side-by-side comparison (e.g. to confirm two datasets differ only in the way expected, before paying for a GPU run on the new one)")
    args = parser.parse_args()

    if args.describe:
        assert args.dataset, "--describe needs --dataset=<name>"
        describe(prepared_dir(args.dataset))
        if args.deep:
            info, splits = deep_scan(prepared_dir(args.dataset))
            _print_deep_stats(args.dataset, info, splits)
            if args.compare_to:
                describe(prepared_dir(args.compare_to))
                info2, splits2 = deep_scan(prepared_dir(args.compare_to))
                _print_deep_stats(args.compare_to, info2, splits2)
                _print_deep_stats_compare(args.dataset, splits, args.compare_to, splits2)
        return

    tokenizer = get_tokenizer()
    if args.kind == "base":
        dataset_dir = prepare_base(args, tokenizer)
    else:
        dataset_dir = prepare_sft(args, tokenizer)
    print0(f"Done. Use --dataset={os.path.basename(dataset_dir)} at train time.")


if __name__ == "__main__":
    main()
