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
    packer = BestFitPadPacker(bos_token_id=bos_id, buffer_size=args.buffer_size)
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
    parser.add_argument("--mmlu-epochs", type=int, default=1, help="--kind=sft: MMLU auxiliary_train epochs in the training mixture")
    parser.add_argument("--gsm8k-epochs", type=int, default=1, help="--kind=sft: GSM8K train epochs in the training mixture")
    parser.add_argument("--max-tokens-per-conversation", type=int, default=2048, help="--kind=sft: truncate a rendered conversation to this many tokens (matches RustBPETokenizer.render_conversation's own default)")
    parser.add_argument("--describe", action="store_true", help="print an existing prepared dataset's info and exit (needs --dataset)")
    args = parser.parse_args()

    if args.describe:
        assert args.dataset, "--describe needs --dataset=<name>"
        describe(prepared_dir(args.dataset))
        return

    tokenizer = get_tokenizer()
    if args.kind == "base":
        dataset_dir = prepare_base(args, tokenizer)
    else:
        dataset_dir = prepare_sft(args, tokenizer)
    print0(f"Done. Use --dataset={os.path.basename(dataset_dir)} at train time.")


if __name__ == "__main__":
    main()
