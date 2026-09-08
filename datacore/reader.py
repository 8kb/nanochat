"""
reader.py: opens a prepared dataset's manifest, memmaps its volumes, and drives the cursor-based,
DDP-sharded, resumable batch iterator. The only module in datacore that imports torch -- and only
inside batches(), lazily -- since prepare()/writer.py never need a tensor. See
datacore/docs/architecture.md's "Read order, DDP, and resume" section for the design this
implements.
"""
from dataclasses import dataclass

import numpy as np

FORMAT = "datacore.v1"


@dataclass(frozen=True)
class DatasetInfo:
    """What DataManager.open returns for presentation -- the host application formats it (same
    rule as modelcore.ModelStats: a value type, not a print statement, crosses the boundary)."""
    sequence_len: int
    vocab_size: int
    dtype: str
    has_mask: bool
    tokenizer_fingerprint: str
    packer_name: str
    splits: dict  # split_name -> {"num_sequences", "num_tokens", "num_documents",
                  #                "num_documents_dropped", "num_tokens_encoded", "num_tokens_dropped"}


class SplitIndex:
    """Maps a global sequence index (0..N-1) to (volume_index, local_row) via a prefix sum of
    per-volume row counts, and memmaps volumes on demand behind a small LRU cache."""

    def __init__(self, store, volumes, has_mask, cache_size=2):
        self.store = store
        self.volumes = volumes  # list of {"file", "rows", "mask_file"} dicts, manifest order
        self.has_mask = has_mask
        prefix = np.zeros(len(volumes) + 1, dtype=np.int64)
        for i, v in enumerate(volumes):
            prefix[i + 1] = prefix[i] + v["rows"]
        self._prefix = prefix
        self.num_sequences = int(prefix[-1])
        self._cache_size = cache_size
        self._cache = {}
        self._cache_order = []

    def _open(self, volume_index):
        if volume_index in self._cache:
            self._cache_order.remove(volume_index)
            self._cache_order.append(volume_index)
            return self._cache[volume_index]
        v = self.volumes[volume_index]
        tokens = self.store.open_volume(v["file"], mmap=True)
        mask = self.store.open_volume(v["mask_file"], mmap=True) if (self.has_mask and v.get("mask_file")) else None
        self._cache[volume_index] = (tokens, mask)
        self._cache_order.append(volume_index)
        if len(self._cache_order) > self._cache_size:
            del self._cache[self._cache_order.pop(0)]
        return tokens, mask

    def locate(self, global_index):
        volume_index = int(np.searchsorted(self._prefix, global_index, side="right")) - 1
        local_row = global_index - int(self._prefix[volume_index])
        return volume_index, local_row

    def read_contiguous(self, start, count):
        """[start, start+count) must not wrap past num_sequences -- callers split a wrapping
        rank-batch into non-wrapping runs before calling this. May still span several volumes;
        handled with a small loop, not a special-cased 2-volume assumption."""
        assert 0 <= start and start + count <= self.num_sequences, \
            f"read_contiguous({start}, {count}) out of range for {self.num_sequences} sequences"
        assert count > 0, "read_contiguous(start, 0) is not meaningful -- caller bug"
        chunks_t, chunks_m = [], []
        pos, remaining = start, count
        while remaining > 0:
            vol_idx, local = self.locate(pos)
            tokens, mask = self._open(vol_idx)
            vol_rows = self.volumes[vol_idx]["rows"]
            take = min(remaining, vol_rows - local)
            chunks_t.append(np.asarray(tokens[local:local + take]))
            chunks_m.append(np.asarray(mask[local:local + take]) if mask is not None else None)
            pos += take
            remaining -= take
        tokens_out = chunks_t[0] if len(chunks_t) == 1 else np.concatenate(chunks_t, axis=0)
        mask_out = None
        if self.has_mask:
            width = tokens_out.shape[1]
            filled = [m if m is not None else np.ones((c.shape[0], width), dtype=np.uint8)
                      for c, m in zip(chunks_t, chunks_m)]
            mask_out = filled[0] if len(filled) == 1 else np.concatenate(filled, axis=0)
        return tokens_out, mask_out


class Dataset:
    def __init__(self, store, manifest):
        self.store = store
        self.manifest = manifest
        has_mask = manifest["has_mask"]
        self._split_indices = {
            split: SplitIndex(store, data["volumes"], has_mask)
            for split, data in manifest["splits"].items()
        }
        self.info = DatasetInfo(
            sequence_len=manifest["sequence_len"],
            vocab_size=manifest["vocab_size"],
            dtype=manifest["dtype"],
            has_mask=has_mask,
            tokenizer_fingerprint=manifest["tokenizer_fingerprint"],
            packer_name=manifest["packer"]["name"],
            splits={
                split: {**{k: v for k, v in data.items() if k != "volumes"}, "num_volumes": len(data["volumes"])}
                for split, data in manifest["splits"].items()
            },
        )

    def num_sequences(self, split: str) -> int:
        return self._split_indices[split].num_sequences


def open_dataset(store) -> Dataset:
    manifest = store.read_manifest()
    if manifest is None:
        raise FileNotFoundError(
            "no manifest found -- either prepare() was never run against this store, or it "
            "didn't finish (the manifest is written last, so a half-prepared directory has none)"
        )
    return Dataset(store, manifest)


def _lazy_torch():
    try:
        import torch
    except ImportError as e:
        raise ImportError(
            "datacore.reader.batches() needs torch (pip install 'datacore[torch]', or just torch "
            "-- prepare()/write_split() never need it, only reading batches does)"
        ) from e
    return torch


def batches(dataset: Dataset, split: str, batch_size: int, *, rank: int = 0, world_size: int = 1,
            device=None, resume: dict | None = None, infinite: bool = True):
    """Yields (inputs, targets, state) tuples.

    inputs: torch.int32 (batch_size, sequence_len). targets: torch.int64 (batch_size,
    sequence_len), with -1 (ignore_index) wherever the dataset's mask says 0 (a plain pretraining
    dataset with no mask supervises every position). state: the resumable cursor dict -- persist
    it (e.g. in checkpoint meta) and pass it back as `resume` to continue exactly where this rank
    left off.

    rank/world_size are explicit parameters, never read from the environment -- see
    datacore/docs/architecture.md's "no ambient globals" rule (the same one modelcore.runtime
    applies to compute dtype). A batch may straddle the end of the split, mixing rows from epoch e
    and e+1 -- deliberate: it keeps the entire iterator state one integer (`cursor`), and resuming
    at a DIFFERENT world_size than the run that saved the state still produces a gap-free,
    duplicate-free continuation of the global stream (only bit-exact reproduction of the batch
    contents needs matching batch_size/world_size).
    """
    idx = dataset._split_indices[split]
    N = idx.num_sequences
    if N == 0:
        raise ValueError(f"split {split!r} has zero sequences")
    B, W = batch_size, world_size

    if resume is not None:
        if resume.get("format") != FORMAT:
            raise ValueError(f"unrecognized resume state format {resume.get('format')!r}, expected {FORMAT!r}")
        if resume.get("num_sequences") != N:
            raise ValueError(
                f"resume state was saved against a dataset with {resume['num_sequences']} "
                f"sequences; this one has {N} -- wrong dataset, or it was re-prepared?"
            )
        cursor = resume["cursor"]
    else:
        cursor = 0

    torch = _lazy_torch()

    while True:
        local_start = (cursor + rank * B) % N
        chunks_t, chunks_m = [], []
        pos, remaining = local_start, B
        while remaining > 0:
            take = min(remaining, N - pos)
            t, m = idx.read_contiguous(pos, take)
            chunks_t.append(t)
            chunks_m.append(m)
            remaining -= take
            pos = (pos + take) % N
        tokens = chunks_t[0] if len(chunks_t) == 1 else np.concatenate(chunks_t, axis=0)

        inputs_np = tokens[:, :-1].astype(np.int32)
        targets_np = tokens[:, 1:].astype(np.int64)
        if idx.has_mask:
            mask = chunks_m[0] if len(chunks_m) == 1 else np.concatenate(chunks_m, axis=0)
            targets_np[mask[:, 1:] == 0] = -1

        inputs = torch.from_numpy(inputs_np)
        targets = torch.from_numpy(targets_np)
        if device is not None:
            # pin_memory()/non_blocking transfer is a CUDA-specific optimization (matches
            # nanochat/dataloader.py's original `use_cuda = device == "cuda"` string check) --
            # MPS rejects a pinned-CPU-storage tensor moved non_blocking with a device-mismatch
            # RuntimeError, and a plain .to(device) is already synchronous there anyway.
            use_cuda = str(device).startswith("cuda")
            if use_cuda:
                inputs = inputs.pin_memory().to(device, non_blocking=True)
                targets = targets.pin_memory().to(device, non_blocking=True)
            else:
                inputs = inputs.to(device)
                targets = targets.to(device)

        cursor += B * W
        state = {
            "format": FORMAT, "cursor": cursor, "epoch": cursor // N,
            "num_sequences": N, "batch_size": B, "world_size": W,
        }
        yield inputs, targets, state

        if not infinite and cursor >= N:
            return
