import numpy as np
import pytest

from datacore.packing import BestFitCropPacker, BestFitPadPacker, EncodedDoc
from datacore.reader import batches, open_dataset
from datacore.store import FileSystemDatasetStore
from datacore.writer import write_split

torch = pytest.importorskip("torch")


def _make_dataset(tmp_path, n_sequences, sequence_len=4, mask=False, volume_cap=3, vocab_size=50):
    """Builds a tiny prepared dataset with rows [k, k+1, ..., k+sequence_len] for k in
    0..n_sequences-1 (via the crop packer on already-right-sized documents), so a row's content
    is trivially checkable against its global index."""
    store = FileSystemDatasetStore(str(tmp_path))
    row_capacity = sequence_len + 1
    docs = [EncodedDoc(ids=list(range(k, k + row_capacity))) for k in range(n_sequences)]
    packer = BestFitCropPacker(buffer_size=n_sequences + 1)
    totals = write_split(store, "train", packer, sequence_len, volume_cap, vocab_size, [("f", docs)])
    assert totals.num_sequences == n_sequences  # every doc is already exactly row_capacity wide
    manifest = {
        "format": "datacore.v1", "sequence_len": sequence_len, "stride": row_capacity,
        "dtype": "uint16", "has_mask": False, "vocab_size": vocab_size, "bos_token_id": 0,
        "tokenizer_fingerprint": "test", "packer": {"name": packer.name, "params": {}},
        "sequences_per_volume": volume_cap,
        "splits": {"train": {
            "volumes": [{"file": v.file, "rows": v.rows, "source": v.source} for v in totals.volumes],
            "num_sequences": totals.num_sequences, "num_tokens": totals.num_tokens,
            "num_documents": totals.num_documents, "num_documents_dropped": 0,
            "num_tokens_encoded": totals.num_tokens_encoded, "num_tokens_dropped": totals.num_tokens_dropped,
        }},
    }
    store.write_manifest(manifest)
    return store


def test_row_content_is_inputs_shift_targets(tmp_path):
    store = _make_dataset(tmp_path, n_sequences=6, sequence_len=4, volume_cap=2)
    dataset = open_dataset(store)
    gen = batches(dataset, "train", batch_size=6, infinite=False)
    inputs, targets, state = next(gen)
    for row in range(6):
        # row k's raw content is [k, k+1, ..., k+4]; inputs = [:-1], targets = [1:]
        assert inputs[row].tolist() == [row, row + 1, row + 2, row + 3]
        assert targets[row].tolist() == [row + 1, row + 2, row + 3, row + 4]
    assert inputs.dtype == torch.int32
    assert targets.dtype == torch.int64


def test_batch_spans_a_volume_boundary(tmp_path):
    # volume_cap=2, 5 sequences -> volumes of size [2,2,1]; a batch_size=4 read starting at 0
    # spans the first two volumes entirely and one row of the third.
    store = _make_dataset(tmp_path, n_sequences=5, sequence_len=3, volume_cap=2)
    dataset = open_dataset(store)
    gen = batches(dataset, "train", batch_size=4, infinite=False)
    inputs, targets, state = next(gen)
    for row in range(4):
        assert inputs[row].tolist() == [row, row + 1, row + 2]


def test_mask_produces_ignore_index_targets(tmp_path):
    store_dir = tmp_path
    store = FileSystemDatasetStore(str(store_dir))
    packer = BestFitPadPacker(bos_token_id=9, buffer_size=4)
    docs = [EncodedDoc(ids=[9, 1, 2], mask=[0, 1, 1])]  # -> row [9,1,2,9,9,9] mask [0,1,1,0,0,0]
    totals = write_split(store, "train", packer, sequence_len=5, sequences_per_volume=10, vocab_size=50,
                          named_document_batches=[("f", docs)])
    manifest = {
        "format": "datacore.v1", "sequence_len": 5, "stride": 6, "dtype": "uint16", "has_mask": True,
        "vocab_size": 50, "bos_token_id": 9, "tokenizer_fingerprint": "t",
        "packer": {"name": packer.name, "params": {}}, "sequences_per_volume": 10,
        "splits": {"train": {
            "volumes": [{"file": v.file, "rows": v.rows, "source": v.source, "mask_file": v.mask_file}
                        for v in totals.volumes],
            "num_sequences": totals.num_sequences, "num_tokens": totals.num_tokens,
            "num_documents": totals.num_documents, "num_documents_dropped": 0,
            "num_tokens_encoded": totals.num_tokens_encoded, "num_tokens_dropped": totals.num_tokens_dropped,
        }},
    }
    store.write_manifest(manifest)
    dataset = open_dataset(store)
    inputs, targets, state = next(batches(dataset, "train", batch_size=1, infinite=False))
    assert inputs[0].tolist() == [9, 1, 2, 9, 9]
    assert targets[0].tolist() == [1, 2, -1, -1, -1]  # mask[1:] = [1,1,0,0,0] -> keep,keep,-1,-1,-1


def test_flexible_batch_size_same_multiset(tmp_path):
    store = _make_dataset(tmp_path, n_sequences=12, sequence_len=3, volume_cap=4)
    dataset = open_dataset(store)
    for B in (1, 2, 5):
        seen = set()
        cursor_state = None
        consumed = 0
        gen = batches(dataset, "train", batch_size=B, resume=cursor_state, infinite=True)
        while consumed < 12:
            inputs, targets, state = next(gen)
            for row in inputs:
                seen.add(row[0].item())
            consumed += B
        assert seen == set(range(12))


def test_disjointness_across_ranks(tmp_path):
    N = 17
    store = _make_dataset(tmp_path, n_sequences=N, sequence_len=2, volume_cap=5)
    dataset = open_dataset(store)
    B, W, K = 3, 4, 10  # N not divisible by B or W
    gens = [batches(dataset, "train", batch_size=B, rank=r, world_size=W, infinite=True) for r in range(W)]
    all_indices = []
    for _ in range(K):
        step_indices = []
        for g in gens:
            inputs, targets, state = next(g)
            step_indices.extend(row[0].item() for row in inputs)
        assert len(step_indices) == len(set(step_indices)), "duplicate index within one step across ranks"
        all_indices.append(step_indices)
    # the union over K steps should be exactly arange(0, K*B*W) % N (as a multiset over N)
    flat = [i for step in all_indices for i in step]
    expected = [(i) % N for i in range(K * B * W)]
    assert flat == expected


def test_exact_resume(tmp_path):
    store = _make_dataset(tmp_path, n_sequences=20, sequence_len=3, volume_cap=3)
    dataset = open_dataset(store)
    B = 3
    gen_a = batches(dataset, "train", batch_size=B, infinite=True)
    for _ in range(5):
        _, _, state = next(gen_a)
    tail_a = [next(gen_a)[0].tolist() for _ in range(5)]

    gen_b = batches(dataset, "train", batch_size=B, resume=state, infinite=True)
    tail_b = [next(gen_b)[0].tolist() for _ in range(5)]

    assert tail_a == tail_b


def test_world_size_change_on_resume_has_no_gaps_or_duplicates(tmp_path):
    N = 24
    store = _make_dataset(tmp_path, n_sequences=N, sequence_len=2, volume_cap=6)
    dataset = open_dataset(store)
    B = 2
    # phase 1: W=2, run a few steps, save state from rank 0 (all ranks share the same cursor logic)
    gens_w2 = [batches(dataset, "train", batch_size=B, rank=r, world_size=2, infinite=True) for r in range(2)]
    seen = set()
    state = None
    for _ in range(3):
        for g in gens_w2:
            inputs, _, state = next(g)
            seen.update(row[0].item() for row in inputs)
    cursor_after_phase1 = state["cursor"]

    # phase 2: resume at W=4 from the saved cursor
    gens_w4 = [batches(dataset, "train", batch_size=B, rank=r, world_size=4, resume=state, infinite=True)
               for r in range(4)]
    for _ in range(3):
        step_indices = []
        for g in gens_w4:
            inputs, _, state = next(g)
            step_indices.extend(row[0].item() for row in inputs)
        assert len(step_indices) == len(set(step_indices))
        seen.update(step_indices)

    # every index consumed across both phases, mapped back to the cursor's own generation order,
    # should equal the contiguous range the cursor arithmetic predicts -- no gaps, no dupes overall
    total_consumed = cursor_after_phase1 + 3 * B * 4
    expected = set((i) % N for i in range(total_consumed))
    assert seen == expected


def test_epoch_wraps_and_state_epoch_field(tmp_path):
    N = 10
    store = _make_dataset(tmp_path, n_sequences=N, sequence_len=2, volume_cap=4)
    dataset = open_dataset(store)
    gen = batches(dataset, "train", batch_size=4, infinite=True)
    _, _, s1 = next(gen)  # cursor=4, epoch=0
    _, _, s2 = next(gen)  # cursor=8, epoch=0
    inputs3, _, s3 = next(gen)  # cursor=12, epoch=1, this batch wraps N=10
    assert s1["epoch"] == 0 and s2["epoch"] == 0
    assert s3["epoch"] == 1
    # rows 8,9 then wrap to 0,1
    assert inputs3[:, 0].tolist() == [8, 9, 0, 1]


def test_resume_rejects_mismatched_dataset_size(tmp_path):
    store = _make_dataset(tmp_path, n_sequences=10, sequence_len=2, volume_cap=4)
    dataset = open_dataset(store)
    bad_state = {"format": "datacore.v1", "cursor": 3, "num_sequences": 999, "batch_size": 2, "world_size": 1}
    with pytest.raises(ValueError):
        next(batches(dataset, "train", batch_size=2, resume=bad_state))


def test_resume_rejects_unrecognized_format(tmp_path):
    store = _make_dataset(tmp_path, n_sequences=10, sequence_len=2, volume_cap=4)
    dataset = open_dataset(store)
    bad_state = {"format": "some-other-format", "cursor": 3}
    with pytest.raises(ValueError):
        next(batches(dataset, "train", batch_size=2, resume=bad_state))


def test_device_transfer_works_on_mps(tmp_path):
    # Regression test: pin_memory()+non_blocking is a CUDA-only optimization: MPS raises
    # "Attempted to set the storage of a tensor on device cpu to a storage on different device
    # mps:0" if a pinned CPU tensor is moved non_blocking. Only meaningful on a Mac with MPS.
    if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        pytest.skip("MPS not available on this machine")
    store = _make_dataset(tmp_path, n_sequences=6, sequence_len=3, volume_cap=2)
    dataset = open_dataset(store)
    inputs, targets, _ = next(batches(dataset, "train", batch_size=2, device=torch.device("mps")))
    assert inputs.device.type == "mps"
    assert targets.device.type == "mps"


def test_open_dataset_raises_on_missing_manifest(tmp_path):
    store = FileSystemDatasetStore(str(tmp_path))
    with pytest.raises(FileNotFoundError):
        open_dataset(store)
