import numpy as np

from datacore.packing import BestFitCropPacker, BestFitPadPacker, EncodedDoc
from datacore.store import FileSystemDatasetStore
from datacore.writer import write_split


def _docs(lengths, bos=1):
    return [EncodedDoc(ids=[bos] + list(range(2, 2 + n))) for n in lengths]


def test_volume_rolls_at_cap(tmp_path):
    store = FileSystemDatasetStore(str(tmp_path))
    packer = BestFitCropPacker(buffer_size=8)
    # row_capacity=4 (sequence_len=3); many tiny docs so several full rows are producible
    named_batches = [("shard0", _docs([3] * 20))]
    totals = write_split(store, "train", packer, sequence_len=3, sequences_per_volume=2,
                          vocab_size=100, named_document_batches=named_batches)
    assert totals.num_sequences > 2  # more than one volume's worth
    assert all(v.rows <= 2 for v in totals.volumes)
    assert len(totals.volumes) >= 2
    for v in totals.volumes:
        loaded = store.open_volume(v.file, mmap=False)
        assert loaded.shape == (v.rows, 4)


def test_source_boundary_forces_a_flush_even_below_cap(tmp_path):
    store = FileSystemDatasetStore(str(tmp_path))
    packer = BestFitCropPacker(buffer_size=8)
    # cap=100 (never hit); two files, each producing exactly 1 row -> two separate (short) volumes
    named_batches = [("fileA", _docs([3])), ("fileB", _docs([3]))]
    totals = write_split(store, "train", packer, sequence_len=3, sequences_per_volume=100,
                          vocab_size=100, named_document_batches=named_batches)
    assert len(totals.volumes) == 2
    assert totals.volumes[0].source == "fileA"
    assert totals.volumes[1].source == "fileB"
    assert all(v.rows == 1 for v in totals.volumes)


def test_pad_packer_writes_mask_volumes(tmp_path):
    store = FileSystemDatasetStore(str(tmp_path))
    packer = BestFitPadPacker(bos_token_id=1, buffer_size=8)
    docs = [EncodedDoc(ids=[1, 2, 3], mask=[0, 1, 1])]
    totals = write_split(store, "train", packer, sequence_len=5, sequences_per_volume=10,
                          vocab_size=100, named_document_batches=[("f", docs)])
    assert len(totals.volumes) == 1
    v = totals.volumes[0]
    assert v.mask_file is not None
    mask = store.open_volume(v.mask_file, mmap=False)
    assert mask.shape == (v.rows, 6)


def test_crop_packer_writes_no_mask_volumes(tmp_path):
    store = FileSystemDatasetStore(str(tmp_path))
    packer = BestFitCropPacker(buffer_size=8)
    totals = write_split(store, "train", packer, sequence_len=3, sequences_per_volume=10,
                          vocab_size=100, named_document_batches=[("f", _docs([3] * 5))])
    for v in totals.volumes:
        assert v.mask_file is None


def test_dropped_oversized_documents_are_counted(tmp_path):
    store = FileSystemDatasetStore(str(tmp_path))
    packer = BestFitPadPacker(bos_token_id=1, buffer_size=8)
    docs = [
        EncodedDoc(ids=[1, 2, 3], mask=[0, 1, 1]),
        EncodedDoc(ids=list(range(50)), mask=[1] * 50),  # oversized for row_capacity=6
    ]
    totals = write_split(store, "train", packer, sequence_len=5, sequences_per_volume=10,
                          vocab_size=100, named_document_batches=[("f", docs)])
    assert totals.num_documents == 2
    assert totals.num_documents_dropped == 1


def test_token_accounting_reflects_cropping_loss(tmp_path):
    store = FileSystemDatasetStore(str(tmp_path))
    packer = BestFitCropPacker(buffer_size=8)
    docs = _docs([3] * 7)  # 7 docs of len 4 (incl bos) = 28 tokens in; row_capacity=4 -> exactly packable
    totals = write_split(store, "train", packer, sequence_len=3, sequences_per_volume=10,
                          vocab_size=100, named_document_batches=[("f", docs)])
    assert totals.num_tokens_encoded == 28
    assert totals.num_tokens == totals.num_sequences * 4
    assert totals.num_tokens_dropped == totals.num_tokens_encoded - totals.num_tokens
    assert totals.num_tokens_dropped >= 0
