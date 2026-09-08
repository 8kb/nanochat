import numpy as np
import pytest

from datacore.store import FileSystemDatasetStore, FORMAT, token_dtype


def test_token_dtype_boundary():
    assert token_dtype(65535) == np.dtype("uint16")
    assert token_dtype(65536) == np.dtype("uint32")
    assert token_dtype(32768) == np.dtype("uint16")


def test_manifest_round_trip(tmp_path):
    store = FileSystemDatasetStore(str(tmp_path))
    assert store.read_manifest() is None
    manifest = {"format": FORMAT, "sequence_len": 128, "splits": {}}
    store.write_manifest(manifest)
    assert store.read_manifest() == manifest


def test_write_manifest_requires_correct_format_stamp(tmp_path):
    store = FileSystemDatasetStore(str(tmp_path))
    with pytest.raises(AssertionError):
        store.write_manifest({"format": "something-else"})


def test_read_manifest_rejects_wrong_format(tmp_path):
    import json
    import os
    os.makedirs(tmp_path, exist_ok=True)
    with open(tmp_path / "manifest.json", "w") as f:
        json.dump({"format": "not-datacore"}, f)
    store = FileSystemDatasetStore(str(tmp_path))
    with pytest.raises(ValueError):
        store.read_manifest()


def test_manifest_is_not_left_partial_on_disk_until_written(tmp_path):
    store = FileSystemDatasetStore(str(tmp_path))
    store.write_manifest({"format": FORMAT})
    # no .tmp file left behind
    assert not (tmp_path / "manifest.json.tmp").exists()
    assert (tmp_path / "manifest.json").exists()


def test_volume_write_read_round_trip(tmp_path):
    store = FileSystemDatasetStore(str(tmp_path))
    arr = np.arange(20, dtype=np.uint16).reshape(4, 5)
    filename = store.volume_filename("train", 0)
    store.write_volume(filename, arr)
    loaded = store.open_volume(filename)
    assert np.array_equal(np.asarray(loaded), arr)
    assert loaded.dtype == np.uint16


def test_volume_write_leaves_no_tmp_file(tmp_path):
    store = FileSystemDatasetStore(str(tmp_path))
    arr = np.zeros((2, 3), dtype=np.uint8)
    filename = store.volume_filename("val", 0, mask=True)
    assert filename.endswith(".mask.npy")
    store.write_volume(filename, arr)
    assert not (tmp_path / (filename + ".tmp.npy")).exists()


def test_volume_filenames_are_stable_and_split_qualified():
    store = FileSystemDatasetStore("/irrelevant")
    assert store.volume_filename("train", 3) == "train_000003.npy"
    assert store.volume_filename("val", 0, mask=True) == "val_000000.mask.npy"


def test_open_volume_is_memmapped(tmp_path):
    store = FileSystemDatasetStore(str(tmp_path))
    arr = np.arange(100, dtype=np.uint16).reshape(10, 10)
    filename = store.volume_filename("train", 0)
    store.write_volume(filename, arr)
    loaded = store.open_volume(filename, mmap=True)
    assert isinstance(loaded, np.memmap)
