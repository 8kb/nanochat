"""
DatasetStore: what DataManager reads/writes a prepared dataset's manifest and volumes through.
datacore owns the artifact *format* (a manifest plus plain .npy volumes); a store just knows where
the bytes live. FileSystemDatasetStore is a directory convention any host application can point at
its own prepared-data directory -- mirrors modelcore/store.py's ArtifactStore/FileSystemStore
split.

A store is deliberately narrow: read/write the manifest dict, and read/write one volume (a numpy
array) by filename. Anything about a dataset's *identity* -- which corpus, which packer, which
tokenizer, which splits exist -- is a manifest field decided by whoever calls
DataManager.prepare, not a store concern.
"""
import json
import os

import numpy as np

FORMAT = "datacore.v1"


def token_dtype(vocab_size: int) -> np.dtype:
    """uint16 covers any tokenizer vocab up to 65535 (every tokenizer this repo has used, and
    plenty of headroom); uint32 covers the rest. -1 (ignore_index) is never stored on disk -- see
    reader.py, which derives it from the mask plane at read time -- so an unsigned dtype is
    always safe here, unlike in a torch tensor headed for nn.Embedding."""
    return np.dtype("uint16") if vocab_size <= 65535 else np.dtype("uint32")


class DatasetStore:
    """Protocol every store implements. Not an ABC -- duck typing is enough, and datacore has no
    business enforcing what a caller's store subclasses from."""

    def read_manifest(self):
        raise NotImplementedError

    def write_manifest(self, manifest: dict) -> None:
        raise NotImplementedError

    def write_volume(self, filename: str, array: np.ndarray) -> None:
        raise NotImplementedError

    def open_volume(self, filename: str, mmap: bool = True) -> np.ndarray:
        raise NotImplementedError


class FileSystemDatasetStore(DatasetStore):
    """One directory: manifest.json plus `<split>_<index:06d>.npy` (+ a sibling `.mask.npy` when
    the packer emits masks) volumes. Every write goes through a `.tmp` file then `os.replace` --
    including the manifest, which is written LAST by DataManager.prepare, so a half-prepared
    dataset directory can never be mistaken for a complete one (same idiom as
    nanochat.dataset.download_single_file and tasks/common.py's "manifest written last")."""

    def __init__(self, dataset_dir: str):
        self.dataset_dir = dataset_dir

    def _manifest_path(self) -> str:
        return os.path.join(self.dataset_dir, "manifest.json")

    def read_manifest(self):
        path = self._manifest_path()
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        if manifest.get("format") != FORMAT:
            raise ValueError(f"{path}: unrecognized format {manifest.get('format')!r}, expected {FORMAT!r}")
        return manifest

    def write_manifest(self, manifest: dict) -> None:
        assert manifest.get("format") == FORMAT, f"manifest must be stamped format={FORMAT!r}"
        os.makedirs(self.dataset_dir, exist_ok=True)
        path = self._manifest_path()
        tmp_path = path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        os.replace(tmp_path, path)

    def volume_filename(self, split: str, index: int, *, mask: bool = False) -> str:
        suffix = ".mask.npy" if mask else ".npy"
        return f"{split}_{index:06d}{suffix}"

    def write_volume(self, filename: str, array: np.ndarray) -> None:
        os.makedirs(self.dataset_dir, exist_ok=True)
        path = os.path.join(self.dataset_dir, filename)
        tmp_path = path + ".tmp.npy"
        np.save(tmp_path, array)
        os.replace(tmp_path, path)

    def open_volume(self, filename: str, mmap: bool = True) -> np.ndarray:
        path = os.path.join(self.dataset_dir, filename)
        return np.load(path, mmap_mode="r" if mmap else None)
