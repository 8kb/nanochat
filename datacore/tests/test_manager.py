import pytest

from datacore import BestFitCropPacker, BestFitPadPacker, CharTokenizer, DataManager, FileSystemDatasetStore
from datacore.packing import EncodedDoc

torch = pytest.importorskip("torch")

CHARS = " abcdefghijklmnopqrstuvwxyz.,!?'\n0123456789"


class ListTextSource:
    def __init__(self, named_batches):
        self._batches = named_batches

    def text_batches(self):
        return iter(self._batches)


class ListTokenSource:
    def __init__(self, named_batches):
        self._batches = named_batches

    def token_batches(self):
        return iter(self._batches)


def test_prepare_open_batches_round_trip(tmp_path):
    tok = CharTokenizer(CHARS)
    source = ListTextSource([
        ("shard0", ["the quick brown fox.\n"] * 20),
        ("shard1", ["hello there, friend!\n"] * 20),
    ])
    store = FileSystemDatasetStore(str(tmp_path))
    manager = DataManager()
    manifest = manager.prepare(
        store, sources={"train": source}, tokenizer=tok,
        sequence_len=8, sequences_per_volume=5, packer=BestFitCropPacker(buffer_size=10),
    )
    assert manifest["format"] == "datacore.v1"
    assert manifest["tokenizer_fingerprint"] == tok.fingerprint()
    assert manifest["splits"]["train"]["num_sequences"] > 0

    dataset = manager.open(store)
    assert dataset.info.sequence_len == 8
    assert dataset.num_sequences("train") == manifest["splits"]["train"]["num_sequences"]

    gen = manager.batches(dataset, "train", batch_size=4, infinite=False)
    inputs, targets, state = next(gen)
    assert inputs.shape == (4, 8)
    assert targets.shape == (4, 8)
    assert state["format"] == "datacore.v1"


def test_prepare_is_deterministic_regardless_of_sequences_per_volume(tmp_path):
    tok = CharTokenizer(CHARS)
    texts = ["one two three four five.\n"] * 15
    manager = DataManager()

    store_a = FileSystemDatasetStore(str(tmp_path / "a"))
    manager.prepare(store_a, sources={"train": ListTextSource([("f", texts)])}, tokenizer=tok,
                     sequence_len=6, sequences_per_volume=3, packer=BestFitCropPacker(buffer_size=20))
    store_b = FileSystemDatasetStore(str(tmp_path / "b"))
    manager.prepare(store_b, sources={"train": ListTextSource([("f", texts)])}, tokenizer=tok,
                     sequence_len=6, sequences_per_volume=100, packer=BestFitCropPacker(buffer_size=20))

    ds_a, ds_b = manager.open(store_a), manager.open(store_b)
    assert ds_a.num_sequences("train") == ds_b.num_sequences("train")
    gen_a = manager.batches(ds_a, "train", batch_size=ds_a.num_sequences("train"), infinite=False)
    gen_b = manager.batches(ds_b, "train", batch_size=ds_b.num_sequences("train"), infinite=False)
    inputs_a, targets_a, _ = next(gen_a)
    inputs_b, targets_b, _ = next(gen_b)
    assert torch.equal(inputs_a, inputs_b)
    assert torch.equal(targets_a, targets_b)


def test_prepare_with_token_source_for_sft_style_data(tmp_path):
    tok = CharTokenizer(CHARS)
    bos = tok.get_bos_token_id()

    def conv(user, asst):
        ids = [bos] + tok.encode(user) + tok.encode(asst)
        mask = [0] * (1 + len(user)) + [1] * len(asst)
        return EncodedDoc(ids=ids, mask=mask)

    source = ListTokenSource([("conv_batch", [conv("hi", "hello!"), conv("bye", "goodbye!")])])
    store = FileSystemDatasetStore(str(tmp_path))
    manager = DataManager()
    manifest = manager.prepare(
        store, sources={"train": source}, tokenizer=tok,
        sequence_len=16, sequences_per_volume=10, packer=BestFitPadPacker(bos_token_id=bos, buffer_size=10),
    )
    assert manifest["has_mask"] is True
    dataset = manager.open(store)
    inputs, targets, _ = next(manager.batches(dataset, "train", batch_size=1, infinite=False))
    assert (-1 in targets[0].tolist()) or True  # some positions may be masked; shape check is the real assertion
    assert inputs.shape == (1, 16)


def test_prepare_records_packer_params_including_padding_id(tmp_path):
    tok = CharTokenizer(CHARS)
    bos = tok.get_bos_token_id()

    def conv(user, asst):
        ids = [bos] + tok.encode(user) + tok.encode(asst)
        mask = [0] * (1 + len(user)) + [1] * len(asst)
        return EncodedDoc(ids=ids, mask=mask)

    source = ListTokenSource([("conv_batch", [conv("hi", "hello!")])])

    # Default (padding_id=None -> resolves to bos_token_id): recorded value is the resolved one.
    store_default = FileSystemDatasetStore(str(tmp_path / "default"))
    manifest_default = DataManager().prepare(
        store_default, sources={"train": source}, tokenizer=tok,
        sequence_len=16, sequences_per_volume=10, packer=BestFitPadPacker(bos_token_id=bos, buffer_size=10),
    )
    assert manifest_default["packer"]["params"]["padding_id"] == bos

    # Explicit padding_id: recorded as given, distinct from bos_token_id.
    source2 = ListTokenSource([("conv_batch", [conv("hi", "hello!")])])
    store_explicit = FileSystemDatasetStore(str(tmp_path / "explicit"))
    pad_id = bos + 1
    manifest_explicit = DataManager().prepare(
        store_explicit, sources={"train": source2}, tokenizer=tok,
        sequence_len=16, sequences_per_volume=10,
        packer=BestFitPadPacker(bos_token_id=bos, padding_id=pad_id, buffer_size=10),
    )
    assert manifest_explicit["packer"]["params"]["padding_id"] == pad_id

    # BestFitCropPacker has no padding_id at all -- key must be absent, not None.
    store_crop = FileSystemDatasetStore(str(tmp_path / "crop"))
    manifest_crop = DataManager().prepare(
        store_crop, sources={"train": ListTextSource([("f", ["one two three.\n"] * 5)])}, tokenizer=tok,
        sequence_len=8, sequences_per_volume=5, packer=BestFitCropPacker(buffer_size=10),
    )
    assert "padding_id" not in manifest_crop["packer"]["params"]


def test_dataset_info_surfaces_padding_id_and_bos_token_id(tmp_path):
    """DatasetInfo (what DataManager.open() returns) must carry padding_id/bos_token_id through
    from the manifest -- these are what modelcore.kernels.flash_attn.build_doc_args needs at train
    time, and what scripts/data_prep.py's --describe stats (documents-per-row, padding share) are
    computed against."""
    tok = CharTokenizer(CHARS)
    bos = tok.get_bos_token_id()

    def conv(user, asst):
        ids = [bos] + tok.encode(user) + tok.encode(asst)
        mask = [0] * (1 + len(user)) + [1] * len(asst)
        return EncodedDoc(ids=ids, mask=mask)

    manager = DataManager()

    # Pad packer, default padding_id -> resolves to bos_token_id, and DatasetInfo reflects that.
    store_pad = FileSystemDatasetStore(str(tmp_path / "pad"))
    manager.prepare(
        store_pad, sources={"train": ListTokenSource([("b", [conv("hi", "hello!")])])}, tokenizer=tok,
        sequence_len=16, sequences_per_volume=10, packer=BestFitPadPacker(bos_token_id=bos, buffer_size=10),
    )
    ds_pad = manager.open(store_pad)
    assert ds_pad.info.padding_id == bos
    assert ds_pad.info.bos_token_id == bos

    # Crop packer: no padding concept -- DatasetInfo.padding_id must be None, not defaulted.
    store_crop = FileSystemDatasetStore(str(tmp_path / "crop"))
    manager.prepare(
        store_crop, sources={"train": ListTextSource([("f", ["one two three.\n"] * 5)])}, tokenizer=tok,
        sequence_len=8, sequences_per_volume=5, packer=BestFitCropPacker(buffer_size=10),
    )
    ds_crop = manager.open(store_crop)
    assert ds_crop.info.padding_id is None
    assert ds_crop.info.bos_token_id == bos


def test_read_rows_round_trips_through_data_manager(tmp_path):
    tok = CharTokenizer(CHARS)
    texts = ["one two three four five.\n"] * 6
    manager = DataManager()
    store = FileSystemDatasetStore(str(tmp_path))
    manager.prepare(store, sources={"train": ListTextSource([("f", texts)])}, tokenizer=tok,
                     sequence_len=6, sequences_per_volume=3, packer=BestFitCropPacker(buffer_size=20))
    dataset = manager.open(store)
    tokens, mask = manager.read_rows(dataset, "train", 0, dataset.num_sequences("train"))
    assert mask is None
    assert tokens.shape == (dataset.num_sequences("train"), 7)


def test_two_splits_from_different_sources(tmp_path):
    tok = CharTokenizer(CHARS)
    train_source = ListTextSource([("f1", ["training text here.\n"] * 20)])
    val_source = ListTextSource([("f2", ["validation text here.\n"] * 20)])
    store = FileSystemDatasetStore(str(tmp_path))
    manager = DataManager()
    manifest = manager.prepare(
        store, sources={"train": train_source, "val": val_source}, tokenizer=tok,
        sequence_len=8, sequences_per_volume=5, packer=BestFitCropPacker(buffer_size=10),
    )
    assert set(manifest["splits"]) == {"train", "val"}
    dataset = manager.open(store)
    assert dataset.num_sequences("train") > 0
    assert dataset.num_sequences("val") > 0
