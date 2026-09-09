"""scripts/data_prep.py's --describe --deep stats scan, exercised against manifests it must stay
correct against even though it never wrote them: an old-format bestfit_pad manifest (predating the
padding_id field) and a bestfit_crop manifest (no padding concept at all). Both report
DatasetInfo.padding_id == None, for two different reasons -- deep_scan must not conflate them.
See the deep_scan() docstring/comment in scripts/data_prep.py for the bug this pins: treating an
old-format bestfit_pad manifest as "no padding" undercounts real padding to 0% and miscounts each
padded row's bos-valued tail as a spurious extra one-token document.
"""
from datacore import BestFitCropPacker, BestFitPadPacker, EncodedDoc, FileSystemDatasetStore
from datacore.writer import write_split
from scripts.data_prep import deep_scan


def _write_manifest(tmp_path, packer, docs, sequence_len, bos, *, packer_params, has_mask):
    store = FileSystemDatasetStore(str(tmp_path))
    totals = write_split(store, "train", packer, sequence_len, sequences_per_volume=10, vocab_size=50,
                          named_document_batches=[("f", docs)])
    volumes = [{"file": v.file, "rows": v.rows, "source": v.source} for v in totals.volumes]
    if has_mask:
        for v, rec in zip(totals.volumes, volumes):
            rec["mask_file"] = v.mask_file
    manifest = {
        "format": "datacore.v1", "sequence_len": sequence_len, "stride": sequence_len + 1,
        "dtype": "uint16", "has_mask": has_mask, "vocab_size": 50, "bos_token_id": bos,
        "tokenizer_fingerprint": "t", "packer": {"name": packer.name, "params": packer_params},
        "sequences_per_volume": 10,
        "splits": {"train": {
            "volumes": volumes, "num_sequences": totals.num_sequences, "num_tokens": totals.num_tokens,
            "num_documents": totals.num_documents, "num_documents_dropped": 0,
            "num_tokens_encoded": totals.num_tokens_encoded, "num_tokens_dropped": totals.num_tokens_dropped,
        }},
    }
    store.write_manifest(manifest)


def test_deep_scan_old_pad_manifest_without_padding_id_still_detects_padding(tmp_path):
    """A bestfit_pad manifest written before the padding_id field existed has no "padding_id" key
    at all in packer.params -- NOT the same as a packer with no padding concept. The packer still
    padded every row with bos_token_id (its documented default), so deep_scan must fall back to
    that, not report zero padding."""
    bos = 99
    docs = [EncodedDoc(ids=[bos, 1, 2], mask=[0, 1, 1])]  # row_capacity=6 -> [bos,1,2,bos,bos,bos]
    packer = BestFitPadPacker(bos_token_id=bos, buffer_size=10)  # padding_id defaults to bos
    _write_manifest(tmp_path, packer, docs, sequence_len=5, bos=bos,
                     packer_params={"buffer_size": packer.buffer_size}, has_mask=True)  # no padding_id key

    info, splits = deep_scan(str(tmp_path))
    assert info.padding_id is None  # the manifest genuinely lacks the key
    assert info.packer_name == "bestfit_pad"
    s = splits["train"]
    assert s["num_documents"] == 1  # NOT 2 -- the padded tail must not be miscounted as its own document
    assert s["doc_len_max"] == 3  # [bos, 1, 2] -- padding excluded from document content
    assert s["pad_tokens_total"] == 3  # the 3 trailing bos tokens
    assert s["pad_token_share"] > 0
    assert s["rows_with_padding"] == 1


def test_deep_scan_crop_manifest_has_no_padding_id_and_reports_no_padding(tmp_path):
    """bestfit_crop never pads -- padding_id is None here for a genuinely different reason (no
    padding concept at all), and deep_scan must not invent padding for it."""
    bos = 99
    docs = [EncodedDoc(ids=[bos, 1, 2, 3, 4])]  # row_capacity=5, exact fit, no crop needed
    packer = BestFitCropPacker(buffer_size=10)
    _write_manifest(tmp_path, packer, docs, sequence_len=4, bos=bos,
                     packer_params={"buffer_size": packer.buffer_size}, has_mask=False)

    info, splits = deep_scan(str(tmp_path))
    assert info.padding_id is None
    assert info.packer_name == "bestfit_crop"
    s = splits["train"]
    assert s["pad_tokens_total"] == 0
    assert s["rows_with_padding"] == 0
    assert s["num_documents"] == 1
    assert s["doc_len_max"] == 5
