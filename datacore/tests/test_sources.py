import pytest

from datacore import CharTokenizer
from datacore.packing import EncodedDoc
from datacore.sources import named_document_batches

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

CHARS = " abcdefghijklmnopqrstuvwxyz.,!?'\n0123456789"


class ListTextSource:
    def __init__(self, batches):
        self._batches = batches

    def text_batches(self):
        return iter(self._batches)


class ListTokenSource:
    def __init__(self, batches):
        self._batches = batches

    def token_batches(self):
        return iter(self._batches)


def test_text_source_gets_bos_prepended():
    tok = CharTokenizer(CHARS)
    source = ListTextSource([("f", ["hi", "bye"])])
    out = list(named_document_batches(source, tok))
    assert len(out) == 1
    name, docs = out[0]
    docs = list(docs)
    assert name == "f"
    assert docs[0].ids[0] == tok.get_bos_token_id()
    assert docs[1].ids[0] == tok.get_bos_token_id()


def test_token_source_passes_through_unchanged():
    tok = CharTokenizer(CHARS)
    doc = EncodedDoc(ids=[1, 2, 3], mask=[0, 1, 1])
    source = ListTokenSource([("f", [doc])])
    out = list(named_document_batches(source, tok))
    name, docs = out[0]
    docs = list(docs)
    assert docs[0] is doc


def test_unrecognized_source_raises():
    tok = CharTokenizer(CHARS)
    with pytest.raises(TypeError):
        list(named_document_batches(object(), tok))


def test_parquet_directory_source(tmp_path):
    from datacore.sources import ParquetDirectorySource
    path = tmp_path / "shard_00000.parquet"
    table = pa.Table.from_pydict({"text": ["doc one", "doc two", "doc three"]})
    pq.write_table(table, str(path), row_group_size=2)
    source = ParquetDirectorySource(paths=[str(path)])
    batches = list(source.text_batches())
    assert len(batches) == 1
    name, texts = batches[0]
    assert name == str(path)
    assert list(texts) == ["doc one", "doc two", "doc three"]
