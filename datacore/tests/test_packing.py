from datacore.packing import BestFitCropPacker, BestFitPadPacker, EncodedDoc


def _docs(lengths, bos=99):
    return [EncodedDoc(ids=[bos] + list(range(n))) for n in lengths]


def test_crop_every_row_is_full_width():
    packer = BestFitCropPacker(buffer_size=4)
    rows = list(packer.pack(_docs([3, 5, 2, 10, 1, 4, 6, 3, 2]), row_capacity=8))
    assert len(rows) > 0
    assert all(len(r.ids) == 8 for r in rows)
    assert all(r.mask is None for r in rows)


def test_crop_every_row_starts_with_bos():
    packer = BestFitCropPacker(buffer_size=4)
    rows = list(packer.pack(_docs([3, 5, 2, 10, 1, 4, 6, 3, 2], bos=99), row_capacity=8))
    assert all(r.ids[0] == 99 for r in rows)


def test_crop_tie_break_crops_the_shortest_doc_not_the_longest():
    # buffer_size=2, row_capacity=3: two docs of len 1 (with bos -> len 2) and 1 (len 2), neither
    # fits after a first doc of len 3 (bos+2) is placed leaving remaining=0... construct directly:
    # buffer holds two docs, both too big to fit remaining=2: lengths 5 and 3 (post-bos).
    # remaining=2, nothing fits (5>2, 3>2) -> must crop the SHORTEST (len 3), not the longest (len 5).
    packer = BestFitCropPacker(buffer_size=2)
    long_doc = EncodedDoc(ids=[0, 1, 2, 3, 4])   # len 5
    short_doc = EncodedDoc(ids=[9, 1, 2])         # len 3
    docs = [EncodedDoc(ids=[7, 7]), long_doc, short_doc]  # first doc (len 2) fills remaining=2 first
    rows = list(packer.pack(iter(docs), row_capacity=4))
    # row_capacity=4: first doc [7,7] (len2) placed, remaining=2. Neither long_doc(5) nor
    # short_doc(3) fits (both > 2) -> crop the shortest (short_doc=[9,1,2][:2]=[9,1]).
    assert rows[0].ids == [7, 7, 9, 1]


def test_crop_drops_partial_row_when_source_runs_dry_mid_row():
    packer = BestFitCropPacker(buffer_size=4)
    rows = list(packer.pack(_docs([2]), row_capacity=8))  # one doc, len 3 (incl bos) < 8, can't fill a row
    assert rows == []


def test_pad_never_crops_and_masks_padding():
    bos = 99
    packer = BestFitPadPacker(bos_token_id=bos, buffer_size=4)
    docs = [EncodedDoc(ids=[bos, 1, 2], mask=[0, 1, 1])]  # len 3
    rows = list(packer.pack(iter(docs), row_capacity=6))
    assert len(rows) == 1
    row = rows[0]
    assert row.ids == [bos, 1, 2, bos, bos, bos]
    assert row.mask == [0, 1, 1, 0, 0, 0]


def test_pad_docs_without_mask_default_to_fully_supervised():
    bos = 99
    packer = BestFitPadPacker(bos_token_id=bos, buffer_size=4)
    docs = [EncodedDoc(ids=[bos, 1, 2])]  # no mask given
    rows = list(packer.pack(iter(docs), row_capacity=5))
    assert rows[0].mask == [1, 1, 1, 0, 0]


def test_pad_drops_oversized_documents_instead_of_looping_forever():
    # A document longer than row_capacity can never fit a row (padding never crops it) -- left in
    # the buffer it would block forever once nothing else remains. Regression test for exactly
    # that: buffer_size=1 means the oversized doc would be the ONLY thing ever in the buffer.
    bos = 99
    packer = BestFitPadPacker(bos_token_id=bos, buffer_size=1)
    docs = [EncodedDoc(ids=list(range(20)), mask=[1] * 20)]
    rows = list(packer.pack(iter(docs), row_capacity=8))  # must terminate, not hang
    assert rows == []
    assert packer.num_documents_dropped == 1
    assert packer.num_tokens_dropped == 20


def test_pad_drops_oversized_document_but_keeps_packing_the_rest():
    bos = 99
    packer = BestFitPadPacker(bos_token_id=bos, buffer_size=4)
    docs = [
        EncodedDoc(ids=[bos, 1, 2], mask=[0, 1, 1]),          # fits
        EncodedDoc(ids=list(range(50)), mask=[1] * 50),        # oversized, dropped
        EncodedDoc(ids=[bos, 3, 4], mask=[0, 1, 1]),          # fits
    ]
    rows = list(packer.pack(iter(docs), row_capacity=8))
    assert packer.num_documents_dropped == 1
    assert packer.num_tokens_dropped == 50
    all_ids = [i for row in rows for i in row.ids]
    assert 1 in all_ids and 2 in all_ids and 3 in all_ids and 4 in all_ids


def test_pack_is_a_generator_and_source_is_consumed_exactly_once():
    packer = BestFitCropPacker(buffer_size=2)
    calls = []

    def gen():
        for d in _docs([3, 3, 3, 3]):
            calls.append(1)
            yield d

    list(packer.pack(gen(), row_capacity=4))
    assert len(calls) == 4
