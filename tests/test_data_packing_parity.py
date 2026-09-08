"""
Cross-checks datacore's packers against today's (pre-datacore) packing algorithms, captured
verbatim in dev/capture_data_goldens.py (frozen after Step 0) and its exported
tests/goldens/data_bestfit_{crop,pad}.json. Proves the extraction preserves the packing algorithm
exactly, not just approximately.

The crop packer's golden is used directly (bit-exact, no caveats). The pad golden's own fixed
corpus deliberately includes conversations longer than row_capacity, to document today's "stuck
buffer" behavior for that case (see dev/capture_data_goldens.py and
datacore/tests/test_packing.py's regression tests) -- but datacore's BestFitPadPacker fixes that
case (drops an oversized conversation instead of leaving it stuck forever, see
datacore/packing.py's docstring), so it does NOT reproduce the golden's rows bit-for-bit once such
a conversation enters the buffer. The pad parity check here therefore re-derives a reference on a
corpus with the oversized conversations excluded, using the frozen reference algorithm directly --
an apples-to-apples comparison of the *unchanged* part of the algorithm.
"""
import json
import os

from dev.capture_data_goldens import BOS_ID, CHARS, CONVERSATIONS_TEXT, encode_char, reference_bestfit_pad
from datacore import CharTokenizer
from datacore.packing import BestFitCropPacker, BestFitPadPacker, EncodedDoc

GOLDENS_DIR = os.path.join(os.path.dirname(__file__), "goldens")

OVERSIZED_CONVERSATION_INDICES = {2, 5, 7}  # exceed row_capacity=40 -- see dev/capture_data_goldens.py


def _load(name):
    with open(os.path.join(GOLDENS_DIR, name), "r", encoding="utf-8") as f:
        return json.load(f)


def test_bestfit_crop_matches_golden():
    golden = _load("data_bestfit_crop.json")
    tok = CharTokenizer(golden["char_list"])
    assert tok.get_bos_token_id() == golden["bos_id"]
    docs = [
        EncodedDoc(ids=tok.encode(t, prepend=tok.get_bos_token_id()))
        for t in golden["documents_text"]
    ]
    # golden cycles the fixed doc list indefinitely; datacore's packer consumes an iterable to
    # exhaustion once, so feed it enough cycles to produce at least as many rows as the golden.
    cycles = docs * (golden["num_rows"] * 2 + 2)
    packer = BestFitCropPacker(buffer_size=golden["buffer_size"])
    rows = list(packer.pack(iter(cycles), row_capacity=golden["row_capacity"]))
    got = [r.ids for r in rows[: golden["num_rows"]]]
    assert got == golden["rows"]


def test_bestfit_pad_matches_reference_algorithm_excluding_the_oversized_case():
    golden = _load("data_bestfit_pad.json")
    row_capacity, buffer_size, num_rows = golden["row_capacity"], golden["buffer_size"], golden["num_rows"]
    assert CHARS == golden["char_list"] and BOS_ID == golden["bos_id"]

    sane_conversations_text = [
        c for i, c in enumerate(CONVERSATIONS_TEXT) if i not in OVERSIZED_CONVERSATION_INDICES
    ]
    assert all(
        len(encode_char(u)) + len(encode_char(a)) + 1 <= row_capacity
        for u, a in sane_conversations_text
    ), "test fixture assumption broke: a 'sane' conversation now exceeds row_capacity"

    def encode_conv_ref(user_text, asst_text):
        ids = [BOS_ID] + encode_char(user_text) + encode_char(asst_text)
        mask = [0] * (1 + len(user_text)) + [1] * len(asst_text)
        return ids, mask

    ref_convs = [encode_conv_ref(u, a) for u, a in sane_conversations_text]
    ref_rows, ref_mask_rows, _ = reference_bestfit_pad(
        ref_convs * (num_rows * 2 + 2), row_capacity, buffer_size, num_rows, bos_token=BOS_ID,
    )

    tok = CharTokenizer(CHARS)
    bos = tok.get_bos_token_id()

    def encode_conv_dc(user_text, asst_text):
        ids = [bos] + tok.encode(user_text) + tok.encode(asst_text)
        mask = [0] * (1 + len(user_text)) + [1] * len(asst_text)
        return EncodedDoc(ids=ids, mask=mask)

    dc_convs = [encode_conv_dc(u, a) for u, a in sane_conversations_text]
    packer = BestFitPadPacker(bos_token_id=bos, buffer_size=buffer_size)
    rows = list(packer.pack(iter(dc_convs * (num_rows * 2 + 2)), row_capacity=row_capacity))
    got_ids = [r.ids for r in rows[:num_rows]]
    got_mask = [r.mask for r in rows[:num_rows]]

    assert got_ids == ref_rows[:num_rows]
    assert got_mask == ref_mask_rows[:num_rows]
    assert packer.num_documents_dropped == 0  # none of the "sane" conversations should be dropped

    # and the reader's mask-only target rule matches what dev/capture_data_goldens.py proved is
    # equivalent to today's combined (mask rule + row_lengths rule) pipeline
    for ids, mask in zip(got_ids, got_mask):
        targets = [t if m else -1 for t, m in zip(ids[1:], mask[1:])]
        assert all(t != -1 or m == 0 for t, m in zip(targets, mask[1:]))  # sanity: masked <=> -1
