"""
Captures byte-exact reference output of today's (pre-datacore) BOS-aligned best-fit packing
algorithms -- nanochat/dataloader.py's crop packer and scripts/chat_sft.py's pad packer -- before
either is touched. Frozen after this commit, like dev/capture_model_goldens.py: a point-in-time
snapshot, not meant to track future refactors.

Deliberately standalone: no dependency on nanochat.tokenizer (rustbpe/tiktoken) or on datacore
(which doesn't exist yet at this commit). A tiny fixed char-to-id table stands in for a real
tokenizer -- the packing algorithm only cares about token id sequence lengths, not what produced
them. This table's shape (id 0 = <unk>, ids 1..N = a fixed char list, id N+1 = BOS) previews
datacore.tokenizer.CharTokenizer; tests/test_data_packing_parity.py uses the real CharTokenizer
with the same char list to replay these goldens against datacore's packers.

Writes tests/goldens/data_bestfit_crop.json and data_bestfit_pad.json.

python -m dev.capture_data_goldens          # (re)writes the goldens
python -m dev.capture_data_goldens --check  # recomputes and asserts byte-identical to what's on disk
"""
import argparse
import json
import os

import numpy as np

GOLDENS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests", "goldens")

# -----------------------------------------------------------------------------
# Tiny deterministic tokenizer (previews datacore.tokenizer.CharTokenizer's id scheme)

CHARS = " abcdefghijklmnopqrstuvwxyz.,!?'\n0123456789"  # 43 chars, fixed order
BOS_ID = len(CHARS) + 1  # id 0 reserved for <unk>, ids 1..len(CHARS) are CHARS in this order


def encode_char(text):
    return [CHARS.index(c) + 1 if c in CHARS else 0 for c in text]


# -----------------------------------------------------------------------------
# Reference packers -- verbatim transcriptions of the algorithms at
# nanochat/dataloader.py:122-151 (crop) and scripts/chat_sft.py:227-268 (pad), generalized to
# operate on a fixed, cyclable buffer of pre-tokenized documents instead of a live parquet stream.

def reference_bestfit_crop(documents, row_capacity, buffer_size, num_rows):
    """documents: list[list[int]], already BOS-prepended. Cycled indefinitely, matching the
    original's multi-epoch refill_buffer(). Every row is filled to exactly row_capacity (100%
    utilization) -- see nanochat/dataloader.py's module docstring."""
    doc_buffer = []
    cursor = 0

    def refill():
        nonlocal cursor
        while len(doc_buffer) < buffer_size:
            doc_buffer.append(list(documents[cursor % len(documents)]))
            cursor += 1

    rows = []
    for _ in range(num_rows):
        row = []
        pos = 0
        while pos < row_capacity:
            while len(doc_buffer) < buffer_size:
                refill()
            remaining = row_capacity - pos
            # largest doc that fits entirely
            best_idx, best_len = -1, 0
            for i, doc in enumerate(doc_buffer):
                doc_len = len(doc)
                if doc_len <= remaining and doc_len > best_len:
                    best_idx, best_len = i, doc_len
            if best_idx >= 0:
                doc = doc_buffer.pop(best_idx)
                row.extend(doc)
                pos += len(doc)
            else:
                # nothing fits: crop the SHORTEST doc in the buffer (not the longest -- this is
                # the tie-break dataloader.py:147-150 actually uses; freeze it, don't "fix" it)
                shortest_idx = min(range(len(doc_buffer)), key=lambda i: len(doc_buffer[i]))
                doc = doc_buffer.pop(shortest_idx)
                row.extend(doc[:remaining])
                pos += remaining
        rows.append(row)
    return rows


def reference_bestfit_pad(conversations, row_capacity, buffer_size, num_rows, bos_token):
    """conversations: list[(ids, mask)]. Cycled indefinitely. Pads instead of cropping -- no
    token is ever discarded, unmasked padding at the tail instead."""
    conv_buffer = []
    cursor = 0

    def refill():
        nonlocal cursor
        while len(conv_buffer) < buffer_size:
            ids, mask = conversations[cursor % len(conversations)]
            conv_buffer.append((list(ids), list(mask)))
            cursor += 1

    rows, mask_rows, row_lengths = [], [], []
    for _ in range(num_rows):
        row, mask_row = [], []
        padded = False
        content_len = row_capacity
        while len(row) < row_capacity:
            while len(conv_buffer) < buffer_size:
                refill()
            remaining = row_capacity - len(row)
            best_idx, best_len = -1, 0
            for i, (conv, _) in enumerate(conv_buffer):
                conv_len = len(conv)
                if conv_len <= remaining and conv_len > best_len:
                    best_idx, best_len = i, conv_len
            if best_idx >= 0:
                conv, conv_mask = conv_buffer.pop(best_idx)
                row.extend(conv)
                mask_row.extend(conv_mask)
            else:
                content_len = len(row)
                row.extend([bos_token] * remaining)
                mask_row.extend([0] * remaining)
                padded = True
                break
        row_lengths.append(content_len if padded else row_capacity)
        rows.append(row[:row_capacity])
        mask_rows.append(mask_row[:row_capacity])
    return rows, mask_rows, row_lengths


def compute_targets_dual(rows, mask_rows, row_lengths):
    """scripts/chat_sft.py:295-303 applies the mask rule, THEN the row_lengths rule, on top of
    it, in sequence. The real equivalence claim (see the plan) is that the second is redundant
    once the first has run -- not that either rule alone reproduces the other. This computes the
    mask rule alone (targets_mask_rule, what datacore's reader implements) and the mask rule
    followed by the row_lengths rule (targets_combined, today's actual behavior) and reports
    where they differ -- they shouldn't, anywhere, which is exactly what makes the row_lengths
    rule dead code today. (Applied *alone*, without the mask rule first, the row_lengths rule
    would mis-handle content_len==0 via numpy's negative-slice semantics -- that's the latent bug
    the plan calls out, but it never manifests in the combined pipeline because the mask rule
    already zeroed those positions.)"""
    rows = np.array(rows, dtype=np.int64)
    mask_rows = np.array(mask_rows, dtype=np.int64)
    targets = rows[:, 1:]
    mask_targets = mask_rows[:, 1:]
    targets_mask_rule = targets.copy()
    targets_mask_rule[mask_targets == 0] = -1
    targets_combined = targets_mask_rule.copy()
    for i, content_len in enumerate(row_lengths):
        if content_len < len(rows[i]):  # scripts/chat_sft.py:301's guard: `if content_len < row_capacity`
            targets_combined[i, content_len - 1:] = -1
    matches = [
        bool(np.array_equal(targets_mask_rule[i], targets_combined[i]))
        for i in range(len(rows))
    ]
    return targets_mask_rule, matches


# -----------------------------------------------------------------------------
# Fixed corpus

DOCS_TEXT = [
    "the quick brown fox jumps over the lazy dog.\n",
    "hello world!",
    "a",
    "to be or not to be, that is the question.\n",
    "1 2 3 4 5 6 7 8 9 0",
    "short",
    "this document is deliberately long enough to exceed the row capacity that we will use for "
    "testing the crop tie break behavior when nothing else fits into the remaining space.\n",
    "another medium length document for variety and coverage of the packer.\n",
    "x",
    "yes? no! maybe...\n",
    "the year is 2024 and we are testing packers.\n",
    "z",
    "UPPER and unicode: café — these characters are not in CHARS and map to <unk>.\n",
]

CONVERSATIONS_TEXT = [
    ("what is 2 plus 2", "it is 4."),
    ("hi", "hello there!"),
    ("tell me a joke", "why did the chicken cross the road? to get to the other side.\n"),
    ("what color is the sky", "blue."),
    ("a", "b"),
    ("this is a long user message meant to pad things out a fair bit further than usual",
     "and this is a correspondingly long assistant reply so that together, once packed, this "
     "single conversation alone exceeds the row capacity we use for the pad packer golden, which "
     "exercises the content_len == 0 edge case documented in the plan.\n"),
    ("short", "ok"),
    ("another question here", "another answer here, somewhat longer than the question itself.\n"),
]


def _build_conversations():
    out = []
    for user_text, asst_text in CONVERSATIONS_TEXT:
        ids = [BOS_ID] + encode_char(user_text) + encode_char(asst_text)
        mask = [0] * (1 + len(user_text)) + [1] * len(asst_text)
        out.append((ids, mask))
    return out


# -----------------------------------------------------------------------------

def build_crop_golden():
    row_capacity = 32
    buffer_size = 6
    num_rows = 10
    documents = [[BOS_ID] + encode_char(t) for t in DOCS_TEXT]
    rows = reference_bestfit_crop(documents, row_capacity, buffer_size, num_rows)
    return {
        "row_capacity": row_capacity,
        "buffer_size": buffer_size,
        "num_rows": num_rows,
        "char_list": CHARS,
        "bos_id": BOS_ID,
        "documents_text": DOCS_TEXT,
        "rows": rows,
    }


def build_pad_golden():
    row_capacity = 40
    buffer_size = 4
    num_rows = 8
    conversations = _build_conversations()
    rows, mask_rows, row_lengths = reference_bestfit_pad(
        conversations, row_capacity, buffer_size, num_rows, bos_token=BOS_ID,
    )
    targets_mask_rule, dual_matches = compute_targets_dual(rows, mask_rows, row_lengths)
    return {
        "row_capacity": row_capacity,
        "buffer_size": buffer_size,
        "num_rows": num_rows,
        "char_list": CHARS,
        "bos_id": BOS_ID,
        "conversations_text": [[u, a] for u, a in CONVERSATIONS_TEXT],
        "rows": rows,
        "mask_rows": mask_rows,
        "row_lengths": row_lengths,
        "targets_mask_rule": targets_mask_rule.tolist(),
        "dual_mechanism_matches_mask_rule": dual_matches,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="recompute and assert identical to what's on disk, don't write")
    args = parser.parse_args()

    goldens = {
        "data_bestfit_crop.json": build_crop_golden(),
        "data_bestfit_pad.json": build_pad_golden(),
    }

    pad_golden = goldens["data_bestfit_pad.json"]
    num_mismatch = pad_golden["dual_mechanism_matches_mask_rule"].count(False)
    if num_mismatch:
        raise SystemExit(
            f"pad packer: {num_mismatch}/{pad_golden['num_rows']} rows where today's combined "
            f"pipeline (mask rule, then row_lengths rule) disagrees with the mask rule alone -- "
            f"the row_lengths rule is supposed to be fully redundant given the mask rule already "
            f"ran; a mismatch here means datacore's mask-only reader would NOT reproduce today's "
            f"targets and the plan's equivalence claim is wrong."
        )
    print(f"pad packer: mask rule alone reproduces today's combined pipeline on all "
          f"{pad_golden['num_rows']} rows -- the row_lengths mechanism is confirmed dead code.")

    os.makedirs(GOLDENS_DIR, exist_ok=True)
    for filename, data in goldens.items():
        path = os.path.join(GOLDENS_DIR, filename)
        new_content = json.dumps(data, indent=2, sort_keys=True) + "\n"
        if args.check:
            if not os.path.exists(path):
                raise SystemExit(f"{path} does not exist -- run without --check first")
            with open(path, "r", encoding="utf-8") as f:
                old_content = f.read()
            if old_content != new_content:
                raise SystemExit(f"{path} is NOT byte-identical to a fresh capture")
            print(f"{path}: OK (byte-identical)")
        else:
            with open(path, "w", encoding="utf-8") as f:
                f.write(new_content)
            print(f"wrote {path}")


if __name__ == "__main__":
    main()
