from datacore import CharTokenizer

CHARS = " abcdefghijklmnopqrstuvwxyz.,!?'\n0123456789"


def test_unknown_char_maps_to_zero():
    tok = CharTokenizer(CHARS)
    ids = tok.encode("hello UNKNOWN 你好")
    known = set(range(1, len(CHARS) + 1))
    for c, i in zip("hello UNKNOWN 你好", ids):
        if c.islower() or c in " ":
            assert i in known
        else:
            assert i == 0  # uppercase and non-ASCII are not in CHARS


def test_roundtrip_known_chars():
    tok = CharTokenizer(CHARS)
    text = "the quick brown fox, 123!\n"
    ids = tok.encode(text)
    assert tok.decode(ids) == text


def test_bos_is_last_id():
    tok = CharTokenizer(CHARS)
    assert tok.get_bos_token_id() == len(CHARS) + 1
    assert tok.get_vocab_size() == len(CHARS) + 2


def test_prepend_bos():
    tok = CharTokenizer(CHARS)
    ids = tok.encode("hi", prepend=tok.get_bos_token_id())
    assert ids[0] == tok.get_bos_token_id()
    assert tok.decode(ids[1:]) == "hi"


def test_batch_encode_shape():
    tok = CharTokenizer(CHARS)
    batch = tok.encode(["a", "bb", "ccc"], prepend=tok.get_bos_token_id())
    assert [len(row) for row in batch] == [2, 3, 4]


def test_fingerprint_deterministic_and_sensitive_to_chars():
    tok_a = CharTokenizer(CHARS)
    tok_b = CharTokenizer(CHARS)
    tok_c = CharTokenizer(CHARS + "@")
    assert tok_a.fingerprint() == tok_b.fingerprint()
    assert tok_a.fingerprint() != tok_c.fingerprint()
    assert isinstance(tok_a.fingerprint(), str) and len(tok_a.fingerprint()) == 16


def test_satisfies_tokenizer_protocol():
    from datacore import Tokenizer
    assert isinstance(CharTokenizer(CHARS), Tokenizer)
