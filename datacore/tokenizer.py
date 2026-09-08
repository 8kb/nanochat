"""
Tokenizer: the protocol datacore.manager.DataManager.prepare needs from a tokenizer, plus
CharTokenizer, a tiny fixed-vocabulary implementation that lets datacore's own tests (and any
host application's smoke tests) run with no real BPE tokenizer installed at all.

Not an ABC -- duck typing is enough, and datacore has no business enforcing what a caller's
tokenizer subclasses from. nanochat.tokenizer.RustBPETokenizer already satisfies this protocol
unmodified: encode(text, prepend=, num_threads=), get_bos_token_id(), get_vocab_size(),
fingerprint().
"""
import hashlib
from typing import Protocol, runtime_checkable


@runtime_checkable
class Tokenizer(Protocol):
    def encode(self, text, prepend=None, num_threads=8):
        """text: str or list[str]. Returns list[int] or list[list[int]] to match. `prepend`, if
        given, is an int token id or a special-token string resolved by the tokenizer itself."""
        ...

    def get_bos_token_id(self) -> int:
        ...

    def get_vocab_size(self) -> int:
        ...

    def fingerprint(self) -> str:
        """Content hash identifying what a token id means -- see
        nanochat.tokenizer.RustBPETokenizer.fingerprint for the convention this mirrors."""
        ...


class CharTokenizer:
    """A fixed character list, id 0 reserved for <unk>, BOS the last id. Deterministic,
    dependency-free (no rustbpe/tiktoken) -- the tokenizer datacore's own suite and a host
    application's CI/smoke tests use instead of a real BPE tokenizer.

    id 0            -> <unk> (any character not in `chars`)
    id 1..len(chars) -> chars[i-1], in the given order
    id len(chars)+1  -> <|bos|>
    """

    def __init__(self, chars: str):
        assert len(chars) == len(set(chars)), "chars must not contain duplicates"
        self.chars = chars
        self._char_to_id = {c: i + 1 for i, c in enumerate(chars)}
        self.bos_token_id = len(chars) + 1

    def get_vocab_size(self) -> int:
        return len(self.chars) + 2  # <unk> + chars + <|bos|>

    def get_bos_token_id(self) -> int:
        return self.bos_token_id

    def _encode_one(self, text: str, prepend_id):
        ids = [self._char_to_id.get(c, 0) for c in text]
        if prepend_id is not None:
            ids.insert(0, prepend_id)
        return ids

    def encode(self, text, prepend=None, num_threads=8):
        prepend_id = None
        if prepend is not None:
            prepend_id = prepend if isinstance(prepend, int) else self.get_bos_token_id()
        if isinstance(text, str):
            return self._encode_one(text, prepend_id)
        elif isinstance(text, list):
            return [self._encode_one(t, prepend_id) for t in text]
        else:
            raise ValueError(f"Invalid input type: {type(text)}")

    def decode(self, ids):
        return "".join(self.chars[i - 1] if 1 <= i <= len(self.chars) else "" for i in ids)

    def fingerprint(self) -> str:
        h = hashlib.sha256(self.chars.encode("utf-8"))
        return h.hexdigest()[:16]
