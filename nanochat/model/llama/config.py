from dataclasses import dataclass

from nanochat.model.base import BaseModelConfig


@dataclass
class LlamaConfig(BaseModelConfig):
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6  # number of query heads
    n_kv_head: int = 6  # number of key/value heads (GQA)
    n_embd: int = 768
    # Sliding window attention pattern (see nanochat.model.components.windows), default "L" (full
    # context) since Llama doesn't traditionally use sliding windows -- but the field stays
    # tunable, same convention as GPTConfig.
    window_pattern: str = "L"

    @classmethod
    def from_depth(cls, depth, aspect_ratio=64, head_dim=128, max_seq_len=2048, vocab_size=32768, window_pattern="L"):
        """Same muP-style depth dial as GPTConfig.from_depth (nanochat/model/gpt/config.py),
        duplicated rather than factored into a shared helper -- see docs/architecture.md."""
        base_dim = depth * aspect_ratio
        model_dim = ((base_dim + head_dim - 1) // head_dim) * head_dim
        num_heads = model_dim // head_dim
        return cls(
            sequence_len=max_seq_len, vocab_size=vocab_size,
            n_layer=depth, n_head=num_heads, n_kv_head=num_heads, n_embd=model_dim,
            window_pattern=window_pattern,
        )
