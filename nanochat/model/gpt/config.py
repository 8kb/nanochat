from dataclasses import dataclass

from nanochat.model.base import BaseModelConfig


@dataclass
class GPTConfig(BaseModelConfig):
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6 # number of query heads
    n_kv_head: int = 6 # number of key/value heads (GQA)
    n_embd: int = 768
    # Sliding window attention pattern string, tiled across layers. Final layer always L.
    # Characters: L=long (full context), S=short (quarter context)
    # Examples: "L"=all full context, "SL"=alternating, "SSL"=two short then one long
    window_pattern: str = "SSSL"

    @classmethod
    def from_depth(cls, depth, aspect_ratio=64, head_dim=128, max_seq_len=2048, vocab_size=32768, window_pattern="SSSL"):
        """
        Derive a compute-optimal GPTConfig from a single depth dial (muP-style), matching the
        derivation in scripts/base_train.py's build_model_meta: model_dim is nudged up to the
        nearest multiple of head_dim for clean division (FA3 requires head_dim divisible by 8,
        and this guarantees head_dim == the requested head_dim exactly).
        """
        base_dim = depth * aspect_ratio
        model_dim = ((base_dim + head_dim - 1) // head_dim) * head_dim
        num_heads = model_dim // head_dim
        return cls(
            sequence_len=max_seq_len, vocab_size=vocab_size,
            n_layer=depth, n_head=num_heads, n_kv_head=num_heads, n_embd=model_dim,
            window_pattern=window_pattern,
        )
