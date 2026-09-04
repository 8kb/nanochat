from dataclasses import dataclass

from nanochat.model.llama_kvshare.config import LlamaKVShareConfig


@dataclass
class LlamaKVShareWinConfig(LlamaKVShareConfig):
    """LlamaKVShareConfig with sliding-window attention turned on by default: the one and only
    difference from LlamaKVShareConfig is window_pattern defaulting to "SSSL" (GPTConfig's
    pattern) instead of LlamaConfig's "L" (full context). kv_share_frac is unchanged -- this
    config isolates the windowing variable against llama_kvshare's already-measured baseline
    rather than introducing a third pattern into the comparison.

    from_depth must be overridden (not just this field default) because LlamaConfig.from_depth
    hardcodes window_pattern="L" in its own signature default -- without the override here, a
    --depth run would silently build a full-context model despite this class's field default
    saying otherwise. See nanochat.model.llama_kvshare_win.model for why no model.py logic is
    needed at all: LlamaKVShare.__init__ already reads config.window_pattern."""
    window_pattern: str = "SSSL"

    @classmethod
    def from_depth(cls, depth, aspect_ratio=64, head_dim=128, max_seq_len=2048, vocab_size=32768, window_pattern="SSSL"):
        return super().from_depth(depth, aspect_ratio=aspect_ratio, head_dim=head_dim,
                                   max_seq_len=max_seq_len, vocab_size=vocab_size, window_pattern=window_pattern)
