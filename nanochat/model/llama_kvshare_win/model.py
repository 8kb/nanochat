"""
LlamaKVShare plus sliding-window attention (SWA): trades attention span for more tokens trained
at the same FLOPs budget, with cross-layer KV sharing held constant against
nanochat.model.llama_kvshare's already-measured baseline so only the windowing variable moves.

No new model logic at all -- LlamaKVShare.__init__ already reads config.window_pattern and calls
nanochat.model.components.windows.compute_window_sizes with it; LlamaKVShareWinConfig
(nanochat.model.llama_kvshare_win.config) just changes that field's default from "L" to "SSSL".
This class exists (rather than reusing LlamaKVShare directly) purely so the registry, checkpoint
meta, and tracebacks name "llama_kvshare_win" instead of aliasing an unrelated arch name onto
LlamaKVShare.

Windowing and KV sharing compose correctly because the window is a mask applied at attention time
(CausalSelfAttention.forward passes window_size=(self.window, 0) per layer), not a truncation of
what a producer layer stores -- a short-window producer shared by a long-window consumer works in
both the training path (kv_bus) and the cached-inference path (get_slot_cache).
"""

from nanochat.model.registry import register_model
from nanochat.model.llama_kvshare.model import LlamaKVShare
from nanochat.model.llama_kvshare_win.config import LlamaKVShareWinConfig


@register_model("llama_kvshare_win", LlamaKVShareWinConfig)
class LlamaKVShareWin(LlamaKVShare):
    pass
