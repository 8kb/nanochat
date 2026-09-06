"""
Registers nanochat/model/components/'s existing classes (and this package's own composers) as
composed-architecture component types, under the type names a config's "#type" field names. See
docs/architecture.md's "Composed architectures" section. No new module code for the reused
components -- gpt_block/plain_block/token_embedding/lm_head/rotary are the exact same classes
gpt/llama/llama_kvshare(_win) build directly.
"""
from nanochat.model.components.embedding import TokenEmbedding
from nanochat.model.components.unembedding import LMHead
from nanochat.model.components.rotary import RotaryEmbedding
from nanochat.model.components.block import Block, PlainBlock
from nanochat.model.composed.registry import register_component
from nanochat.model.composed.composers import StackComposer, BackoutComposer

register_component("token_embedding", needs=("padded_vocab_size", "n_embd"))(TokenEmbedding)
register_component("lm_head", needs=("n_embd", "vocab_size", "padded_vocab_size"))(LMHead)
register_component("rotary", needs=("sequence_len",))(RotaryEmbedding)
register_component("gpt_block", needs=("n_embd", "padded_vocab_size", "rope", "n_layer"))(Block)
register_component("plain_block", needs=("n_embd", "padded_vocab_size", "rope"))(PlainBlock)
register_component("stack")(StackComposer)
register_component("backout")(BackoutComposer)
