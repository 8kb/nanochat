"""
nanochat.model — pluggable transformer architectures.

Import GPT/GPTConfig here (or from nanochat.gpt, kept as a backward-compat shim so upstream
diffs that `from nanochat.gpt import ...` keep applying) to build the default architecture.
Use the registry (register_model / get_model_class / get_config_class / config_from_dict) to
add or look up other architectures. See docs/architecture.md for the full contract.
"""

from nanochat.model.base import BaseModel, BaseModelConfig, AttentionLayerSpec, BaseEmbedding, BaseBlock, BaseUnembedding
from nanochat.model.registry import register_model, get_model_class, get_config_class, config_from_dict

# Import architecture packages so their @register_model decorators run and populate the registry.
from nanochat.model.gpt import GPT, GPTConfig
from nanochat.model.llama import Llama, LlamaConfig

__all__ = [
    "BaseModel", "BaseModelConfig", "AttentionLayerSpec", "BaseEmbedding", "BaseBlock", "BaseUnembedding",
    "register_model", "get_model_class", "get_config_class", "config_from_dict",
    "GPT", "GPTConfig", "Llama", "LlamaConfig",
]
