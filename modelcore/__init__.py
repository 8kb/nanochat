"""
modelcore -- a standalone model subsystem: configs, architectures-as-data, and the machinery to
create/load/save models and optimizers and compute their stats. Knows nothing about nanochat,
checkpoints, tokenizers, or CLI flags; see nanochat/architectures/ and
nanochat/checkpoint_manager.py for the layer that does.

ModelManager is the one entrypoint; ModelConfig/ComponentSpec, ModelStats, ValidationReport,
OptimizerHparams, ArtifactStore/FileSystemStore, and KVCache are the value types that cross its
boundary. Everything else (components, composers, catalog, roles) is internal.

Importing this package (or modelcore.manager) triggers every built-in component/composer's
@register_component decorator, via modelcore.components/modelcore.composers -- see catalog.py.
"""
import modelcore.components  # noqa: F401 -- import for @register_component side effects
import modelcore.composers  # noqa: F401 -- import for @register_component side effects

from modelcore.cache import KVCache
from modelcore.config.spec import AttentionLayerSpec, ComponentSpec, ModelConfig
from modelcore.errors import ConfigError, ValidationReport
from modelcore.manager import ModelManager, OptimizerHparams
from modelcore.model import Model
from modelcore.runtime import DEFAULT_RUNTIME, Runtime
from modelcore.stats import ModelStats
from modelcore.store import ArtifactStore, FileSystemStore

__all__ = [
    "ModelManager", "OptimizerHparams",
    "ModelConfig", "ComponentSpec", "AttentionLayerSpec",
    "Model", "ModelStats", "KVCache",
    "ConfigError", "ValidationReport",
    "ArtifactStore", "FileSystemStore",
    "Runtime", "DEFAULT_RUNTIME",
]
