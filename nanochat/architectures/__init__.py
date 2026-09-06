"""
nanochat.architectures -- everything outside modelcore that knows an architecture *by name*:
turning a --depth dial into a materialized modelcore.ModelConfig (presets.py, backed by the
derivation rules in derive.py), and migrating an old checkpoint predating modelcore into current
config/state-dict/optimizer-state shape (legacy.py). modelcore itself never imports this package.
"""
