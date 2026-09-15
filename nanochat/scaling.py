"""
Backward-compat shim: muP scaling-law horizon derivation now lives in modelcore/scaling.py (this
was pure math with no nanochat dependency -- tinylab carried an identical copy). Kept here so
existing `from nanochat.scaling import derive_training_plan, B_REF, TrainingPlan` call sites keep
working unchanged.
"""
from modelcore.scaling import B_REF, TrainingPlan, derive_training_plan

__all__ = ["B_REF", "TrainingPlan", "derive_training_plan"]
