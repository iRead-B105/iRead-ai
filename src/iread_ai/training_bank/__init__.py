from __future__ import annotations

from functools import lru_cache

from .generator import RULE_BASED_TYPES, RuleBasedBasicTrainingGenerator
from .repository import SQLiteLearningUnitRepository


@lru_cache(maxsize=1)
def default_basic_training_generator() -> RuleBasedBasicTrainingGenerator:
    return RuleBasedBasicTrainingGenerator(SQLiteLearningUnitRepository())


__all__ = [
    "RULE_BASED_TYPES",
    "RuleBasedBasicTrainingGenerator",
    "SQLiteLearningUnitRepository",
    "default_basic_training_generator",
]
