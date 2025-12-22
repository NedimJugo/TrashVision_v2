"""
Agent Runners

Agent runneri - Sense→Think→Act→Learn ciklus.

⭐ OVO JE NAJVAŽNIJI DIO PROJEKTA! ⭐

Runneri:
- ClassificationAgentRunner: Klasifikuje slike (Sense→Think→Act)
- LearningAgentRunner: Retrenira model (Sense→Think→Act→Learn)
"""

from .classification_runner import ClassificationAgentRunner, ClassificationResult
from .learning_runner import LearningAgentRunner, LearningResult

__all__ = [
    "ClassificationAgentRunner",
    "ClassificationResult",
    "LearningAgentRunner",
    "LearningResult",
]