"""
AiAgents.Core - Framework sloj

Generičke apstrakcije za izgradnju AI agenata.
Ne sadrži biznis logiku - samo interfejse i bazne klase.

Izvoz:
- SoftwareAgent: Bazna klasa za agente (Sense→Think→Act)
- LearningAgent: Proširenje za agente koji uče (+ Learn)
- IPerceptionSource: Interface za izvore percepcije
- IPolicy: Interface za pravila odlučivanja
- IActuator: Interface za izvršioce akcija
- ILearningComponent: Interface za učenje
"""

from .software_agent import SoftwareAgent, LearningAgent, NoOpAgent
from .perception_source import IPerceptionSource, NoOpPerceptionSource
from .policy import IPolicy, NoOpPolicy
from .actuator import IActuator, NoOpActuator
from .learning_component import ILearningComponent, NoOpLearningComponent

__all__ = [
    # Main classes
    "SoftwareAgent",
    "LearningAgent",
    
    # Interfaces
    "IPerceptionSource",
    "IPolicy",
    "IActuator",
    "ILearningComponent",
    
    # Test implementations
    "NoOpAgent",
    "NoOpPerceptionSource",
    "NoOpPolicy",
    "NoOpActuator",
    "NoOpLearningComponent",
]

__version__ = "1.0.0"