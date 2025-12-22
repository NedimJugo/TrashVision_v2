"""
AiAgents.TrashAgent.Domain

Domain sloj - entiteti, enum-i, value objects, biznis pravila.

NE SADRŽI:
- Web logiku (routing, controllers)
- Infrastructure logiku (DB, ML)
- Application logiku (use-cases, runnere)

SADRŽI:
- Entitete sa identitetom (WasteImage, Prediction, Review)
- Enum-e (ImageStatus, WasteCategory)
- Value objects (RecyclingInfo, ClassificationDecision)
- Biznis pravila (validacije, invarijante)
"""

# Enums
from .enums import (
    ImageStatus,
    WasteCategory,
    ReviewStatus,
    ModelType,
    TrainingMode,
)

# Value Objects
from .value_objects import (
    RecyclingInfo,
    ClassificationDecision,
    TrainingDecision,
)

# Entities
from .entities import (
    WasteImage,
    Prediction,
    Review,
    ModelVersion,
    SystemSettings,
)

__all__ = [
    # Enums
    "ImageStatus",
    "WasteCategory",
    "ReviewStatus",
    "ModelType",
    "TrainingMode",
    
    # Value Objects
    "RecyclingInfo",
    "ClassificationDecision",
    "TrainingDecision",
    
    # Entities
    "WasteImage",
    "Prediction",
    "Review",
    "ModelVersion",
    "SystemSettings",
]