"""
AiAgents.TrashAgent.Infrastructure

Infrastructure sloj - DB, ML, File Storage.

Komponente:
- database: SQLAlchemy modeli + CRUD
- waste_classifier: ML classifier interface + YOLO implementacija
- yolo_classifier: YOLOv8 implementacija + trainer
- file_storage: File management helper
"""

from .database import (
    DatabaseHelper,
    init_db,
    WasteImageModel,
    PredictionModel,
    ReviewModel,
    ModelVersionModel,
    SystemSettingsModel,
)
from .waste_classifier import IWasteClassifier
from .yolo_classifier import YoloWasteClassifier, YoloTrainer, IModelTrainer
from .file_storage import FileStorage

__all__ = [
    # Database
    "DatabaseHelper",
    "init_db",
    "WasteImageModel",
    "PredictionModel",
    "ReviewModel",
    "ModelVersionModel",
    "SystemSettingsModel",
    
    # Classifiers
    "IWasteClassifier",
    "YoloWasteClassifier",
    "IModelTrainer",
    "YoloTrainer",
    
    # File Storage
    "FileStorage",
]