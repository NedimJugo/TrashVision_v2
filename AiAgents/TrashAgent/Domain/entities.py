"""
Domain Entities

Entiteti sa identitetom (imaju ID).
Ovo su glavni objekti sa kojima agent radi.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from .enums import ImageStatus, WasteCategory, ReviewStatus, ModelType, TrainingMode


@dataclass
class WasteImage:
    """
    Slika otpada koju agent procesira.
    
    Lifecycle:
    Upload → Queued → Processing → Classified/PendingReview → (Review) → Reviewed
    """
    id: Optional[int] = None
    filepath: str = ""
    filename: str = ""
    status: ImageStatus = ImageStatus.QUEUED
    
    # Metadata
    uploaded_at: datetime = field(default_factory=datetime.now)
    processed_at: Optional[datetime] = None
    reviewed_at: Optional[datetime] = None
    
    # File info
    file_size_bytes: int = 0
    width: int = 0
    height: int = 0
    
    def mark_as_processing(self):
        """Agent počinje procesirati"""
        self.status = ImageStatus.PROCESSING
    
    def mark_as_classified(self):
        """Agent završio klasifikaciju (visok confidence)"""
        self.status = ImageStatus.CLASSIFIED
        self.processed_at = datetime.now()
    
    def mark_as_pending_review(self):
        """Agent nije siguran - treba review"""
        self.status = ImageStatus.PENDING_REVIEW
        self.processed_at = datetime.now()
    
    def mark_as_reviewed(self):
        """User pregledao sliku"""
        self.status = ImageStatus.REVIEWED
        self.reviewed_at = datetime.now()
    
    def mark_as_failed(self):
        """Greška pri procesiranju"""
        self.status = ImageStatus.FAILED
        self.processed_at = datetime.now()
    
    @property
    def is_processed(self) -> bool:
        return self.status in [
            ImageStatus.CLASSIFIED,
            ImageStatus.PENDING_REVIEW,
            ImageStatus.REVIEWED
        ]


@dataclass
class Prediction:
    """
    Predikcija modela za jednu sliku.
    
    Agent kreira prediction nakon klasifikacije.
    """
    id: Optional[int] = None
    image_id: int = 0
    
    # Prediction results
    predicted_category: WasteCategory = WasteCategory.TRASH
    confidence: float = 0.0
    
    # Top 3 alternative predictions
    top2_category: Optional[WasteCategory] = None
    top2_confidence: float = 0.0
    top3_category: Optional[WasteCategory] = None
    top3_confidence: float = 0.0
    
    # Model info
    model_version: str = "unknown"
    model_type: ModelType = ModelType.YOLOV8_NANO
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    inference_time_ms: float = 0.0
    
    @property
    def is_confident(self) -> bool:
        """Da li je model siguran (>70%)"""
        return self.confidence >= 0.70
    
    @property
    def needs_review(self) -> bool:
        """Da li treba user review (<70%)"""
        return not self.is_confident


@dataclass
class Review:
    """
    User review/feedback za predikciju.
    
    Agent koristi ovo za continuous learning.
    """
    id: Optional[int] = None
    image_id: int = 0
    prediction_id: int = 0
    
    # User feedback
    user_confirmed_category: WasteCategory = WasteCategory.TRASH
    review_status: ReviewStatus = ReviewStatus.CORRECT
    
    # Analysis
    was_correct: bool = True
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    user_comment: Optional[str] = None
    
    @property
    def is_correction(self) -> bool:
        """Da li je user ispravio predikciju"""
        return self.review_status == ReviewStatus.CORRECTED


@dataclass
class ModelVersion:
    """
    Verzija ML modela.
    
    Agent kreira novu verziju nakon svakog retraining-a.
    """
    id: Optional[int] = None
    
    # Version info
    version_number: int = 1
    model_path: str = ""
    model_type: ModelType = ModelType.YOLOV8_NANO
    
    # Training info
    training_mode: TrainingMode = TrainingMode.INITIAL
    epochs: int = 0
    training_samples_count: int = 0
    
    # Metrics
    accuracy: float = 0.0
    top5_accuracy: float = 0.0
    loss: float = 0.0
    
    # Status
    is_active: bool = False
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    trained_by: str = "system"
    notes: Optional[str] = None
    
    def activate(self):
        """Aktiviraj ovu verziju modela"""
        self.is_active = True
    
    def deactivate(self):
        """Deaktiviraj ovu verziju"""
        self.is_active = False


@dataclass
class SystemSettings:
    """
    Konfiguracione postavke sistema/agenta.
    
    Agent čita ovo da bi odlučio kada treba učenje.
    """
    id: Optional[int] = None
    
    # Learning settings
    auto_retrain_enabled: bool = True
    retrain_threshold: int = 100  # Broj novih uzoraka prije retraining-a
    new_samples_count: int = 0    # Trenutni brojač
    
    # Model settings
    active_model_version: int = 1
    min_confidence_threshold: float = 0.70  # Ispod ovoga ide na review
    
    # Retraining settings
    default_training_mode: TrainingMode = TrainingMode.INCREMENTAL
    incremental_epochs: int = 10
    full_epochs: int = 30
    
    # Metadata
    last_retrain_at: Optional[datetime] = None
    retrain_count: int = 0
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def increment_samples(self):
        """Agent dodao novi uzorak"""
        self.new_samples_count += 1
        self.updated_at = datetime.now()
    
    def should_trigger_retraining(self) -> bool:
        """Da li je vrijeme za retraining"""
        return (
            self.auto_retrain_enabled and 
            self.new_samples_count >= self.retrain_threshold
        )
    
    @property
    def progress_percentage(self) -> float:
        """Progress prema retraining threshold-u (0-100%)"""
        if self.retrain_threshold <= 0:
            return 0.0
        return (self.new_samples_count / self.retrain_threshold) * 100
    
    def reset_samples_counter(self):
        """Reset nakon retraining-a"""
        self.new_samples_count = 0
        self.retrain_count += 1
        self.last_retrain_at = datetime.now()
        self.updated_at = datetime.now()


if __name__ == "__main__":
    # Test
    print("✅ Domain Entities loaded")
    
    # Test WasteImage lifecycle
    img = WasteImage(id=1, filepath="/uploads/test.jpg")
    print(f"\n   Image status: {img.status}")
    
    img.mark_as_processing()
    print(f"   → Processing: {img.status}")
    
    img.mark_as_classified()
    print(f"   → Classified: {img.status}")
    print(f"   Is processed: {img.is_processed}")
    
    # Test SystemSettings
    settings = SystemSettings(retrain_threshold=100)
    settings.new_samples_count = 95
    print(f"\n   New samples: {settings.new_samples_count}/{settings.retrain_threshold}")
    print(f"   Should retrain: {settings.should_trigger_retraining()}")