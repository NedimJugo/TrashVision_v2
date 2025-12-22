"""
Database Infrastructure

SQLAlchemy modeli i CRUD operacije.
Mapira Domain entitete na DB tabele.
"""

from datetime import datetime
from typing import Optional, List
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Float,
    Boolean,
    DateTime,
    Text,
    ForeignKey,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Session

from ..Domain import (
    WasteImage,
    Prediction,
    Review,
    ModelVersion,
    SystemSettings,
    ImageStatus,
    WasteCategory,
    ReviewStatus,
    ModelType,
    TrainingMode,
)

Base = declarative_base()


# ========================================
# SQLALCHEMY MODELI (DB tabele)
# ========================================

class WasteImageModel(Base):
    """DB tabela za slike"""
    __tablename__ = "waste_images"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    filepath = Column(String(500), nullable=False)
    filename = Column(String(255), nullable=False)
    status = Column(String(50), nullable=False, default="queued")
    
    # Metadata
    uploaded_at = Column(DateTime, nullable=False, default=datetime.now)
    processed_at = Column(DateTime, nullable=True)
    reviewed_at = Column(DateTime, nullable=True)
    
    # File info
    file_size_bytes = Column(Integer, default=0)
    width = Column(Integer, default=0)
    height = Column(Integer, default=0)
    
    # Relationships
    predictions = relationship("PredictionModel", back_populates="image")
    reviews = relationship("ReviewModel", back_populates="image")
    
    def to_domain(self) -> WasteImage:
        """Konvertuj DB model u Domain entitet"""
        return WasteImage(
            id=self.id,
            filepath=self.filepath,
            filename=self.filename,
            status=ImageStatus(self.status),
            uploaded_at=self.uploaded_at,
            processed_at=self.processed_at,
            reviewed_at=self.reviewed_at,
            file_size_bytes=self.file_size_bytes,
            width=self.width,
            height=self.height,
        )
    
    @staticmethod
    def from_domain(entity: WasteImage) -> "WasteImageModel":
        """Konvertuj Domain entitet u DB model"""
        return WasteImageModel(
            id=entity.id,
            filepath=entity.filepath,
            filename=entity.filename,
            status=entity.status.value,
            uploaded_at=entity.uploaded_at,
            processed_at=entity.processed_at,
            reviewed_at=entity.reviewed_at,
            file_size_bytes=entity.file_size_bytes,
            width=entity.width,
            height=entity.height,
        )


class PredictionModel(Base):
    """DB tabela za predikcije"""
    __tablename__ = "predictions"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    image_id = Column(Integer, ForeignKey("waste_images.id"), nullable=False)
    
    # Prediction results
    predicted_category = Column(String(50), nullable=False)
    confidence = Column(Float, nullable=False)
    
    # Top 3
    top2_category = Column(String(50), nullable=True)
    top2_confidence = Column(Float, default=0.0)
    top3_category = Column(String(50), nullable=True)
    top3_confidence = Column(Float, default=0.0)
    
    # Model info
    model_version = Column(String(50), default="unknown")
    model_type = Column(String(50), default="yolov8n")
    
    # Metadata
    created_at = Column(DateTime, nullable=False, default=datetime.now)
    inference_time_ms = Column(Float, default=0.0)
    
    # Relationships
    image = relationship("WasteImageModel", back_populates="predictions")
    reviews = relationship("ReviewModel", back_populates="prediction")
    
    def to_domain(self) -> Prediction:
        """Konvertuj u Domain entitet"""
        return Prediction(
            id=self.id,
            image_id=self.image_id,
            predicted_category=WasteCategory(self.predicted_category),
            confidence=self.confidence,
            top2_category=WasteCategory(self.top2_category) if self.top2_category else None,
            top2_confidence=self.top2_confidence,
            top3_category=WasteCategory(self.top3_category) if self.top3_category else None,
            top3_confidence=self.top3_confidence,
            model_version=self.model_version,
            model_type=ModelType(self.model_type),
            created_at=self.created_at,
            inference_time_ms=self.inference_time_ms,
        )


class ReviewModel(Base):
    """DB tabela za review-e"""
    __tablename__ = "reviews"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    image_id = Column(Integer, ForeignKey("waste_images.id"), nullable=False)
    prediction_id = Column(Integer, ForeignKey("predictions.id"), nullable=False)
    
    # User feedback
    user_confirmed_category = Column(String(50), nullable=False)
    review_status = Column(String(50), nullable=False)
    was_correct = Column(Boolean, nullable=False)
    
    # Metadata
    created_at = Column(DateTime, nullable=False, default=datetime.now)
    user_comment = Column(Text, nullable=True)
    
    # Relationships
    image = relationship("WasteImageModel", back_populates="reviews")
    prediction = relationship("PredictionModel", back_populates="reviews")
    
    def to_domain(self) -> Review:
        """Konvertuj u Domain entitet"""
        return Review(
            id=self.id,
            image_id=self.image_id,
            prediction_id=self.prediction_id,
            user_confirmed_category=WasteCategory(self.user_confirmed_category),
            review_status=ReviewStatus(self.review_status),
            was_correct=self.was_correct,
            created_at=self.created_at,
            user_comment=self.user_comment,
        )


class ModelVersionModel(Base):
    """DB tabela za verzije modela"""
    __tablename__ = "model_versions"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    version_number = Column(Integer, nullable=False, unique=True)
    model_path = Column(String(500), nullable=False)
    model_type = Column(String(50), default="yolov8n")
    
    # Training info
    training_mode = Column(String(50), default="initial")
    epochs = Column(Integer, default=0)
    training_samples_count = Column(Integer, default=0)
    
    # Metrics
    accuracy = Column(Float, default=0.0)
    top5_accuracy = Column(Float, default=0.0)
    loss = Column(Float, default=0.0)
    
    # Status
    is_active = Column(Boolean, default=False)
    
    # Metadata
    created_at = Column(DateTime, nullable=False, default=datetime.now)
    trained_by = Column(String(100), default="system")
    notes = Column(Text, nullable=True)
    
    def to_domain(self) -> ModelVersion:
        """Konvertuj u Domain entitet"""
        return ModelVersion(
            id=self.id,
            version_number=self.version_number,
            model_path=self.model_path,
            model_type=ModelType(self.model_type),
            training_mode=TrainingMode(self.training_mode),
            epochs=self.epochs,
            training_samples_count=self.training_samples_count,
            accuracy=self.accuracy,
            top5_accuracy=self.top5_accuracy,
            loss=self.loss,
            is_active=self.is_active,
            created_at=self.created_at,
            trained_by=self.trained_by,
            notes=self.notes,
        )


class SystemSettingsModel(Base):
    """DB tabela za system settings (jedna row)"""
    __tablename__ = "system_settings"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Learning settings
    auto_retrain_enabled = Column(Boolean, default=True)
    retrain_threshold = Column(Integer, default=100)
    new_samples_count = Column(Integer, default=0)
    
    # Model settings
    active_model_version = Column(Integer, default=1)
    min_confidence_threshold = Column(Float, default=0.70)
    
    # Retraining settings
    default_training_mode = Column(String(50), default="incremental")
    incremental_epochs = Column(Integer, default=10)
    full_epochs = Column(Integer, default=30)
    
    # Metadata
    last_retrain_at = Column(DateTime, nullable=True)
    retrain_count = Column(Integer, default=0)
    created_at = Column(DateTime, nullable=False, default=datetime.now)
    updated_at = Column(DateTime, nullable=False, default=datetime.now)
    
    def to_domain(self) -> SystemSettings:
        """Konvertuj u Domain entitet"""
        return SystemSettings(
            id=self.id,
            auto_retrain_enabled=self.auto_retrain_enabled,
            retrain_threshold=self.retrain_threshold,
            new_samples_count=self.new_samples_count,
            active_model_version=self.active_model_version,
            min_confidence_threshold=self.min_confidence_threshold,
            default_training_mode=TrainingMode(self.default_training_mode),
            incremental_epochs=self.incremental_epochs,
            full_epochs=self.full_epochs,
            last_retrain_at=self.last_retrain_at,
            retrain_count=self.retrain_count,
            created_at=self.created_at,
            updated_at=self.updated_at,
        )


# ========================================
# DATABASE HELPER CLASS
# ========================================

class DatabaseHelper:
    """
    Helper za DB operacije.
    
    Omogućava:
    - Kreiranje/brisanje baze
    - Session management
    - CRUD operacije
    """
    
    def __init__(self, db_url: str = "sqlite:///trashvision.db"):
        self.engine = create_engine(db_url, echo=False)
        self.SessionLocal = sessionmaker(bind=self.engine)
    
    def create_all_tables(self):
        """Kreiraj sve tabele"""
        Base.metadata.create_all(self.engine)
        print("✅ Database tables created")
    
    def drop_all_tables(self):
        """Obriši sve tabele (OPASNO!)"""
        Base.metadata.drop_all(self.engine)
        print("⚠️  Database tables dropped")
    
    def get_session(self) -> Session:
        """Dobij DB session"""
        return self.SessionLocal()
    
    def init_default_settings(self):
        """Kreiraj default SystemSettings ako ne postoji"""
        session = self.get_session()
        try:
            existing = session.query(SystemSettingsModel).first()
            if not existing:
                default_settings = SystemSettingsModel()
                session.add(default_settings)
                session.commit()
                print("✅ Default SystemSettings created")
        finally:
            session.close()


# ========================================
# INITIALIZATION FUNCTION
# ========================================

def init_db(db_url: str = "sqlite:///trashvision.db"):
    """
    Inicijalizuj bazu - pozovi ovo jednom pri setup-u.
    
    Args:
        db_url: SQLAlchemy connection string
    """
    print("🗄️  Initializing database...")
    
    db = DatabaseHelper(db_url)
    db.create_all_tables()
    db.init_default_settings()
    
    print("✅ Database ready!")
    return db


if __name__ == "__main__":
    # Test: Kreiraj bazu
    init_db()