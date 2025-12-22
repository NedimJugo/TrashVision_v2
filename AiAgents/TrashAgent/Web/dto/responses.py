"""
DTO (Data Transfer Objects) za API responses.

Ovo su klase koje se vraćaju Web klijentima (JSON format).
"""

from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel


# ========================================
# IMAGE RESPONSES
# ========================================

class ImageUploadResponse(BaseModel):
    """Response nakon upload-a slike"""
    success: bool
    image_id: int
    filename: str
    status: str
    message: str


class ImageStatusResponse(BaseModel):
    """Response za status slike"""
    image_id: int
    filename: str
    status: str
    uploaded_at: datetime
    processed_at: Optional[datetime] = None


class PredictionResponse(BaseModel):
    """Response sa predikcijom"""
    class_name: str
    confidence: float
    emoji: str
    recyclable: bool
    disposal_instruction: str
    container_color: Optional[str]


class ClassificationResultResponse(BaseModel):
    """Kompletan rezultat klasifikacije"""
    image_id: int
    filename: str
    status: str
    prediction: Optional[PredictionResponse] = None
    top3: List[dict] = []
    processed_at: Optional[datetime] = None
    needs_review: bool = False


# ========================================
# REVIEW RESPONSES
# ========================================

class ReviewSubmitResponse(BaseModel):
    """Response nakon submit-a review-a"""
    success: bool
    message: str
    should_retrain: bool
    new_samples_count: int
    threshold: int
    progress_percentage: float


# ========================================
# LEARNING/TRAINING RESPONSES
# ========================================

class LearningStatsResponse(BaseModel):
    """Statistika continuous learning-a"""
    new_samples_count: int
    threshold: int
    progress_percentage: float
    auto_retrain_enabled: bool
    last_retrain_at: Optional[datetime]
    retrain_count: int


class RetrainStatusResponse(BaseModel):
    """Status retraining-a"""
    success: bool
    message: str
    mode: str
    new_model_version: Optional[int] = None
    training_time_seconds: Optional[float] = None


# ========================================
# AGENT STATUS RESPONSES
# ========================================

class AgentStatusResponse(BaseModel):
    """Status agent-a"""
    agent_name: str
    is_running: bool
    tick_count: int
    last_tick_at: Optional[datetime]
    queue_size: int


class SystemStatusResponse(BaseModel):
    """Kompletan system status"""
    classification_agent: AgentStatusResponse
    learning_agent: AgentStatusResponse
    database_connected: bool
    model_loaded: bool
    active_model_version: int


# ========================================
# STATISTICS RESPONSES
# ========================================

class QueueStatsResponse(BaseModel):
    """Statistika queue-a"""
    queued: int
    processing: int
    classified: int
    pending_review: int
    total: int


class ClassificationStatsResponse(BaseModel):
    """Statistika klasifikacija"""
    total_predictions: int
    avg_confidence: float
    high_confidence_count: int
    review_needed_count: int


# ========================================
# ERROR RESPONSES
# ========================================

class ErrorResponse(BaseModel):
    """Error response"""
    success: bool = False
    error: str
    details: Optional[str] = None


if __name__ == "__main__":
    print("✅ DTO responses loaded")