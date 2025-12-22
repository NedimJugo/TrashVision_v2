"""
Learning Controller

Controller za continuous learning i retraining operacije.
"""

from fastapi import APIRouter, HTTPException, Depends

from ..dto import (
    LearningStatsResponse,
    RetrainStatusResponse,
)
from ...Domain import SystemSettings
from ...Application.Services import TrainingService

router = APIRouter(prefix="/api/learning", tags=["Learning"])


# ========================================
# DEPENDENCY INJECTION
# ========================================

def get_training_service():
    """DI: TrainingService"""
    raise NotImplementedError("Configure DI in main.py")


def get_system_settings():
    """DI: SystemSettings"""
    raise NotImplementedError("Configure DI in main.py")


# ========================================
# ENDPOINTS
# ========================================

@router.get("/stats", response_model=LearningStatsResponse)
async def get_learning_stats(
    settings: SystemSettings = Depends(get_system_settings)
):
    """
    Statistika continuous learning-a.
    
    Returns:
        LearningStatsResponse: Counters, thresholds, progress
    """
    progress = (settings.new_samples_count / settings.retrain_threshold * 100) if settings.retrain_threshold > 0 else 0
    
    return LearningStatsResponse(
        new_samples_count=settings.new_samples_count,
        threshold=settings.retrain_threshold,
        progress_percentage=progress,
        auto_retrain_enabled=settings.auto_retrain_enabled,
        last_retrain_at=settings.last_retrain_at,
        retrain_count=settings.retrain_count,
    )


@router.post("/retrain", response_model=RetrainStatusResponse)
async def trigger_manual_retrain(
    mode: str = "incremental",
    epochs: int = 10,
    training_service: TrainingService = Depends(get_training_service),
    settings: SystemSettings = Depends(get_system_settings)
):
    """
    Manuelno pokreni retraining (za admin).
    
    Args:
        mode: "incremental" ili "full"
        epochs: Broj epoha
    
    Returns:
        RetrainStatusResponse: Status retraining-a
    """
    try:
        if mode not in ["incremental", "full"]:
            raise HTTPException(status_code=400, detail="Mode must be 'incremental' or 'full'")
        
        if epochs < 1 or epochs > 100:
            raise HTTPException(status_code=400, detail="Epochs must be between 1 and 100")
        
        # Pokreni retraining (može trajati dugo!)
        from ...Domain import TrainingMode
        training_mode = TrainingMode.INCREMENTAL if mode == "incremental" else TrainingMode.FULL
        
        import time
        start_time = time.time()
        
        new_version = await training_service.retrain_model(
            mode=training_mode,
            epochs=epochs,
            settings=settings
        )
        
        training_time = time.time() - start_time
        
        return RetrainStatusResponse(
            success=True,
            message=f"Retraining completed successfully ({mode} mode)",
            mode=mode,
            new_model_version=new_version.version_number,
            training_time_seconds=training_time,
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/config", response_model=dict)
async def get_learning_config(
    settings: SystemSettings = Depends(get_system_settings)
):
    """
    Trenutna konfiguracija learning-a.
    
    Returns:
        dict: Sve settings
    """
    return {
        "auto_retrain_enabled": settings.auto_retrain_enabled,
        "retrain_threshold": settings.retrain_threshold,
        "new_samples_count": settings.new_samples_count,
        "min_confidence_threshold": settings.min_confidence_threshold,
        "default_training_mode": settings.default_training_mode.value,
        "incremental_epochs": settings.incremental_epochs,
        "full_epochs": settings.full_epochs,
    }


@router.post("/config", response_model=dict)
async def update_learning_config(
    config: dict,
    settings: SystemSettings = Depends(get_system_settings)
):
    """
    Ažuriraj learning konfiguraciju.
    
    Args:
        config: Novi settings
    
    Returns:
        dict: Ažurirani settings
    """
    try:
        # Validate i ažuriraj settings
        if "retrain_threshold" in config:
            settings.retrain_threshold = int(config["retrain_threshold"])
        
        if "auto_retrain_enabled" in config:
            settings.auto_retrain_enabled = bool(config["auto_retrain_enabled"])
        
        if "min_confidence_threshold" in config:
            threshold = float(config["min_confidence_threshold"])
            if 0.0 <= threshold <= 1.0:
                settings.min_confidence_threshold = threshold
        
        # TODO: Save to DB
        
        return {
            "success": True,
            "message": "Config updated successfully"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    print("✅ LearningController loaded")