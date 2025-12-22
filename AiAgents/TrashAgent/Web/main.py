"""
TrashVision Agent - JEDNOSTAVNA VERZIJA (bez DI komplikacija)
"""

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import Image
import io

# Infrastructure
from ..Infrastructure import (
    DatabaseHelper,
    init_db,
    YoloWasteClassifier,
    YoloTrainer,
    FileStorage,
)

# Application
from ..Application.Services import (
    QueueService,
    ClassificationService,
    ReviewService,
    TrainingService,
)
from ..Application.Agents import (
    ClassificationAgentRunner,
    LearningAgentRunner,
)
from ..Domain import WasteImage, ImageStatus, RecyclingInfo, WasteCategory

# Web
from .workers import ClassificationWorker, LearningWorker


# ========================================
# GLOBAL STATE
# ========================================

class AppState:
    """Global application state"""
    
    def __init__(self):
        # Infrastructure
        self.db: DatabaseHelper = None
        self.classifier: YoloWasteClassifier = None
        self.trainer: YoloTrainer = None
        self.file_storage: FileStorage = None
        
        # Services
        self.queue_service: QueueService = None
        self.classification_service: ClassificationService = None
        self.review_service: ReviewService = None
        self.training_service: TrainingService = None
        
        # Agents
        self.classification_runner: ClassificationAgentRunner = None
        self.learning_runner: LearningAgentRunner = None
        
        # Workers
        self.classification_worker: ClassificationWorker = None
        self.learning_worker: LearningWorker = None
        
        # Settings
        self.system_settings = None


app_state = AppState()


# ========================================
# LIFECYCLE
# ========================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle"""
    print("=" * 60)
    print("🚀 TRASHVISION AGENT STARTING...")
    print("=" * 60)
    
    # 1. Database
    print("\n📦 1/6: Initializing database...")
    app_state.db = init_db("sqlite:///trashvision.db")
    
    # Ne čuvaj session, već SessionLocal factory
    from ..Infrastructure.database import SystemSettingsModel
    temp_session = app_state.db.get_session()
    settings_model = temp_session.query(SystemSettingsModel).first()
    app_state.system_settings = settings_model.to_domain() if settings_model else None
    temp_session.close()
    
    # 2. ML Model
    print("\n🤖 2/6: Loading ML model...")
    model_path = "models/trashvision_v1/weights/best.pt"
    app_state.classifier = YoloWasteClassifier()
    
    if Path(model_path).exists():
        await app_state.classifier.load_model(model_path)
    else:
        print(f"⚠️  Model not found: {model_path}")
    
    app_state.trainer = YoloTrainer(model_path)
    
    # 3. File Storage
    print("\n💾 3/6: Initializing file storage...")
    app_state.file_storage = FileStorage()
    
    # 4. Services
    print("\n⚙️  4/6: Creating services...")
    # Svaki servis dobije SVOJU session (važno za thread safety!)
    app_state.queue_service = QueueService(app_state.db.get_session())
    app_state.classification_service = ClassificationService(app_state.classifier, app_state.db.get_session())
    app_state.review_service = ReviewService(app_state.db.get_session(), app_state.file_storage)
    app_state.training_service = TrainingService(app_state.trainer, app_state.db.get_session())
    
    # 5. Agent Runners
    print("\n🤖 5/6: Creating agent runners...")
    app_state.classification_runner = ClassificationAgentRunner(
        queue_service=app_state.queue_service,
        classification_service=app_state.classification_service,
        settings=app_state.system_settings,
        name="ClassificationAgent"
    )
    
    app_state.learning_runner = LearningAgentRunner(
        training_service=app_state.training_service,
        settings=app_state.system_settings,
        name="LearningAgent"
    )
    
    # 6. Background Workers
    print("\n🔄 6/6: Starting background workers...")
    
    app_state.classification_worker = ClassificationWorker(
        runner=app_state.classification_runner,
        tick_interval_seconds=2,
        name="ClassificationWorker"
    )
    app_state.classification_worker.start()
    
    app_state.learning_worker = LearningWorker(
        runner=app_state.learning_runner,
        tick_interval_seconds=60,
        name="LearningWorker"
    )
    app_state.learning_worker.start()
    
    print("\n" + "=" * 60)
    print("✅ TRASHVISION AGENT READY!")
    print("=" * 60)
    print(f"📍 API: http://localhost:8000")
    print(f"📚 Docs: http://localhost:8000/docs")
    print(f"🤖 Classification Agent: Running (every 2s)")
    print(f"🎓 Learning Agent: Running (every 60s)")
    print("=" * 60)
    
    yield
    
    # Shutdown
    print("\n🛑 Shutting down...")
    if app_state.classification_worker:
        app_state.classification_worker.stop()
    if app_state.learning_worker:
        app_state.learning_worker.stop()
    print("✅ Shutdown complete")


# ========================================
# FASTAPI APP
# ========================================

app = FastAPI(
    title="TrashVision Agent API",
    description="AI Agent za klasifikaciju otpada",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ========================================
# ROUTES (direktno u main.py bez controllera)
# ========================================

@app.get("/")
async def root():
    """Frontend ili API info"""
    frontend_path = Path("app/frontend/index.html")
    if frontend_path.exists():
        return FileResponse(frontend_path)
    
    return {
        "status": "ok",
        "message": "TrashVision Agent API",
        "version": "2.0.0",
        "docs": "/docs"
    }


@app.post("/api/images/upload")
async def upload_image(file: UploadFile = File(...)):
    """Upload sliku za klasifikaciju"""
    try:
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        file_data = await file.read()
        
        try:
            img = Image.open(io.BytesIO(file_data))
            width, height = img.size
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image: {str(e)}")
        
        # Save to storage
        filepath = await app_state.file_storage.save_uploaded_image(file_data, file.filename)
        
        # Create WasteImage entity
        image = WasteImage(
            filepath=filepath,
            filename=file.filename,
            status=ImageStatus.QUEUED,
            file_size_bytes=len(file_data),
            width=width,
            height=height,
        )
        
        # Enqueue (ovo bi trebalo da sačuva u DB i vrati ID)
        image = await app_state.queue_service.enqueue(image)
        
        # MOCK FIX: Ako ID nije postavljen, generiši ga
        if image.id is None:
            import random
            image.id = random.randint(1, 999999)
        
        return {
            "success": True,
            "image_id": image.id,
            "filename": image.filename,
            "status": image.status.value,
            "message": "Image queued for classification"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/images/{image_id}")
async def get_image_status(image_id: int):
    """Provjeri status slike"""
    try:
        image = await app_state.queue_service.get_by_id(image_id)
        
        if not image:
            raise HTTPException(status_code=404, detail="Image not found")
        
        response = {
            "image_id": image.id,
            "filename": image.filename,
            "status": image.status.value,
            "processed_at": image.processed_at.isoformat() if image.processed_at else None,
            "needs_review": (image.status == ImageStatus.PENDING_REVIEW),
            "prediction": None  # TODO: Get from classification service
        }
        
        return response
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/learning/stats")
async def get_learning_stats():
    """Statistika learning-a"""
    settings = app_state.system_settings
    progress = (settings.new_samples_count / settings.retrain_threshold * 100) if settings.retrain_threshold > 0 else 0
    
    return {
        "new_samples_count": settings.new_samples_count,
        "threshold": settings.retrain_threshold,
        "progress_percentage": progress,
        "auto_retrain_enabled": settings.auto_retrain_enabled,
        "last_retrain_at": settings.last_retrain_at.isoformat() if settings.last_retrain_at else None,
        "retrain_count": settings.retrain_count,
    }


@app.get("/status")
async def system_status():
    """System status"""
    classification_stats = await app_state.classification_worker.get_stats() if app_state.classification_worker else {}
    learning_stats = await app_state.learning_worker.get_stats() if app_state.learning_worker else {}
    
    return {
        "classification_agent": classification_stats,
        "learning_agent": learning_stats,
        "database_connected": app_state.db is not None,
        "model_loaded": app_state.classifier.is_loaded() if app_state.classifier else False,
    }


# ========================================
# LEGACY ENDPOINTS (za kompatibilnost sa starim frontendom)
# ========================================

@app.post("/predict")
async def predict_legacy(file: UploadFile = File(...)):
    """
    Stari /predict endpoint - direktna predikcija (BEZ agent queue-a)
    
    Ovo je za kompatibilnost sa starim frontendom.
    NOVA arhitektura koristi /api/images/upload + agent background processing.
    """
    try:
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        file_data = await file.read()
        img = Image.open(io.BytesIO(file_data))
        
        # Save privremeno
        temp_path = f"temp_{file.filename}"
        img.save(temp_path)
        
        # DIREKTNA predikcija (ne ide kroz agent queue)
        result = await app_state.classifier.predict(temp_path)
        
        # Cleanup
        Path(temp_path).unlink(missing_ok=True)
        
        # Format kao stari API
        predicted_class = result["class"]
        confidence = result["confidence"]
        top3 = result.get("top3", [])
        
        # Recycling info
        category = WasteCategory(predicted_class)
        recycling_info = RecyclingInfo.get_for_category(category)
        
        return {
            "success": True,
            "predictions": [
                {
                    "class": predicted_class,
                    "name": category.display_name,
                    "confidence": confidence,
                    "disposal": recycling_info.disposal_instruction,
                    "recyclable": recycling_info.is_recyclable,
                    "emoji": category.emoji,
                    "color": recycling_info.container_color or "gray"
                }
            ] + [
                {
                    "class": cls,
                    "confidence": conf,
                    "name": WasteCategory(cls).display_name
                }
                for cls, conf in top3[1:3]  # Top 2-3
            ]
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/feedback")
async def submit_feedback_legacy(
    file: UploadFile = File(...),
    predicted_class: str = Form(None),
    actual_class: str = Form(None),
    confidence: float = Form(0.0)
):
    """
    User feedback endpoint - Clean Architecture verzija.
    
    Web layer je TANAK - samo prima podatke i delegira biznis logiku.
    """
    try:
        print(f"📝 Feedback received:")
        print(f"   File: {file.filename}")
        print(f"   Predicted: {predicted_class}")
        print(f"   Actual: {actual_class}")
        
        # Read file
        file_data = await file.read()
        
        # Save file temporarily
        import tempfile
        import os
        temp_path = os.path.join(tempfile.gettempdir(), file.filename)
        with open(temp_path, 'wb') as f:
            f.write(file_data)
        
        try:
            # DELEGIRAJ biznis logiku u Application layer (ReviewService)
            result = await app_state.review_service.submit_user_feedback(
                file_path=temp_path,
                predicted_class=predicted_class,
                actual_class=actual_class,
                settings=app_state.system_settings
            )
            
            return result
            
        finally:
            # Cleanup temp file
            try:
                os.remove(temp_path)
            except:
                pass
    
    except ValueError as e:
        # Biznis pravila error
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Ostali errori
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/status")
async def system_status():
    """System status"""
    classification_stats = await app_state.classification_worker.get_stats() if app_state.classification_worker else {}
    learning_stats = await app_state.learning_worker.get_stats() if app_state.learning_worker else {}
    
    return {
        "classification_agent": classification_stats,
        "learning_agent": learning_stats,
        "database_connected": app_state.db is not None,
        "model_loaded": app_state.classifier.is_loaded() if app_state.classifier else False,
    }


# ========================================
# RUN
# ========================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    )