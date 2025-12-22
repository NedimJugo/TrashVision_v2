"""
Prediction Controller

Controller za image upload i prediction operacije.

VAŽNO: Controller je TANAK - samo DTO mapping i pozivanje servisa.
NE SADRŽI biznis logiku!
"""

from fastapi import APIRouter, File, UploadFile, HTTPException, Depends
from PIL import Image
import io

from ..dto import (
    ImageUploadResponse,
    ImageStatusResponse,
    ClassificationResultResponse,
    PredictionResponse,
    ErrorResponse,
)
from ...Domain import WasteImage, ImageStatus, RecyclingInfo
from ...Application.Services import QueueService
from ...Infrastructure import FileStorage

router = APIRouter(prefix="/api/images", tags=["Images"])


# ========================================
# DEPENDENCY INJECTION (će biti konfigurisano u main.py)
# ========================================

def get_queue_service():
    """DI: QueueService"""
    # TODO: Inject pravi servis iz DI container-a
    raise NotImplementedError("Configure DI in main.py")


def get_file_storage():
    """DI: FileStorage"""
    raise NotImplementedError("Configure DI in main.py")


# ========================================
# ENDPOINTS
# ========================================

@router.post("/upload", response_model=ImageUploadResponse)
async def upload_image(
    file: UploadFile = File(...),
    queue_service: QueueService = Depends(get_queue_service),
    file_storage: FileStorage = Depends(get_file_storage)
):
    """
    Upload sliku za klasifikaciju.
    
    Slika se stavlja u queue - agent će je procesirati u background-u.
    
    Returns:
        ImageUploadResponse: Status upload-a
    """
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read file
        file_data = await file.read()
        
        # Validate image
        try:
            img = Image.open(io.BytesIO(file_data))
            width, height = img.size
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image: {str(e)}")
        
        # Save to storage
        filepath = await file_storage.save_uploaded_image(file_data, file.filename)
        
        # Create WasteImage entity
        image = WasteImage(
            filepath=filepath,
            filename=file.filename,
            status=ImageStatus.QUEUED,
            file_size_bytes=len(file_data),
            width=width,
            height=height,
        )
        
        # Enqueue (agent će procesirati)
        image = await queue_service.enqueue(image)
        
        return ImageUploadResponse(
            success=True,
            image_id=image.id,
            filename=image.filename,
            status=image.status.value,
            message="Image uploaded successfully. Agent will process it soon."
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{image_id}", response_model=ClassificationResultResponse)
async def get_image_status(
    image_id: int,
    queue_service: QueueService = Depends(get_queue_service)
):
    """
    Provjeri status slike i predikciju.
    
    Args:
        image_id: ID slike
    
    Returns:
        ClassificationResultResponse: Status i predikcija (ako postoji)
    """
    try:
        # Get image
        image = await queue_service.get_by_id(image_id)
        
        if not image:
            raise HTTPException(status_code=404, detail="Image not found")
        
        # Base response
        response = ClassificationResultResponse(
            image_id=image.id,
            filename=image.filename,
            status=image.status.value,
            processed_at=image.processed_at,
            needs_review=(image.status == ImageStatus.PENDING_REVIEW),
        )
        
        # Ako je procesirana, dodaj predikciju
        if image.is_processed:
            # TODO: Get prediction from ClassificationService
            # Za sada mock
            response.prediction = None
        
        return response
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/", response_model=dict)
async def list_images(
    status: str = "all",
    limit: int = 50,
    queue_service: QueueService = Depends(get_queue_service)
):
    """
    Lista slika sa filterom po statusu.
    
    Args:
        status: Filter (queued, classified, pending_review, all)
        limit: Max broj rezultata
    
    Returns:
        dict: {"images": [...], "total": 123}
    """
    try:
        if status == "all":
            # Get all
            images = []  # TODO: Implement
        else:
            # Filter by status
            image_status = ImageStatus(status)
            images = await queue_service.get_all_by_status(image_status, limit)
        
        return {
            "images": [
                {
                    "id": img.id,
                    "filename": img.filename,
                    "status": img.status.value,
                    "uploaded_at": img.uploaded_at.isoformat(),
                }
                for img in images
            ],
            "total": len(images)
        }
    
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid status: {status}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    print("✅ PredictionController loaded")