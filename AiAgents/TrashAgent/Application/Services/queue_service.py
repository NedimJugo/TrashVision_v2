"""
Queue Service - WORKING VERSION

Implementacija sa pravim DB operacijama.
"""

from typing import Optional, List
from datetime import datetime
from sqlalchemy.orm import Session

from ...Domain import WasteImage, ImageStatus
from ...Infrastructure.database import WasteImageModel


class QueueService:
    """
    Servis za queue operacije - SA PRAVIM DB!
    """
    
    def __init__(self, db_session: Session):
        self.db = db_session
    
    async def enqueue(self, image: WasteImage) -> WasteImage:
        """
        Dodaj sliku u queue (SNIMI U DB).
        """
        # Postavi status na QUEUED
        image.status = ImageStatus.QUEUED
        image.uploaded_at = datetime.now()
        
        # Konvertuj u DB model
        db_image = WasteImageModel.from_domain(image)
        
        # Sačuvaj u bazu
        self.db.add(db_image)
        self.db.commit()
        self.db.refresh(db_image)
        
        # Konvertuj nazad u Domain entitet
        saved_image = db_image.to_domain()
        
        print(f"📥 Enqueued image: {saved_image.filename} (ID: {saved_image.id})")
        
        return saved_image
    
    async def dequeue_next(self, status: ImageStatus = ImageStatus.QUEUED) -> Optional[WasteImage]:
        """
        Uzmi sljedeću sliku iz queue-a.
        """
        # Query za sljedeću QUEUED sliku
        db_image = (
            self.db.query(WasteImageModel)
            .filter(WasteImageModel.status == status.value)
            .order_by(WasteImageModel.uploaded_at)
            .first()
        )
        
        if not db_image:
            return None
        
        # Konvertuj u Domain
        image = db_image.to_domain()
        
        return image
    
    async def update_status(
        self,
        image_id: int,
        new_status: ImageStatus,
        processed_at: Optional[datetime] = None
    ) -> bool:
        """
        Ažuriraj status slike.
        """
        db_image = self.db.query(WasteImageModel).filter(WasteImageModel.id == image_id).first()
        
        if not db_image:
            return False
        
        db_image.status = new_status.value
        if processed_at:
            db_image.processed_at = processed_at
        
        self.db.commit()
        
        print(f"🔄 Updated image {image_id} status: {new_status.value}")
        return True
    
    async def get_queue_size(self, status: ImageStatus = ImageStatus.QUEUED) -> int:
        """
        Broj slika u queue-u.
        """
        count = (
            self.db.query(WasteImageModel)
            .filter(WasteImageModel.status == status.value)
            .count()
        )
        return count
    
    async def get_all_by_status(
        self,
        status: ImageStatus,
        limit: int = 100
    ) -> List[WasteImage]:
        """
        Sve slike sa određenim statusom.
        """
        db_images = (
            self.db.query(WasteImageModel)
            .filter(WasteImageModel.status == status.value)
            .order_by(WasteImageModel.uploaded_at.desc())
            .limit(limit)
            .all()
        )
        
        return [img.to_domain() for img in db_images]
    
    async def get_by_id(self, image_id: int) -> Optional[WasteImage]:
        """
        Jedna slika po ID-u.
        """
        db_image = self.db.query(WasteImageModel).filter(WasteImageModel.id == image_id).first()
        
        if not db_image:
            return None
        
        return db_image.to_domain()
    
    async def mark_as_failed(self, image_id: int, error: str) -> bool:
        """
        Označi sliku kao FAILED.
        """
        print(f"❌ Image {image_id} failed: {error}")
        return await self.update_status(image_id, ImageStatus.FAILED)


if __name__ == "__main__":
    print("✅ QueueService (WORKING VERSION) loaded")