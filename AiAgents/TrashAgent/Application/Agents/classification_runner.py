"""
Classification Agent Runner

GLAVNI AGENT - Klasifikuje slike kroz Sense→Think→Act ciklus.

Ovo je KLJUČNA klasa koja pokazuje agent arhitekturu!
"""

from typing import Optional
from dataclasses import dataclass
from datetime import datetime

from AiAgents.Core import SoftwareAgent
from ...Domain import (
    WasteImage,
    ClassificationDecision,
    ImageStatus,
    SystemSettings,
)
from ..Services import QueueService, ClassificationService


@dataclass
class ClassificationResult:
    """
    Rezultat jednog agent tick-a.
    
    Ovo se vraća Web sloju za real-time updates.
    """
    image_id: int
    filename: str
    predicted_category: str
    confidence: float
    new_status: str
    needs_review: bool
    processed_at: datetime
    
    # Top 3 predictions
    top3: list[tuple[str, float]]


class ClassificationAgentRunner(SoftwareAgent[WasteImage, ClassificationDecision, ClassificationResult]):
    """
    Classification Agent - Sense→Think→Act
    
    Lifecycle:
    1. SENSE: Preuzmi sljedeću sliku iz queue-a (status=QUEUED)
    2. THINK: Klasifikuj sliku + primijeni policy (confidence threshold)
    3. ACT: Sačuvaj predikciju + ažuriraj status
    
    Runner NE SADRŽI:
    - Web logiku (routing, DTO mapping)
    - Infrastructure logiku (direktne DB/ML calls)
    - Hardkodovane threshold-e (čita iz SystemSettings)
    """
    
    def __init__(
        self,
        queue_service: QueueService,
        classification_service: ClassificationService,
        settings: SystemSettings,
        name: str = "ClassificationAgent"
    ):
        super().__init__(name=name)
        
        self._queue = queue_service
        self._classifier = classification_service
        self._settings = settings
        
        self._model_version = "v1"  # TODO: Učitaj iz active ModelVersion
        
        # Context holder - držimo trenutnu sliku kroz cijeli ciklus
        self._current_image: Optional[WasteImage] = None
    
    # ========================================
    # SENSE - Opazi okolinu
    # ========================================
    
    async def sense(self) -> Optional[WasteImage]:
        """
        SENSE: Preuzmi sljedeću sliku iz queue-a.
        
        Returns:
            Optional[WasteImage]: Slika ako ima posla, None ako nema
        
        Logika:
        - Query: SELECT * FROM images WHERE status='queued' ORDER BY uploaded_at LIMIT 1
        - Postavlja status na PROCESSING (lock)
        """
        # Provjeri da li ima queued slika
        queue_size = await self._queue.get_queue_size(ImageStatus.QUEUED)
        
        if queue_size == 0:
            # Nema posla - agent miruje
            return None
        
        # Dequeue sljedeću sliku
        image = await self._queue.dequeue_next(ImageStatus.QUEUED)
        
        if image is None:
            return None
        
        # ČUVAJ u kontekstu za think/act faze
        self._current_image = image
        
        # Označi kao PROCESSING (agent lock)
        image.mark_as_processing()
        await self._queue.update_status(image.id, ImageStatus.PROCESSING)
        
        print(f"👁️  SENSE: Image {image.id} ({image.filename})")
        
        return image
    
    # ========================================
    # THINK - Donesi odluku
    # ========================================
    
    async def think(self, image: WasteImage) -> ClassificationDecision:
        """
        THINK: Klasifikuj sliku i donesi odluku.
        
        Args:
            image: Slika iz SENSE faze
        
        Returns:
            ClassificationDecision: Odluka šta raditi
        
        Logika:
        1. Klasifikuj sliku preko ML modela
        2. Primijeni policy:
           - confidence >= 70% → status = CLASSIFIED
           - confidence < 70%  → status = PENDING_REVIEW
        3. Vrati odluku
        """
        # Klasifikuj sliku
        decision = await self._classifier.classify_image(image, self._settings)
        
        print(f"🧠 THINK: {decision.predicted_category.value} ({decision.confidence:.0%})")
        print(f"         Status: {decision.new_status.value}")
        
        return decision
    
    # ========================================
    # ACT - Izvrši akciju
    # ========================================
    
    async def act(self, action: ClassificationDecision) -> ClassificationResult:
        """
        ACT: Sačuvaj predikciju i ažuriraj status.
        
        Args:
            action: Odluka iz THINK faze
        
        Returns:
            ClassificationResult: Rezultat za Web sloj
        
        Logika:
        1. Sačuvaj Prediction u DB
        2. Ažuriraj Image status (CLASSIFIED ili PENDING_REVIEW)
        3. Vrati rezultat
        """
        # Koristi sliku iz context-a
        if not self._current_image:
            raise RuntimeError("No current image in context!")
        
        image = self._current_image
        
        # 1. Sačuvaj predikciju
        prediction = await self._classifier.save_prediction(
            image=image,
            decision=action,
            model_version=self._model_version,
            inference_time_ms=0.0
        )
        
        # 2. Ažuriraj status slike
        await self._queue.update_status(
            image_id=image.id,
            new_status=action.new_status,
            processed_at=datetime.now()
        )
        
        # 3. Kreiraj rezultat za Web sloj
        result = ClassificationResult(
            image_id=image.id,
            filename=image.filename,
            predicted_category=action.predicted_category.value,
            confidence=action.confidence,
            new_status=action.new_status.value,
            needs_review=action.needs_review,
            processed_at=datetime.now(),
            top3=[(cat.value, conf) for cat, conf in action.top3_predictions]
        )
        
        print(f"✅ ACT: Image {image.id} processed")
        print(f"       Result: {result.predicted_category} ({result.confidence:.0%})")
        
        # Clear context
        self._current_image = None
        
        return result
    
    # ========================================
    # HELPER METHODS
    # ========================================
    
    async def get_queue_stats(self) -> dict:
        """Statistika queue-a (za monitoring)"""
        return {
            "queued": await self._queue.get_queue_size(ImageStatus.QUEUED),
            "processing": await self._queue.get_queue_size(ImageStatus.PROCESSING),
            "classified": await self._queue.get_queue_size(ImageStatus.CLASSIFIED),
            "pending_review": await self._queue.get_queue_size(ImageStatus.PENDING_REVIEW),
        }


if __name__ == "__main__":
    print("✅ ClassificationAgentRunner loaded")
    print("   - Sense: Dequeue sliku iz queue-a")
    print("   - Think: Klasifikuj + policy")
    print("   - Act: Sačuvaj predikciju + ažuriraj status")