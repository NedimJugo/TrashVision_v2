"""
Classification Service

Servis za klasifikaciju slika i donošenje odluka.
Agent koristi ovo u THINK i ACT fazama.

NOVO: Koristi DecisionOptimizer za cost-aware decisions!
"""

from typing import Optional
from datetime import datetime
from ...Domain import (
    WasteImage,
    Prediction,
    WasteCategory,
    ImageStatus,
    ClassificationDecision,
    SystemSettings,
)
from ...Domain.decision_optimizer import DecisionOptimizer, OptimizedDecision


class ClassificationService:
    """
    Servis za klasifikaciju slika.
    
    Omogućava:
    - Klasifikaciju slike preko ML modela
    - Donošenje odluke (da li ide na review)
    - Čuvanje predikcije u DB
    - Cost-aware decision optimization
    """
    
    def __init__(self, classifier, db_session, use_optimizer: bool = True):
        """
        Args:
            classifier: ML classifier (IWasteClassifier interface)
            db_session: SQLAlchemy session
            use_optimizer: Da li koristiti decision optimizer (default: True)
        """
        self.classifier = classifier
        self.db = db_session
        self.use_optimizer = use_optimizer
        
        # Kreiraj decision optimizer
        if use_optimizer:
            self.optimizer = DecisionOptimizer(
                min_confidence_threshold=0.70,
                review_threshold=0.50,
                max_acceptable_cost=1.5,
                cost_weight=0.3
            )
    
    async def classify_image(
        self,
        image: WasteImage,
        settings: SystemSettings
    ) -> ClassificationDecision:
        """
        Klasifikuj sliku i donesi odluku.
        
        Args:
            image: Slika za klasifikaciju
            settings: System settings (za threshold-e)
        
        Returns:
            ClassificationDecision: Odluka šta raditi
        
        THINK faza:
        1. Klasifikuj sliku preko ML modela
        2. Primijeni policy (confidence threshold + cost optimization)
        3. Vrati odluku
        """
        # 1. KLASIFIKACIJA
        prediction_result = await self.classifier.predict(image.filepath)
        
        # 2. OPTIMIZACIJA (ako je uključena)
        if self.use_optimizer:
            optimized = self.optimizer.optimize_decision(prediction_result)
            
            # Konvertuj u ClassificationDecision
            decision = ClassificationDecision(
                predicted_category=optimized.predicted_category,
                confidence=optimized.confidence,
                new_status=optimized.status,
                top3_predictions=optimized.top3_predictions
            )
            
            print(f"🎯 OPTIMIZED Classification: {optimized.predicted_category.value} ({optimized.confidence:.2%})")
            print(f"   Expected Cost: {optimized.expected_cost:.2f}")
            print(f"   Status: {optimized.status.value}")
            print(f"   Reasoning: {optimized.reasoning}")
            
            if optimized.is_fallback:
                print(f"   ⚠️ Fallback: {optimized.fallback_reason}")
            
            if optimized.original_top1 != optimized.predicted_category:
                print(f"   🔄 Changed: {optimized.original_top1.value} → {optimized.predicted_category.value}")
        
        else:
            # 2. LEGACY POLICY - Jednostavno odluči o statusu
            confidence = prediction_result["confidence"]
            predicted_category = WasteCategory(prediction_result["class"])
            
            # Pravilo: Ako confidence < 70% → ide na review
            if confidence < settings.min_confidence_threshold:
                new_status = ImageStatus.PENDING_REVIEW
            else:
                new_status = ImageStatus.CLASSIFIED
            
            # 3. Top 3 predictions
            top3 = prediction_result.get("top3", [])
            top3_predictions = [
                (WasteCategory(cls), conf)
                for cls, conf in top3
            ]
            
            # 4. Kreiraj odluku
            decision = ClassificationDecision(
                predicted_category=predicted_category,
                confidence=confidence,
                new_status=new_status,
                top3_predictions=top3_predictions
            )
            
            print(f"🔮 Classification: {predicted_category.value} ({confidence:.2%})")
            print(f"   Status: {new_status.value}")
        
        return decision
    
    async def save_prediction(
        self,
        image: WasteImage,
        decision: ClassificationDecision,
        model_version: str,
        inference_time_ms: float = 0.0
    ) -> Prediction:
        """
        Sačuvaj predikciju u bazu.
        
        Args:
            image: Slika
            decision: Odluka agenta
            model_version: Verzija modela
            inference_time_ms: Vrijeme izvršavanja (ms)
        
        Returns:
            Prediction: Sačuvana predikcija
        
        ACT faza:
        - Persist prediction u DB
        """
        prediction = Prediction(
            image_id=image.id,
            predicted_category=decision.predicted_category,
            confidence=decision.confidence,
            model_version=model_version,
            inference_time_ms=inference_time_ms,
            created_at=datetime.now()
        )
        
        # Ako ima top3
        if len(decision.top3_predictions) >= 2:
            prediction.top2_category = decision.top3_predictions[1][0]
            prediction.top2_confidence = decision.top3_predictions[1][1]
        
        if len(decision.top3_predictions) >= 3:
            prediction.top3_category = decision.top3_predictions[2][0]
            prediction.top3_confidence = decision.top3_predictions[2][1]
        
        # Save to DB (mock za sada)
        print(f"💾 Saved prediction: {prediction.predicted_category.value}")
        
        return prediction
    
    async def get_prediction_for_image(self, image_id: int) -> Optional[Prediction]:
        """
        Vrati predikciju za sliku.
        
        Args:
            image_id: ID slike
        
        Returns:
            Optional[Prediction]: Predikcija ili None
        """
        # SELECT * FROM predictions WHERE image_id = ? ORDER BY created_at DESC LIMIT 1
        return None
    
    async def get_statistics(self) -> dict:
        """
        Statistika klasifikacija.
        
        Returns:
            dict: Statistike (total, per category, avg confidence...)
        """
        return {
            "total_predictions": 0,
            "avg_confidence": 0.0,
            "high_confidence_count": 0,  # >70%
            "review_needed_count": 0,    # <70%
        }


if __name__ == "__main__":
    print("✅ ClassificationService loaded")