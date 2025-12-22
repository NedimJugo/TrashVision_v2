"""
Learning Agent Runner

LEARNING AGENT - Prati nove uzorke i odlučuje kada retrenirati model.

Sense→Think→Act→Learn ciklus.
"""

from typing import Optional
from dataclasses import dataclass
from datetime import datetime

from AiAgents.Core import LearningAgent
from ...Domain import (
    SystemSettings,
    TrainingDecision,
    TrainingMode,
    ModelVersion,
)
from ..Services import TrainingService


@dataclass
class LearningResult:
    """
    Rezultat jednog learning agent tick-a.
    """
    checked_at: datetime
    new_samples_count: int
    threshold: int
    should_retrain: bool
    retrain_triggered: bool
    mode: Optional[str] = None
    new_model_version: Optional[int] = None


class LearningAgentRunner(LearningAgent[SystemSettings, TrainingDecision, LearningResult]):
    """
    Learning Agent - Sense→Think→Act→Learn
    
    Lifecycle:
    1. SENSE: Pročitaj SystemSettings (counters, thresholds)
    2. THINK: Odluči da li treba retraining (primijeni pravila)
    3. ACT: Ako treba, pokreni retraining
    4. LEARN: Reset counters, log metrike
    
    Runner NE SADRŽI:
    - Hardkodovane threshold-e (čita iz SystemSettings)
    - Direktne ML calls (koristi TrainingService)
    - Web logiku
    """
    
    def __init__(
        self,
        training_service: TrainingService,
        settings: SystemSettings,
        name: str = "LearningAgent"
    ):
        super().__init__(name=name)
        
        self._trainer = training_service
        self._settings = settings
    
    # ========================================
    # SENSE - Opazi okolinu
    # ========================================
    
    async def sense(self) -> Optional[SystemSettings]:
        """
        SENSE: Pročitaj system settings.
        
        Returns:
            Optional[SystemSettings]: Settings ako treba provjeriti, None ako ne
        
        Logika:
        - Uvijek vrati settings (treba provjeriti counters)
        - Možemo dodati cache da ne query-ujemo svaki put
        """
        # U production sistemu: SELECT * FROM system_settings LIMIT 1
        # Za sada vraćamo self._settings
        
        if not self._settings.auto_retrain_enabled:
            # Auto retraining isključen
            return None
        
        print(f"👁️  SENSE: New samples: {self._settings.new_samples_count}/{self._settings.retrain_threshold}")
        
        return self._settings
    
    # ========================================
    # THINK - Donesi odluku
    # ========================================
    
    async def think(self, settings: SystemSettings) -> TrainingDecision:
        """
        THINK: Odluči da li treba retraining.
        
        Args:
            settings: System settings iz SENSE faze
        
        Returns:
            TrainingDecision: Odluka da li i kako retrenirati
        
        Logika (PRAVILA):
        1. Ako new_samples >= threshold → treba retraining
        2. Ako new_samples < 500 → incremental mode (brzo)
        3. Ako new_samples >= 500 → full mode (preciznije)
        """
        should_retrain = settings.should_trigger_retraining()
        
        # Odluči o modu
        if settings.new_samples_count < 500:
            mode = TrainingMode.INCREMENTAL
            reason = f"Incremental (< 500 samples)"
        else:
            mode = TrainingMode.FULL
            reason = f"Full (>= 500 samples)"
        
        decision = TrainingDecision(
            should_retrain=should_retrain,
            new_samples_count=settings.new_samples_count,
            threshold=settings.retrain_threshold,
            mode=mode.value,
            reason=reason if should_retrain else "Threshold not reached"
        )
        
        if should_retrain:
            print(f"🧠 THINK: RETRAINING NEEDED!")
            print(f"         Mode: {mode.value}")
            print(f"         Samples: {settings.new_samples_count}")
        else:
            print(f"🧠 THINK: No retraining needed ({decision.progress_percentage:.1f}% to threshold)")
        
        return decision
    
    # ========================================
    # ACT - Izvrši akciju
    # ========================================
    
    async def act(self, action: TrainingDecision) -> LearningResult:
        """
        ACT: Pokreni retraining ako je potrebno.
        
        Args:
            action: Odluka iz THINK faze
        
        Returns:
            LearningResult: Rezultat za Web sloj
        
        Logika:
        1. Ako ne treba retraining → early exit
        2. Pokreni retraining (može trajati dugo!)
        3. Vrati rezultat
        
        VAŽNO:
        - Ovo može trajati 5-60 minuta!
        - U production sistemu ovo treba biti async job (Celery, background task)
        - Agent će biti "blokiran" tokom treniranja
        """
        result = LearningResult(
            checked_at=datetime.now(),
            new_samples_count=action.new_samples_count,
            threshold=action.threshold,
            should_retrain=action.should_retrain,
            retrain_triggered=False,
        )
        
        if not action.should_retrain:
            # Nema retraining-a
            print(f"✅ ACT: No action needed")
            return result
        
        # RETRAINING!
        print(f"🔥 ACT: Starting retraining ({action.mode} mode)...")
        
        mode = TrainingMode(action.mode)
        epochs = (
            self._settings.incremental_epochs if mode == TrainingMode.INCREMENTAL
            else self._settings.full_epochs
        )
        
        # Pokreni retraining (DUGO TRAJE!)
        new_version = await self._trainer.retrain_model(
            mode=mode,
            epochs=epochs,
            settings=self._settings
        )
        
        result.retrain_triggered = True
        result.mode = action.mode
        result.new_model_version = new_version.version_number
        
        print(f"✅ ACT: Retraining completed - v{new_version.version_number}")
        
        return result
    
    # ========================================
    # LEARN - Ažuriraj znanje
    # ========================================
    
    async def learn(self, result: LearningResult) -> None:
        """
        LEARN: Ažuriraj metrike i log.
        
        Args:
            result: Rezultat ACT faze
        
        Logika:
        - Log training results
        - Ažuriraj metrike (accuracy improvement, training time)
        - Eventualno notifikuj admina
        """
        if result.retrain_triggered:
            print(f"📚 LEARN: Model retrained successfully")
            print(f"         New version: v{result.new_model_version}")
            print(f"         Mode: {result.mode}")
            
            # TODO: Log metrics, send notifications, etc.
        else:
            # Ništa za učiti
            pass
    
    # ========================================
    # HELPER METHODS
    # ========================================
    
    async def get_learning_stats(self) -> dict:
        """Statistika learning-a (za monitoring)"""
        return {
            "new_samples_count": self._settings.new_samples_count,
            "threshold": self._settings.retrain_threshold,
            "progress_percentage": (
                self._settings.new_samples_count / self._settings.retrain_threshold * 100
            ),
            "auto_retrain_enabled": self._settings.auto_retrain_enabled,
            "last_retrain_at": self._settings.last_retrain_at.isoformat() if self._settings.last_retrain_at else None,
            "retrain_count": self._settings.retrain_count,
        }


if __name__ == "__main__":
    print("✅ LearningAgentRunner loaded")
    print("   - Sense: Pročitaj SystemSettings")
    print("   - Think: Odluči da li treba retraining")
    print("   - Act: Pokreni retraining")
    print("   - Learn: Log metrics + reset counters")