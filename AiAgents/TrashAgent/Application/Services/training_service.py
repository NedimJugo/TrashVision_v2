"""
Training Service

Servis za treniranje i retreniranje ML modela.
Learning agent koristi ovo u LEARN fazi.
"""

from typing import Optional
from datetime import datetime
from ...Domain import (
    ModelVersion,
    SystemSettings,
    TrainingMode,
    ModelType,
)


class TrainingService:
    """
    Servis za model training/retraining.
    
    Omogućava:
    - Retreniranje modela (incremental ili full)
    - Kreiranje nove verzije modela
    - Aktivaciju nove verzije
    """
    
    def __init__(self, trainer, db_session):
        """
        Args:
            trainer: ML trainer (IModelTrainer interface)
            db_session: SQLAlchemy session
        """
        self.trainer = trainer
        self.db = db_session
    
    async def retrain_model(
        self,
        mode: TrainingMode,
        epochs: int,
        settings: SystemSettings
    ) -> ModelVersion:
        """
        Retreniranje modela.
        
        Args:
            mode: INCREMENTAL (brzo) ili FULL (sporo, ali preciznije)
            epochs: Broj epoha
            settings: System settings
        
        Returns:
            ModelVersion: Nova verzija modela
        
        LEARN faza (može trajati dugo!):
        1. Pripremi dataset
        2. Treniraj model
        3. Kreiraj novu verziju
        4. Aktiviraj novu verziju
        5. Reset counter
        """
        print(f"🔥 Starting retraining: {mode.value} mode, {epochs} epochs")
        
        # 1. Pripremi dataset
        if mode == TrainingMode.INCREMENTAL:
            dataset_path = await self.trainer.prepare_incremental_dataset()
            training_samples = await self._count_new_samples()
        else:
            dataset_path = await self.trainer.prepare_full_dataset()
            training_samples = await self._count_all_samples()
        
        print(f"📦 Dataset prepared: {training_samples} samples")
        
        # 2. Treniraj model
        start_time = datetime.now()
        
        training_results = await self.trainer.train(
            dataset_path=dataset_path,
            epochs=epochs,
            mode=mode
        )
        
        duration = (datetime.now() - start_time).total_seconds()
        print(f"✅ Training completed in {duration:.1f}s")
        
        # 3. Kreiraj ModelVersion
        current_version = await self._get_latest_version_number()
        new_version = ModelVersion(
            version_number=current_version + 1,
            model_path=training_results["model_path"],
            model_type=ModelType.YOLOV8_NANO,
            training_mode=mode,
            epochs=epochs,
            training_samples_count=training_samples,
            accuracy=training_results.get("accuracy", 0.0),
            top5_accuracy=training_results.get("top5_accuracy", 0.0),
            loss=training_results.get("loss", 0.0),
            is_active=False,  # Još nije aktivna
            created_at=datetime.now(),
        )
        
        # Save ModelVersion
        print(f"💾 Created model version: v{new_version.version_number}")
        
        # 4. Aktiviraj novu verziju
        await self.activate_version(new_version)
        
        # 5. Reset counters
        settings.reset_samples_counter()
        
        return new_version
    
    async def activate_version(self, version: ModelVersion) -> bool:
        """
        Aktiviraj verziju modela.
        
        Args:
            version: Verzija za aktivaciju
        
        Returns:
            bool: True ako uspješno
        """
        # 1. Deaktiviraj sve ostale
        # UPDATE model_versions SET is_active = FALSE
        
        # 2. Aktiviraj novu
        version.activate()
        # UPDATE model_versions SET is_active = TRUE WHERE id = ?
        
        # 3. Ažuriraj settings
        # UPDATE system_settings SET active_model_version = ?
        
        print(f"✅ Activated model version: v{version.version_number}")
        return True
    
    async def get_active_version(self) -> Optional[ModelVersion]:
        """
        Trenutno aktivna verzija modela.
        
        Returns:
            Optional[ModelVersion]: Aktivna verzija ili None
        """
        # SELECT * FROM model_versions WHERE is_active = TRUE LIMIT 1
        return None
    
    async def get_all_versions(self) -> list[ModelVersion]:
        """
        Sve verzije modela (za history).
        
        Returns:
            list[ModelVersion]: Lista verzija
        """
        # SELECT * FROM model_versions ORDER BY version_number DESC
        return []
    
    async def _get_latest_version_number(self) -> int:
        """Helper: Broj zadnje verzije"""
        # SELECT MAX(version_number) FROM model_versions
        return 0
    
    async def _count_new_samples(self) -> int:
        """Helper: Broj novih uzoraka"""
        # Count files in data/new_samples/
        return 100
    
    async def _count_all_samples(self) -> int:
        """Helper: Broj svih uzoraka"""
        # Count files in data/processed/train/
        return 1000
    
    async def should_retrain(self, settings: SystemSettings) -> bool:
        """
        Da li je vrijeme za retraining.
        
        Args:
            settings: System settings
        
        Returns:
            bool: True ako treba retrenirati
        """
        return settings.should_trigger_retraining()


if __name__ == "__main__":
    print("✅ TrainingService loaded")