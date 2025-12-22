"""
YOLO Waste Classifier

Implementacija IWasteClassifier interface-a koristeći YOLOv8.
"""

import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from ultralytics import YOLO

from .waste_classifier import IWasteClassifier
from ..Domain import WasteCategory


class YoloWasteClassifier(IWasteClassifier):
    """
    YOLOv8 classifier za waste classification.
    
    Koristi Ultralytics YOLO za klasifikaciju.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Args:
            model_path: Putanja do .pt modela (opciono)
        """
        self.model: Optional[YOLO] = None
        self.model_path: Optional[str] = None
        self.classes = [cat.value for cat in WasteCategory]
        
        # NE učitavaj model u __init__ jer smo u async kontekstu
        # Model će biti učitan eksplicitno u main.py startup-u
        self._model_path_to_load = model_path
    
    async def predict(self, image_path: str) -> Dict:
        """
        Klasifikuj sliku koristeći YOLO.
        
        Args:
            image_path: Putanja do slike
        
        Returns:
            Dict sa predikcijom
        
        Raises:
            RuntimeError: Ako model nije učitan
        """
        if not self.is_loaded():
            raise RuntimeError("Model nije učitan! Pozovi load_model() prvo.")
        
        # Provjeri da li slika postoji
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Slika ne postoji: {image_path}")
        
        # Timing
        start_time = time.time()
        
        # YOLO predikcija
        results = self.model(image_path, verbose=False)[0]
        probs = results.probs
        
        # Inference time
        inference_time_ms = (time.time() - start_time) * 1000
        
        # Top 1 prediction
        top1_idx = int(probs.top1)
        top1_conf = float(probs.top1conf)
        top1_class = self.classes[top1_idx]
        
        # Top 3 predictions
        top5_indices = probs.top5[:3]  # Uzmi top 3
        top3_predictions = []
        
        for idx in top5_indices:
            idx = int(idx)
            conf = float(probs.data[idx])
            cls = self.classes[idx]
            top3_predictions.append((cls, conf))
        
        result = {
            "class": top1_class,
            "confidence": top1_conf,
            "top3": top3_predictions,
            "inference_time_ms": inference_time_ms,
        }
        
        print(f"🔮 YOLO Prediction: {top1_class} ({top1_conf:.2%}) - {inference_time_ms:.1f}ms")
        
        return result
    
    async def load_model(self, model_path: str) -> bool:
        """
        Učitaj YOLO model.
        
        Args:
            model_path: Putanja do .pt fajla
        
        Returns:
            bool: True ako uspješno
        
        Raises:
            FileNotFoundError: Ako model ne postoji
        """
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model ne postoji: {model_path}")
        
        print(f"📥 Loading YOLO model: {model_path}")
        
        # Učitaj model
        self.model = YOLO(str(model_path))
        self.model_path = str(model_path)
        
        print(f"✅ YOLO model loaded successfully")
        
        return True
    
    async def get_model_info(self) -> Dict:
        """
        Informacije o modelu.
        
        Returns:
            Dict sa info-ima
        """
        if not self.is_loaded():
            return {
                "model_type": "yolov8n",
                "version": "unknown",
                "loaded": False,
                "classes": self.classes,
                "num_classes": len(self.classes),
            }
        
        return {
            "model_type": "yolov8n-cls",  # Classification model
            "version": Path(self.model_path).stem,
            "loaded": True,
            "model_path": self.model_path,
            "classes": self.classes,
            "num_classes": len(self.classes),
        }
    
    def is_loaded(self) -> bool:
        """
        Da li je model učitan.
        
        Returns:
            bool: True ako je spreman
        """
        return self.model is not None


# ========================================
# TRAINER INTERFACE (za retraining)
# ========================================

class IModelTrainer:
    """
    Interface za model training/retraining.
    
    Koristi ga TrainingService.
    """
    
    async def prepare_incremental_dataset(self) -> str:
        """
        Pripremi dataset sa SAMO novim podacima.
        
        Returns:
            str: Putanja do dataseta
        """
        pass
    
    async def prepare_full_dataset(self) -> str:
        """
        Pripremi kompletan dataset (stari + novi).
        
        Returns:
            str: Putanja do dataseta
        """
        pass
    
    async def train(
        self,
        dataset_path: str,
        epochs: int,
        mode: str  # "incremental" ili "full"
    ) -> Dict:
        """
        Treniraj model.
        
        Args:
            dataset_path: Putanja do dataseta
            epochs: Broj epoha
            mode: Training mode
        
        Returns:
            Dict: {
                "model_path": "models/v2/weights/best.pt",
                "accuracy": 0.87,
                "top5_accuracy": 0.96,
                "loss": 0.23,
                "training_time_seconds": 342.5
            }
        """
        pass


class YoloTrainer(IModelTrainer):
    """
    YOLO trainer implementacija.
    
    Koristi continuous_learning.py logiku iz starog projekta.
    """
    
    def __init__(self, base_model_path: str):
        self.base_model_path = base_model_path
    
    async def prepare_incremental_dataset(self) -> str:
        """Pripremi samo nove podatke"""
        # TODO: Implementiraj (kopiraj iz continuous_learning.py)
        return "data/incremental_train"
    
    async def prepare_full_dataset(self) -> str:
        """Pripremi sve podatke"""
        # TODO: Implementiraj
        return "data/processed"
    
    async def train(self, dataset_path: str, epochs: int, mode: str) -> Dict:
        """Pokreni YOLO training"""
        # TODO: Implementiraj (kopiraj iz continuous_learning.py)
        print(f"🔥 Training YOLO model: {mode} mode, {epochs} epochs")
        print(f"   Dataset: {dataset_path}")
        
        # Mock results
        return {
            "model_path": "models/trashvision_v2/weights/best.pt",
            "accuracy": 0.87,
            "top5_accuracy": 0.96,
            "loss": 0.23,
            "training_time_seconds": 300.0
        }


if __name__ == "__main__":
    import asyncio
    
    async def test():
        # Test classifier
        classifier = YoloWasteClassifier()
        
        # Load model (ako postoji)
        model_path = "models/trashvision_v1/weights/best.pt"
        if Path(model_path).exists():
            await classifier.load_model(model_path)
            print(f"✅ Model loaded: {model_path}")
            
            # Info
            info = await classifier.get_model_info()
            print(f"   Classes: {info['num_classes']}")
        else:
            print(f"⚠️  Model not found: {model_path}")
    
    asyncio.run(test())