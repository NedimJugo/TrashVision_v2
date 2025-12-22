"""
Waste Classifier Interface

Interface za ML classifier.
Omogućava zamjenu različitih ML implementacija (YOLO, ResNet, etc.)
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple


class IWasteClassifier(ABC):
    """
    Interface za waste classification modele.
    
    Omogućava:
    - Predikciju (predict)
    - Training/retraining
    - Model loading
    """
    
    @abstractmethod
    async def predict(self, image_path: str) -> Dict:
        """
        Klasifikuj sliku.
        
        Args:
            image_path: Putanja do slike
        
        Returns:
            Dict: {
                "class": "plastic",
                "confidence": 0.92,
                "top3": [("plastic", 0.92), ("metal", 0.05), ("glass", 0.02)],
                "inference_time_ms": 45.2
            }
        """
        pass
    
    @abstractmethod
    async def load_model(self, model_path: str) -> bool:
        """
        Učitaj model sa diska.
        
        Args:
            model_path: Putanja do .pt fajla
        
        Returns:
            bool: True ako uspješno
        """
        pass
    
    @abstractmethod
    async def get_model_info(self) -> Dict:
        """
        Informacije o trenutno učitanom modelu.
        
        Returns:
            Dict: {
                "model_type": "yolov8n",
                "version": "v1",
                "loaded": True,
                "classes": ["battery", "biological", ...],
                "num_classes": 10
            }
        """
        pass
    
    @abstractmethod
    def is_loaded(self) -> bool:
        """
        Da li je model učitan.
        
        Returns:
            bool: True ako je model spreman za predikciju
        """
        pass


if __name__ == "__main__":
    print("✅ IWasteClassifier interface loaded")