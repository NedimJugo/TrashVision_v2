"""
Learning Component Interface

Interface za komponente koje implementiraju učenje.
Learning component ažurira znanje agenta na osnovu iskustava.

Primjeri:
- ModelRetrainingComponent: Retrenira ML model
- QTableComponent: Ažurira Q-tabelu (reinforcement learning)
- CounterComponent: Broji uzorke za incremental learning
- MetricsComponent: Prati i analizira performanse
"""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Any

TExperience = TypeVar('TExperience')


class ILearningComponent(ABC, Generic[TExperience]):
    """
    Interface za learning komponente.
    
    Learning component prima iskustvo (experience) i ažurira znanje.
    """
    
    @abstractmethod
    async def learn(self, experience: TExperience) -> None:
        """
        Uči iz iskustva.
        
        Args:
            experience: Iskustvo (obično rezultat akcije + feedback)
        
        Primjer:
            - Model retraining:
              experience = UserFeedback(image_id=1, correct_class="plastic")
              → Dodaj u training set, ako ima 100+ uzoraka → retrain
            
            - Q-learning:
              experience = (state, action, reward, next_state)
              → Ažuriraj Q(s,a) prema Bellman equation
        
        VAŽNO:
        - Learn može biti async (retraining traje dugo)
        - Learn ne smije blokirati agent loop (run in background)
        - Learn treba čuvati stanje (counters, metrics)
        """
        pass
    
    @abstractmethod
    async def should_trigger_update(self) -> bool:
        """
        Provjeri da li je vrijeme za ažuriranje znanja.
        
        Returns:
            bool: True ako treba pokrenuti learning, False inače
        
        Primjer:
            - Model retraining: Da li ima 100+ novih uzoraka?
            - Q-table: Da li je prošlo 1000 koraka?
            - Metrics: Da li je prošao 1 sat od zadnjeg updata?
        """
        pass
    
    @abstractmethod
    async def get_learning_stats(self) -> dict[str, Any]:
        """
        Vrati statistiku učenja (za monitoring).
        
        Returns:
            dict: Statistike (counters, metrics, timestamps)
        
        Primjer:
            {
                "new_samples_count": 45,
                "last_retrain": "2025-12-22T10:30:00",
                "retrain_count": 3,
                "accuracy_improvement": 0.02
            }
        """
        pass


class NoOpLearningComponent(ILearningComponent[str]):
    """Test learning component koji ne uči ništa"""
    
    async def learn(self, experience: str) -> None:
        pass  # No-op
    
    async def should_trigger_update(self) -> bool:
        return False
    
    async def get_learning_stats(self) -> dict[str, Any]:
        return {"type": "noop"}


if __name__ == "__main__":
    print("✅ ILearningComponent interface loaded successfully")