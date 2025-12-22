"""
Policy Interface

Interface za sve policy klase (pravila odlučivanja).
Policy odlučuje šta agent treba da uradi na osnovu percepta.

Primjeri:
- ThresholdPolicy: Odlučuje na osnovu pragova (confidence > 0.7)
- RuleBasedPolicy: Skup if-else pravila
- MLPolicy: Koristi ML model za odluku
- HybridPolicy: Kombinacija više pristupa
"""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

TPercept = TypeVar('TPercept')
TAction = TypeVar('TAction')


class IPolicy(ABC, Generic[TPercept, TAction]):
    """
    Interface za policy (pravila odlučivanja).
    
    Policy prima percept i vraća akciju koju agent treba izvršiti.
    """
    
    @abstractmethod
    async def decide(self, percept: TPercept) -> TAction:
        """
        Donesi odluku na osnovu percepta.
        
        Args:
            percept: Opažanje iz okoline
        
        Returns:
            TAction: Odluka šta uraditi
        
        Primjer:
            - Image classification policy:
              percept = (image, prediction)
              action = (status, save_location, confidence_check)
            
            - Spam filtering policy:
              percept = (message, spam_score)
              action = (Allow/Block/Review, folder)
        
        VAŽNO:
        - Policy NE smije menjati stanje sistema (readonly)
        - Policy NE smije pozivati bazu/API (samo logika)
        - Policy treba biti deterministički (isti input = isti output)
        """
        pass
    
    @abstractmethod
    def get_rules(self) -> dict:
        """
        Vrati trenutna pravila (za monitoring/debug).
        
        Returns:
            dict: Rječnik sa pravilima
        
        Primjer:
            {
                "confidence_threshold": 0.7,
                "review_threshold": 0.5,
                "max_retries": 3
            }
        """
        pass


class NoOpPolicy(IPolicy[str, str]):
    """Test policy koji uvijek vraća istu akciju"""
    
    async def decide(self, percept: str) -> str:
        return "noop"
    
    def get_rules(self) -> dict:
        return {"type": "noop"}


if __name__ == "__main__":
    print("✅ IPolicy interface loaded successfully")