"""
Perception Source Interface

Interface za sve izvore percepcije (opažanja).
Agent koristi perception source da bi "opazio" okolinu.

Primjeri:
- QueuePerceptionSource: čita iz queue-a
- DatabasePerceptionSource: čita iz baze
- FileSystemPerceptionSource: skenira foldere
- ApiPerceptionSource: čita iz external API-ja
"""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Optional

TPercept = TypeVar('TPercept')


class IPerceptionSource(ABC, Generic[TPercept]):
    """
    Interface za izvore percepcije.
    
    Agent poziva has_next() i get_next() da bi opazio okolinu.
    """
    
    @abstractmethod
    async def has_next(self) -> bool:
        """
        Provjeri da li ima sljedećeg percepta.
        
        Returns:
            bool: True ako ima još percepta, False ako nema
        
        Primjer:
            - Queue: Da li ima poruka u queued statusu?
            - File system: Da li ima novih fajlova?
            - Database: Da li ima neprocesiranih redova?
        """
        pass
    
    @abstractmethod
    async def get_next(self) -> Optional[TPercept]:
        """
        Uzmi sljedeći percept.
        
        Returns:
            Optional[TPercept]: Percept ako postoji, None ako nema
        
        VAŽNO:
        - Ova metoda obično "dequeue-uje" ili "lock-uje" percept
        - Pozivaj samo ako has_next() == True
        - Treba biti thread-safe ako ima više agenata
        """
        pass
    
    @abstractmethod
    async def count(self) -> int:
        """
        Broj dostupnih percepta (za monitoring).
        
        Returns:
            int: Broj percepta koji čekaju na obradu
        """
        pass


class NoOpPerceptionSource(IPerceptionSource[str]):
    """Test perception source koji uvijek vraća None"""
    
    async def has_next(self) -> bool:
        return False
    
    async def get_next(self) -> Optional[str]:
        return None
    
    async def count(self) -> int:
        return 0


if __name__ == "__main__":
    print("✅ IPerceptionSource interface loaded successfully")