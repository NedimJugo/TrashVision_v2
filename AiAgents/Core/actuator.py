"""
Actuator Interface

Interface za sve actuatore (izvršioce akcija).
Actuator prima akciju i mijenja stanje sistema.

Primjeri:
- DatabaseActuator: Piše u bazu
- FileSystemActuator: Kreira/briše fajlove
- ApiActuator: Šalje HTTP requests
- EmailActuator: Šalje emailove
"""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

TAction = TypeVar('TAction')
TResult = TypeVar('TResult')


class IActuator(ABC, Generic[TAction, TResult]):
    """
    Interface za actuatore (izvršioce akcija).
    
    Actuator prima akciju i izvršava je, vraćajući rezultat.
    """
    
    @abstractmethod
    async def execute(self, action: TAction) -> TResult:
        """
        Izvrši akciju i vrati rezultat.
        
        Args:
            action: Akcija koju treba izvršiti
        
        Returns:
            TResult: Rezultat izvršene akcije
        
        Primjer:
            - Database actuator:
              action = SavePrediction(image_id=1, class="plastic", ...)
              result = PredictionSaved(success=True, id=42)
            
            - Email actuator:
              action = SendEmail(to="user@example.com", subject="...")
              result = EmailSent(success=True, message_id="...")
        
        VAŽNO:
        - Actuator JE jedino mjesto gdje se mijenja stanje sistema
        - Treba biti idempotentan ako se može (safe za retry)
        - Treba logirati sve akcije (za audit)
        """
        pass
    
    @abstractmethod
    async def can_execute(self, action: TAction) -> bool:
        """
        Provjeri da li se akcija može izvršiti (pre-check).
        
        Args:
            action: Akcija koju želimo izvršiti
        
        Returns:
            bool: True ako se može izvršiti, False ako ne
        
        Primjer:
            - Provjeri da li postoji resursi (disk space, memory)
            - Provjeri da li je akcija validna (consistent sa DB)
            - Provjeri da li korisnik ima permisije
        """
        pass


class NoOpActuator(IActuator[str, str]):
    """Test actuator koji ne radi ništa"""
    
    async def execute(self, action: str) -> str:
        return f"executed: {action}"
    
    async def can_execute(self, action: str) -> bool:
        return True


if __name__ == "__main__":
    print("✅ IActuator interface loaded successfully")