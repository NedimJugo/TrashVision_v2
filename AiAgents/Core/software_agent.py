"""
Software Agent Base Class

Generička bazna klasa za sve software agente.
Implementira osnovni Sense → Think → Act ciklus.

Type parametri:
- TPercept: Tip percepta (šta agent opaža)
- TAction: Tip akcije (šta agent odluči)
- TResult: Tip rezultata (šta agent postigne)
"""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Optional
import asyncio

# Generic type parametri
TPercept = TypeVar('TPercept')
TAction = TypeVar('TAction')
TResult = TypeVar('TResult')


class SoftwareAgent(ABC, Generic[TPercept, TAction, TResult]):
    """
    Bazna klasa za sve software agente.
    
    Svaki agent mora implementirati:
    - sense(): Opažanje okoline
    - think(): Donošenje odluke
    - act(): Izvršavanje akcije
    
    Opciono:
    - learn(): Učenje iz iskustva (implementiraju samo learning agenti)
    """
    
    def __init__(self, name: str = "Agent"):
        self.name = name
        self._is_running = False
    
    # ========================================
    # APSTRAKTNE METODE (svaki agent mora implementirati)
    # ========================================
    
    @abstractmethod
    async def sense(self) -> Optional[TPercept]:
        """
        SENSE: Opazi okolinu i vrati percept.
        
        Returns:
            Optional[TPercept]: Percept ako ima posla, None ako nema
        
        Primjer:
            - Image classification agent: vrati sljedeću sliku iz queue-a
            - Spam agent: vrati sljedeću poruku za obradu
            - Trading agent: vrati trenutno stanje marketa
        """
        pass
    
    @abstractmethod
    async def think(self, percept: TPercept) -> TAction:
        """
        THINK: Na osnovu percepta donesi odluku.
        
        Args:
            percept: Opažanje iz okoline
        
        Returns:
            TAction: Odluka šta uraditi
        
        Primjer:
            - Image agent: klasifikuj sliku + odluči o statusu
            - Spam agent: izračunaj spam score + odluči (Allow/Block/Review)
            - Trading agent: donesi buy/sell/hold odluku
        """
        pass
    
    @abstractmethod
    async def act(self, action: TAction) -> TResult:
        """
        ACT: Izvrši akciju i vrati rezultat.
        
        Args:
            action: Odluka iz think() faze
        
        Returns:
            TResult: Rezultat izvršene akcije
        
        Primjer:
            - Image agent: sačuvaj predikciju + ažuriraj status
            - Spam agent: pomjeri poruku u inbox/spam folder
            - Trading agent: izvrši trade na berzi
        """
        pass
    
    # ========================================
    # AGENT TICK (jedan korak agenta)
    # ========================================
    
    async def step_async(self) -> Optional[TResult]:
        """
        Jedan tick/step agentskog ciklusa.
        
        Redoslijed:
        1. SENSE: Opazi okolinu
        2. THINK: Donesi odluku
        3. ACT: Izvrši akciju
        
        Returns:
            Optional[TResult]: Rezultat ako je agent radio, None ako nema posla
        
        VAŽNO:
        - Ako sense() vrati None (nema posla), vraća None odmah
        - Tick mora biti kratak i atomaran (ne radi "sve odjednom")
        - Tick mora biti idempotentan (može se ponoviti bez štete)
        """
        # SENSE
        percept = await self.sense()
        
        if percept is None:
            # Nema posla - izlaz bez štete
            return None
        
        # THINK
        action = await self.think(percept)
        
        # ACT
        result = await self.act(action)
        
        return result
    
    # ========================================
    # HELPER METODE
    # ========================================
    
    def start(self):
        """Označi da je agent pokrenut (za monitoring)"""
        self._is_running = True
    
    def stop(self):
        """Zaustavi agenta (za graceful shutdown)"""
        self._is_running = False
    
    @property
    def is_running(self) -> bool:
        """Da li agent trenutno radi"""
        return self._is_running
    
    def __repr__(self):
        return f"<{self.__class__.__name__} name='{self.name}' running={self.is_running}>"


class LearningAgent(SoftwareAgent[TPercept, TAction, TResult]):
    """
    Proširenje SoftwareAgent klase za agente koji uče.
    
    Dodaje:
    - learn(): Učenje iz iskustva
    """
    
    @abstractmethod
    async def learn(self, result: TResult) -> None:
        """
        LEARN: Ažuriraj znanje na osnovu rezultata.
        
        Args:
            result: Rezultat prethodne akcije
        
        Primjer:
            - Image agent: ažuriraj counters za retraining
            - Spam agent: ažuriraj model ako ima dovoljno gold labels
            - Trading agent: ažuriraj strategy parametre
        """
        pass
    
    async def step_async(self) -> Optional[TResult]:
        """
        Tick sa učenjem: Sense → Think → Act → Learn
        """
        # SENSE + THINK + ACT (isto kao bazna klasa)
        result = await super().step_async()
        
        # LEARN (samo ako je agent nešto uradio)
        if result is not None:
            await self.learn(result)
        
        return result


# ========================================
# UTILITY: No-op agent za testiranje
# ========================================

class NoOpAgent(SoftwareAgent[str, str, str]):
    """Test agent koji ne radi ništa (za debug)"""
    
    async def sense(self) -> Optional[str]:
        return None  # Uvijek nema posla
    
    async def think(self, percept: str) -> str:
        return "noop"
    
    async def act(self, action: str) -> str:
        return "done"


if __name__ == "__main__":
    # Test: Provjeri da klasa radi
    print("✅ SoftwareAgent base class loaded successfully")
    print(f"   - SoftwareAgent")
    print(f"   - LearningAgent")
    print(f"   - NoOpAgent (test)")