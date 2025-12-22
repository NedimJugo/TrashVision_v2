"""
Classification Worker

Background worker koji pokreće Classification Agent u loop-u.

⭐ OVO JE KLJUČNA KLASA - pokazuje kako agent radi u background-u!
"""

import asyncio
from datetime import datetime
from typing import Optional

from ...Application.Agents import ClassificationAgentRunner, ClassificationResult


class ClassificationWorker:
    """
    Background worker za Classification Agent.
    
    Pokreće agent tick() svakih N sekundi.
    
    VAŽNO:
    - Worker JE TANAK - samo poziva runner.step_async()
    - Worker NE SADRŽI biznis logiku
    - Worker je odgovoran samo za loop + delay + error handling
    """
    
    def __init__(
        self,
        runner: ClassificationAgentRunner,
        tick_interval_seconds: int = 2,
        name: str = "ClassificationWorker"
    ):
        """
        Args:
            runner: Agent runner
            tick_interval_seconds: Koliko često ticka agent (sekunde)
            name: Ime worker-a
        """
        self.runner = runner
        self.tick_interval = tick_interval_seconds
        self.name = name
        
        # State
        self._is_running = False
        self._tick_count = 0
        self._last_tick_at: Optional[datetime] = None
        self._task: Optional[asyncio.Task] = None
        
        # Callbacks (za real-time events)
        self._result_callback = None
    
    # ========================================
    # WORKER LIFECYCLE
    # ========================================
    
    def start(self):
        """
        Pokreni worker (non-blocking).
        """
        if self._is_running:
            print(f"⚠️  {self.name} already running")
            return
        
        self._is_running = True
        self.runner.start()
        
        # Pokreni async task
        self._task = asyncio.create_task(self._run_loop())
        
        print(f"✅ {self.name} started (tick every {self.tick_interval}s)")
    
    def stop(self):
        """
        Zaustavi worker (graceful shutdown).
        """
        if not self._is_running:
            return
        
        self._is_running = False
        self.runner.stop()
        
        if self._task:
            self._task.cancel()
        
        print(f"🛑 {self.name} stopped")
    
    @property
    def is_running(self) -> bool:
        """Da li worker radi"""
        return self._is_running
    
    @property
    def tick_count(self) -> int:
        """Broj tick-ova do sada"""
        return self._tick_count
    
    @property
    def last_tick_at(self) -> Optional[datetime]:
        """Vrijeme zadnjeg tick-a"""
        return self._last_tick_at
    
    # ========================================
    # AGENT LOOP (privatna metoda)
    # ========================================
    
    async def _run_loop(self):
        """
        Glavni agent loop.
        
        while running:
            1. Pozovi runner.step_async()
            2. Ako ima rezultat → emit event
            3. Delay
            4. Repeat
        """
        print(f"🔄 {self.name} loop started")
        
        while self._is_running:
            try:
                # ⭐ AGENT TICK - pozovi runner
                result = await self.runner.step_async()
                
                self._tick_count += 1
                self._last_tick_at = datetime.now()
                
                if result:
                    # Agent procesirao sliku!
                    print(f"✅ Agent tick #{self._tick_count}: Image {result.image_id} classified")
                    
                    # Emit event (ako ima callback)
                    if self._result_callback:
                        await self._result_callback(result)
                else:
                    # Nema posla - agent miruje
                    # (ne logiramo da ne spamujemo console)
                    pass
                
            except Exception as e:
                # Error handling - loguj ali nastavi
                print(f"❌ {self.name} error in tick #{self._tick_count}: {e}")
                import traceback
                traceback.print_exc()
            
            # Delay prije sljedećeg tick-a
            await asyncio.sleep(self.tick_interval)
        
        print(f"🛑 {self.name} loop stopped")
    
    # ========================================
    # EVENT CALLBACKS (za real-time updates)
    # ========================================
    
    def on_result(self, callback):
        """
        Registruj callback za rezultate.
        
        Args:
            callback: async funkcija koja prima ClassificationResult
        
        Usage:
            worker.on_result(emit_to_websocket)
        """
        self._result_callback = callback
    
    # ========================================
    # STATISTICS
    # ========================================
    
    async def get_stats(self) -> dict:
        """Statistika worker-a"""
        queue_stats = await self.runner.get_queue_stats()
        
        return {
            "name": self.name,
            "is_running": self.is_running,
            "tick_count": self.tick_count,
            "last_tick_at": self.last_tick_at.isoformat() if self.last_tick_at else None,
            "tick_interval_seconds": self.tick_interval,
            "queue": queue_stats,
        }


if __name__ == "__main__":
    print("✅ ClassificationWorker loaded")
    print("   - Pokreće Classification Agent u background loop-u")
    print("   - Tick interval: konfigurabilan")
    print("   - Graceful start/stop")