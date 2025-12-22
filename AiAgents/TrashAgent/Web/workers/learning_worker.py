"""
Learning Worker

Background worker koji pokreće Learning Agent u loop-u.

Provjerava da li treba retraining svakih N sekundi (obično 60s).
"""

import asyncio
from datetime import datetime
from typing import Optional

from ...Application.Agents import LearningAgentRunner, LearningResult


class LearningWorker:
    """
    Background worker za Learning Agent.
    
    Provjerava retraining threshold i pokreće retraining kada je potrebno.
    """
    
    def __init__(
        self,
        runner: LearningAgentRunner,
        tick_interval_seconds: int = 60,  # Provjeri svaki minut
        name: str = "LearningWorker"
    ):
        """
        Args:
            runner: Learning agent runner
            tick_interval_seconds: Koliko često provjerava (sekunde)
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
        
        # Callbacks
        self._retrain_callback = None
    
    # ========================================
    # WORKER LIFECYCLE
    # ========================================
    
    def start(self):
        """Pokreni worker"""
        if self._is_running:
            print(f"⚠️  {self.name} already running")
            return
        
        self._is_running = True
        self.runner.start()
        
        self._task = asyncio.create_task(self._run_loop())
        
        print(f"✅ {self.name} started (check every {self.tick_interval}s)")
    
    def stop(self):
        """Zaustavi worker"""
        if not self._is_running:
            return
        
        self._is_running = False
        self.runner.stop()
        
        if self._task:
            self._task.cancel()
        
        print(f"🛑 {self.name} stopped")
    
    @property
    def is_running(self) -> bool:
        return self._is_running
    
    @property
    def tick_count(self) -> int:
        return self._tick_count
    
    @property
    def last_tick_at(self) -> Optional[datetime]:
        return self._last_tick_at
    
    # ========================================
    # AGENT LOOP
    # ========================================
    
    async def _run_loop(self):
        """
        Learning agent loop.
        
        Provjerava:
        1. Da li ima dovoljno novih uzoraka
        2. Ako da → pokreni retraining
        3. Delay (obično duži nego classification - 60s)
        """
        print(f"🔄 {self.name} loop started")
        
        while self._is_running:
            try:
                # ⭐ AGENT TICK
                result = await self.runner.step_async()
                
                self._tick_count += 1
                self._last_tick_at = datetime.now()
                
                if result:
                    if result.retrain_triggered:
                        # Retraining pokrenut!
                        print(f"🔥 Learning tick #{self._tick_count}: Retraining completed!")
                        print(f"   New model: v{result.new_model_version}")
                        print(f"   Mode: {result.mode}")
                        
                        # Emit event
                        if self._retrain_callback:
                            await self._retrain_callback(result)
                    else:
                        # Provjera ali nema retraining-a
                        progress = (result.new_samples_count / result.threshold * 100) if result.threshold > 0 else 0
                        print(f"📊 Learning tick #{self._tick_count}: No retraining needed ({progress:.1f}%)")
                else:
                    # Auto-retrain disabled ili nema provjere
                    pass
                
            except Exception as e:
                print(f"❌ {self.name} error in tick #{self._tick_count}: {e}")
                import traceback
                traceback.print_exc()
            
            # Delay (obično duži nego classification)
            await asyncio.sleep(self.tick_interval)
        
        print(f"🛑 {self.name} loop stopped")
    
    # ========================================
    # EVENT CALLBACKS
    # ========================================
    
    def on_retrain(self, callback):
        """
        Registruj callback za retraining event.
        
        Args:
            callback: async funkcija koja prima LearningResult
        """
        self._retrain_callback = callback
    
    # ========================================
    # STATISTICS
    # ========================================
    
    async def get_stats(self) -> dict:
        """Statistika worker-a"""
        learning_stats = await self.runner.get_learning_stats()
        
        return {
            "name": self.name,
            "is_running": self.is_running,
            "tick_count": self.tick_count,
            "last_tick_at": self.last_tick_at.isoformat() if self.last_tick_at else None,
            "tick_interval_seconds": self.tick_interval,
            "learning_stats": learning_stats,
        }


if __name__ == "__main__":
    print("✅ LearningWorker loaded")
    print("   - Pokreće Learning Agent u background loop-u")
    print("   - Provjerava retraining threshold")
    print("   - Automatski retrenira model kada je potrebno")