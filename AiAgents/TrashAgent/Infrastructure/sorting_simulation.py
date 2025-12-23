"""
Sorting Simulation Module

Simulira robotsku ruku i transportnu traku za sortiranje otpada.
Prati troškove greške i efikasnost sortiranja.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime
from enum import Enum
import time
import random
from ..Domain.enums import WasteCategory
from ..Domain.error_costs import ErrorCostMatrix


class RobotState(str, Enum):
    """Status robotske ruke"""
    IDLE = "idle"
    SCANNING = "scanning"
    PICKING = "picking"
    MOVING = "moving"
    DROPPING = "dropping"
    ERROR = "error"


class BinType(str, Enum):
    """Tipovi kontejnera"""
    BATTERY = "battery"
    BIOLOGICAL = "biological"
    CARDBOARD = "cardboard"
    CLOTHES = "clothes"
    GLASS = "glass"
    METAL = "metal"
    PAPER = "paper"
    PLASTIC = "plastic"
    SHOES = "shoes"
    TRASH = "trash"
    UNCERTAIN = "uncertain"  # Za items sa niskim confidence


@dataclass
class WasteItem:
    """Jedan komad otpada na traci"""
    id: int
    true_category: WasteCategory
    position: float  # Pozicija na traci (0.0 - 100.0)
    size: str = "medium"  # small, medium, large
    weight_kg: float = 0.5
    
    # Predikcija
    predicted_category: Optional[WasteCategory] = None
    confidence: float = 0.0
    scan_time: Optional[datetime] = None
    
    # Sortiranje
    sorted_into_bin: Optional[BinType] = None
    sorted_at: Optional[datetime] = None
    is_correctly_sorted: bool = False
    sorting_cost: float = 0.0


@dataclass
class SortingBin:
    """Kontejner za sortiranje"""
    bin_type: BinType
    capacity_kg: float = 50.0
    current_weight_kg: float = 0.0
    items: List[WasteItem] = field(default_factory=list)
    contamination_count: int = 0  # Broj pogrešno sortiranih
    
    @property
    def is_full(self) -> bool:
        return self.current_weight_kg >= self.capacity_kg
    
    @property
    def fill_percentage(self) -> float:
        return (self.current_weight_kg / self.capacity_kg) * 100
    
    def add_item(self, item: WasteItem):
        """Dodaj item u kontejner"""
        self.items.append(item)
        self.current_weight_kg += item.weight_kg
        
        # Provjeri kontaminaciju
        if item.true_category.value != self.bin_type.value:
            self.contamination_count += 1


@dataclass
class ConveyorBelt:
    """Transportna traka"""
    length_m: float = 10.0
    speed_m_per_sec: float = 0.2  # 20cm/s
    items: List[WasteItem] = field(default_factory=list)
    items_processed: int = 0
    
    def add_item(self, item: WasteItem):
        """Dodaj novi item na početak trake"""
        item.position = 0.0
        self.items.append(item)
    
    def update(self, delta_time: float):
        """Pomjeri traku naprijed"""
        distance = self.speed_m_per_sec * delta_time
        
        # Pomjeri sve items
        for item in self.items:
            item.position += distance
        
        # Ukloni items koji su pali sa kraja
        self.items = [
            item for item in self.items
            if item.position < self.length_m
        ]
    
    def get_items_in_scan_zone(self, scan_zone_start: float = 3.0, scan_zone_end: float = 4.0) -> List[WasteItem]:
        """Vrati items u scan zoni"""
        return [
            item for item in self.items
            if scan_zone_start <= item.position <= scan_zone_end
            and item.predicted_category is None
        ]
    
    def get_items_in_pickup_zone(self, pickup_zone_start: float = 5.0, pickup_zone_end: float = 6.0) -> List[WasteItem]:
        """Vrati items u pickup zoni"""
        return [
            item for item in self.items
            if pickup_zone_start <= item.position <= pickup_zone_end
            and item.predicted_category is not None
            and item.sorted_into_bin is None
        ]


@dataclass
class RoboticArm:
    """Robotska ruka za sortiranje"""
    state: RobotState = RobotState.IDLE
    current_item: Optional[WasteItem] = None
    target_bin: Optional[BinType] = None
    
    # Performance metrics
    picks_per_minute: int = 15
    scan_time_sec: float = 0.3
    pick_time_sec: float = 0.5
    move_time_sec: float = 1.0
    drop_time_sec: float = 0.3
    
    action_start_time: Optional[float] = None
    
    def can_perform_action(self, current_time: float) -> bool:
        """Da li je završila trenutna akcija"""
        if self.action_start_time is None:
            return True
        
        elapsed = current_time - self.action_start_time
        
        if self.state == RobotState.SCANNING:
            return elapsed >= self.scan_time_sec
        elif self.state == RobotState.PICKING:
            return elapsed >= self.pick_time_sec
        elif self.state == RobotState.MOVING:
            return elapsed >= self.move_time_sec
        elif self.state == RobotState.DROPPING:
            return elapsed >= self.drop_time_sec
        
        return True


class SortingSimulation:
    """
    Kompletna simulacija sortiranja.
    
    Komponente:
    - Conveyor belt (transportna traka)
    - Robotic arm (robotska ruka)
    - Sorting bins (kontejneri)
    - Classifier (ML model za predikciju)
    """
    
    def __init__(
        self,
        classifier,
        belt_speed: float = 0.2,
        num_bins: int = 11,  # 10 kategorija + uncertain
        enable_uncertain_bin: bool = True
    ):
        """
        Args:
            classifier: ML classifier
            belt_speed: Brzina trake (m/s)
            num_bins: Broj kontejnera
            enable_uncertain_bin: Da li imati "uncertain" kontejner
        """
        self.classifier = classifier
        self.cost_matrix = ErrorCostMatrix()
        
        # Komponente
        self.belt = ConveyorBelt(speed_m_per_sec=belt_speed)
        self.robot = RoboticArm()
        self.bins = self._create_bins(enable_uncertain_bin)
        
        # Statistika
        self.total_items_processed = 0
        self.total_cost = 0.0
        self.correct_sorts = 0
        self.incorrect_sorts = 0
        self.uncertain_sorts = 0
        
        self.simulation_time = 0.0
        self.is_running = False
    
    def _create_bins(self, enable_uncertain: bool) -> Dict[BinType, SortingBin]:
        """Kreiraj kontejnere"""
        bins = {}
        
        for bin_type in BinType:
            if bin_type == BinType.UNCERTAIN and not enable_uncertain:
                continue
            
            bins[bin_type] = SortingBin(
                bin_type=bin_type,
                capacity_kg=50.0
            )
        
        return bins
    
    def add_waste_item(
        self,
        true_category: WasteCategory,
        size: str = "medium",
        weight_kg: float = 0.5
    ):
        """Dodaj novi item na traku"""
        item = WasteItem(
            id=self.total_items_processed + len(self.belt.items) + 1,
            true_category=true_category,
            position=0.0,
            size=size,
            weight_kg=weight_kg
        )
        
        self.belt.add_item(item)
        print(f"➕ Added {true_category.value} to belt (ID: {item.id})")
    
    async def step(self, delta_time: float = 0.1):
        """
        Jedan korak simulacije.
        
        Args:
            delta_time: Vrijeme koraka (sekunde)
        """
        self.simulation_time += delta_time
        
        # 1. Pomjeri traku
        self.belt.update(delta_time)
        
        # 2. Robot akcije
        await self._robot_step()
    
    async def _robot_step(self):
        """Korak robotske ruke"""
        current_time = self.simulation_time
        
        # Provjeri da li je završila trenutna akcija
        if not self.robot.can_perform_action(current_time):
            return
        
        # State machine
        if self.robot.state == RobotState.IDLE:
            # Potraži item za skeniranje
            scan_items = self.belt.get_items_in_scan_zone()
            if scan_items:
                await self._start_scanning(scan_items[0])
        
        elif self.robot.state == RobotState.SCANNING:
            # Završeno skeniranje
            await self._finish_scanning()
        
        elif self.robot.state == RobotState.PICKING:
            # Završeno picking
            self._finish_picking()
        
        elif self.robot.state == RobotState.MOVING:
            # Završeno moving
            self._finish_moving()
        
        elif self.robot.state == RobotState.DROPPING:
            # Završeno dropping
            self._finish_dropping()
    
    async def _start_scanning(self, item: WasteItem):
        """Započni skeniranje item-a"""
        self.robot.state = RobotState.SCANNING
        self.robot.current_item = item
        self.robot.action_start_time = self.simulation_time
        
        print(f"🔍 Scanning item {item.id} ({item.true_category.value})...")
    
    async def _finish_scanning(self):
        """Završi skeniranje - klasifikuj item"""
        item = self.robot.current_item
        
        # Simuliraj ML predikciju (mock - u pravoj verziji bi zvao classifier)
        # Za sada, dodaj noise
        if random.random() < 0.85:  # 85% tačnost
            predicted = item.true_category
            confidence = random.uniform(0.75, 0.98)
        else:
            # Pogrešna predikcija
            other_categories = [cat for cat in WasteCategory if cat != item.true_category]
            predicted = random.choice(other_categories)
            confidence = random.uniform(0.50, 0.85)
        
        item.predicted_category = predicted
        item.confidence = confidence
        item.scan_time = datetime.now()
        
        print(f"   ✅ Scanned: {predicted.value} ({confidence:.1%} conf)")
        
        # Prijeđi u picking
        pickup_items = self.belt.get_items_in_pickup_zone()
        if item in pickup_items:
            self._start_picking(item)
        else:
            self.robot.state = RobotState.IDLE
            self.robot.current_item = None
    
    def _start_picking(self, item: WasteItem):
        """Započni picking"""
        self.robot.state = RobotState.PICKING
        self.robot.current_item = item
        self.robot.action_start_time = self.simulation_time
        
        # Odluči u koji bin
        if item.confidence < 0.60:
            target_bin = BinType.UNCERTAIN
        else:
            target_bin = BinType(item.predicted_category.value)
        
        self.robot.target_bin = target_bin
        
        print(f"🤖 Picking item {item.id} → {target_bin.value} bin")
    
    def _finish_picking(self):
        """Završi picking - pomjeri ruku"""
        self.robot.state = RobotState.MOVING
        self.robot.action_start_time = self.simulation_time
    
    def _finish_moving(self):
        """Završi moving - dropuj item"""
        self.robot.state = RobotState.DROPPING
        self.robot.action_start_time = self.simulation_time
    
    def _finish_dropping(self):
        """Završi dropping - sortiranje gotovo"""
        item = self.robot.current_item
        target_bin = self.robot.target_bin
        
        # Dodaj u kontejner
        item.sorted_into_bin = target_bin
        item.sorted_at = datetime.now()
        
        bin_obj = self.bins[target_bin]
        bin_obj.add_item(item)
        
        # Ukloni sa trake
        if item in self.belt.items:
            self.belt.items.remove(item)
        
        # Izračunaj trošak
        if target_bin == BinType.UNCERTAIN:
            # Uncertain ide na manual review - baseline trošak
            cost = 0.5
            is_correct = False
            self.uncertain_sorts += 1
        else:
            # Provjeri tačnost
            is_correct = (item.true_category.value == target_bin.value)
            
            if is_correct:
                cost = 0.0
                self.correct_sorts += 1
            else:
                cost = self.cost_matrix.get_cost(
                    item.true_category,
                    item.predicted_category
                )
                self.incorrect_sorts += 1
        
        item.is_correctly_sorted = is_correct
        item.sorting_cost = cost
        
        self.total_items_processed += 1
        self.total_cost += cost
        
        # Status
        status_icon = "✅" if is_correct else "❌"
        print(f"   {status_icon} Sorted into {target_bin.value} (cost: {cost:.2f})")
        
        # Reset robot
        self.robot.state = RobotState.IDLE
        self.robot.current_item = None
        self.robot.target_bin = None
    
    def get_statistics(self) -> Dict:
        """Vrati statistiku simulacije"""
        total = self.total_items_processed
        
        if total == 0:
            accuracy = 0.0
            avg_cost = 0.0
        else:
            accuracy = (self.correct_sorts / total) * 100
            avg_cost = self.total_cost / total
        
        return {
            "simulation_time_sec": self.simulation_time,
            "total_processed": total,
            "correct_sorts": self.correct_sorts,
            "incorrect_sorts": self.incorrect_sorts,
            "uncertain_sorts": self.uncertain_sorts,
            "accuracy_percent": accuracy,
            "total_cost": self.total_cost,
            "average_cost_per_item": avg_cost,
            "items_on_belt": len(self.belt.items),
            "robot_state": self.robot.state.value,
            "bins": {
                bin_type.value: {
                    "items": len(bin_obj.items),
                    "weight_kg": bin_obj.current_weight_kg,
                    "fill_percent": bin_obj.fill_percentage,
                    "contamination_count": bin_obj.contamination_count
                }
                for bin_type, bin_obj in self.bins.items()
            }
        }
    
    def print_statistics(self):
        """Printaj statistiku"""
        stats = self.get_statistics()
        
        print("\n" + "="*70)
        print("📊 SIMULATION STATISTICS")
        print("="*70)
        print(f"⏱️  Time: {stats['simulation_time_sec']:.1f}s")
        print(f"📦 Total Processed: {stats['total_processed']}")
        print(f"✅ Correct: {stats['correct_sorts']}")
        print(f"❌ Incorrect: {stats['incorrect_sorts']}")
        print(f"❓ Uncertain: {stats['uncertain_sorts']}")
        print(f"🎯 Accuracy: {stats['accuracy_percent']:.1f}%")
        print(f"💰 Total Cost: {stats['total_cost']:.2f}")
        print(f"💵 Avg Cost/Item: {stats['average_cost_per_item']:.2f}")
        print(f"🤖 Robot State: {stats['robot_state']}")
        print(f"🚚 Items on Belt: {stats['items_on_belt']}")
        
        print("\n🗑️  BINS:")
        for bin_name, bin_stats in stats['bins'].items():
            if bin_stats['items'] > 0:
                contamination = bin_stats['contamination_count']
                contam_str = f" (⚠️ {contamination} contaminated!)" if contamination > 0 else ""
                print(f"   {bin_name}: {bin_stats['items']} items, "
                      f"{bin_stats['weight_kg']:.1f}kg ({bin_stats['fill_percent']:.0f}%)"
                      f"{contam_str}")
        
        print("="*70 + "\n")


if __name__ == "__main__":
    print("✅ SortingSimulation module loaded")
    print("💡 Example usage:")
    print("   sim = SortingSimulation(classifier)")
    print("   sim.add_waste_item(WasteCategory.PLASTIC)")
    print("   await sim.step()")
