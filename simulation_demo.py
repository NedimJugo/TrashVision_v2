"""
Simulation Demo

Demonstracija sorting simulacije.
Pokreće simulaciju i prikazuje rezultate.
"""

import asyncio
import random
from AiAgents.TrashAgent.Infrastructure.sorting_simulation import (
    SortingSimulation,
    WasteCategory
)
from AiAgents.TrashAgent.Domain.error_costs import ErrorCostMatrix


class MockClassifier:
    """Mock classifier za demo"""
    
    async def predict(self, image_path: str):
        """Mock predikcija"""
        # Ovo bi zvalo pravi ML model
        return {
            "class": "plastic",
            "confidence": 0.85,
            "top3": [("plastic", 0.85), ("metal", 0.10), ("glass", 0.03)]
        }


async def run_simulation_demo():
    """
    Demonstracija sorting simulacije.
    
    Simulira:
    - 50 waste items
    - Različite kategorije
    - Praćenje troškova
    """
    print("🚀 Starting Sorting Simulation Demo")
    print("=" * 70)
    
    # 1. Kreiraj simulator
    classifier = MockClassifier()
    sim = SortingSimulation(
        classifier=classifier,
        belt_speed=0.3,  # 30cm/s
        enable_uncertain_bin=True
    )
    
    # 2. Prikaži error cost matrix
    print("\n💰 ERROR COST MATRIX:")
    sim.cost_matrix.print_matrix()
    
    # 3. Dodaj random waste items
    print("\n📦 Adding waste items to conveyor belt...")
    
    categories = list(WasteCategory)
    num_items = 50
    
    for i in range(num_items):
        category = random.choice(categories)
        weight = random.uniform(0.2, 2.0)
        sim.add_waste_item(
            true_category=category,
            size=random.choice(["small", "medium", "large"]),
            weight_kg=weight
        )
        
        # Dodaj pauzu da traka ne bude prepuna
        if i < 10:
            await sim.step(delta_time=0.5)
    
    print(f"✅ Added {num_items} items to belt\n")
    
    # 4. Pokreni simulaciju
    print("🤖 Running simulation...")
    print("-" * 70)
    
    # Simuliraj 2 minuta
    max_time = 120.0  # 2 minute
    step_time = 0.1   # 100ms
    
    steps = 0
    max_steps = int(max_time / step_time)
    
    while steps < max_steps:
        await sim.step(delta_time=step_time)
        steps += 1
        
        # Prikaži progress svakih 10 sekundi
        if steps % 100 == 0:
            elapsed = sim.simulation_time
            processed = sim.total_items_processed
            on_belt = len(sim.belt.items)
            print(f"⏱️  {elapsed:.0f}s | Processed: {processed}/{num_items} | On belt: {on_belt}")
        
        # Ako nema više items, završi
        if sim.total_items_processed >= num_items and len(sim.belt.items) == 0:
            break
    
    print("-" * 70)
    
    # 5. Prikaži rezultate
    print("\n✅ Simulation completed!")
    sim.print_statistics()
    
    # 6. Analiza kontaminacije
    print("\n⚠️  CONTAMINATION ANALYSIS:")
    for bin_type, bin_obj in sim.bins.items():
        if bin_obj.contamination_count > 0:
            print(f"   {bin_type.value}: {bin_obj.contamination_count} contaminated items")
            
            # Prikaži top 3 najskuplje greške u ovom bin-u
            costly_items = sorted(
                [item for item in bin_obj.items if not item.is_correctly_sorted],
                key=lambda x: x.sorting_cost,
                reverse=True
            )[:3]
            
            for item in costly_items:
                print(f"      - Item {item.id}: {item.true_category.value} classified as "
                      f"{item.predicted_category.value} (cost: {item.sorting_cost:.2f})")
    
    # 7. Savjeti za poboljšanje
    print("\n💡 IMPROVEMENT SUGGESTIONS:")
    stats = sim.get_statistics()
    
    if stats['accuracy_percent'] < 90:
        print("   - Accuracy is low. Consider:")
        print("     * Retraining model with more data")
        print("     * Using higher confidence threshold")
        print("     * Sending more items to uncertain bin")
    
    if stats['average_cost_per_item'] > 0.5:
        print("   - High average cost. Consider:")
        print("     * Using decision optimizer to minimize expected cost")
        print("     * Implementing fallback strategies")
        print("     * Manual review for uncertain items")
    
    if stats['uncertain_sorts'] > num_items * 0.3:
        print("   - Too many uncertain items. Consider:")
        print("     * Lowering confidence threshold slightly")
        print("     * Improving model accuracy")
    
    print("\n" + "=" * 70)
    print("🎉 Demo completed!")


async def run_cost_comparison():
    """
    Demonstracija razlike između:
    1. Klasične klasifikacije (max probability)
    2. Cost-aware klasifikacije (minimize expected cost)
    """
    print("\n" + "=" * 70)
    print("💰 COST COMPARISON: Classic vs Cost-Aware")
    print("=" * 70)
    
    cost_matrix = ErrorCostMatrix()
    
    # Scenario 1: Metal vs Paper (KRITIČNO!)
    print("\n🧪 SCENARIO 1: Uncertain between METAL and PAPER")
    print("-" * 70)
    print("Model says: 55% metal, 40% paper")
    print()
    
    # Classic approach: uzmi max
    print("Classic approach (max probability):")
    print("   Decision: METAL (55%)")
    print("   Expected cost if true=metal: 0.0")
    print("   Expected cost if true=paper: ???")
    print("   Expected: 0.55*0 + 0.40*cost(paper→metal) = 0.55*0 + 0.40*1.0 = 0.40")
    
    metal_cost = 0.55 * 0 + 0.40 * cost_matrix.get_cost(WasteCategory.PAPER, WasteCategory.METAL)
    print(f"   EXPECTED COST: {metal_cost:.2f}")
    print()
    
    # Cost-aware approach
    print("Cost-aware approach (minimize expected cost):")
    print("   Decision: PAPER (40%)")
    print("   Expected cost if true=paper: 0.0")
    print("   Expected cost if true=metal: cost(metal→paper) = 3.0 (HIGH!)")
    paper_cost = 0.40 * 0 + 0.55 * cost_matrix.get_cost(WasteCategory.METAL, WasteCategory.PAPER)
    print(f"   EXPECTED COST: {paper_cost:.2f}")
    print()
    
    if paper_cost < metal_cost:
        print(f"   ❌ PAPER is worse! Cost {paper_cost:.2f} vs {metal_cost:.2f}")
        print("   In this case, better to classify as METAL or send to UNCERTAIN!")
    else:
        print(f"   ✅ PAPER is better! Cost {paper_cost:.2f} vs {metal_cost:.2f}")
    
    # Scenario 2: Paper vs Cardboard (slično)
    print("\n🧪 SCENARIO 2: Uncertain between PAPER and CARDBOARD")
    print("-" * 70)
    print("Model says: 60% paper, 35% cardboard")
    print()
    
    print("Classic approach: PAPER (60%)")
    paper_cost = 0.60 * 0 + 0.35 * cost_matrix.get_cost(WasteCategory.CARDBOARD, WasteCategory.PAPER)
    print(f"   Expected cost: {paper_cost:.2f}")
    
    print("\nCost-aware approach: CARDBOARD (35%)")
    cardboard_cost = 0.35 * 0 + 0.60 * cost_matrix.get_cost(WasteCategory.PAPER, WasteCategory.CARDBOARD)
    print(f"   Expected cost: {cardboard_cost:.2f}")
    
    if abs(paper_cost - cardboard_cost) < 0.1:
        print("\n   ✅ Costs are similar - both choices are acceptable!")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    print("""
╔═══════════════════════════════════════════════════════════════════╗
║                  TRASHVISION SORTING SIMULATION                   ║
║                                                                   ║
║  Demonstracija robotskog sortiranja otpada sa:                   ║
║  ✓ Transportnom trakom                                           ║
║  ✓ Robotskom rukom                                               ║
║  ✓ Error cost matrix                                             ║
║  ✓ Decision optimization                                         ║
║  ✓ Statistikom i cost tracking-om                                ║
╚═══════════════════════════════════════════════════════════════════╝
    """)
    
    # Pokreni demo
    asyncio.run(run_simulation_demo())
    
    # Pokreni cost comparison
    asyncio.run(run_cost_comparison())
