"""
Error Cost Matrix

Definira trošak za svaku grešku u klasifikaciji.
Npr: Metal u papir je skuplji trošak nego papir u metal (kontaminacija).
"""

from typing import Dict, Tuple
from .enums import WasteCategory
import numpy as np


class ErrorCostMatrix:
    """
    Matrica troškova za pogrešnu klasifikaciju.
    
    cost[actual][predicted] = trošak kada je stvarna kategorija 'actual',
                              a model kaže 'predicted'
    
    Principi:
    1. Tačna klasifikacija = 0 trošak
    2. Teška kontaminacija (metal u papir) = visok trošak
    3. Blaga kontaminacija (papir u karton) = nizak trošak
    4. Reciklabilni u trash = srednji trošak
    5. Opasan otpad (baterije) u pogrešnu kategoriju = VEOMA VISOK trošak
    """
    
    def __init__(self):
        """Inicijalizuj matricu troškova"""
        self.categories = [
            WasteCategory.BATTERY,
            WasteCategory.BIOLOGICAL,
            WasteCategory.CARDBOARD,
            WasteCategory.CLOTHES,
            WasteCategory.GLASS,
            WasteCategory.METAL,
            WasteCategory.PAPER,
            WasteCategory.PLASTIC,
            WasteCategory.SHOES,
            WasteCategory.TRASH,
        ]
        
        self.category_to_idx = {cat: i for i, cat in enumerate(self.categories)}
        self._build_cost_matrix()
    
    def _build_cost_matrix(self):
        """
        Napravi matricu troškova.
        
        Vrijednosti su relativne težine (baseline = 1.0):
        - 0.0 = tačna klasifikacija
        - 0.5 = blaga greška (slične kategorije)
        - 1.0 = standardna greška
        - 2.0 = teža greška (kontaminacija)
        - 5.0 = kritična greška (opasan otpad)
        """
        n = len(self.categories)
        self.cost_matrix = np.ones((n, n))
        
        # Dijagonala = 0 (tačna predikcija nema trošak)
        np.fill_diagonal(self.cost_matrix, 0.0)
        
        # Definiši specifične troškove za kritične greške
        self._set_critical_costs()
        self._set_contamination_costs()
        self._set_similar_category_costs()
    
    def _set_critical_costs(self):
        """Kritični troškovi - opasan otpad"""
        bat_idx = self.category_to_idx[WasteCategory.BATTERY]
        
        # Baterije u bilo šta (osim trash) = VEOMA OPASNO
        for cat in self.categories:
            if cat != WasteCategory.BATTERY and cat != WasteCategory.TRASH:
                other_idx = self.category_to_idx[cat]
                self.cost_matrix[bat_idx][other_idx] = 5.0  # KRITIČNO
        
        # Baterije u trash = OK (3x niži trošak)
        trash_idx = self.category_to_idx[WasteCategory.TRASH]
        self.cost_matrix[bat_idx][trash_idx] = 1.5
    
    def _set_contamination_costs(self):
        """Kontaminacija - metal/staklo u papir/karton = skupo"""
        
        # Metal ili glass u paper/cardboard = VISOK TROŠAK (kontaminacija)
        metal_idx = self.category_to_idx[WasteCategory.METAL]
        glass_idx = self.category_to_idx[WasteCategory.GLASS]
        paper_idx = self.category_to_idx[WasteCategory.PAPER]
        cardboard_idx = self.category_to_idx[WasteCategory.CARDBOARD]
        
        self.cost_matrix[metal_idx][paper_idx] = 3.0
        self.cost_matrix[metal_idx][cardboard_idx] = 3.0
        self.cost_matrix[glass_idx][paper_idx] = 3.0
        self.cost_matrix[glass_idx][cardboard_idx] = 3.0
        
        # Papir u metal/glass = manji trošak (lakše se ukloni)
        self.cost_matrix[paper_idx][metal_idx] = 1.0
        self.cost_matrix[paper_idx][glass_idx] = 1.0
        self.cost_matrix[cardboard_idx][metal_idx] = 1.0
        self.cost_matrix[cardboard_idx][glass_idx] = 1.0
        
        # Plastika u metal = problematično
        plastic_idx = self.category_to_idx[WasteCategory.PLASTIC]
        self.cost_matrix[plastic_idx][metal_idx] = 2.5
        
        # Metal u plastiku = malo manji problem
        self.cost_matrix[metal_idx][plastic_idx] = 2.0
    
    def _set_similar_category_costs(self):
        """Slične kategorije - nizak trošak"""
        
        # Paper i Cardboard su slični
        paper_idx = self.category_to_idx[WasteCategory.PAPER]
        cardboard_idx = self.category_to_idx[WasteCategory.CARDBOARD]
        self.cost_matrix[paper_idx][cardboard_idx] = 0.3
        self.cost_matrix[cardboard_idx][paper_idx] = 0.3
        
        # Clothes i Shoes su slični
        clothes_idx = self.category_to_idx[WasteCategory.CLOTHES]
        shoes_idx = self.category_to_idx[WasteCategory.SHOES]
        self.cost_matrix[clothes_idx][shoes_idx] = 0.4
        self.cost_matrix[shoes_idx][clothes_idx] = 0.4
        
        # Biological u trash = ok
        bio_idx = self.category_to_idx[WasteCategory.BIOLOGICAL]
        trash_idx = self.category_to_idx[WasteCategory.TRASH]
        self.cost_matrix[bio_idx][trash_idx] = 0.5
        self.cost_matrix[trash_idx][bio_idx] = 0.8  # Trash u bio = malo gori
    
    def get_cost(
        self,
        true_category: WasteCategory,
        predicted_category: WasteCategory
    ) -> float:
        """
        Dobij trošak greške.
        
        Args:
            true_category: Stvarna kategorija
            predicted_category: Predviđena kategorija
        
        Returns:
            float: Trošak greške (0 = tačno, 5 = kritično)
        """
        true_idx = self.category_to_idx[true_category]
        pred_idx = self.category_to_idx[predicted_category]
        return float(self.cost_matrix[true_idx][pred_idx])
    
    def get_expected_cost(
        self,
        predicted_category: WasteCategory,
        probability_distribution: Dict[WasteCategory, float]
    ) -> float:
        """
        Očekivani trošak za datu predikciju.
        
        Args:
            predicted_category: Šta želimo da predvidimo
            probability_distribution: Vjerovatnoće za svaku kategoriju
        
        Returns:
            float: Očekivani trošak = sum(P(true_cat) * cost(true_cat, pred_cat))
        
        Primjer:
            Ako model kaže 80% metal, 15% plastic, 5% glass
            a mi odlučimo da klasifikujemo kao METAL:
            
            expected_cost = 0.80 * cost(metal→metal) +
                           0.15 * cost(plastic→metal) +
                           0.05 * cost(glass→metal)
        """
        expected_cost = 0.0
        pred_idx = self.category_to_idx[predicted_category]
        
        for true_category, probability in probability_distribution.items():
            true_idx = self.category_to_idx[true_category]
            cost = self.cost_matrix[true_idx][pred_idx]
            expected_cost += probability * cost
        
        return expected_cost
    
    def get_all_expected_costs(
        self,
        probability_distribution: Dict[WasteCategory, float]
    ) -> Dict[WasteCategory, float]:
        """
        Očekivani trošak za SVE moguće odluke.
        
        Args:
            probability_distribution: Model outputs (probabilities)
        
        Returns:
            Dict: {category: expected_cost} za svaku kategoriju
        
        Ovo koristimo u decision optimizer-u da nađemo najbolju odluku.
        """
        costs = {}
        
        for predicted_category in self.categories:
            costs[predicted_category] = self.get_expected_cost(
                predicted_category,
                probability_distribution
            )
        
        return costs
    
    def print_matrix(self):
        """Printaj matricu troškova (za debugging)"""
        print("\n🎯 ERROR COST MATRIX")
        print("=" * 80)
        print("Rows = TRUE category, Cols = PREDICTED category")
        print()
        
        # Header
        header = "TRUE \\ PRED".ljust(15)
        for cat in self.categories:
            header += cat.value[:6].ljust(8)
        print(header)
        print("-" * 80)
        
        # Rows
        for i, true_cat in enumerate(self.categories):
            row = true_cat.value.ljust(15)
            for j, pred_cat in enumerate(self.categories):
                cost = self.cost_matrix[i][j]
                row += f"{cost:.1f}".ljust(8)
            print(row)
        
        print("=" * 80)
        print("Legend: 0.0=correct, 0.5=minor, 1.0=normal, 2-3=high, 5.0=CRITICAL")
        print()


# Global instance
error_cost_matrix = ErrorCostMatrix()


if __name__ == "__main__":
    # Test
    matrix = ErrorCostMatrix()
    matrix.print_matrix()
    
    # Primjer
    print("\n📊 PRIMJERI:")
    print(f"Metal → Paper: {matrix.get_cost(WasteCategory.METAL, WasteCategory.PAPER):.1f} (SKUPO - kontaminacija)")
    print(f"Paper → Metal: {matrix.get_cost(WasteCategory.PAPER, WasteCategory.METAL):.1f} (OK - lakše ukloniti)")
    print(f"Battery → Plastic: {matrix.get_cost(WasteCategory.BATTERY, WasteCategory.PLASTIC):.1f} (KRITIČNO!)")
    print(f"Paper → Cardboard: {matrix.get_cost(WasteCategory.PAPER, WasteCategory.CARDBOARD):.1f} (Blago)")
