"""
Decision Optimizer

Optimizira odluke klasifikacije korištenjem:
1. Confidence thresholds
2. Error cost matrix
3. Expected cost minimization
4. Fallback strategies
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from .enums import WasteCategory, ImageStatus
from .error_costs import ErrorCostMatrix


@dataclass
class OptimizedDecision:
    """
    Optimizirana odluka klasifikacije.
    
    Sadrži:
    - Optimizovanu kategoriju (može biti različita od top1!)
    - Confidence i expected cost
    - Razlog odluke
    - Fallback status
    """
    predicted_category: WasteCategory
    confidence: float
    expected_cost: float
    status: ImageStatus
    reasoning: str
    
    # Alternative predictions
    top3_predictions: List[Tuple[WasteCategory, float]]
    all_costs: Dict[WasteCategory, float]
    
    # Da li je primjenjen fallback
    is_fallback: bool = False
    fallback_reason: Optional[str] = None
    
    # Original top1 (prije optimizacije)
    original_top1: WasteCategory = None
    original_confidence: float = 0.0
    
    @property
    def is_confident(self) -> bool:
        """Da li je odluka pouzdana"""
        return self.status == ImageStatus.CLASSIFIED
    
    @property
    def needs_review(self) -> bool:
        """Da li treba human review"""
        return self.status == ImageStatus.PENDING_REVIEW


class DecisionOptimizer:
    """
    Optimizira odluke klasifikacije.
    
    Umjesto da uvijek uzme max(probability), optimizuje korištenjem:
    - Error cost matrix
    - Confidence thresholds
    - Expected cost minimization
    """
    
    def __init__(
        self,
        min_confidence_threshold: float = 0.70,
        review_threshold: float = 0.50,
        max_acceptable_cost: float = 1.0,
        cost_weight: float = 0.3
    ):
        """
        Args:
            min_confidence_threshold: Minimum za auto-klasifikaciju (default 70%)
            review_threshold: Ispod ovoga IDE NA REVIEW (default 50%)
            max_acceptable_cost: Maksimalni prihvatljiv expected cost
            cost_weight: Težina troška u odluci (0-1, default 0.3)
        """
        self.min_confidence_threshold = min_confidence_threshold
        self.review_threshold = review_threshold
        self.max_acceptable_cost = max_acceptable_cost
        self.cost_weight = cost_weight
        
        self.cost_matrix = ErrorCostMatrix()
    
    def optimize_decision(
        self,
        prediction_result: Dict
    ) -> OptimizedDecision:
        """
        Optimizuj odluku klasifikacije.
        
        Args:
            prediction_result: Output od ML modela:
                {
                    "class": "metal",
                    "confidence": 0.85,
                    "top3": [("metal", 0.85), ("plastic", 0.10), ("glass", 0.03)]
                }
        
        Returns:
            OptimizedDecision: Optimizovana odluka
        
        Proces:
        1. Provijeri confidence thresholds
        2. Ako je confidence OK, računaj expected costs
        3. Odluči da li koristiti top1 ili neku alternativu
        4. Primijeni fallback ako je potrebno
        """
        original_class = prediction_result["class"]
        original_confidence = prediction_result["confidence"]
        top3 = prediction_result.get("top3", [])
        
        # Konvertuj u WasteCategory
        original_category = WasteCategory(original_class)
        top3_predictions = [
            (WasteCategory(cls), conf)
            for cls, conf in top3
        ]
        
        # 1. Confidence threshold check
        if original_confidence < self.review_threshold:
            return self._apply_fallback_low_confidence(
                original_category,
                original_confidence,
                top3_predictions
            )
        
        # 2. Napravi probability distribution
        prob_dist = self._build_probability_distribution(top3)
        
        # 3. Izračunaj expected costs za sve kategorije
        all_costs = self.cost_matrix.get_all_expected_costs(prob_dist)
        
        # 4. Nađi najbolju kategoriju (minimizuj expected cost)
        best_category = self._find_optimal_category(
            original_category,
            original_confidence,
            all_costs,
            prob_dist
        )
        
        best_cost = all_costs[best_category]
        best_confidence = prob_dist.get(best_category, 0.0)
        
        # 5. Odluči status
        status, reasoning = self._decide_status(
            best_category,
            best_confidence,
            best_cost,
            original_category
        )
        
        return OptimizedDecision(
            predicted_category=best_category,
            confidence=best_confidence,
            expected_cost=best_cost,
            status=status,
            reasoning=reasoning,
            top3_predictions=top3_predictions,
            all_costs=all_costs,
            is_fallback=False,
            original_top1=original_category,
            original_confidence=original_confidence
        )
    
    def _build_probability_distribution(
        self,
        top3: List[Tuple[str, float]]
    ) -> Dict[WasteCategory, float]:
        """Napravi probability distribution od top3"""
        prob_dist = {}
        
        for cls, conf in top3:
            category = WasteCategory(cls)
            prob_dist[category] = conf
        
        return prob_dist
    
    def _find_optimal_category(
        self,
        original_category: WasteCategory,
        original_confidence: float,
        all_costs: Dict[WasteCategory, float],
        prob_dist: Dict[WasteCategory, float]
    ) -> WasteCategory:
        """
        Nađi optimalnu kategoriju koja minimizuje očekivani trošak.
        
        Balansira između:
        - Vjerojatnosti (høgher is better)
        - Expected cost (lower is better)
        
        Score = (1 - cost_weight) * confidence - cost_weight * cost
        """
        scores = {}
        
        for category in prob_dist.keys():
            confidence = prob_dist[category]
            cost = all_costs[category]
            
            # Weighted score (veća je bolja)
            score = (1 - self.cost_weight) * confidence - self.cost_weight * cost
            scores[category] = score
        
        # Nađi kategoriju sa najboljim score-om
        best_category = max(scores, key=scores.get)
        
        return best_category
    
    def _decide_status(
        self,
        category: WasteCategory,
        confidence: float,
        expected_cost: float,
        original_category: WasteCategory
    ) -> Tuple[ImageStatus, str]:
        """
        Odluči status slike.
        
        Returns:
            (ImageStatus, reasoning)
        """
        # 1. Low confidence → review
        if confidence < self.min_confidence_threshold:
            return (
                ImageStatus.PENDING_REVIEW,
                f"Low confidence ({confidence:.1%})"
            )
        
        # 2. High expected cost → review
        if expected_cost > self.max_acceptable_cost:
            return (
                ImageStatus.PENDING_REVIEW,
                f"High expected cost ({expected_cost:.2f})"
            )
        
        # 3. Changed from original → classified with note
        if category != original_category:
            return (
                ImageStatus.CLASSIFIED,
                f"Cost-optimized: {original_category.value}→{category.value}"
            )
        
        # 4. All good
        return (
            ImageStatus.CLASSIFIED,
            f"High confidence ({confidence:.1%})"
        )
    
    def _apply_fallback_low_confidence(
        self,
        original_category: WasteCategory,
        original_confidence: float,
        top3_predictions: List[Tuple[WasteCategory, float]]
    ) -> OptimizedDecision:
        """
        Fallback strategija za nizak confidence.
        
        Opcije:
        1. Confidence < 30% → Sigurno TRASH (safe default)
        2. Confidence 30-50% → PENDING_REVIEW
        """
        # Veoma nizak confidence → safe default (TRASH)
        if original_confidence < 0.30:
            return OptimizedDecision(
                predicted_category=WasteCategory.TRASH,
                confidence=original_confidence,
                expected_cost=999.0,  # Unknown
                status=ImageStatus.PENDING_REVIEW,
                reasoning="Very low confidence - fallback to TRASH",
                top3_predictions=top3_predictions,
                all_costs={},
                is_fallback=True,
                fallback_reason="confidence < 30%",
                original_top1=original_category,
                original_confidence=original_confidence
            )
        
        # Nizak confidence → review
        return OptimizedDecision(
            predicted_category=original_category,
            confidence=original_confidence,
            expected_cost=999.0,  # Unknown
            status=ImageStatus.PENDING_REVIEW,
            reasoning=f"Low confidence ({original_confidence:.1%}) - needs review",
            top3_predictions=top3_predictions,
            all_costs={},
            is_fallback=True,
            fallback_reason=f"confidence < {self.review_threshold:.0%}",
            original_top1=original_category,
            original_confidence=original_confidence
        )
    
    def print_decision(self, decision: OptimizedDecision):
        """Pretty print odluke (za debugging)"""
        print("\n🎯 OPTIMIZED DECISION")
        print("=" * 60)
        print(f"📦 Category: {decision.predicted_category.value}")
        print(f"✅ Confidence: {decision.confidence:.1%}")
        print(f"💰 Expected Cost: {decision.expected_cost:.2f}")
        print(f"📊 Status: {decision.status.value}")
        print(f"💭 Reasoning: {decision.reasoning}")
        
        if decision.is_fallback:
            print(f"⚠️  FALLBACK: {decision.fallback_reason}")
            print(f"   Original: {decision.original_top1.value} ({decision.original_confidence:.1%})")
        
        if decision.original_top1 != decision.predicted_category:
            print(f"🔄 Changed from original:")
            print(f"   {decision.original_top1.value} ({decision.original_confidence:.1%})")
            print(f"   → {decision.predicted_category.value} ({decision.confidence:.1%})")
        
        print("\n📈 Top 3 Predictions:")
        for i, (cat, conf) in enumerate(decision.top3_predictions[:3], 1):
            print(f"   {i}. {cat.value}: {conf:.1%}")
        
        if decision.all_costs:
            print("\n💵 Expected Costs:")
            sorted_costs = sorted(
                decision.all_costs.items(),
                key=lambda x: x[1]
            )
            for cat, cost in sorted_costs[:5]:
                marker = "✓" if cat == decision.predicted_category else " "
                print(f"   {marker} {cat.value}: {cost:.2f}")
        
        print("=" * 60)


if __name__ == "__main__":
    # Test
    optimizer = DecisionOptimizer(
        min_confidence_threshold=0.70,
        review_threshold=0.50,
        cost_weight=0.3
    )
    
    # Test case 1: High confidence metal
    print("\n🧪 TEST 1: High confidence METAL")
    result1 = {
        "class": "metal",
        "confidence": 0.85,
        "top3": [("metal", 0.85), ("plastic", 0.10), ("glass", 0.03)]
    }
    decision1 = optimizer.optimize_decision(result1)
    optimizer.print_decision(decision1)
    
    # Test case 2: Uncertain between metal and paper (KRITIČNO!)
    print("\n🧪 TEST 2: Uncertain METAL vs PAPER (HIGH COST!)")
    result2 = {
        "class": "metal",
        "confidence": 0.55,
        "top3": [("metal", 0.55), ("paper", 0.35), ("cardboard", 0.08)]
    }
    decision2 = optimizer.optimize_decision(result2)
    optimizer.print_decision(decision2)
    
    # Test case 3: Very low confidence
    print("\n🧪 TEST 3: Very low confidence")
    result3 = {
        "class": "plastic",
        "confidence": 0.25,
        "top3": [("plastic", 0.25), ("metal", 0.20), ("glass", 0.18)]
    }
    decision3 = optimizer.optimize_decision(result3)
    optimizer.print_decision(decision3)
