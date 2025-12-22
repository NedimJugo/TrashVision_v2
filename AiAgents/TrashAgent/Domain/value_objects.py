"""
Domain Value Objects

Immutable objekti koji predstavljaju koncepte bez identiteta.
Value objects se porede po sadržaju, ne po ID-u.
"""

from dataclasses import dataclass
from typing import Optional
from .enums import WasteCategory, ImageStatus


@dataclass(frozen=True)
class RecyclingInfo:
    """
    Informacije o reciklaži za jednu kategoriju.
    
    Immutable - biznis pravila koja se ne mijenjaju često.
    """
    category: WasteCategory
    disposal_instruction: str
    container_color: Optional[str] = None
    special_notes: Optional[str] = None
    
    @property
    def is_recyclable(self) -> bool:
        return self.category.is_recyclable
    
    @staticmethod
    def get_for_category(category: WasteCategory) -> "RecyclingInfo":
        """Factory metoda - vrati recycling info za kategoriju"""
        
        recycling_map = {
            WasteCategory.BATTERY: RecyclingInfo(
                category=WasteCategory.BATTERY,
                disposal_instruction="Poseban kontejner za baterije ili reciklažno dvorište",
                container_color="orange",
                special_notes="OPASNO: Ne bacati u opći otpad!"
            ),
            WasteCategory.BIOLOGICAL: RecyclingInfo(
                category=WasteCategory.BIOLOGICAL,
                disposal_instruction="Braon/zelena kanta za kompost",
                container_color="brown",
                special_notes="Idealno za kompostiranje"
            ),
            WasteCategory.CARDBOARD: RecyclingInfo(
                category=WasteCategory.CARDBOARD,
                disposal_instruction="Plavi kontejner za papir i karton",
                container_color="blue",
                special_notes="Spljoštiti prije odlaganja"
            ),
            WasteCategory.CLOTHES: RecyclingInfo(
                category=WasteCategory.CLOTHES,
                disposal_instruction="Donirati ili odvesti u kontejner za tekstil",
                container_color="white",
                special_notes="Čista odjeća se može donirati"
            ),
            WasteCategory.GLASS: RecyclingInfo(
                category=WasteCategory.GLASS,
                disposal_instruction="Zeleni kontejner za staklo",
                container_color="green",
                special_notes="Odvojiti poklopce"
            ),
            WasteCategory.METAL: RecyclingInfo(
                category=WasteCategory.METAL,
                disposal_instruction="Žuti kontejner za metal",
                container_color="yellow",
                special_notes="Limenke isprati prije odlaganja"
            ),
            WasteCategory.PAPER: RecyclingInfo(
                category=WasteCategory.PAPER,
                disposal_instruction="Plavi kontejner za papir",
                container_color="blue",
                special_notes="Bez masnoće i vlage"
            ),
            WasteCategory.PLASTIC: RecyclingInfo(
                category=WasteCategory.PLASTIC,
                disposal_instruction="Žuti kontejner za plastiku",
                container_color="yellow",
                special_notes="Provjeriti oznaku reciklaže (PET, HDPE...)"
            ),
            WasteCategory.SHOES: RecyclingInfo(
                category=WasteCategory.SHOES,
                disposal_instruction="Donirati ili odvesti u kontejner za tekstil",
                container_color="white",
                special_notes="Dobra obuća se može donirati"
            ),
            WasteCategory.TRASH: RecyclingInfo(
                category=WasteCategory.TRASH,
                disposal_instruction="Crna/siva kanta za opći otpad",
                container_color="gray",
                special_notes="NE MOŽE se reciklirati"
            ),
        }
        
        return recycling_map[category]


@dataclass(frozen=True)
class ClassificationDecision:
    """
    Odluka agenta nakon klasifikacije.
    
    Sadrži:
    - Predviđenu kategoriju
    - Confidence
    - Status (da li ide na review ili ne)
    """
    predicted_category: WasteCategory
    confidence: float
    new_status: ImageStatus
    top3_predictions: list[tuple[WasteCategory, float]]
    
    def __post_init__(self):
        """Validacija"""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence mora biti 0-1, dobio: {self.confidence}")
        
        if self.new_status not in [ImageStatus.CLASSIFIED, ImageStatus.PENDING_REVIEW]:
            raise ValueError(f"new_status mora biti CLASSIFIED ili PENDING_REVIEW")
    
    @property
    def needs_review(self) -> bool:
        """Da li treba user review"""
        return self.new_status == ImageStatus.PENDING_REVIEW
    
    @property
    def is_confident(self) -> bool:
        """Da li je agent siguran u predikciju (>70%)"""
        return self.confidence >= 0.70


@dataclass(frozen=True)
class TrainingDecision:
    """
    Odluka learning agenta da li treba retrenirati model.
    """
    should_retrain: bool
    new_samples_count: int
    threshold: int
    mode: str  # "incremental" ili "full"
    reason: str
    
    @property
    def progress_percentage(self) -> float:
        """Koliko % do retraining-a"""
        return min(100.0, (self.new_samples_count / self.threshold) * 100)


if __name__ == "__main__":
    # Test
    print("✅ Domain Value Objects loaded")
    
    # Test RecyclingInfo
    plastic_info = RecyclingInfo.get_for_category(WasteCategory.PLASTIC)
    print(f"\n   {WasteCategory.PLASTIC.emoji} {WasteCategory.PLASTIC.display_name}")
    print(f"   Disposal: {plastic_info.disposal_instruction}")
    print(f"   Container: {plastic_info.container_color}")
    
    # Test ClassificationDecision
    decision = ClassificationDecision(
        predicted_category=WasteCategory.PLASTIC,
        confidence=0.92,
        new_status=ImageStatus.CLASSIFIED,
        top3_predictions=[]
    )
    print(f"\n   Decision: {decision.predicted_category} ({decision.confidence:.0%})")
    print(f"   Needs review: {decision.needs_review}")