"""
Domain Enums

Sve enumeracije za TrashVision Domain.
Definišu legalne statusе, kategorije, i druge tipove.
"""

from enum import Enum


class ImageStatus(str, Enum):
    """
    Status slike kroz agent lifecycle.
    
    Flow:
    QUEUED → PROCESSING → CLASSIFIED → (PENDING_REVIEW) → REVIEWED
                        ↓
                      FAILED
    """
    QUEUED = "queued"                    # Uploaded, čeka na agent
    PROCESSING = "processing"            # Agent trenutno procesira
    CLASSIFIED = "classified"            # Klasifikovana (visok confidence)
    PENDING_REVIEW = "pending_review"    # Treba user review (nizak confidence)
    REVIEWED = "reviewed"                # User potvrdio/ispravio
    FAILED = "failed"                    # Greška pri procesiranju


class WasteCategory(str, Enum):
    """
    10 kategorija otpada.
    
    VAŽNO: Redoslijed MORA odgovarati YOLO model labelama!
    (Provijeri u data/processed/labels.txt)
    """
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
    
    @property
    def display_name(self) -> str:
        """User-friendly ime"""
        names = {
            "battery": "Battery (Baterija)",
            "biological": "Biological (Organski otpad)",
            "cardboard": "Cardboard (Karton)",
            "clothes": "Clothes (Odjeća)",
            "glass": "Glass (Staklo)",
            "metal": "Metal",
            "paper": "Paper (Papir)",
            "plastic": "Plastic (Plastika)",
            "shoes": "Shoes (Obuća)",
            "trash": "Trash (Mješoviti otpad)",
        }
        return names[self.value]
    
    @property
    def is_recyclable(self) -> bool:
        """Da li je reciklabilno"""
        return self != WasteCategory.TRASH
    
    @property
    def emoji(self) -> str:
        """Emoji za kategoriju"""
        emojis = {
            "battery": "🔋",
            "biological": "🌱",
            "cardboard": "📦",
            "clothes": "👕",
            "glass": "🍾",
            "metal": "🔩",
            "paper": "📄",
            "plastic": "🧴",
            "shoes": "👟",
            "trash": "🗑️",
        }
        return emojis[self.value]


class ReviewStatus(str, Enum):
    """Status user review-a"""
    CORRECT = "correct"      # User potvrdio da je predikcija tačna
    CORRECTED = "corrected"  # User ispravio predikciju
    SKIPPED = "skipped"      # User preskočio review


class ModelType(str, Enum):
    """Tip modela"""
    YOLOV8_NANO = "yolov8n"
    YOLOV8_SMALL = "yolov8s"
    YOLOV8_MEDIUM = "yolov8m"


class TrainingMode(str, Enum):
    """Mod treniranja modela"""
    INITIAL = "initial"          # Prvo treniranje od nule
    INCREMENTAL = "incremental"  # Fine-tuning na novim podacima
    FULL = "full"                # Potpuno retreniranje sa svim podacima


if __name__ == "__main__":
    # Test
    print("✅ Domain Enums loaded")
    print(f"   ImageStatus: {len(ImageStatus)} statusa")
    print(f"   WasteCategory: {len(WasteCategory)} kategorija")
    
    # Test enum properties
    plastic = WasteCategory.PLASTIC
    print(f"\n   {plastic.emoji} {plastic.display_name}")
    print(f"   Recyclable: {plastic.is_recyclable}")