"""
File Storage Helper

Helper za upravljanje fajlovima (slike, modeli).
"""

import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional


class FileStorage:
    """
    Helper za file storage operacije.
    
    Omogućava:
    - Čuvanje uploaded slika
    - Kopiranje u learning dataset
    - Organizaciju file strukture
    """
    
    def __init__(
        self,
        uploads_dir: str = "data/uploads",
        learning_dir: str = "data/new_samples",
        models_dir: str = "models"
    ):
        self.uploads_dir = Path(uploads_dir)
        self.learning_dir = Path(learning_dir)
        self.models_dir = Path(models_dir)
        
        # Kreiraj direktorije
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.learning_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
    
    async def save_uploaded_image(
        self,
        file_data: bytes,
        filename: str
    ) -> str:
        """
        Sačuvaj uploaded sliku.
        
        Args:
            file_data: Bytes slike
            filename: Original filename
        
        Returns:
            str: Putanja do sačuvane slike
        """
        # Generiši unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        ext = Path(filename).suffix
        new_filename = f"upload_{timestamp}{ext}"
        
        filepath = self.uploads_dir / new_filename
        
        # Sačuvaj
        with open(filepath, 'wb') as f:
            f.write(file_data)
        
        print(f"💾 Saved upload: {filepath}")
        
        return str(filepath)
    
    async def copy_to_learning_set(
        self,
        source_path: str,
        category: str
    ) -> str:
        """
        Kopiraj sliku u learning dataset.
        
        Args:
            source_path: Originalna slika
            category: Kategorija (npr. "plastic")
        
        Returns:
            str: Nova putanja
        """
        # Kreiraj category folder
        category_dir = self.learning_dir / category
        category_dir.mkdir(parents=True, exist_ok=True)
        
        # Generiši novi filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        ext = Path(source_path).suffix
        new_filename = f"{category}_{timestamp}{ext}"
        
        target_path = category_dir / new_filename
        
        # Kopiraj
        shutil.copy(source_path, target_path)
        
        print(f"📚 Copied to learning set: {target_path}")
        
        return str(target_path)
    
    async def count_learning_samples(self) -> dict[str, int]:
        """
        Broj uzoraka po kategoriji u learning datasetu.
        
        Returns:
            dict: {"plastic": 45, "glass": 12, ...}
        """
        counts = {}
        
        for category_dir in self.learning_dir.iterdir():
            if category_dir.is_dir():
                count = len(list(category_dir.glob("*.jpg"))) + len(list(category_dir.glob("*.png")))
                counts[category_dir.name] = count
        
        return counts
    
    async def get_total_learning_samples(self) -> int:
        """
        Ukupan broj uzoraka u learning datasetu.
        
        Returns:
            int: Total count
        """
        counts = await self.count_learning_samples()
        return sum(counts.values())
    
    async def archive_learning_samples(self) -> str:
        """
        Arhiviraj learning samples (nakon retraining-a).
        
        Returns:
            str: Putanja do arhive
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_dir = Path("data/archived_samples") / timestamp
        
        # Pomjeri folder
        if self.learning_dir.exists():
            shutil.move(str(self.learning_dir), str(archive_dir))
            print(f"📦 Archived learning samples: {archive_dir}")
        
        # Kreiraj novi prazan folder
        self.learning_dir.mkdir(parents=True, exist_ok=True)
        
        return str(archive_dir)
    
    async def get_model_path(self, version: int) -> Optional[str]:
        """
        Putanja do modela za verziju.
        
        Args:
            version: Broj verzije
        
        Returns:
            Optional[str]: Putanja ili None ako ne postoji
        """
        # models/trashvision_v{version}/weights/best.pt
        model_path = self.models_dir / f"trashvision_v{version}" / "weights" / "best.pt"
        
        if model_path.exists():
            return str(model_path)
        
        return None
    
    async def cleanup_old_uploads(self, days: int = 7):
        """
        Obriši stare uploaded fajlove.
        
        Args:
            days: Briši fajlove starije od X dana
        """
        cutoff = datetime.now().timestamp() - (days * 24 * 60 * 60)
        deleted = 0
        
        for file in self.uploads_dir.glob("*"):
            if file.is_file() and file.stat().st_mtime < cutoff:
                file.unlink()
                deleted += 1
        
        if deleted > 0:
            print(f"🧹 Cleaned up {deleted} old uploads")


if __name__ == "__main__":
    import asyncio
    
    async def test():
        storage = FileStorage()
        
        # Test counts
        counts = await storage.count_learning_samples()
        print(f"✅ Learning samples: {counts}")
        
        total = await storage.get_total_learning_samples()
        print(f"   Total: {total}")
    
    asyncio.run(test())