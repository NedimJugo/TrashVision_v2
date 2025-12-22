"""
Review Service

Servis za user review/feedback.
Agent koristi ovo za continuous learning.
"""

from datetime import datetime
from typing import Optional
from ...Domain import (
    WasteImage,
    Prediction,
    Review,
    ReviewStatus,
    WasteCategory,
    SystemSettings,
)


class ReviewService:
    """
    Servis za user reviews i feedback.
    
    Omogućava:
    - Kreiranje review-a
    - Čuvanje feedback-a za learning
    - Ažuriranje countera za retraining
    """
    
    def __init__(self, db_session, file_storage):
        """
        Args:
            db_session: SQLAlchemy session
            file_storage: File storage za čuvanje slika
        """
        self.db = db_session
        self.file_storage = file_storage
    
    async def submit_review(
        self,
        image: WasteImage,
        prediction: Prediction,
        user_confirmed_category: WasteCategory,
        comment: Optional[str] = None
    ) -> Review:
        """
        User review slike.
        
        Args:
            image: Slika
            prediction: Predikcija agenta
            user_confirmed_category: Kategorija koju user potvrdi
            comment: Opcioni komentar
        
        Returns:
            Review: Kreirani review
        """
        # Provjeri da li je user potvrdio ili ispravio
        was_correct = (prediction.predicted_category == user_confirmed_category)
        
        if was_correct:
            review_status = ReviewStatus.CORRECT
        else:
            review_status = ReviewStatus.CORRECTED
        
        review = Review(
            image_id=image.id,
            prediction_id=prediction.id,
            user_confirmed_category=user_confirmed_category,
            review_status=review_status,
            was_correct=was_correct,
            user_comment=comment,
            created_at=datetime.now()
        )
        
        # Save review
        print(f"✍️  Review: {review_status.value} - {user_confirmed_category.value}")
        
        return review
    
    async def add_to_learning_dataset(
        self,
        image: WasteImage,
        review: Review,
        settings: SystemSettings
    ) -> bool:
        """
        Dodaj sliku u learning dataset.
        
        Args:
            image: Slika
            review: Review sa correct labelom
            settings: System settings
        
        Returns:
            bool: True ako uspješno dodato
        
        LEARNING faza:
        1. Kopiraj sliku u data/new_samples/{category}/
        2. Increment counter
        3. Provjeri da li treba retraining
        """
        # 1. Kopiraj sliku
        target_category = review.user_confirmed_category
        new_path = await self.file_storage.copy_to_learning_set(
            image.filepath,
            target_category.value
        )
        
        # 2. Increment counter
        settings.increment_samples()
        
        print(f"📚 Added to learning dataset: {target_category.value}")
        print(f"   New samples: {settings.new_samples_count}/{settings.retrain_threshold}")
        
        # 3. Provjeri retraining threshold
        if settings.should_trigger_retraining():
            print(f"🔔 RETRAINING THRESHOLD REACHED!")
            return True
        
        return False
    
    async def get_review_for_image(self, image_id: int) -> Optional[Review]:
        """
        Vrati review za sliku.
        
        Args:
            image_id: ID slike
        
        Returns:
            Optional[Review]: Review ili None
        """
        # SELECT * FROM reviews WHERE image_id = ? ORDER BY created_at DESC LIMIT 1
        return None
    
    async def get_review_statistics(self) -> dict:
        """
        Statistika review-a.
        
        Returns:
            dict: Statistike
        """
        return {
            "total_reviews": 0,
            "correct_predictions": 0,
            "corrections": 0,
            "accuracy": 0.0,
        }
    
    async def get_pending_reviews(self, limit: int = 10) -> list:
        """
        Slike koje čekaju review.
        
        Args:
            limit: Max broj rezultata
        
        Returns:
            list: Lista (image, prediction) tuple-a
        """
        # SELECT * FROM images WHERE status = 'pending_review' LIMIT ?
        return []
    
    async def submit_user_feedback(
        self,
        file_path: str,
        predicted_class: Optional[str],
        actual_class: Optional[str],
        settings: SystemSettings
    ) -> dict:
        """
        Procesira user feedback i dodaje u learning dataset.
        
        Args:
            file_path: Path do slike (temp file)
            predicted_class: Agent predikcija
            actual_class: User korekcija
            settings: System settings za counter update
        
        Returns:
            dict: Rezultat sa statistikama i retraining info
        
        BIZNIS LOGIKA:
        1. Odredi target kategoriju (actual ili predicted)
        2. Kopiraj sliku u learning dataset
        3. Increment sample counter
        4. Persist counter u DB
        5. Provjeri retraining threshold
        """
        # 1. Odredi target kategoriju
        target_class_str = actual_class or predicted_class
        
        if not target_class_str:
            raise ValueError("No category provided")
        
        try:
            target_category = WasteCategory(target_class_str)
        except ValueError:
            raise ValueError(f"Invalid category: {target_class_str}")
        
        print(f"📝 User Feedback: {target_category.value}")
        
        # 2. Kopiraj u learning dataset
        await self.file_storage.copy_to_learning_set(
            file_path,
            target_category.value
        )
        
        print(f"✅ Added to learning dataset: {target_category.value}")
        
        # 3. Increment counter
        settings.increment_samples()
        
        # 4. Persist u DB
        from ...Infrastructure.database import SystemSettingsModel
        db_settings = self.db.query(SystemSettingsModel).first()
        if db_settings:
            db_settings.new_samples_count = settings.new_samples_count
            self.db.commit()
        
        # 5. Provjeri retraining threshold
        should_retrain = settings.should_trigger_retraining()
        
        print(f"📊 New samples: {settings.new_samples_count}/{settings.retrain_threshold}")
        
        if should_retrain:
            print(f"🔔 RETRAINING THRESHOLD REACHED!")
        
        return {
            "success": True,
            "message": "Hvala na feedbacku!",
            "category": target_category.value,
            "should_retrain": should_retrain,
            "new_samples_count": settings.new_samples_count,
            "threshold": settings.retrain_threshold,
            "progress_percentage": (
                settings.new_samples_count / settings.retrain_threshold * 100
            )
        }


if __name__ == "__main__":
    print("✅ ReviewService loaded")