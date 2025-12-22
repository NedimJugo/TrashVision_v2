"""
Debug script - provjeri strukturu projekta
"""

import os
from pathlib import Path

def check_structure():
    """Provjeri da li svi potrebni fajlovi postoje"""
    
    root = Path(".")
    
    required_files = [
        # Core
        "AiAgents/Core/__init__.py",
        "AiAgents/Core/software_agent.py",
        "AiAgents/Core/perception_source.py",
        "AiAgents/Core/policy.py",
        "AiAgents/Core/actuator.py",
        "AiAgents/Core/learning_component.py",
        
        # Domain
        "AiAgents/TrashAgent/Domain/__init__.py",
        "AiAgents/TrashAgent/Domain/enums.py",
        "AiAgents/TrashAgent/Domain/entities.py",
        "AiAgents/TrashAgent/Domain/value_objects.py",
        
        # Application - Services
        "AiAgents/TrashAgent/Application/__init__.py",
        "AiAgents/TrashAgent/Application/Services/__init__.py",
        "AiAgents/TrashAgent/Application/Services/queue_service.py",
        "AiAgents/TrashAgent/Application/Services/classification_service.py",
        "AiAgents/TrashAgent/Application/Services/review_service.py",
        "AiAgents/TrashAgent/Application/Services/training_service.py",
        
        # Application - Agents
        "AiAgents/TrashAgent/Application/Agents/__init__.py",
        "AiAgents/TrashAgent/Application/Agents/classification_runner.py",
        "AiAgents/TrashAgent/Application/Agents/learning_runner.py",
        
        # Infrastructure
        "AiAgents/TrashAgent/Infrastructure/__init__.py",
        "AiAgents/TrashAgent/Infrastructure/database.py",
        "AiAgents/TrashAgent/Infrastructure/waste_classifier.py",
        "AiAgents/TrashAgent/Infrastructure/yolo_classifier.py",
        "AiAgents/TrashAgent/Infrastructure/file_storage.py",
        
        # Web
        "AiAgents/TrashAgent/Web/__init__.py",
        "AiAgents/TrashAgent/Web/main.py",
        "AiAgents/TrashAgent/Web/dto/__init__.py",
        "AiAgents/TrashAgent/Web/dto/responses.py",
        "AiAgents/TrashAgent/Web/controllers/__init__.py",
        "AiAgents/TrashAgent/Web/controllers/prediction_controller.py",
        "AiAgents/TrashAgent/Web/controllers/learning_controller.py",
        "AiAgents/TrashAgent/Web/workers/__init__.py",
        "AiAgents/TrashAgent/Web/workers/classification_worker.py",
        "AiAgents/TrashAgent/Web/workers/learning_worker.py",
    ]
    
    print("🔍 Checking project structure...\n")
    
    missing = []
    existing = []
    
    for file_path in required_files:
        full_path = root / file_path
        if full_path.exists():
            existing.append(file_path)
            print(f"✅ {file_path}")
        else:
            missing.append(file_path)
            print(f"❌ MISSING: {file_path}")
    
    print("\n" + "=" * 60)
    print(f"✅ Existing: {len(existing)}/{len(required_files)}")
    print(f"❌ Missing: {len(missing)}/{len(required_files)}")
    print("=" * 60)
    
    if missing:
        print("\n⚠️  Missing files:")
        for file in missing:
            print(f"   - {file}")
        return False
    else:
        print("\n🎉 All files present!")
        return True

if __name__ == "__main__":
    check_structure()