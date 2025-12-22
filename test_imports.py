"""
Test Imports - Provjeri da li svi moduli rade

Koristi ovo PRIJE pokretanja aplikacije da provjeriš da li sve radi.
"""

import sys
from pathlib import Path

# Dodaj root u path
root = Path(__file__).parent
sys.path.insert(0, str(root))

print("=" * 60)
print("🧪 TESTING IMPORTS...")
print("=" * 60)

tests = []

# Test 1: Core
print("\n1️⃣  Testing Core...")
try:
    from AiAgents.Core import SoftwareAgent, LearningAgent
    print("   ✅ Core.SoftwareAgent")
    print("   ✅ Core.LearningAgent")
    tests.append(("Core", True))
except Exception as e:
    print(f"   ❌ Core: {e}")
    tests.append(("Core", False))

# Test 2: Domain
print("\n2️⃣  Testing Domain...")
try:
    from AiAgents.TrashAgent.Domain import WasteImage, WasteCategory, ImageStatus
    print("   ✅ Domain.WasteImage")
    print("   ✅ Domain.WasteCategory")
    print("   ✅ Domain.ImageStatus")
    tests.append(("Domain", True))
except Exception as e:
    print(f"   ❌ Domain: {e}")
    tests.append(("Domain", False))

# Test 3: Application - Services
print("\n3️⃣  Testing Application.Services...")
try:
    from AiAgents.TrashAgent.Application.Services import (
        QueueService,
        ClassificationService,
        ReviewService,
        TrainingService
    )
    print("   ✅ Services.QueueService")
    print("   ✅ Services.ClassificationService")
    print("   ✅ Services.ReviewService")
    print("   ✅ Services.TrainingService")
    tests.append(("Services", True))
except Exception as e:
    print(f"   ❌ Services: {e}")
    tests.append(("Services", False))

# Test 4: Application - Agents (NAJVAŽNIJE!)
print("\n4️⃣  Testing Application.Agents (KLJUČNO!)...")
try:
    from AiAgents.TrashAgent.Application.Agents import (
        ClassificationAgentRunner,
        LearningAgentRunner
    )
    print("   ✅ Agents.ClassificationAgentRunner")
    print("   ✅ Agents.LearningAgentRunner")
    tests.append(("Agents", True))
except Exception as e:
    print(f"   ❌ Agents: {e}")
    tests.append(("Agents", False))

# Test 5: Infrastructure
print("\n5️⃣  Testing Infrastructure...")
try:
    from AiAgents.TrashAgent.Infrastructure import (
        DatabaseHelper,
        YoloWasteClassifier,
        FileStorage
    )
    print("   ✅ Infrastructure.DatabaseHelper")
    print("   ✅ Infrastructure.YoloWasteClassifier")
    print("   ✅ Infrastructure.FileStorage")
    tests.append(("Infrastructure", True))
except Exception as e:
    print(f"   ❌ Infrastructure: {e}")
    tests.append(("Infrastructure", False))

# Test 6: Web - Controllers
print("\n6️⃣  Testing Web.Controllers...")
try:
    from AiAgents.TrashAgent.Web.controllers import (
        prediction_router,
        learning_router
    )
    print("   ✅ Controllers.prediction_router")
    print("   ✅ Controllers.learning_router")
    tests.append(("Controllers", True))
except Exception as e:
    print(f"   ❌ Controllers: {e}")
    tests.append(("Controllers", False))

# Test 7: Web - Workers (KLJUČNO!)
print("\n7️⃣  Testing Web.Workers (KLJUČNO!)...")
try:
    from AiAgents.TrashAgent.Web.workers import (
        ClassificationWorker,
        LearningWorker
    )
    print("   ✅ Workers.ClassificationWorker")
    print("   ✅ Workers.LearningWorker")
    tests.append(("Workers", True))
except Exception as e:
    print(f"   ❌ Workers: {e}")
    tests.append(("Workers", False))

# Test 8: Web - Main App
print("\n8️⃣  Testing Web.Main...")
try:
    from AiAgents.TrashAgent.Web.main import app
    print("   ✅ Main.app (FastAPI)")
    tests.append(("Main", True))
except Exception as e:
    print(f"   ❌ Main: {e}")
    tests.append(("Main", False))

# Summary
print("\n" + "=" * 60)
print("📊 SUMMARY")
print("=" * 60)

passed = sum(1 for _, result in tests if result)
total = len(tests)

for name, result in tests:
    status = "✅" if result else "❌"
    print(f"{status} {name}")

print("\n" + "=" * 60)

if passed == total:
    print(f"🎉 ALL TESTS PASSED! ({passed}/{total})")
    print("=" * 60)
    print("\n✅ Ready to run: python run_agent.py")
else:
    print(f"⚠️  {total - passed} TESTS FAILED ({passed}/{total} passed)")
    print("=" * 60)
    print("\n❌ Fix errors above before running!")

print()