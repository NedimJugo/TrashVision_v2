"""
AiAgents.TrashAgent.Web

Web sloj - tanak API host koji poziva agent runnere.

Komponente:
- main.py: FastAPI app + DI + agent startup
- controllers/: API endpoints (samo DTO mapping)
- workers/: Background agent loops
- dto/: Request/Response modeli
"""

__version__ = "2.0.0"