"""
TrashVision Agent Launcher

Pokreće FastAPI aplikaciju sa agent runner-ima.

Usage:
    python run_agent.py
"""

import sys
from pathlib import Path

# Dodaj root direktorij u Python path
root_dir = Path(__file__).parent
sys.path.insert(0, str(root_dir))

# Pokreni aplikaciju
if __name__ == "__main__":
    import uvicorn
    
    print("=" * 60)
    print("🚀 Starting TrashVision Agent...")
    print("=" * 60)
    
    # Import app direktno
    from AiAgents.TrashAgent.Web.main import app
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False,  # Set to True for development
        log_level="info"
    )