"""
Direct Server Runner for Trial 1 (Sub-200ms Pipeline)
Can be run with: python trial_1/run_server.py
"""

import sys
from pathlib import Path
import uvicorn

# Add project root and trial_1 directory to sys.path
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(BASE_DIR))

if __name__ == "__main__":
    uvicorn.run("trial_1.app.main:app", host="0.0.0.0", port=8001, reload=True)
