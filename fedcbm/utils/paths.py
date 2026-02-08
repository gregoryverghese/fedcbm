"""Path utilities with environment variable support."""
import os
from pathlib import Path
from typing import Optional

def get_project_root() -> Path:
    """Get the project root directory."""
    # Try environment variable first
    root = os.getenv("MINOTAUR_ROOT")
    if root:
        return Path(root)
    
    # Fallback: assume we're in MINOTAUR/MINOTAUR, go up one level
    current = Path(__file__).resolve()
    # Go up from minotaur/utils/paths.py -> minotaur/utils -> minotaur -> MINOTAUR -> MINOTAUR
    # Actually, if installed as package, this might be different
    # For now, use a simple heuristic
    if "MINOTAUR" in str(current):
        parts = current.parts
        try:
            minotaur_idx = next(i for i, p in enumerate(parts) if p == "MINOTAUR")
            return Path(*parts[:minotaur_idx + 1])
        except StopIteration:
            pass
    
    # Last resort: current directory
    return Path.cwd()

PROJECT_ROOT = get_project_root()

def get_db_path() -> Optional[Path]:
    """Get database path from environment or None."""
    db_path = os.getenv("DB_PATH")
    return Path(db_path) if db_path else None


